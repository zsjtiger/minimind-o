# Baize-V 工程结构

本文说明 `baize-v` 仓库的目录划分、各模块职责和依赖方向。

## 目录总览

```text
baize-v/                        （仓库目录名沿用 minimind-o）
├── baize_v/                    核心 Python 包（发行名 baize-v，import 名 baize_v）
│   ├── config.py               BaizeConfig / OmniConfig（结构参数）
│   ├── paths.py                项目路径常量，统一以仓库根为基准
│   ├── tokens.py               音频 token 约定（pad / stop / spk / 词表大小）
│   ├── models/                 网络结构
│   │   ├── llm.py              基础文本 LLM（Attention、MoE、Block、CausalLM）
│   │   ├── talker.py           Talker（音频 codec 码本的自回归头）
│   │   ├── projectors.py       音频 / 视觉 projector
│   │   ├── encoders.py         SenseVoice / SigLIP 的加载与特征前端包装
│   │   ├── fusion.py           音频 / 视觉特征注入到文本 embedding 的占位区间
│   │   ├── summary.py          参数量统计、load_state_dict 结果报告
│   │   └── omni.py             BaizeOmni：thinker + talker 的完整模型
│   ├── data/                   数据处理
│   │   ├── dataset.py          OmniDataset（parquet → 训练样本）
│   │   ├── chat.py             系统提示注入、思考块后处理
│   │   ├── labels.py           文本标签构造、scheduled sampling
│   │   ├── augment.py          音频 / fbank 数据增强
│   │   └── collate.py          batch 拼接
│   ├── train/                  训练支撑
│   │   ├── init.py             模型初始化、加载权重、冻结策略
│   │   ├── loop.py             TrainContext 与单个 epoch 的训练循环
│   │   ├── losses.py           文本 + 音频 loss
│   │   ├── checkpoint.py       续训文件的保存 / 读取
│   │   ├── schedule.py         学习率曲线、跳过已训 batch 的 sampler
│   │   └── dist.py             分布式初始化、随机种子、日志
│   └── inference/              推理支撑
│       ├── generate.py         流式生成循环（文本 token + 8 层音频码）
│       ├── mimi.py             Mimi 码本整理与解码（Web / Gradio / 评测共用）
│       ├── realtime.py         Silero VAD 与实时会话状态机
│       └── loader.py           扫描 transformers 模型、加载 Omni 模型
├── scripts/                    命令行入口（薄封装，只做参数解析）
│   ├── train_sft.py            SFT 训练
│   ├── evaluate.py             离线评测（6 种模式，每种一个函数）
│   ├── convert.py              PyTorch ⇄ transformers 权重互转
│   └── recipes/                分阶段训练配方（mini / full_dense / full_moe）
├── apps/web/                   Web 应用（Flask + WebSocket，支持实时语音通话）
│   ├── server.py               路由与 RealtimeConnection
│   ├── voices.py               音色注册表（内置 / 未见 / 克隆）
│   ├── consent.py              音色克隆授权确认与审计日志
│   └── static/web_demo.html
├── examples/gradio_demo.py     Gradio 非实时示例
├── tests/                      pytest 测试
├── assets/                     随仓库分发的小资产（tokenizer、VAD、音色、评测音频）
├── docs/                       文档与图片
├── pyproject.toml              包元数据、依赖、dependency-groups、pytest / ruff 配置
├── uv.lock                     uv 生成的完整依赖锁文件
└── .python-version             uv 使用的 Python 版本
```

运行时产生或需要下载的目录（已在 `.gitignore` 中忽略）：

| 目录 | 内容 |
| --- | --- |
| `pretrained/` | SenseVoice-Small、SigLIP2、Mimi、CAM++ 等外部预训练权重 |
| `out/` | 训练输出的权重、导出的 transformers 模型、评测音频 |
| `checkpoints/` | 续训用的完整状态（模型 + 优化器 + step） |
| `dataset/` | 训练用 parquet 数据 |

## 依赖方向

依赖只能自上而下流动，下层不能 import 上层：

```text
scripts/ , apps/ , examples/
        │
        ▼
baize_v.train  ·  baize_v.inference  ·  baize_v.data
        │                 │                 │
        └────────► baize_v.models ◄─────────┘
                          │
                  baize_v.config / baize_v.paths
```

- `models/` 不依赖 `train/`、`data/`、`inference/`，可以单独用于推理或导出。`inference/generate.py` 依赖 `models` 之外的任何模块，只接收模型对象。
- `inference/realtime.py` 只依赖 `onnxruntime`，与模型解耦。
- 入口脚本不包含业务逻辑，便于在 notebook 或其他服务中直接调用 `baize_v`。

## 主干：循环深度 Transformer（RDT）

文本主干（thinker）采用循环深度结构，参考 [LLMBaize/baize](https://github.com/LLMBaize/baize) 的 Recurrent-Depth Transformer 实现：

```text
tokens → embed → [Prelude × P] → e ──┐
                                      ▼
                     ┌──── Recurrent Block（共享权重，循环 T 圈）────┐
                     │  h ← A·h + B·e + Block(RMSNorm(h+e)) + LoRA_t(RMSNorm(h+e))  │
                     └───────────────────────────────────────────┘
                                      ▼ bridge（循环块输出，供 talker 使用）
                                [Coda × C] → final norm → lm_head
```

| 机制 | 实现 | 位置 |
| --- | --- | --- |
| 输入注入 | Prelude 输出 `e` 在每圈重新注入，防止隐状态随圈数漂移 | `RecurrentBlock.forward` |
| LTI 稳定注入 | `A = exp(-exp(log_dt + log_A)) ∈ (0,1)`，谱半径小于 1 由参数化保证 | `LTIInjection` |
| 圈数嵌入 | 正弦嵌入加到前 D/8 通道，让共享权重在不同圈执行不同功能 | `loop_index_embedding` |
| 深度 LoRA | 跨圈共享 `down` / `B`，每圈一个 scale；超过训练圈数时 scale 钳制到最后一圈 | `LoRAAdapter` |
| ACT 早停 | 每位置预测停机概率并加权累积各圈隐状态；ponder cost 鼓励早停 | `ACTHalting`，`act_*` 配置 |
| 每圈 KV cache | 扁平列表顺序为 `[prelude…, 第 0..T-1 圈, coda…]`，解码时各圈各自缓存 | `BaizeModel.backbone` |

与 LLMBaize 的差异：

- 注意力：`attn_type="gqa"`（默认，分组查询注意力）或 `attn_type="mla"`（多头潜变量注意力，参考 LLMBaize 的 MLAttention）。MLA 只缓存压缩隐变量 `c_kv` 与共享的 RoPE key，缓存量为 `kv_lora_rank + qk_rope_head_dim`，接口与 GQA 一致，因此循环块与 KV cache 的逻辑不变。
- FFN：稠密 SwiGLU，或 MoE（由 `use_moe` 决定，循环块与 Prelude/Coda 一致）。
- 激活重计算：`--grad_checkpoint 1`，训练时按层、按圈重算中间激活。
- 分布式：DDP，或 `--fsdp 1` 使用 FSDP2 逐块分片参数、梯度与优化器状态（需 `torchrun` 启动，`--dtype` 不能用 float16）。

talker 的桥接信号由“中间层输出”改为“循环块输出”，talker 本身仍是标准的堆叠层。

配置项见 `BaizeConfig`：`prelude_layers`（P）、`coda_layers`（C）、`max_loop_iters`（T）、`lora_rank`、`use_act`、`act_threshold`、`act_init_bias`、`act_ponder_coef`、`loop_emb_frac`。

## 关键约定

- **路径**：所有默认路径来自 `baize_v/paths.py`，以仓库根目录为基准，与当前工作目录无关。外部权重目录可用环境变量 `BAIZE_PRETRAINED_DIR` 覆盖。
- **权重键名**：`thinker`、`talker`、`audio_proj`、`vision_proj`、`lm_head` 等属性名属于 checkpoint 格式的一部分，不能随意重命名。thinker 的键名为 `model.prelude.*`、`model.recurrent.*`、`model.coda.*`。`tests/fixtures/omni_tiny_state_dict.json` 记录了完整键名与形状，`tests/test_model.py` 会校验。
- **模型类型**：`OmniConfig.model_type = "baize_v"` 保持不变，已导出的 transformers 模型依赖它识别。
