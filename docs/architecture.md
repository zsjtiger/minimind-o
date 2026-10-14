# Baize-V 工程结构

本文说明 `baize-v` 仓库的目录划分、各模块职责和依赖方向。

## 目录总览

```text
baize-v/                        （仓库目录名沿用 minimind-o）
├── baize_v/                    核心 Python 包（发行名 baize-v，import 名 baize_v）
│   ├── config.py               MiniMindConfig / OmniConfig
│   ├── paths.py                项目路径常量，统一以仓库根为基准
│   ├── models/                 网络结构
│   │   ├── minimind.py         基础文本 LLM（Attention、MoE、Block、CausalLM）
│   │   ├── talker.py           Talker（音频 codec 码本的自回归头）
│   │   ├── projectors.py       音频 / 视觉 projector
│   │   ├── encoders.py         SenseVoice 特征前端的包装
│   │   └── omni.py             BaizeOmni：thinker + talker 的完整模型与生成逻辑
│   ├── data/                   数据处理
│   │   ├── dataset.py          OmniDataset（parquet → 训练样本）
│   │   ├── chat.py             系统提示注入、思考块后处理
│   │   ├── labels.py           文本 / 音频标签构造、scheduled sampling
│   │   ├── augment.py          音频 / fbank 数据增强
│   │   └── collate.py          batch 拼接
│   ├── train/                  训练支撑
│   │   ├── init.py             模型初始化、加载权重、冻结策略、参数统计
│   │   ├── loop.py             单个 epoch 的训练循环
│   │   ├── losses.py           文本 + 音频 loss
│   │   ├── checkpoint.py       checkpoint 保存 / 续训读取
│   │   ├── schedule.py         学习率曲线、跳过已训 batch 的 sampler
│   │   └── dist.py             分布式初始化、随机种子、日志
│   └── inference/              推理支撑
│       ├── realtime.py         Silero VAD 与实时会话状态机
│       └── loader.py           扫描目录中的 transformers 模型
├── scripts/                    命令行入口（薄封装，只做参数解析）
│   ├── train_sft.py            SFT 训练
│   ├── evaluate.py             离线评测（文本 / 语音 / 图像样例）
│   ├── convert.py              PyTorch ⇄ transformers 权重互转
│   └── recipes/                分阶段训练配方（mini / full_dense / full_moe）
├── apps/web/                   Web 应用（Flask + WebSocket，支持实时语音通话）
│   ├── server.py
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

- `models/` 不依赖 `train/`、`data/`、`inference/`，可以单独用于推理或导出。
- `inference/realtime.py` 只依赖 `onnxruntime`，与模型解耦。
- 入口脚本不包含业务逻辑，便于在 notebook 或其他服务中直接调用 `baize_v`。

## 关键约定

- **路径**：所有默认路径来自 `baize_v/paths.py`，以仓库根目录为基准，与当前工作目录无关。外部权重目录可用环境变量 `BAIZE_PRETRAINED_DIR` 覆盖。
- **权重键名**：`thinker`、`talker`、`audio_proj`、`vision_proj`、`lm_head` 等属性名属于 checkpoint 格式的一部分，不能随意重命名。`tests/fixtures/omni_tiny_state_dict.json` 记录了完整键名与形状，`tests/test_model.py` 会校验。
- **模型类型**：`OmniConfig.model_type = "minimind-o"` 保持不变，已导出的 transformers 模型依赖它识别。
