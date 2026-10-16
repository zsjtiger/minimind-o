# 用开源语言模型热启动多模态模型（BaizeOmni）

本文说明如何用 Qwen3 / Llama 风格的开源因果语言模型初始化 BaizeOmni 的 thinker，并继续训练多模态能力。
相关代码：`baize_v/models/init_from_lm.py`、`scripts/init_from_lm.py`、`tests/test_init_from_lm.py`。

## 1. 结论先行

- 这是**近似热启动**，不是等价迁移。初始化后的输出与源模型不一致，必须经过训练才能使用。
- 源模型的文本能力通过 prelude / 循环块 / coda 三段进入模型，循环块共享同一份权重。
- 音频、图像占位符会追加到词表，多模态投影、音频/视觉编码器、talker 不来自源模型。
- 当前仓库的测试只用本地随机模型验证映射正确性，**没有用真实开源权重验证效果**。

## 2. 实现逻辑

整体流程分四步：

```
源 HF 模型 ──► ① 配置对齐 ──► ② 词表补齐 ──► ③ 层映射 ──► ④ 加载到 BaizeOmni ──► 保存 pth / json / tokenizer
```

### ① 配置对齐 `omni_config_from_hf`

源模型的注意力和 MLP 形状必须与 BaizeOmni 一致，否则权重无法装入。以下字段直接从源配置复制：

| 字段 | 来源 |
| --- | --- |
| `hidden_size`、`intermediate_size` | 源 `hidden_size`、`intermediate_size` |
| `num_attention_heads`、`num_key_value_heads`、`head_dim` | 源配置（`head_dim` 缺省时为 hidden/heads） |
| `rms_norm_eps`、`rope_theta`、`max_position_embeddings` | 源配置 |
| `tie_word_embeddings` | 源配置 |

以下字段由用户指定：`prelude_layers`（默认 2）、`coda_layers`（默认 2）、`max_loop_iters`（默认 4）。

源层数必须至少为 `prelude + coda + 1`，否则报错。

### ② 词表补齐 `ensure_omni_tokens`

BaizeOmni 需要两个占位符 `<|audio_pad|>`、`<|image_pad|>`。若源 tokenizer 中不存在，则追加到词表末尾。
`resize_rows` 把嵌入矩阵扩展到新词表大小，新行用原嵌入的均值初始化。`lm_head` 若不与嵌入绑定，同样扩展。

### ③ 层映射 `lm_state_to_omni`

复用仓库已有的 `remap_legacy_state_dict`：

| 源层 | 目标 |
| --- | --- |
| 第 0 .. P-1 层 | `model.prelude.*`，各自独立 |
| 第 P 层 | `model.recurrent.*`，循环块的**共享权重**，每圈都使用 |
| 最后 C 层 | `model.coda.*` |
| 中间层 | 丢弃 |

这意味着源模型的中间层不参与初始化，循环块则把源第 P 层的权重复用于多圈迭代。

### ④ 加载 `init_omni_from_lm`

用 `BaizeOmni(config)` 建模型，编码器路径设为不存在，因此音频、视觉编码器不加载。
然后 `load_state_dict(strict=False)`，返回的 `missing_keys` 与 `unexpected_keys` 用于检查映射是否完整。

### 为什么是近似

循环块的单圈更新为：

```
h' = A·h + B·e + f,   A = exp(-exp(log_dt + log_A)),   初始 A ≈ 0.37,  B = 0.1
```

标准残差层是 `h' = h + f`。两者不等价，因此即使权重完全映射，输出也不会与源模型一致。
另外 LTI（A、B、log_dt）、LoRA、ACT、循环块自带的 norm 在源模型中没有对应物，保持随机或默认初始化。

## 3. 使用说明

### 3.1 环境

```bash
uv sync                      # 依赖见 pyproject.toml
```

需要能访问模型源（Hugging Face 或本地目录）。训练需要 GPU，推荐显存 24 GB 以上，视 `max_loop_iters` 与序列长度而定。

### 3.2 生成热启动权重

```bash
uv run python scripts/init_from_lm.py \
    --base Qwen/Qwen3-0.6B \
    --save_weight omni_init \
    --prelude_layers 2 --coda_layers 2 --max_loop_iters 4 \
    --dtype bfloat16
```

脚本会打印 load 结果（缺失/多余的键），并输出：

| 产物 | 说明 |
| --- | --- |
| `out/omni_init_<hidden>.pth` | fp16 权重，可作为 pretrain / SFT 的 `--from_weight` |
| `out/omni_init_<hidden>.json` | OmniConfig，训练结构参数以它为准 |
| `out/omni_init_tokenizer/` | 补齐占位符后的 tokenizer |

脚本最后会打印对应的训练参数，例如：

```
--hidden_size 1024 --prelude_layers 2 --coda_layers 2 --max_loop_iters 4 --from_weight omni_init
```

### 3.3 接入训练（需要手动处理 tokenizer）

**注意**：`scripts/pretrain.py`、`scripts/dpo.py` 以及 SFT 的 `init_omni_model` 默认从 `assets/tokenizer`（`baize_v/paths.py` 中的 `TOKENIZER_DIR`）加载 tokenizer。热启动产生的 tokenizer 位于 `out/omni_init_tokenizer/`，两者不一致时：

- 词表大小与嵌入不匹配，加载或训练会出错；
- 占位符 id 错误，图像和音频注入会落到错误的位置。

在把训练脚本指向新 tokenizer 之前，需要：

```bash
rm -rf assets/tokenizer && cp -r out/omni_init_tokenizer assets/tokenizer    # 临时方案
```

更稳妥的做法是给训练脚本增加 `--tokenizer` 参数，这一改动尚未完成。

训练命令示例：

```bash
# 纯文本继续预训练
uv run python scripts/pretrain.py --data dataset/pretrain.jsonl --save_weight omni_pt \
    --hidden_size 1024 --prelude_layers 2 --coda_layers 2 --max_loop_iters 4

# 随后做多模态 SFT（数据需已按补齐后的 tokenizer 准备）
uv run torchrun --nproc_per_node 1 scripts/train_sft.py --data_path dataset/sft_mm.parquet \
    --from_weight omni_pt --hidden_size 1024 --prelude_layers 2 --coda_layers 2 --max_loop_iters 4
```

注意 `--hidden_size` 等结构参数必须与 `out/omni_init_<hidden>.json` 一致。

### 3.4 数据

- SFT 数据中的图像标签 `<image>` 与占位符绑定是按顺序进行的（见 `docs/data_pipeline.md`）。
- 数据预处理使用的 tokenizer 必须与热启动产生的 tokenizer 相同，否则占位符 id 会错位。

## 4. 测试

```bash
uv run pytest tests/test_init_from_lm.py -q
```

6 项测试，使用本地随机初始化的 Qwen3 小模型（不需要网络）：

| 测试 | 验证内容 |
| --- | --- |
| `test_special_tokens_are_appended_once` | 占位符只追加一次，id 正确 |
| `test_resize_rows_uses_mean_for_new_rows` | 原有行保持不变，新行为均值 |
| `test_config_copies_structure_from_source` | 结构字段正确复制，占位符 id 正确 |
| `test_too_few_layers_is_rejected` | 层数不足时报错 |
| `test_layers_map_to_prelude_recurrent_coda` | 源第 0、1、2、4、5 层映射到正确位置 |
| `test_init_loads_backbone_and_runs_forward` | 缺失键只剩新增参数；前向输出形状与有限性正确 |

全部测试 118 项（含本模块）在本地通过。

## 5. 限制与未验证项

1. **未用真实开源权重验证**。容器无法访问 Hugging Face / ModelScope，也没有 GPU，因此没有跑过真实模型的热启动、命令行脚本端到端，也没有评估初始化后的效果。
2. **输出不等价**。近似误差没有被量化。可以增加一个对照测试：`max_loop_iters=1` 且层数恰好匹配时，比较与源模型输出的偏差。
3. **tokenizer 接入需手动处理**（见 3.3）。
4. **只支持 Qwen3 / Llama 风格结构**。RMSNorm + q/k norm + SwiGLU、无注意力偏置。带偏置的结构（如 Qwen2）需要额外处理。
5. **循环块只复用一层权重**。源模型的大部分中间层被丢弃，文本能力的保留程度取决于后续训练。

## 6. 建议的后续步骤

1. 在有网络和 GPU 的机器上运行 `scripts/init_from_lm.py`，确认 load 结果中没有意外的缺失键。
2. 为训练脚本增加 `--tokenizer` 参数，去掉对 `assets/tokenizer` 的硬编码依赖。
3. 增加对照测试，量化 `max_loop_iters=1` 下与源模型输出的偏差。
4. 用少量文本数据做继续预训练，观察 loss 是否快速回落，再进入多模态 SFT。
