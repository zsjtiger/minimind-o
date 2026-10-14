# 开发指南

## 环境

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 版本、虚拟环境和依赖。

- Python 版本由 `.python-version` 固定为 3.12；`pydub` 依赖的 `audioop` 在 3.13 中已移除，因此不支持 3.13。
- 依赖声明在 `pyproject.toml`，完整的解析结果在 `uv.lock`，修改依赖后请运行 `uv lock` 并一并提交。

```bash
uv sync                          # 创建 .venv，安装依赖（默认包含 dev 组：pytest / pyflakes / ruff）
uv sync --group demo             # 额外安装 Gradio 示例所需依赖
uv run python -c "import baize_v; print(baize_v.__version__)"
```

依赖组：

| 组 | 内容 | 默认安装 |
| --- | --- | --- |
| `dev` | pytest、pyflakes、ruff | 是 |
| `demo` | gradio | 否 |

所有命令都通过 `uv run` 执行，无需手动激活虚拟环境。

## 测试

```bash
uv run pytest                   # 全部测试
uv run pytest tests/test_model.py -q    # 单个文件
```

测试不依赖外部预训练权重，全部使用 tiny 配置（hidden 64、2 层）和不存在的编码器路径。覆盖范围：

| 文件 | 内容 |
| --- | --- |
| `tests/test_structure.py` | 所有模块可导入、资产存在、路径相对仓库根、没有残留的旧导入 |
| `tests/test_model.py` | 权重键名与形状与 `tests/fixtures/omni_tiny_state_dict.json` 一致；前向形状；MoE 辅助损失 |
| `tests/test_data.py` | 聊天模板、标签构造、scheduled sampling、音频增强、数据集样本与 collate、音频加载路径 |
| `tests/test_train.py` | 学习率曲线、跳过 batch 的 sampler、loss、checkpoint 往返、一轮训练、模型初始化 |
| `tests/test_inference.py` | 模型目录扫描、流式生成、实时 VAD 静音判断 |

### 关于 `state_dict` 键名测试

`tests/fixtures/omni_tiny_state_dict.json` 记录了结构迁移前代码导出的键名与形状。它的作用是防止无意中改动属性名导致已有权重无法加载。如果有意修改了网络结构，需要同时：

1. 在 PR 说明中写明旧权重是否还能加载；
2. 重新生成该文件，并在 `docs/migration.md` 中记录变更。

## Lint

```bash
uv run python -m pyflakes baize_v scripts tests
uv run ruff check baize_v tests scripts
```

`ruff` 的规则在 `pyproject.toml` 的 `[tool.ruff.lint]` 中，目前只检查语法错误和 pyflakes 类问题。

## 导入规则

- 核心逻辑只放在 `baize_v/` 中，入口脚本只做参数解析和调用。
- 不使用 `from x import *`。
- 路径一律从 `baize_v.paths` 引用，不要在代码中写 `../model` 之类的相对路径。
- 新增 checkpoint 字段时，要保证旧的 checkpoint 仍能读取（缺失字段用默认值）。

## 验证迁移的方法

结构调整时可以用同一个 tiny 配置，在改动前后分别导出 `state_dict` 的键名和同种子下的前向输出，比较两者是否完全相同。这套方法已经用于本次迁移（96 个键、logits 逐元素一致）。

## 重构与修复记录

以下事项已在本轮完成，并用固定种子比对模型输出与数据样本，确保行为不变或有意说明：

- 编码器加载、参数统计、音频 token 常量收拢到 `encoders.py` / `summary.py` / `tokens.py`。
- Mimi 解码、HF 模型加载在 Web 与 Gradio 入口之间共用（`inference/mimi.py`、`inference/loader.py`）。
- `evaluate.py` 拆分为每种模式一个函数；`OmniDataset.__getitem__` 拆分为多个私有方法，随机数调用顺序保持不变。
- 注入逻辑移到 `models/fusion.py`，生成循环移到 `inference/generate.py`，`count_vision_proj` 改名为 `inject_vision_features`（保留旧名）。
- 训练上下文改为 `TrainContext` dataclass；音色状态封装为 `VoiceRegistry`；实时 WebSocket 处理封装为 `RealtimeConnection`。
- `load_state_dict(strict=False)` 的缺失键与多余键会打印。
- 续训时不再额外写一份 `checkpoints/<w>.pth` 纯权重副本。
- RoPE 缓冲区检查只在第一次 forward 时执行，避免每步 GPU 同步。
- `torch.cuda.amp` 替换为 `torch.amp`；`--use_wandb` 统一为 `--use_swanlab`（旧名保留为别名）。
- Web 服务切换模型时复用已加载的 SenseVoice / SigLIP 编码器（Gradio 示例同样处理）。

## 暂不处理的事项

- **Parquet 整表读入内存**：改为按需读取需要按 row group 解码。训练是随机采样，row group 缓存命中率低，每个样本都可能重新解码整个 row group，训练速度会明显下降。目前保留整表加载，每个进程一份内存。如果数据集超过内存，建议先用更小的 row group 切分数据，再评估按需读取的方案。
- **`OmniConfig` 拆分**：配置字段会写入已导出的 `config.json`，拆分会影响旧模型加载，需要单独设计迁移方案。
- **Gradio 示例的全局变量与长函数**：它只是示例入口，优先级低，暂未拆分。
