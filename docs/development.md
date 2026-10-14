# 开发指南

## 环境

- Python 3.10–3.12（`pydub` 依赖的 `audioop` 在 3.13 中已移除，因此不支持 3.13）。
- 先按 PyTorch 官网安装与本机 CUDA 匹配的 `torch` / `torchaudio`，再安装项目依赖：

```bash
pip install -r requirements.txt
pip install -e ".[dev]"          # 可编辑安装 baize-v，并安装 pytest / pyflakes / ruff
pip install -e ".[demo]"         # 可选：Gradio 示例
```

## 测试

```bash
pytest                           # 全部测试
pytest tests/test_model.py -q    # 单个文件
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
python -m pyflakes baize_v scripts tests
python -m ruff check baize_v tests scripts
```

`ruff` 的规则在 `pyproject.toml` 的 `[tool.ruff.lint]` 中，目前只检查语法错误和 pyflakes 类问题。

## 导入规则

- 核心逻辑只放在 `baize_v/` 中，入口脚本只做参数解析和调用。
- 不使用 `from x import *`。
- 路径一律从 `baize_v.paths` 引用，不要在代码中写 `../model` 之类的相对路径。
- 新增 checkpoint 字段时，要保证旧的 checkpoint 仍能读取（缺失字段用默认值）。

## 验证迁移的方法

结构调整时可以用同一个 tiny 配置，在改动前后分别导出 `state_dict` 的键名和同种子下的前向输出，比较两者是否完全相同。这套方法已经用于本次迁移（96 个键、logits 逐元素一致）。

## 后续计划

本次迁移只调整结构，没有改变训练和推理的行为。以下问题会改变 checkpoint 格式或数值结果，计划分开处理：

1. **权重加载**：`load_state_dict(strict=False)` 时打印 missing / unexpected 键。
2. **checkpoint 布局**：模型权重只保存一份；`--save_dir` 与 `checkpoints/` 统一。续训文件只存优化器、scaler 等状态。
3. **性能**：去掉每次 forward 中的 `freqs_cos[0, 0] == 0` 检查（GPU 同步）；`MiniMindOmni.forward` 中的 dummy gradient 改为更清晰的方式。
4. **Web 应用**：切换模型时复用已加载的 SenseVoice 与 SigLIP，不重复读盘。
5. **数据集**：避免整表读入内存，改为按需读取 parquet。
6. **训练脚本**：`--use_wandb` 实际使用 SwanLab，参数名与日志名需统一；将 `torch.cuda.amp` 替换为 `torch.amp`。
7. **Demo 去重**：Flask 与 Gradio 两个入口之间的 Mimi 解码、音色加载逻辑还有重复，可进一步合并到 `baize_v/inference/`。
8. **生成逻辑**：`MiniMindOmni.stream_generate` 约 80 行，可拆分为采样函数，保持 `model.generate` 接口不变。
