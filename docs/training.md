# 训练指南

训练入口是 `scripts/train_sft.py`，配方脚本位于 `scripts/recipes/`。所有命令都在仓库根目录执行。

```bash
uv sync
bash scripts/recipes/mini.sh          # 单卡 mini 数据集
```

## 训练阶段与配方

| 配方 | 数据 | 卡数 | 用途 |
| --- | --- | --- | --- |
| `scripts/recipes/mini.sh` | `sft_t2a_mini` → `sft_a2a_mini` | 1 | 入门，依次完成文本到语音、语音到语音的对齐 |
| `scripts/recipes/full_dense.sh` | 全量 `sft_t2a` / `sft_a2a` / `sft_i2t` | 4 | Dense 架构完整训练 |
| `scripts/recipes/full_moe.sh` | 同上 | 4 | MoE 架构完整训练（`--use_moe 1`） |

每个配方内的命令按顺序执行，后一条通常以前一条的 `--save_weight` 作为 `--from_weight`。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data_path` | `dataset/train_t2a_mini.parquet` | 训练数据，parquet 格式（见 [data_format.md](data_format.md)） |
| `--from_weight` | `llm` | 初始化权重前缀，`none` 表示随机初始化；从 `--save_dir/<from_weight>_<hidden>.pth` 读取 |
| `--save_weight` | `sft_omni` | 输出权重前缀 |
| `--save_dir` | `out/` | 权重输出目录 |
| `--mode` | `all` | `all` 全量训练；`audio_proj` 只训练音频 projector；`vision_proj` 只训练视觉 projector |
| `--freeze_backbone` | `none` | `none` 不冻结；`all` 冻结主干；`last1` 只训练最后一层 |
| `--from_resume` | `0` | 设为 `1` 时从 `checkpoints/` 中的续训文件恢复 |
| `--prelude_layers` / `--coda_layers` | `2` / `2` | 循环前 / 后的标准层数 P、C |
| `--max_loop_iters` | `8` | 循环圈数 T |
| `--max_seq_len` | `512` | 样本截断长度 |
| `--batch_size` | `32` | 单卡 batch size |
| `--learning_rate` | `5e-4` | 峰值学习率，按余弦曲线衰减到 0.1 倍 |
| `--use_compile` | `0` | 设为 `1` 启用 `torch.compile` |
| `--use_moe` | `0` | 设为 `1` 使用 MoE 架构 |
| `--use_swanlab` | 关 | 启用 SwanLab 记录（项目名 `--swanlab_project`）。旧参数 `--use_wandb` / `--wandb_project` 仍可用 |

运行 `uv run python scripts/train_sft.py --help` 查看全部参数。

## 续训

加上 `--from_resume 1` 重新运行同一条命令即可。

- 续训文件：`checkpoints/<save_weight>_<hidden>[_moe]_resume.pth`，包含模型权重、优化器、scaler、epoch、step 和 SwanLab run_id。旧版本保存的 `wandb_id` 仍可读取。
- 推理用的权重：`--save_dir`（默认 `out/`）下的 `<save_weight>_<hidden>[_moe].pth`，每次保存只写一份，`checkpoints/` 里不再有重复的纯权重副本。
- 加载权重时如果存在缺失或多余的参数，会在日志中打印数量和示例键名。
- GPU 数量变化时，step 会按比例换算。

## 冻结与损失

- 训练中 SenseVoice、SigLIP2、Mimi 始终冻结，只从 `pretrained/` 加载。
- 损失 = 文本交叉熵 + 8 层音频交叉熵的平均 + MoE 辅助损失。音频损失对 stop token 加权 10 倍（见 `baize_v/train/losses.py`）。
- 训练 checkpoint 为 fp16 权重；续训时额外保存优化器状态。
