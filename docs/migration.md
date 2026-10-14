# 迁移说明：从 minimind-o 旧结构到 baize-v 新结构

本次改动把原来散落在 `model/`、`trainer/`、`dataset/`、`webui/`、`scripts/` 中的代码整理为 `baize_v/` 包，入口脚本与应用分别放在 `scripts/`、`apps/`、`examples/`。**代码逻辑保持不变**，只调整了位置、导入方式和默认路径。

## 文件对应关系

| 旧位置 | 新位置 |
| --- | --- |
| `model/model_minimind.py` | `baize_v/models/minimind.py`（配置类移至 `baize_v/config.py`） |
| `model/model_omni.py` 中的 `OmniConfig` | `baize_v/config.py` |
| `model/model_omni.py` 中的 projector | `baize_v/models/projectors.py` |
| `model/model_omni.py` 中的 Talker | `baize_v/models/talker.py` |
| `model/model_omni.py` 中的 `SenseVoiceAudioProcessor` | `baize_v/models/encoders.py` |
| `model/model_omni.py` 中的 `BaizeOmni` | `baize_v/models/omni.py` |
| `model/model_omni.py` 中的 `SileroVAD`、`RealtimeSession` | `baize_v/inference/realtime.py` |
| `dataset/omni_dataset.py` | `baize_v/data/dataset.py`，其中的聊天模板、增强、标签函数分别移至 `chat.py`、`augment.py`、`labels.py` |
| `trainer/trainer_utils.py` | `baize_v/train/{dist,schedule,init,checkpoint}.py`，`vlm_collate_fn` 移至 `data/collate.py` |
| `trainer/train_sft_omni.py` 的 `train_epoch` | `baize_v/train/loop.py`，loss 计算移至 `train/losses.py` |
| `trainer/train_sft_omni.py` 的 `omni_collate_fn` | `baize_v/data/collate.py` |
| `trainer/train_sft_omni.py` 的入口部分 | `scripts/train_sft.py` |
| `eval_omni.py` | `scripts/evaluate.py` |
| `scripts/convert_omni.py` | `scripts/convert.py` |
| `trainer/train.sh` | `scripts/recipes/{mini,full_dense,full_moe}.sh` |
| `webui/web_demo.py` | `apps/web/server.py` |
| `webui/web_demo.html` | `apps/web/static/web_demo.html` |
| `scripts/web_demo_omni.py` | `examples/gradio_demo.py` |
| `model/tokenizer*.json` | `assets/tokenizer/` |
| `model/vad/silero_vad.onnx` | `assets/vad/silero_vad.onnx` |
| `model/speaker/` | `assets/speaker/` |
| `dataset/eval_omni/` | `assets/eval_omni/` |
| `images/` | `docs/images/` |

两个 Web 入口中重复的 `scan_hf_models` 已合并为 `baize_v/inference/loader.py`。

## 需要用户手动处理的变化

### 0. 类名、模型类型与导出目录更名

- 类名：`MiniMindConfig` / `MiniMindModel` / `MiniMindForCausalLM` / `MiniMindBlock` / `MiniMindOmni` 分别更名为 `BaizeConfig` / `BaizeModel` / `BaizeForCausalLM` / `BaizeBlock` / `BaizeOmni`。旧名称均保留别名，旧代码无需立即修改。
- 基座模块文件：`baize_v/models/minimind.py` 更名为 `baize_v/models/llm.py`。
- `model_type`：`minimind` → `baize`，`minimind-o` → `baize_v`。**已导出的 transformers 模型**的 `config.json` 仍写着旧值；加载时请依赖其自带的 `auto_map` 与 `trust_remote_code=True`。若出现无法识别的情况，手动把 `config.json` 中的 `model_type` 改为新值即可，权重无需转换。
- 训练数据的系统提示词由 “你是minimind” 改为 “你是Baize”。这会改变基于该提示词的训练分布，若要复现旧模型请保留旧提示词。
- 本地导出目录：`out/minimind-3o` 改为 `out/baize-v`。Hugging Face 上的仓库名 `jingyaogong/minimind-3o` 不变。

### 1. 外部预训练权重目录改为 `pretrained/`

原来下载到 `./model/` 下的 SenseVoice、SigLIP2、Mimi、CAM++ 需要移动或重新下载到 `pretrained/`：

```bash
mkdir -p pretrained
mv model/SenseVoiceSmall model/siglip2-base-p32-256-ve model/mimi model/campplus pretrained/
```

如果权重放在其他位置，可以设置环境变量 `BAIZE_PRETRAINED_DIR=/path/to/weights` 代替移动文件。

### 2. 运行方式改为从仓库根目录执行

旧版脚本需要 `cd trainer` 或 `cd scripts` 才能运行，依赖相对路径 `../model`、`../out`。新版统一从仓库根目录运行，并先安装一次包：

```bash
uv sync
uv run python scripts/train_sft.py --help
uv run python scripts/evaluate.py --help
```

### 3. 评测脚本的 `--load_from` 参数语义

- 旧：`--load_from model` 表示原生 PyTorch 权重，任何路径中包含 `model` 的目录都会被当作 PyTorch 权重。
- 新：`--load_from torch` 表示原生 PyTorch 权重（默认值），其他取值一律按 transformers 目录加载。

### 4. 导出目录

`scripts/convert.py` 的默认输出从 `../minimind-3o` 改为 `out/baize-v`。

### 5. 应用入口

- 实时 Web 应用：`uv run python apps/web/server.py --load_from out`
- Gradio 示例：`uv run python examples/gradio_demo.py --load_from out`

两者都会扫描 `out/` 下包含权重文件的子目录。

## 后续的行为修复

结构迁移之后又做了一轮行为修复，详见 [development.md](development.md#重构与修复记录)。对用户可见的变化包括：

- 续训不再额外写 `checkpoints/<weight>.pth`，推理权重只在 `--save_dir` 中保存一份。
- 加载权重时会打印缺失和多余的参数数量。
- SwanLab 参数改为 `--use_swanlab` / `--swanlab_project`，旧参数仍可用。
- Web 服务切换模型时不再重复加载编码器。
