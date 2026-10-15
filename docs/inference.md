# 推理指南

## 离线评测：`scripts/evaluate.py`

```bash
uv run python scripts/evaluate.py --load_from torch --weight sft_omni     # 原生 PyTorch 权重（out/sft_omni_768.pth）
uv run python scripts/evaluate.py --load_from out/baize-v             # transformers 格式目录
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--load_from` | `torch` | `torch` 表示原生权重；其他值为 transformers 模型目录 |
| `--mode` | `0` | 评测模式，可用逗号组合：`-1` 全部，`0` 文本，`1` 多轮，`2` 语音输入，`3` 音色克隆，`4` 图像输入，`5` 混合 |
| `--audio_dir` / `--image_dir` | `assets/eval_omni/` | 评测输入 |
| `--output_dir` | `./output_audio/` | 生成的 mp3 输出目录 |
| `--open_thinking` | `0` | 开启思考模式（此时不输出语音） |
| `--decode_audio` | `1` | 是否用 Mimi 解码语音并保存 |
| `--temperature` / `--top_p` | `0.7` / `0.85` | Thinker 采样参数 |

解码语音需要 `pretrained/mimi/`。

## Web 应用：`apps/web/server.py`

支持文本、语音、图像输入，以及实时语音通话（WebSocket，基于 Silero VAD 做端点检测和打断）。

```bash
uv run python apps/web/server.py --load_from out --port 7860
```

- `--load_from`：扫描该目录下包含权重文件的子目录，页面中可切换模型。
- 音色保存在 `assets/speaker/`，克隆得到的音色写入 `voice_clone.pt`。
- 克隆前需勾选授权确认，并会写入审计日志，详见 [使用边界与安全说明](safety.md)。
- 启动时会预热一次模型，首次请求延迟较低。

访问 `http://localhost:7860` 打开页面，页面文件位于 `apps/web/static/web_demo.html`。

## Gradio 示例：`examples/gradio_demo.py`

非实时演示，接口与 Web 应用类似，主要用于快速体验，不支持实时打断。

```bash
uv sync --group demo
uv run python examples/gradio_demo.py --load_from out
```

## 在代码中调用

```python
import torch
from baize_v.models import BaizeOmni
from baize_v.config import OmniConfig
from baize_v.paths import TOKENIZER_DIR, SENSEVOICE_DIR, SIGLIP_DIR
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
# 编码器目录不存在时会给出警告并跳过加载，适合只做文本推理
model = BaizeOmni(OmniConfig(), audio_encoder_path=SENSEVOICE_DIR, vision_model_path=SIGLIP_DIR).eval()
model.load_state_dict(torch.load("out/sft_omni_768.pth", map_location="cpu"), strict=False)
```

更多用法参考 `scripts/evaluate.py` 中的 `eval_sample`。
