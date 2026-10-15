"""模型的保存与加载（不依赖 transformers 的远程代码机制）。

目录内容：
    config.json          OmniConfig（model_type = baize_v）
    pytorch_model.bin    fp16 权重，不含冻结的音视频编码器（推理时从预训练目录加载）
    tokenizer 文件       与 assets/tokenizer 相同

为什么不用 trust_remote_code：transformers 的动态模块扫描器无法解析多级相对导入
（如 `from ..inference.generate import ...`），导出目录会加载失败。
"""
import os

import torch
from transformers import AutoTokenizer

from .config import OmniConfig
from .models.omni import BaizeOmni
from .models.llm import remap_legacy_state_dict
from .models.summary import report_load_result
from .paths import TOKENIZER_DIR, SENSEVOICE_DIR, SIGLIP_DIR

WEIGHTS_FILE = 'pytorch_model.bin'
ENCODER_PREFIXES = ('audio_encoder.', 'vision_encoder.')


def save_pretrained(model, out_dir, dtype=torch.float16, tokenizer_dir=TOKENIZER_DIR):
    os.makedirs(out_dir, exist_ok=True)
    model.config.save_pretrained(out_dir)
    state = {k: v.to(dtype).cpu() for k, v in model.state_dict().items() if not k.startswith(ENCODER_PREFIXES)}
    torch.save(state, os.path.join(out_dir, WEIGHTS_FILE))
    AutoTokenizer.from_pretrained(tokenizer_dir).save_pretrained(out_dir)
    return out_dir


def load_pretrained(path, audio_encoder_path=SENSEVOICE_DIR, vision_model_path=SIGLIP_DIR, device='cpu'):
    config = OmniConfig.from_pretrained(path)
    model = BaizeOmni(config, audio_encoder_path=audio_encoder_path, vision_model_path=vision_model_path)
    state = remap_legacy_state_dict(torch.load(os.path.join(path, WEIGHTS_FILE), map_location='cpu'), config)
    report_load_result(model.load_state_dict(state, strict=False), label=os.path.basename(os.path.normpath(path)))
    return model.to(device).eval()
