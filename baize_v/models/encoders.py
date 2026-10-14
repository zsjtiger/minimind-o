import os
import warnings
import logging
import contextlib
import io
from types import SimpleNamespace

import numpy as np
import torch
from transformers import SiglipImageProcessor, SiglipVisionModel, logging as hf_logging


class SenseVoiceAudioProcessor:
    def __init__(self, frontend): self.frontend = frontend
    def __call__(self, wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, **kwargs):
        if isinstance(wav, np.ndarray): wav = torch.from_numpy(wav).float()
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        with torch.no_grad():
            fbank, flen = self.frontend(wav, torch.tensor([wav.size(1)]))
        return SimpleNamespace(input_features=fbank, attention_mask=(torch.arange(fbank.size(1)) < flen[0]).long().unsqueeze(0))





def load_sensevoice(path):
    """加载 SenseVoice 编码器与特征前端。路径不存在时返回 (None, None)。"""
    if not os.path.exists(path):
        warnings.warn(f"[BaizeOmni] SenseVoice path not found: {path}")
        return None, None
    logging.getLogger().setLevel(logging.ERROR)
    hf_logging.set_verbosity_error()
    with contextlib.redirect_stdout(io.StringIO()):
        from funasr import AutoModel
        m = AutoModel(model=path, trust_remote_code=True, disable_update=True, device="cpu")
    encoder, frontend = m.model.encoder, m.kwargs["frontend"]
    for p in encoder.parameters(): p.requires_grad = False
    return encoder.eval().float(), SenseVoiceAudioProcessor(frontend.eval())


def load_vision(path):
    """加载 SigLIP 视觉编码器。路径不存在或加载失败时返回 (None, None)。"""
    if path is None or not os.path.exists(path):
        warnings.warn(f"[BaizeOmni] Vision model path not found: {path}. vision_encoder will be None!")
        return None, None
    hf_logging.set_verbosity_error()
    try:
        model = SiglipVisionModel.from_pretrained(path)
    except (RuntimeError, ValueError):
        return None, None
    processor = SiglipImageProcessor.from_pretrained(path)
    for p in model.parameters():
        p.requires_grad = False
    return model.eval(), processor
