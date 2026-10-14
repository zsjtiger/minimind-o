"""Mimi 音频编解码的共用工具：码本整理、特殊 token 屏蔽、解码。"""
import torch

from ..tokens import AUDIO_PAD_TOKEN, NUM_AUDIO_LAYERS


def frames_to_codes(frames, device=None):
    """把逐帧的 8 层码本列表转为 (1, 8, T) 张量；没有完整帧时返回 None。"""
    codes = [f for f in frames if f and len(f) == NUM_AUDIO_LAYERS]
    if not codes:
        return None
    return torch.tensor(codes, dtype=torch.long, device=device).T.unsqueeze(0)


def mask_special(codes):
    """把 pad / 特殊 token 置为 0，使其能被 Mimi 解码。"""
    return torch.where(codes >= AUDIO_PAD_TOKEN, torch.zeros_like(codes), codes)


def decode_codes(mimi, codes):
    """用 Mimi 解码 (1, 8, T) 码本为波形，返回 numpy 数组（保持模型输出的 dtype）。"""
    with torch.no_grad():
        return mimi.decode(codes).audio_values.squeeze().cpu().numpy()
