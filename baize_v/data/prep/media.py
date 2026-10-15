"""图像与音频的统一预处理。"""
import io

import numpy as np
import soundfile as sf
from PIL import Image

TARGET_SR = 16000


def normalize_image(data, max_side=512, min_side=64, quality=90):
    """解码图像 → RGB → 长边不超过 max_side → JPEG。过小或损坏的图像返回 None。"""
    try:
        img = Image.open(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data).convert('RGB')
    except Exception:
        return None
    if min(img.size) < min_side:
        return None
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def normalize_audio(data, min_sec=0.3, max_sec=30.0):
    """读取任意 soundfile 可解码的音频 → 单声道 → 16 kHz → PCM16 wav 字节。时长不合格返回 None。"""
    try:
        wav, sr = sf.read(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data, dtype='float32')
    except Exception:
        return None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    dur = len(wav) / TARGET_SR
    if dur < min_sec or dur > max_sec:
        return None
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav, -1.0, 1.0), TARGET_SR, format='WAV', subtype='PCM_16')
    return buf.getvalue()
