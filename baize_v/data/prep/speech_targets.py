"""语音输出监督：把文本回答经 TTS 合成语音，再用 Mimi 编码为 8 层码本，写入 answer_audios。

TTS 与 Mimi 编码器都以函数形式注入，便于替换与测试：
    tts_fn(text) -> (waveform: np.ndarray float32, sample_rate: int)
    mimi_encode_fn(waveform_24k: np.ndarray) -> np.ndarray，形状 (8, T) 的整数码本

answer_audios 的格式与 OmniDataset 一致：每个 assistant 轮次是一个扁平列表，按帧交织（第 0 帧的 8 层，第 1 帧的 8 层……）。
"""
import json

import numpy as np

from ...tokens import NUM_AUDIO_LAYERS, MIMI_CODEBOOK_SIZE

MIMI_SR = 24000


def resample_to(wav, sr, target):
    if sr == target:
        return wav.astype(np.float32)
    import librosa
    return librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target)


def interleave_codes(codes):
    """(8, T) → 扁平的帧交织列表，长度 8·T。"""
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[0] != NUM_AUDIO_LAYERS:
        raise ValueError(f'码本形状应为 (8, T)，实际 {codes.shape}')
    return codes.T.reshape(-1).astype(np.int64).tolist()


def build_answer_audio(record, tts_fn, mimi_encode_fn, max_frames=None):
    """为一条只含文本的记录（conversations 最后一轮为 assistant）生成 answer_audios。返回新的记录（原记录不修改）。"""
    conv = json.loads(record['conversations'])
    answer = conv[-1]['content']
    wav, sr = tts_fn(answer)
    wav24 = resample_to(np.asarray(wav), sr, MIMI_SR)
    codes = np.asarray(mimi_encode_fn(wav24))
    if max_frames is not None and codes.shape[1] > max_frames:
        codes = codes[:, :max_frames]
    if codes.size == 0 or codes.min() < 0 or codes.max() >= MIMI_CODEBOOK_SIZE:
        raise ValueError('Mimi 码本越界或为空')
    out = dict(record)
    out['answer_audios'] = [interleave_codes(codes)]
    out['modality'] = 'speech_out'
    return out


def load_callable(spec):
    """从 'package.module:function' 形式加载函数，供命令行注入自己的 TTS 或编码器。"""
    import importlib
    module, _, name = spec.partition(':')
    if not name:
        raise ValueError('格式应为 module:function')
    return getattr(importlib.import_module(module), name)
