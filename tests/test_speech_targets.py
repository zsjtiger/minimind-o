
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from baize_v.data.prep.schema import make_record, validate_record, SCHEMA
from baize_v.data.prep.speech_targets import build_answer_audio, interleave_codes
from baize_v.data import OmniDataset


def fake_tts(text):
    n = 24000 + 100 * len(text)
    return np.sin(np.linspace(0, 200, n)).astype(np.float32), 22050


def fake_mimi(wav24):
    T = max(1, len(wav24) // 1920)            # Mimi 帧率约 12.5 Hz（24 kHz 下每帧 1920 个采样）
    return np.arange(8 * T).reshape(8, T) % 2048


def test_interleave_order_is_frame_major():
    codes = np.array([[0, 1], [10, 11], [20, 21], [30, 31], [40, 41], [50, 51], [60, 61], [70, 71]])
    flat = interleave_codes(codes)
    assert flat[:8] == [0, 10, 20, 30, 40, 50, 60, 70]    # 第 0 帧的 8 层
    assert flat[8:16] == [1, 11, 21, 31, 41, 51, 61, 71]  # 第 1 帧


def test_build_answer_audio_produces_valid_record():
    rec = make_record([{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好，很高兴见到你。'}],
                      source='sft', modality='text')
    out = build_answer_audio(rec, fake_tts, fake_mimi)
    assert out['modality'] == 'speech_out'
    assert len(out['answer_audios']) == 1 and len(out['answer_audios'][0]) % 8 == 0
    assert validate_record(out) == []
    assert rec['answer_audios'] == []   # 原记录不被修改


def test_out_of_range_codes_are_rejected():
    rec = make_record([{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}], source='s', modality='text')
    with pytest.raises(ValueError):
        build_answer_audio(rec, fake_tts, lambda w: np.full((8, 3), 4000))


def test_speech_targets_load_into_omni_dataset(tmp_path, tokenizer):
    rec = make_record([{'role': 'user', 'content': '你好'}, {'role': 'assistant', 'content': '你好，很高兴见到你。'}],
                      source='sft', modality='text')
    out = build_answer_audio(rec, fake_tts, fake_mimi)
    path = tmp_path / 'speech.parquet'
    pq.write_table(pa.Table.from_pylist([out], schema=SCHEMA), path)
    ds = OmniDataset(str(path), tokenizer, max_length=512)
    _, _, audio_labels, *_ = ds[0]
    assert (audio_labels != -100).any(), '语音输出的监督应进入 audio_labels'
