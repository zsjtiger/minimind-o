import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from baize_v.data import OmniDataset, omni_collate_fn
from baize_v.data.augment import augment_mel, augment_wav
from baize_v.data.chat import pre_processing_chat, post_processing_chat
from baize_v.data.labels import generate_text_labels, apply_scheduled_sampling


def test_pre_processing_keeps_tool_conversations():
    conv = [{"role": "user", "content": "hi", "tools": "[]"}]
    assert pre_processing_chat(conv) is conv


def test_post_processing_drops_empty_think_when_ratio_zero():
    text = "<think>\n\n</think>\n\nhello"
    assert post_processing_chat(text, empty_think_ratio=0.0) == "hello"
    assert post_processing_chat(text, empty_think_ratio=1.0) == text


def test_generate_text_labels_covers_only_assistant_span():
    ids = [9, 1, 2, 5, 6, 3, 9]
    labels, ranges = generate_text_labels(ids, bos_id=[1, 2], eos_id=[3], max_length=len(ids))
    assert ranges == [(3, 5)]
    assert labels == [-100, -100, -100, 5, 6, 3, -100]


def test_scheduled_sampling_noop_when_prob_zero():
    ids = torch.randint(0, 100, (9, 10))
    out = apply_scheduled_sampling(ids.clone(), torch.zeros(8, 10, dtype=torch.long), torch.zeros(10, dtype=torch.long),
                                   prob=0.0, audio_vocab_size=2112, text_vocab_size=6400, image_token_id=12)
    assert torch.equal(out, ids)


def test_augment_preserves_length_and_range():
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(16000).astype(np.float32) * 0.1
    out = augment_wav(wav.copy())
    assert out.dtype == np.float32 and np.all(np.abs(out) <= 1.0)
    fbank = torch.randn(50, 560)
    assert augment_mel(fbank.clone()).shape == (50, 560)


@pytest.fixture
def tiny_parquet(tmp_path):
    rows = {"conversations": [json.dumps([{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}], ensure_ascii=False)] * 2}
    path = os.path.join(tmp_path, "tiny.parquet")
    pq.write_table(pa.table(rows), path)
    return path


def test_dataset_item_shapes(tiny_parquet, tokenizer):
    ds = OmniDataset(tiny_parquet, tokenizer, max_length=128)
    assert len(ds) == 2
    input_ids, text_labels, audio_labels, audio_inputs, audio_len, pixel_values, spk_emb = ds[0]
    assert input_ids.shape == (9, 127)
    assert text_labels.shape == (127,)
    assert audio_labels.shape == (8, 127)
    assert audio_inputs is None and pixel_values is None and audio_len == 0
    assert spk_emb.shape == (192,)
    assert (text_labels != -100).any()


def test_collate_stacks_batch(tiny_parquet, tokenizer):
    ds = OmniDataset(tiny_parquet, tokenizer, max_length=128)
    input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = omni_collate_fn([ds[0], ds[1]])
    assert input_ids.shape == (2, 9, 127)
    assert labels.shape == (2, 127)
    assert audio_labels.shape == (2, 8, 127)
    assert audio_inputs is None and pixel_values is None
    assert spk_emb.shape == (2, 192)


def test_audio_path_with_augmentation(tiny_parquet, tokenizer):
    """走一遍 load_audio_inputs（含 augment_wav / augment_mel），用假 processor 代替 SenseVoice。"""
    import io
    import soundfile as sf
    from types import SimpleNamespace

    def fake_processor(wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True):
        frames = max(1, len(wav) // 160)
        return SimpleNamespace(input_features=torch.randn(1, frames, 560),
                               attention_mask=torch.ones(1, frames, dtype=torch.long))

    ds = OmniDataset(tiny_parquet, tokenizer, audio_processor=fake_processor, max_length=128)
    buf = io.BytesIO()
    sf.write(buf, np.random.default_rng(0).standard_normal(16000 * 2).astype(np.float32) * 0.1, 16000, format="WAV")
    mel, valid_len = ds.load_audio_inputs(buf.getvalue())
    assert mel.shape[1] == 560 and valid_len == mel.shape[0]
