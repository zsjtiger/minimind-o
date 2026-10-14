import os

import numpy as np
import torch

from baize_v.inference.loader import scan_hf_models
from baize_v.inference.realtime import RealtimeSession
from baize_v.paths import VAD_PATH


def test_scan_hf_models_finds_only_weight_dirs(tmp_path):
    (tmp_path / "with_weights").mkdir()
    (tmp_path / "with_weights" / "model.safetensors").write_bytes(b"")
    (tmp_path / "no_weights").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "model.safetensors").write_bytes(b"")
    found = scan_hf_models(str(tmp_path))
    assert list(found) == ["with_weights"]
    assert os.path.isabs(found["with_weights"])


def test_generate_text_stream(tiny_model):
    torch.manual_seed(0)
    ids = torch.tensor([[1, 2, 3, 4]])
    chunks = [c for c, _ in tiny_model.stream_generate(ids, eos_token_id=2, max_new_tokens=3, temperature=0.8, top_p=0.9,
                                                       rp=1.0, use_cache=True) if c is not None]
    assert chunks and all(c.shape[0] == 1 for c in chunks)


def test_realtime_session_silence_stays_listening():
    session = RealtimeSession(VAD_PATH)
    state = session.push_chunk(np.zeros(16000, dtype=np.float32))
    assert state == "listening"
