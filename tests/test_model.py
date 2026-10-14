import json
import os

import torch

from baize_v.models import MiniMindOmni
from _helpers import make_tiny_config, MISSING

GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "omni_tiny_state_dict.json")


def test_state_dict_matches_golden_layout(tiny_model):
    """结构迁移后权重键名与形状必须与迁移前的导出完全一致，否则已有 checkpoint 无法加载。"""
    golden = json.load(open(GOLDEN))
    current = {k: list(v.shape) for k, v in tiny_model.state_dict().items()}
    assert current == golden


def test_forward_shapes(tiny_model):
    ids = torch.randint(0, 6000, (2, 9, 12))
    ids[:, :8, :] = 2049
    with torch.no_grad():
        out = tiny_model(ids)
    assert out.logits.shape == (2, 12, 6400)
    assert len(out.audio_logits) == 8
    assert all(a.shape == (2, 12, 2112) for a in out.audio_logits)


def test_moe_forward_has_aux_loss():
    torch.manual_seed(0)
    model = MiniMindOmni(make_tiny_config(use_moe=True, num_experts=2), audio_encoder_path=MISSING, vision_model_path=MISSING).train()
    ids = torch.randint(0, 6000, (1, 9, 8))
    ids[:, :8, :] = 2049
    out = model(ids)
    assert out.aux_loss.dim() == 0 and torch.isfinite(out.aux_loss)


def test_missing_encoders_are_none(tiny_model):
    assert tiny_model.audio_encoder is None
    assert tiny_model.vision_encoder is None
    assert tiny_model.thinker is tiny_model.model
