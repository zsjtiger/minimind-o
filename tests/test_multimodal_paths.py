"""多模态路径测试：用假的编码器走通音频/视觉注入，并验证导出的 transformers 模型可重新加载。"""
import torch
from torch import nn
from types import SimpleNamespace

import pytest

from baize_v.models import BaizeOmni
from _helpers import make_tiny_config, MISSING

AUDIO_MARKER = 16   # OmniConfig.audio_ids[0]
IMAGE_MARKER = 12   # OmniConfig.image_ids[0]


class FakeAudioEncoder(nn.Module):
    """代替 SenseVoice：(B, T, 560) 的 fbank → (B, T', 512) 的特征。"""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(560, 512)

    def forward(self, fbank, lens):
        return self.proj(fbank), lens


class FakeAudioProcessor:
    pass


class FakeVisionEncoder(nn.Module):
    """代替 SigLIP：(B, 3, H, W) → 64 个 768 维 token。"""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 768)

    def forward(self, pixel_values):
        pooled = pixel_values.mean(dim=(2, 3))                       # (B, 3)
        tokens = self.proj(pooled).unsqueeze(1).expand(-1, 64, -1)   # (B, 64, 768)
        return SimpleNamespace(last_hidden_state=tokens)


@pytest.fixture
def mm_model():
    torch.manual_seed(0)
    model = BaizeOmni(make_tiny_config(), audio_encoder_path=MISSING, vision_model_path=MISSING).eval()
    object.__setattr__(model, 'audio_encoder', FakeAudioEncoder().eval())
    object.__setattr__(model, 'audio_processor', FakeAudioProcessor())
    object.__setattr__(model, 'vision_encoder', FakeVisionEncoder().eval())
    object.__setattr__(model, 'vision_processor', None)
    return model


def _audio_pad_inputs(batch=1, seq=12, marker_len=5):
    ids = torch.randint(0, 6000, (batch, 9, seq))
    ids[:, :8, :] = 2049
    ids[:, 8, 2:2 + marker_len] = AUDIO_MARKER
    return ids


def test_audio_features_are_injected(mm_model):
    ids = _audio_pad_inputs()
    a = torch.randn(1, 40, 560)
    b = torch.randn(1, 40, 560)
    with torch.no_grad():
        la = mm_model(ids, audio_inputs=a, audio_lens=torch.tensor([40])).logits
        lb = mm_model(ids, audio_inputs=b, audio_lens=torch.tensor([40])).logits
        none = mm_model(ids).logits
    assert torch.isfinite(la).all()
    assert not torch.allclose(la, lb), '不同音频应产生不同输出'
    assert not torch.allclose(la, none), '注入音频后应与纯文本不同'


def test_image_tokens_are_injected(mm_model):
    ids = torch.randint(0, 6000, (1, 9, 70))
    ids[:, :8, :] = 2049
    ids[:, 8, 3:3 + 64] = IMAGE_MARKER          # 64 个图像占位 token
    img_a = torch.rand(1, 3, 32, 32)
    img_b = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        la = mm_model(ids, pixel_values=img_a).logits
        lb = mm_model(ids, pixel_values=img_b).logits
    assert not torch.allclose(la, lb), '不同图像应产生不同输出'


def test_export_then_reload_matches(tmp_path):
    """导出目录应能被 baize_v.export.load_pretrained 重新加载，且输出与导出前一致（fp32）。"""
    from baize_v.export import save_pretrained, load_pretrained
    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = BaizeOmni(cfg, audio_encoder_path=MISSING, vision_model_path=MISSING).eval()
    ids = torch.randint(0, 6000, (1, 9, 6))
    ids[:, :8, :] = 2049
    with torch.no_grad():
        before = model(ids).logits
    out = save_pretrained(model, str(tmp_path / 'export'), dtype=torch.float32)
    reloaded = load_pretrained(out, audio_encoder_path=MISSING, vision_model_path=MISSING)
    with torch.no_grad():
        after = reloaded(ids).logits
    assert torch.allclose(before, after, atol=1e-6)


def test_convert_script_writes_loadable_directory(tmp_path):
    from scripts.convert import convert_torch2export
    from baize_v.export import load_pretrained
    cfg = make_tiny_config()
    torch.manual_seed(0)
    torch.save(BaizeOmni(cfg, audio_encoder_path=MISSING, vision_model_path=MISSING).state_dict(), tmp_path / 'sft.pth')
    out = convert_torch2export(str(tmp_path / 'sft.pth'), str(tmp_path / 'out'), cfg, dtype=torch.float32)
    model = load_pretrained(out, audio_encoder_path=MISSING, vision_model_path=MISSING)
    assert model.config.model_type == 'baize_v'
