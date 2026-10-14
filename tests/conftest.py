import pytest
import torch
from transformers import AutoTokenizer

from baize_v.models import BaizeOmni
from baize_v.paths import TOKENIZER_DIR
from _helpers import make_tiny_config, MISSING


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    return BaizeOmni(make_tiny_config(), audio_encoder_path=MISSING, vision_model_path=MISSING).eval()
