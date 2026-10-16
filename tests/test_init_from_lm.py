"""用本地随机初始化的 Qwen3 小模型验证热启动映射（不下载任何权重）。"""
import torch
import pytest

from baize_v.models.init_from_lm import (AUDIO_PAD, IMAGE_PAD, ensure_omni_tokens, init_omni_from_lm,
                                         lm_state_to_omni, omni_config_from_hf, resize_rows)

transformers = pytest.importorskip('transformers')


class StubTokenizer:
    """只实现热启动用到的接口的最小 tokenizer。"""

    def __init__(self, n=300):
        self.vocab = {f'tok{i}': i for i in range(n)}
        self.unk_token_id = None

    def __len__(self):
        return len(self.vocab)

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)

    def add_special_tokens(self, d):
        for t in d['additional_special_tokens']:
            self.vocab[t] = len(self.vocab)


def _tiny_qwen(layers=6, vocab=300):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      intermediate_size=128, num_hidden_layers=layers, vocab_size=vocab,
                      max_position_embeddings=256, tie_word_embeddings=True)
    return Qwen3ForCausalLM(cfg).eval()


def test_special_tokens_are_appended_once():
    tok = StubTokenizer(300)
    tok.unk_token_id = -1
    assert ensure_omni_tokens(tok) == 2
    assert tok.convert_tokens_to_ids(AUDIO_PAD) == 300 and tok.convert_tokens_to_ids(IMAGE_PAD) == 301
    assert ensure_omni_tokens(tok) == 0


def test_resize_rows_uses_mean_for_new_rows():
    w = torch.arange(12.0).reshape(4, 3)
    big = resize_rows(w, 6)
    assert big.shape == (6, 3) and torch.equal(big[:4], w)
    assert torch.allclose(big[4], w.mean(0)) and torch.allclose(big[5], w.mean(0))
    assert resize_rows(w, 2).shape == (2, 3)


def test_config_copies_structure_from_source():
    tok = StubTokenizer(300)
    tok.unk_token_id = -1
    ensure_omni_tokens(tok)
    hf = _tiny_qwen()
    cfg = omni_config_from_hf(hf.config, tok, prelude_layers=2, coda_layers=2, max_loop_iters=3)
    assert (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) == (64, 4, 2, 16)
    assert cfg.intermediate_size == 128 and cfg.vocab_size == len(tok)
    assert cfg.image_ids == [301] and cfg.audio_ids == [300]


def test_too_few_layers_is_rejected():
    tok = StubTokenizer(300)
    tok.unk_token_id = -1
    ensure_omni_tokens(tok)
    with pytest.raises(ValueError):
        omni_config_from_hf(_tiny_qwen(layers=4).config, tok, prelude_layers=2, coda_layers=2)


def test_layers_map_to_prelude_recurrent_coda():
    tok = StubTokenizer(300)
    tok.unk_token_id = -1
    ensure_omni_tokens(tok)
    hf = _tiny_qwen(layers=6)
    cfg = omni_config_from_hf(hf.config, tok, prelude_layers=2, coda_layers=2, max_loop_iters=3)
    src = hf.state_dict()
    out = lm_state_to_omni(src, cfg)
    # 源层 0,1 → prelude；源层 2 → 循环块；源层 4,5 → coda
    assert torch.equal(out['model.prelude.0.self_attn.q_proj.weight'], src['model.layers.0.self_attn.q_proj.weight'])
    assert torch.equal(out['model.prelude.1.mlp.down_proj.weight'], src['model.layers.1.mlp.down_proj.weight'])
    assert torch.equal(out['model.recurrent.self_attn.q_proj.weight'], src['model.layers.2.self_attn.q_proj.weight'])
    assert torch.equal(out['model.coda.0.input_layernorm.weight'], src['model.layers.4.input_layernorm.weight'])
    assert torch.equal(out['model.coda.1.mlp.up_proj.weight'], src['model.layers.5.mlp.up_proj.weight'])
    assert not any(k.startswith('model.layers.') for k in out)


def test_init_loads_backbone_and_runs_forward():
    tok = StubTokenizer(300)
    tok.unk_token_id = -1
    ensure_omni_tokens(tok)
    hf = _tiny_qwen(layers=6)
    model, result = init_omni_from_lm(hf, tok, prelude_layers=2, coda_layers=2, max_loop_iters=3)
    assert not [k for k in result.unexpected_keys]
    # 缺失的应只剩新增的循环块参数（LTI / LoRA / ACT / 循环块输出归一化）、多模态投影与编码器（编码器不加载）
    new_prefixes = ('model.recurrent.injection', 'model.recurrent.lora', 'model.recurrent.act', 'model.recurrent.norm', 'audio_proj',
                    'vision_proj', 'talker', 'audio_encoder', 'vision_encoder', 'audio_processor', 'vision_processor')
    unexpected_missing = [k for k in result.missing_keys if not k.startswith(new_prefixes)]
    assert unexpected_missing == [], unexpected_missing
    src = hf.state_dict()
    assert torch.equal(model.model.embed_tokens.weight[:300], src['model.embed_tokens.weight'])
    ids = torch.randint(0, 300, (1, 12))
    with torch.no_grad():
        logits = model(ids).logits
    assert logits.shape[-1] == len(tok) and torch.isfinite(logits).all()
