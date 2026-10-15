"""循环深度 Transformer（RDT）的结构与机制测试。"""
import torch
import pytest

from baize_v.models.llm import BaizeForCausalLM, RecurrentBlock, remap_legacy_state_dict
from _helpers import make_tiny_config


def _text_model(seed=0, **overrides):
    torch.manual_seed(seed)
    return BaizeForCausalLM(make_tiny_config(**overrides)).eval()


@pytest.mark.parametrize("use_act", [False, True])
def test_incremental_decoding_matches_full_forward(use_act):
    """prefill + 单步解码（带每圈 KV cache）的最后一个位置，应与整段前向一致（含 ACT 早停）。"""
    model = _text_model(use_act=use_act)
    ids = torch.randint(0, 6000, (1, 10))
    with torch.no_grad():
        full = model(ids).logits[:, -1]
        prefill = model(ids[:, :-1], use_cache=True)
        step = model(ids[:, -1:], past_key_values=prefill.past_key_values, use_cache=True)
    assert torch.allclose(full, step.logits[:, -1], atol=1e-4)


def test_cache_has_one_slot_per_loop():
    model = _text_model()
    cfg = model.config
    out = model(torch.randint(0, 6000, (1, 4)), use_cache=True)
    assert len(out.past_key_values) == cfg.prelude_layers + cfg.max_loop_iters + cfg.coda_layers


def test_lti_transition_is_stable():
    """A = exp(-exp(·)) 必须落在 (0, 1)，保证循环稳定。"""
    block = RecurrentBlock(make_tiny_config())
    A = block.injection.get_A()
    assert torch.all(A > 0) and torch.all(A < 1)


def test_more_loops_change_output():
    model = _text_model(use_act=False)
    ids = torch.randint(0, 6000, (1, 6))
    with torch.no_grad():
        a = model(ids, n_loops=1).logits
        b = model(ids, n_loops=3).logits
    assert not torch.allclose(a, b)


def test_depth_extrapolation_beyond_training_loops():
    """推理圈数超过训练圈数时，LoRA scale 钳制到最后一圈，前向应正常完成。"""
    model = _text_model(use_act=False)
    ids = torch.randint(0, 6000, (1, 6))
    with torch.no_grad():
        out = model(ids, n_loops=model.config.max_loop_iters + 2, use_cache=True)
    assert torch.isfinite(out.logits).all()
    assert len(out.past_key_values) == model.config.prelude_layers + model.config.max_loop_iters + 2 + model.config.coda_layers


def test_gradients_reach_shared_recurrent_block():
    model = _text_model().train()
    ids = torch.randint(0, 6000, (2, 8))
    out = model(ids, labels=ids)
    (out.loss + out.aux_loss).backward()
    rec = model.model.recurrent
    assert rec.injection.B.grad is not None and rec.injection.B.grad.abs().sum() > 0
    assert rec.self_attn.q_proj.weight.grad is not None


def test_legacy_flat_layers_are_remapped():
    cfg = make_tiny_config(prelude_layers=1, coda_layers=1)
    legacy = {f'model.layers.{i}.self_attn.q_proj.weight': torch.zeros(1) for i in range(4)}
    legacy['model.norm.weight'] = torch.zeros(1)
    mapped = remap_legacy_state_dict(legacy, cfg)
    assert 'model.prelude.0.self_attn.q_proj.weight' in mapped       # 第 0 层 → prelude
    assert 'model.recurrent.self_attn.q_proj.weight' in mapped       # 第 1 层 → 循环块
    assert 'model.coda.0.self_attn.q_proj.weight' in mapped          # 最后一层 → coda
    assert not any(k.startswith('model.layers.') for k in mapped)
    assert 'model.norm.weight' in mapped                              # 非层权重保持不变


def test_omni_incremental_decoding_matches_full_forward():
    """Omni（thinker 循环 + talker）的文本 logits 在增量解码时与整段前向一致。"""
    from baize_v.models import BaizeOmni
    from _helpers import MISSING
    torch.manual_seed(0)
    model = BaizeOmni(make_tiny_config(use_act=False), audio_encoder_path=MISSING, vision_model_path=MISSING).eval()
    ids = torch.randint(0, 6000, (1, 8))
    audio = torch.full((1, 8, 8), 2049, dtype=torch.long)
    x = torch.cat([audio, ids.unsqueeze(1)], dim=1)
    with torch.no_grad():
        full = model(x).logits[:, -1]
        prefill = model(x[:, :, :-1], use_cache=True)
        step = model(x[:, :, -1:], past_key_values=prefill.past_key_values, use_cache=True)
    assert torch.allclose(full, step.logits[:, -1], atol=1e-4)


def test_omni_generate_runs_with_recurrent_thinker():
    from baize_v.models import BaizeOmni
    from _helpers import MISSING
    torch.manual_seed(0)
    model = BaizeOmni(make_tiny_config(), audio_encoder_path=MISSING, vision_model_path=MISSING).eval()
    chunks = [c for c, _ in model.stream_generate(torch.tensor([[1, 2, 3, 4]]), eos_token_id=2, max_new_tokens=4, temperature=0.8,
                                                  top_p=0.9, rp=1.0, use_cache=True) if c is not None]
    assert chunks


@pytest.mark.parametrize("use_act", [False, True])
def test_mla_incremental_decoding_matches_full_forward(use_act):
    """MLA 压缩缓存下，增量解码与整段前向的最后一个位置一致。"""
    model = _text_model(attn_type="mla", use_act=use_act)
    ids = torch.randint(0, 6000, (1, 10))
    with torch.no_grad():
        full = model(ids).logits[:, -1]
        prefill = model(ids[:, :-1], use_cache=True)
        step = model(ids[:, -1:], past_key_values=prefill.past_key_values, use_cache=True)
    assert torch.allclose(full, step.logits[:, -1], atol=1e-4)


def test_mla_cache_is_compressed():
    """MLA 每层缓存 kv_lora_rank + rope 维，显著小于 GQA 的 K/V 缓存。"""
    cfg = make_tiny_config(attn_type="mla", kv_lora_rank=8, qk_rope_head_dim=4)
    model = BaizeForCausalLM(cfg).eval()
    out = model(torch.randint(0, 6000, (1, 5)), use_cache=True)
    c_kv, k_rope = out.past_key_values[0]
    assert c_kv.shape[-1] == 8 and k_rope.shape[-1] == 4 and k_rope.shape[2] == 1


def test_grad_checkpointing_matches_plain_training():
    """激活重计算不改变前向输出与梯度（同一随机种子）。"""
    torch.manual_seed(0)
    ref = BaizeForCausalLM(make_tiny_config()).train()
    ckpt = BaizeForCausalLM(make_tiny_config()).train()
    ckpt.load_state_dict(ref.state_dict())
    ckpt.model.enable_grad_checkpointing(True)
    ids = torch.randint(0, 6000, (2, 8))
    la = ref(ids, labels=ids).loss
    lb = ckpt(ids, labels=ids).loss
    la.backward()
    lb.backward()
    assert torch.allclose(la, lb, atol=1e-5)
    g1 = ref.model.recurrent.self_attn.q_proj.weight.grad
    g2 = ckpt.model.recurrent.self_attn.q_proj.weight.grad
    assert torch.allclose(g1, g2, atol=1e-5)
