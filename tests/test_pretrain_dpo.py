import torch

from baize_v.models import BaizeForCausalLM
from baize_v.train.pretrain import pack_sequences, train_pretrain
from baize_v.train.dpo import response_logprob, dpo_loss, collate_pairs, train_dpo
from _helpers import make_tiny_config


def test_pack_sequences_shapes_and_eos():
    data = pack_sequences([[5, 6, 7], [8, 9]], seq_len=3, eos_id=2)
    assert data.shape == (1, 4)
    assert data.flatten().tolist() == [5, 6, 7, 2]


def test_pretrain_reduces_loss_on_repeated_text():
    torch.manual_seed(0)
    model = BaizeForCausalLM(make_tiny_config(use_act=False))
    data = pack_sequences([list(range(10, 60))] * 4, seq_len=16, eos_id=2)
    losses = train_pretrain(model, data, batch_size=4, steps=40, lr=3e-3, log_every=0)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


def test_dpo_loss_matches_closed_form():
    pc, pr = torch.tensor([-1.0]), torch.tensor([-2.0])
    rc, rr = torch.tensor([-1.0]), torch.tensor([-2.0])
    loss, acc = dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert abs(float(loss) - torch.log(torch.tensor(2.0)).item()) < 1e-6   # margin 0 → log 2
    assert float(acc) == 0.0


def test_dpo_increases_preference_margin():
    torch.manual_seed(0)
    policy = BaizeForCausalLM(make_tiny_config(use_act=False))
    chosen = ([1, 2, 3, 40, 41, 2], [0, 0, 0, 1, 1, 1])
    rejected = ([1, 2, 3, 50, 51, 2], [0, 0, 0, 1, 1, 1])
    pairs = [(chosen, rejected)]
    history = train_dpo(policy, pairs, pad_id=0, beta=0.5, steps=30, batch_size=1, lr=5e-3)
    assert history[-1][0] < history[0][0]
    (c_ids, c_mask), (r_ids, r_mask) = collate_pairs(pairs, 0)
    with torch.no_grad():
        assert response_logprob(policy, c_ids, c_mask) > response_logprob(policy, r_ids, r_mask)


def test_pretrained_text_weights_load_into_omni():
    """预训练（BaizeForCausalLM）的键名必须是 BaizeOmni 的子集，SFT 的 --from_weight 才能加载。"""
    from baize_v.models import BaizeOmni
    from _helpers import MISSING
    cfg = make_tiny_config()
    text_keys = set(BaizeForCausalLM(cfg).state_dict().keys())
    omni_keys = set(BaizeOmni(cfg, audio_encoder_path=MISSING, vision_model_path=MISSING).state_dict().keys())
    assert text_keys <= omni_keys, text_keys - omni_keys
