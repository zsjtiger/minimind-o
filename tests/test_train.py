import os
from contextlib import nullcontext

import torch
from torch import optim

from baize_v.models.omni import BaizeOmni
from baize_v.train.checkpoint import omni_checkpoint
from baize_v.train.init import init_omni_model
from baize_v.train.losses import compute_losses
from baize_v.train.schedule import get_lr, SkipBatchSampler
from _helpers import make_tiny_config, MISSING


def test_get_lr_schedule_endpoints():
    assert abs(get_lr(0, 100, 1e-3) - 1e-3) < 1e-12
    assert abs(get_lr(100, 100, 1e-3) - 1e-4) < 1e-12


def test_skip_batch_sampler_skips_leading_batches():
    sampler = SkipBatchSampler(list(range(10)), batch_size=3, skip_batches=1)
    assert list(sampler) == [[3, 4, 5], [6, 7, 8], [9]]
    assert len(sampler) == 3


def test_compute_losses_finite(tiny_model):
    ids = torch.randint(0, 6000, (2, 9, 10))
    ids[:, :8, :] = 2049
    labels = torch.randint(0, 6000, (2, 10))
    labels[:, :3] = -100
    audio_labels = torch.randint(0, 2112, (2, 8, 10))
    audio_labels[:, :, :3] = -100
    with torch.no_grad():
        res = tiny_model(ids)
        text_loss, audio_loss = compute_losses(res, labels, audio_labels)
    assert torch.isfinite(text_loss) and torch.isfinite(audio_loss)
    assert text_loss.item() > 0


def test_checkpoint_roundtrip(tmp_path):
    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = BaizeOmni(cfg, audio_encoder_path=MISSING, vision_model_path=MISSING)
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    omni_checkpoint(cfg, weight="t", model=model, optimizer=opt, epoch=1, step=5, save_dir=str(tmp_path))
    assert os.path.exists(tmp_path / "t_64.pth") and os.path.exists(tmp_path / "t_64_resume.pth")

    data = omni_checkpoint(cfg, weight="t", save_dir=str(tmp_path))
    assert data["epoch"] == 1 and data["step"] == 5
    assert set(data["model"]) == set(model.state_dict())
    assert all(v.dtype == torch.float16 for v in data["model"].values())


def test_init_omni_model_without_weights(tmp_path):
    model, tokenizer = init_omni_model(make_tiny_config(), from_weight="none", audio_encoder_path=MISSING,
                                       vision_model_path=MISSING, save_dir=str(tmp_path), device="cpu")
    assert isinstance(model, BaizeOmni)
    assert len(tokenizer) > 0


def test_train_epoch_one_step(tmp_path, monkeypatch, tokenizer):
    """端到端跑一轮：数据 → 前向 → loss → 反向 → 保存，检查点写入临时目录。"""
    from types import SimpleNamespace
    import pyarrow as pa
    import pyarrow.parquet as pq
    import json
    from torch.utils.data import DataLoader
    from baize_v.data import OmniDataset, omni_collate_fn
    from baize_v.train import loop

    rows = {"conversations": [json.dumps([{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}], ensure_ascii=False)] * 4}
    data = tmp_path / "d.parquet"
    pq.write_table(pa.table(rows), data)
    ds = OmniDataset(str(data), tokenizer, max_length=64)
    loader = DataLoader(ds, batch_size=2, collate_fn=omni_collate_fn)

    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = BaizeOmni(cfg, audio_encoder_path=MISSING, vision_model_path=MISSING).train()
    monkeypatch.setattr(loop, "CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    args = SimpleNamespace(device="cpu", epochs=1, learning_rate=1e-4, accumulation_steps=1, grad_clip=1.0,
                           log_interval=1, save_interval=1000, save_dir=str(tmp_path), save_weight="t")
    ctx = SimpleNamespace(args=args, model=model, optimizer=optim.AdamW(model.parameters(), lr=1e-4),
                          scaler=torch.amp.GradScaler("cpu", enabled=False), autocast_ctx=nullcontext(), omni_config=cfg)
    before = {k: v.clone() for k, v in model.state_dict().items() if k == "lm_head.weight"}
    loop.train_epoch(ctx, 0, loader, len(loader), 0, None)

    assert (tmp_path / "t_64.pth").exists()
    assert (tmp_path / "ckpt" / "t_64_resume.pth").exists()
    after = model.state_dict()["lm_head.weight"]
    assert not torch.equal(before["lm_head.weight"], after)
