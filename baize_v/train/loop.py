import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from ..paths import CHECKPOINT_DIR
from .dist import Logger, is_main_process
from .schedule import get_lr
from .checkpoint import save_checkpoint
from .losses import compute_losses


@dataclass
class TrainContext:
    """一次训练运行的共享状态：参数、模型、优化器、混合精度相关对象。"""
    args: Any
    model: Any
    optimizer: Any
    scaler: Any
    autocast_ctx: Any
    omni_config: Any
    fsdp: bool = False  # True 时模型为 FSDP2 分片，保存与梯度同步需要特殊处理


def unwrap(model):
    model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(model, '_orig_mod', model)


def collect_state(ctx: TrainContext):
    """收集完整的模型与优化器状态。FSDP 下是集合通信，所有 rank 都必须调用。"""
    if ctx.fsdp:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, StateDictOptions
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return get_model_state_dict(ctx.model, options=opts), get_optimizer_state_dict(ctx.model, ctx.optimizer, options=opts)
    return unwrap(ctx.model).state_dict(), ctx.optimizer.state_dict()


def run_id_of(tracker):
    if tracker is None:
        return None
    if hasattr(tracker, 'get_run'):
        run = tracker.get_run()
        return getattr(run, 'id', None) if run else None
    return getattr(tracker, 'id', None)


def save_all(ctx: TrainContext, epoch, step, tracker, mode):
    """在所有 rank 上收集状态，只在 rank0 落盘。"""
    args = ctx.args
    model_sd, optim_sd = collect_state(ctx)
    if is_main_process():
        save_checkpoint(ctx.omni_config, args.save_weight, args.save_dir, CHECKPOINT_DIR,
                        model_sd, optim_sd, ctx.scaler.state_dict(), epoch, step,
                        run_id=run_id_of(tracker), mode=mode)


def train_epoch(ctx: TrainContext, epoch, loader, iters, start_step=0, tracker=None):
    args, model, optimizer, scaler, autocast_ctx = ctx.args, ctx.model, ctx.optimizer, ctx.scaler, ctx.autocast_ctx
    mode = 'fsdp' if ctx.fsdp else 'standard'
    start_time = time.time()
    for step, (input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        audio_labels = audio_labels.to(args.device)
        audio_lens = audio_lens.to(args.device)
        if audio_inputs is not None:
            audio_inputs = audio_inputs.to(args.device)
        if pixel_values is not None:
            if hasattr(pixel_values, 'keys'):
                pixel_values = {k: v.to(args.device) for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values.to(args.device)
        spk_emb = spk_emb.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        boundary = step % args.accumulation_steps == 0 or step == iters
        if ctx.fsdp:
            # FSDP 梯度累积：只在累积边界才做 reduce-scatter，中间的 micro-batch 跳过同步
            model.set_requires_gradient_sync(boundary)

        with autocast_ctx:
            res = model(input_ids, audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values, spk_emb=spk_emb)
            text_loss, audio_loss = compute_losses(res, labels, audio_labels)
            loss = (text_loss + audio_loss + res.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()
        if boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            text_loss_val = text_loss.item() if isinstance(text_loss, torch.Tensor) else 0
            audio_loss_val = audio_loss.item() if isinstance(audio_loss, torch.Tensor) else 0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, text: {text_loss_val:.4f}, audio: {audio_loss_val:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if tracker:
                tracker.log({"loss": current_loss, "text_loss": text_loss_val,
                             "audio_loss": audio_loss_val, "lr": current_lr, "epoch_time": eta_min})

        if step % args.save_interval == 0 or step == iters:
            model.eval()
            save_all(ctx, epoch, step, tracker, mode)
            model.train()

        del input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb, res, loss
