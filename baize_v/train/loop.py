import time
import torch
from torch.nn.parallel import DistributedDataParallel

from ..paths import CHECKPOINT_DIR
from .dist import Logger, is_main_process
from .schedule import get_lr
from .checkpoint import omni_checkpoint
from .losses import compute_losses


def train_epoch(ctx, epoch, loader, iters, start_step=0, wandb=None):
    args, model, optimizer, scaler, autocast_ctx, omni_config = ctx.args, ctx.model, ctx.optimizer, ctx.scaler, ctx.autocast_ctx, ctx.omni_config
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

        with autocast_ctx:
            res = model(input_ids, audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values, spk_emb=spk_emb)
            text_loss, audio_loss = compute_losses(res, labels, audio_labels)
            
            loss = (text_loss + audio_loss + res.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0 or step == iters:
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
            if wandb: 
                wandb.log({"loss": current_loss, "text_loss": text_loss_val, 
                          "audio_loss": audio_loss_val, "lr": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if omni_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            clean_state_dict = {k: v for k, v in raw_model.state_dict().items() if not k.startswith('audio_encoder.')}
            torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp)
            omni_checkpoint(omni_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                          epoch=epoch, step=step, wandb=wandb, save_dir=CHECKPOINT_DIR, scaler=scaler)
            model.train()

        del input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb, res, loss
