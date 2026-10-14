"""MiniMind-O SFT 训练入口。

用法（仓库根目录，已 ``pip install -e .``）::

    python scripts/train_sft.py --data_path dataset/sft_a2a_mini.parquet --from_weight llm

完整配方见 docs/training.md 与 scripts/recipes/。
"""
import os
import argparse
import warnings
from types import SimpleNamespace
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from baize_v.config import OmniConfig
from baize_v.data import OmniDataset, omni_collate_fn
from baize_v.train.loop import train_epoch
from baize_v.train.dist import is_main_process, init_distributed_mode, setup_seed, Logger
from baize_v.train.init import init_omni_model, log_model_params
from baize_v.train.checkpoint import omni_checkpoint
from baize_v.train.schedule import SkipBatchSampler
from baize_v.paths import OUT_DIR, CHECKPOINT_DIR, DATA_DIR, SENSEVOICE_DIR, SIGLIP_DIR

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-O SFT")
    parser.add_argument("--save_dir", type=str, default=OUT_DIR, help="模型保存目录")
    parser.add_argument('--save_weight', default='sft_omni', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default=os.path.join(DATA_DIR, "train_t2a_mini.parquet"), help="训练数据路径（parquet格式）")
    parser.add_argument("--audio_encoder_dir", type=str, default=SENSEVOICE_DIR, help="音频encoder路径(SenseVoice)")
    parser.add_argument("--vision_dir", type=str, default=SIGLIP_DIR, help="CLIP视觉模型路径")
    parser.add_argument('--from_weight', default='llm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--freeze_backbone', default='none', type=str, choices=['none', 'all', 'last1'], help="冻结主干模型: none=全量训练, all=只训练audio层, last1=只训练最后1层+audio层")
    parser.add_argument('--mode', default='all', type=str, choices=['all', 'audio_proj', 'vision_proj'], help="训练模式: all=全量训练, audio_proj=只训练audio_proj, vision_proj=只训练vision_proj")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-O-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    omni_config = OmniConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)
    )
    ckp_data = omni_checkpoint(omni_config, weight=args.save_weight, save_dir=CHECKPOINT_DIR) if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-O-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_omni_model(omni_config, from_weight=args.from_weight,
                                        audio_encoder_path=args.audio_encoder_dir,
                                        vision_model_path=args.vision_dir,
                                        save_dir=args.save_dir, device=args.device,
                                        freeze_backbone=args.freeze_backbone, from_resume=args.from_resume)
    
    if args.use_compile == 1:
        model = torch.compile(model)
    
    if model.audio_encoder is not None: model.audio_encoder.to(args.device)
    if model.vision_encoder is not None: model.vision_encoder.to(args.device)
    
    if args.mode == 'audio_proj':
        for p in model.parameters(): p.requires_grad = False
        for p in model.audio_proj.parameters(): p.requires_grad = True
    elif args.mode == 'vision_proj':
        for p in model.parameters(): p.requires_grad = False
        for p in model.vision_proj.parameters(): p.requires_grad = True
    log_model_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f'Trainable: {trainable:.2f}M | Mode: {args.mode} | Freeze: {args.freeze_backbone} | Compile: {"on" if args.use_compile else "off"}')
    
    # scheduled_sampling 现在会自动保护 image/audio token 的连续性
    train_ds = OmniDataset(
        args.data_path, 
        tokenizer, 
        audio_processor=model.audio_processor,
        vision_processor=model.vision_processor,
        max_length=args.max_seq_len,
        image_token_len=model.config.image_token_len
    )
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    ctx = SimpleNamespace(args=args, model=model, optimizer=optimizer, scaler=scaler, autocast_ctx=autocast_ctx, omni_config=omni_config)
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=omni_collate_fn, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(ctx, epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(ctx, epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()

