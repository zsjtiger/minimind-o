"""Baize-V SFT 训练入口。

用法（仓库根目录，已 uv sync）::

    uv run python scripts/train_sft.py --data_path dataset/sft_a2a_mini.parquet --from_weight llm
    uv run torchrun --nproc_per_node 4 scripts/train_sft.py --fsdp 1 --grad_checkpoint 1 ...

完整配方见 docs/training.md 与 scripts/recipes/。
"""
import os
import argparse
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from baize_v.config import OmniConfig
from baize_v.data import OmniDataset, omni_collate_fn
from baize_v.train.loop import train_epoch, TrainContext
from baize_v.train.dist import is_main_process, init_distributed_mode, setup_seed, Logger
from baize_v.train.init import init_omni_model
from baize_v.train.checkpoint import load_checkpoint
from baize_v.train.schedule import SkipBatchSampler
from baize_v.models.summary import log_model_params, report_load_result
from baize_v.models.llm import remap_legacy_state_dict
from baize_v.paths import OUT_DIR, CHECKPOINT_DIR, DATA_DIR, SENSEVOICE_DIR, SIGLIP_DIR

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baize-V SFT")
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
    parser.add_argument('--prelude_layers', default=2, type=int, help="循环前的 Prelude 层数 P")
    parser.add_argument('--coda_layers', default=2, type=int, help="循环后的 Coda 层数 C")
    parser.add_argument('--max_loop_iters', default=8, type=int, help="循环圈数 T（推理可按需覆盖）")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default=os.path.join(DATA_DIR, "train_t2a_mini.parquet"), help="训练数据路径（parquet格式）")
    parser.add_argument("--audio_encoder_dir", type=str, default=SENSEVOICE_DIR, help="音频encoder路径(SenseVoice)")
    parser.add_argument("--vision_dir", type=str, default=SIGLIP_DIR, help="CLIP视觉模型路径")
    parser.add_argument('--from_weight', default='llm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--freeze_backbone', default='none', type=str, choices=['none', 'all', 'last1'], help="冻结主干模型: none=全量训练, all=只训练audio层, last1=只训练最后1层+audio层")
    parser.add_argument('--mode', default='all', type=str, choices=['all', 'audio_proj', 'vision_proj'], help="训练模式: all=全量训练, audio_proj=只训练audio_proj, vision_proj=只训练vision_proj")
    parser.add_argument("--use_swanlab", "--use_wandb", dest="use_swanlab", action="store_true", help="是否使用 SwanLab 记录训练曲线")
    parser.add_argument("--swanlab_project", "--wandb_project", dest="swanlab_project", type=str, default="Baize-V-SFT", help="SwanLab 项目名")
    parser.add_argument('--attn_type', default='gqa', type=str, choices=['gqa', 'mla'], help="注意力类型: gqa=分组查询注意力, mla=多头潜变量注意力")
    parser.add_argument('--grad_checkpoint', default=0, type=int, choices=[0, 1], help="激活重计算（省显存，训练时逐层/逐圈重算）")
    parser.add_argument('--fsdp', default=0, type=int, choices=[0, 1], help="多卡时用 FSDP2 切分参数/梯度/优化器状态（需 torchrun 启动）")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 分布式与随机种子 ==========
    local_rank = init_distributed_mode()
    distributed = dist.is_initialized()
    if distributed:
        args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    setup_seed(42 + (dist.get_rank() if distributed else 0))
    fsdp = bool(args.fsdp) and distributed
    if args.fsdp and not distributed:
        Logger('[warning] --fsdp 1 需要 torchrun 多进程启动，单进程下忽略')
    if fsdp and args.dtype == "float16":
        raise ValueError("FSDP 模式请使用 --dtype bfloat16 或 float32（float16 需要分片 GradScaler）")
    if fsdp and args.use_compile:
        raise ValueError("FSDP 与 --use_compile 暂不同时使用")

    # ========== 2. 目录、模型配置、续训状态 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    omni_config = OmniConfig(
        hidden_size=args.hidden_size,
        prelude_layers=args.prelude_layers, coda_layers=args.coda_layers, max_loop_iters=args.max_loop_iters,
        use_moe=bool(args.use_moe), attn_type=args.attn_type,
    )
    ckp_data = load_checkpoint(omni_config, args.save_weight, CHECKPOINT_DIR) if args.from_resume == 1 else None
    mode = 'fsdp' if fsdp else 'standard'
    restore_optim = ckp_data is not None
    if ckp_data is not None and ckp_data.get('mode', 'standard') != mode:
        Logger(f"[warning] 断点的并行模式（{ckp_data.get('mode', 'standard')}）与当前（{mode}）不同，只恢复模型权重与步数，优化器状态重新开始")
        restore_optim = False

    # ========== 3. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type='cuda', dtype=dtype)

    # ========== 4. SwanLab ==========
    tracker = None
    if args.use_swanlab and is_main_process():
        import swanlab as tracker
        run_id = (ckp_data.get('run_id') or ckp_data.get('wandb_id')) if ckp_data else None  # 兼容旧 checkpoint 的 wandb_id
        resume = 'must' if run_id else None
        run_name = f"Baize-V-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        tracker.init(project=args.swanlab_project, name=run_name, id=run_id, resume=resume)

    # ========== 5. 模型：初始化 → 续训权重 → 冻结 → 梯度检查点（分片之前完成） ==========
    model, tokenizer = init_omni_model(omni_config, from_weight=args.from_weight,
                                       audio_encoder_path=args.audio_encoder_dir,
                                       vision_model_path=args.vision_dir,
                                       save_dir=args.save_dir, device=args.device,
                                       freeze_backbone=args.freeze_backbone, from_resume=args.from_resume)
    if ckp_data is not None:
        report_load_result(model.load_state_dict(remap_legacy_state_dict(ckp_data['model'], omni_config), strict=False), label='resume')

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
    if args.grad_checkpoint:
        model.model.enable_grad_checkpointing(True)
    log_model_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f'Trainable: {trainable:.2f}M | Mode: {args.mode} | Parallel: {mode} | Attn: {args.attn_type} | Freeze: {args.freeze_backbone} | Compile: {"on" if args.use_compile else "off"}')

    # ========== 6. 数据 ==========
    train_ds = OmniDataset(
        args.data_path,
        tokenizer,
        audio_processor=model.audio_processor,
        vision_processor=model.vision_processor,
        max_length=args.max_seq_len,
        image_token_len=model.config.image_token_len
    )
    train_sampler = DistributedSampler(train_ds) if distributed else None

    # ========== 7. 并行包装：FSDP2 逐块分片，或 DDP ==========
    # 各 rank 的随机种子不同，初始化的权重也不同：先把 rank0 的参数与缓冲区广播给所有 rank，
    # 保证分片（FSDP2 不做广播）或副本（DDP）从同一份权重开始
    if distributed:
        for t in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(t.data, src=0)
    if fsdp:
        from torch.distributed.fsdp import fully_shard
        for blk in [*model.model.prelude, model.model.recurrent, *model.model.coda, *model.talker.layers]:
            fully_shard(blk)
        fully_shard(model)
    elif distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device_type == "cuda" else None)

    # 优化器必须在分片之后创建：FSDP2 会把参数替换为 DTensor
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 8. 恢复优化器与进度 ==========
    start_epoch, start_step = 0, 0
    if ckp_data is not None:
        if restore_optim:
            if fsdp:
                from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict, StateDictOptions
                set_optimizer_state_dict(model, optimizer, ckp_data['optimizer'], options=StateDictOptions(full_state_dict=True))
            else:
                optimizer.load_state_dict(ckp_data['optimizer'])
            if ckp_data.get('scaler'):
                scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 9. 训练 ==========
    ctx = TrainContext(args=args, model=model, optimizer=optimizer, scaler=scaler, autocast_ctx=autocast_ctx,
                       omni_config=omni_config, fsdp=fsdp)
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, collate_fn=omni_collate_fn, num_workers=args.num_workers, pin_memory=device_type == "cuda")
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(ctx, epoch, loader, len(loader) + skip, start_step, tracker)
        else:
            train_epoch(ctx, epoch, loader, len(loader), 0, tracker)

    if distributed: dist.destroy_process_group()
