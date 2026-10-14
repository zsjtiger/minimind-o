#!/usr/bin/env bash
# Mini 数据集（单卡，推荐入门）
# 在仓库根目录执行：bash scripts/recipes/mini.sh（请先 uv sync）
# 注释中的时长为参考值（单卡 RTX 3090 估算），仅供参考。
set -e

CUDA_VISIBLE_DEVICES=0 uv run torchrun --master_port 29560 --nproc_per_node 1 scripts/train_sft.py --learning_rate 5e-4 --data_path dataset/sft_t2a_mini.parquet --epochs 1 --batch_size 40 --use_compile 1 --from_weight llm --save_weight sft_zero --max_seq_len 512 --use_wandb --use_moe 0 # epochs * 60min
CUDA_VISIBLE_DEVICES=0 uv run torchrun --master_port 29560 --nproc_per_node 1 scripts/train_sft.py --learning_rate 5e-4 --data_path dataset/sft_a2a_mini.parquet --epochs 1 --batch_size 40 --use_compile 0 --from_weight sft_zero --save_weight sft_zero --max_seq_len 640 --mode audio_proj --use_wandb --use_moe 0 # epochs * 15min
CUDA_VISIBLE_DEVICES=0 uv run torchrun --master_port 29560 --nproc_per_node 1 scripts/train_sft.py --learning_rate 2e-5 --data_path dataset/sft_a2a_mini.parquet --epochs 1 --batch_size 16 --use_compile 0 --from_weight sft_zero --save_weight sft_zero --max_seq_len 768 --use_wandb --use_moe 0 # epochs * 15min
