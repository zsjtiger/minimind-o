#!/usr/bin/env bash
# 训练冒烟：单进程 / DDP / FSDP2 / 续训，全部用 CPU 与极小模型，几分钟内跑完。
# 用法（仓库根目录）：bash scripts/smoke_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=out/smoke
rm -rf "$OUT" checkpoints/smoke_*
uv run python scripts/make_smoke_data.py --out dataset/smoke.parquet

COMMON="--data_path dataset/smoke.parquet --batch_size 2 --max_seq_len 64 --hidden_size 64 \
  --prelude_layers 1 --coda_layers 1 --max_loop_iters 2 --attn_type mla --grad_checkpoint 1 \
  --num_workers 0 --dtype float32 --device cpu --save_dir $OUT --log_interval 1 --from_weight none \
  --freeze_backbone none --accumulation_steps 2"

echo "== [1/4] 单进程 MLA + 梯度检查点"
uv run python scripts/train_sft.py $COMMON --epochs 1 --save_weight smoke_single

echo "== [2/4] DDP（2 进程，gloo）"
uv run torchrun --nproc_per_node 2 scripts/train_sft.py $COMMON --epochs 1 --save_weight smoke_ddp

echo "== [3/4] FSDP2（2 进程，gloo）"
uv run torchrun --nproc_per_node 2 scripts/train_sft.py $COMMON --epochs 1 --fsdp 1 --save_weight smoke_fsdp

echo "== [4/4] FSDP2 续训（从断点再训 1 个 epoch）"
uv run torchrun --nproc_per_node 2 scripts/train_sft.py $COMMON --epochs 2 --fsdp 1 --save_weight smoke_fsdp --from_resume 1

echo "== [5/5] DDP 与 FSDP2 同配置训练一步后权重一致（fp16 舍入误差以内）"
COMMON_PARITY="--data_path dataset/smoke.parquet --batch_size 2 --max_seq_len 64 --hidden_size 64 \
  --prelude_layers 1 --coda_layers 1 --max_loop_iters 2 --attn_type mla --num_workers 0 --dtype float32 \
  --device cpu --save_dir $OUT --log_interval 1 --from_weight none --epochs 1"
uv run torchrun --nproc_per_node 2 scripts/train_sft.py $COMMON_PARITY --save_weight parity_ddp
uv run torchrun --nproc_per_node 2 scripts/train_sft.py $COMMON_PARITY --fsdp 1 --save_weight parity_fsdp
uv run python - <<PY
import torch
a = torch.load("$OUT/parity_ddp_64.pth"); b = torch.load("$OUT/parity_fsdp_64.pth")
assert set(a) == set(b), "键名不一致"
diff = max((a[k].float() - b[k].float()).abs().max().item() for k in a)
print(f"max |DDP - FSDP| = {diff:.3e}")
assert diff < 1e-3, "DDP 与 FSDP 权重不一致"
PY

echo "== 检查产物"
ls -la "$OUT" checkpoints/ | grep smoke
echo "SMOKE OK"
