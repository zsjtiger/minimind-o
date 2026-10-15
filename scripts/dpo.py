"""DPO 偏好对齐入口。输入 jsonl，每行 {"messages": [...], "chosen": str, "rejected": str}。

用法（仓库根目录）：
    uv run python scripts/dpo.py --data dataset/dpo.jsonl --from_weight sft_omni --save_weight dpo_omni
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from baize_v.config import OmniConfig
from baize_v.models import BaizeOmni
from baize_v.models.llm import remap_legacy_state_dict
from baize_v.models.summary import report_load_result
from baize_v.paths import TOKENIZER_DIR, OUT_DIR, SENSEVOICE_DIR, SIGLIP_DIR
from baize_v.train.dist import setup_seed
from baize_v.train.dpo import make_pair, train_dpo


def load_weights_pth(weight_path, cfg, device):
    """从 out/<name>_<hidden>.pth 加载完整模型（编码器从预训练目录加载）。"""
    model = BaizeOmni(cfg, audio_encoder_path=SENSEVOICE_DIR, vision_model_path=SIGLIP_DIR)
    state = remap_legacy_state_dict(torch.load(weight_path, map_location='cpu'), cfg)
    report_load_result(model.load_state_dict(state, strict=False), label=os.path.basename(weight_path))
    return model.to(device).train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--from_weight', required=True, help='out/<name>_<hidden>.pth 的权重前缀')
    parser.add_argument('--save_dir', default=OUT_DIR)
    parser.add_argument('--save_weight', default='dpo_omni')
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--steps', default=500, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--max_len', default=512, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    setup_seed(42)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    with open(args.data, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pairs = [make_pair(tok, r['messages'], r['chosen'], r['rejected'], max_len=args.max_len) for r in rows]

    cfg = OmniConfig(hidden_size=args.hidden_size)
    model = load_weights_pth(os.path.join(args.save_dir, f'{args.from_weight}_{args.hidden_size}.pth'), cfg, args.device)
    history = train_dpo(model, pairs, pad_id=tok.pad_token_id or 0, beta=args.beta, steps=args.steps,
                        batch_size=args.batch_size, lr=args.lr, log_fn=print)
    print(f'final loss {history[-1][0]:.4f} acc {history[-1][1]:.2f}')

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f'{args.save_weight}_{args.hidden_size}.pth')
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items() if not k.startswith(('audio_encoder.', 'vision_encoder.'))}, path)
    print(f'saved {path}')


if __name__ == '__main__':
    main()
