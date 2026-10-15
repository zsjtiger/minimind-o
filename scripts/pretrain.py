"""文本预训练入口。输入 jsonl，每行 {"text": ...}。

用法（仓库根目录）：
    uv run python scripts/pretrain.py --data dataset/pretrain.jsonl --steps 2000 --save_weight pretrain
产物 out/<save_weight>_<hidden>.pth 可作为 SFT 的 --from_weight 使用。
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from baize_v.config import OmniConfig
from baize_v.models import BaizeForCausalLM
from baize_v.paths import TOKENIZER_DIR, OUT_DIR
from baize_v.train.dist import setup_seed
from baize_v.train.pretrain import pack_sequences, train_pretrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--save_dir', default=OUT_DIR)
    parser.add_argument('--save_weight', default='pretrain')
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--prelude_layers', default=2, type=int)
    parser.add_argument('--coda_layers', default=2, type=int)
    parser.add_argument('--max_loop_iters', default=4, type=int)
    parser.add_argument('--seq_len', default=512, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--steps', default=1000, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    setup_seed(42)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    with open(args.data, encoding='utf-8') as f:
        texts = [json.loads(line)['text'] for line in f if line.strip()]
    data = pack_sequences([tok(t, add_special_tokens=False).input_ids for t in texts], args.seq_len, tok.eos_token_id)
    cfg = OmniConfig(hidden_size=args.hidden_size, prelude_layers=args.prelude_layers, coda_layers=args.coda_layers,
                     max_loop_iters=args.max_loop_iters)
    model = BaizeForCausalLM(cfg).to(args.device)
    print(f'packed {data.size(0)} sequences of length {args.seq_len}')
    losses = train_pretrain(model, data.to(args.device), args.batch_size, args.steps, lr=args.lr, log_every=50, log_fn=print)
    print(f'final loss {losses[-1]:.4f}')

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f'{args.save_weight}_{args.hidden_size}.pth')
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, path)
    print(f'saved {path}')


if __name__ == '__main__':
    main()
