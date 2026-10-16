"""用开源因果语言模型热启动 BaizeOmni 的 thinker（Qwen3 / Llama 风格结构）。

用法（仓库根目录，需要能访问模型源，例如 Hugging Face 或本地目录）：
    uv run python scripts/init_from_lm.py --base Qwen/Qwen3-0.6B --save_weight omni_init

产物：
    out/<save_weight>_<hidden>.pth        fp16 权重（可作为 pretrain / SFT 的 --from_weight）
    out/<save_weight>_<hidden>.json       OmniConfig，训练时的结构参数以它为准
    out/<save_weight>_tokenizer/          补齐 <|audio_pad|> / <|image_pad|> 后的 tokenizer

注意：这是近似热启动，初始化后的输出与源模型不一致，需要继续训练。详见 baize_v/models/init_from_lm.py。
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baize_v.models.init_from_lm import ensure_omni_tokens, init_omni_from_lm
from baize_v.models.summary import report_load_result
from baize_v.paths import OUT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, help='HF 模型 id 或本地目录（需为 Qwen3 / Llama 风格结构）')
    parser.add_argument('--save_dir', default=OUT_DIR)
    parser.add_argument('--save_weight', default='omni_init')
    parser.add_argument('--prelude_layers', default=2, type=int)
    parser.add_argument('--coda_layers', default=2, type=int)
    parser.add_argument('--max_loop_iters', default=4, type=int)
    parser.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    ensure_omni_tokens(tokenizer)
    hf_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype).eval()
    model, result = init_omni_from_lm(hf_model, tokenizer, args.prelude_layers, args.coda_layers, args.max_loop_iters)
    report_load_result(result, label=args.base)

    cfg = model.config
    os.makedirs(args.save_dir, exist_ok=True)
    stem = f'{args.save_weight}_{cfg.hidden_size}'
    weight_path = os.path.join(args.save_dir, f'{stem}.pth')
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, weight_path)
    with open(os.path.join(args.save_dir, f'{stem}.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    tok_dir = os.path.join(args.save_dir, f'{args.save_weight}_tokenizer')
    tokenizer.save_pretrained(tok_dir)
    print(f'参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
    print(f'saved {weight_path}, {stem}.json, {tok_dir}')
    print(f'训练参数：--hidden_size {cfg.hidden_size} --prelude_layers {cfg.prelude_layers} '
          f'--coda_layers {cfg.coda_layers} --max_loop_iters {cfg.max_loop_iters} --from_weight {args.save_weight}')


if __name__ == '__main__':
    main()
