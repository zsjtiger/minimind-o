"""自动评测入口。

用法（仓库根目录）：
    uv run python scripts/bench.py --task mc  --data eval/mc.jsonl  --model out/baize-v
    uv run python scripts/bench.py --task vqa --data eval/vqa.jsonl --model out/baize-v --image_dir eval/images
    uv run python scripts/bench.py --task asr --data eval/asr.jsonl            # 只需 ref/hyp，不加载模型
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from baize_v.eval.runner import read_jsonl, run_mc, run_vqa, run_asr
from baize_v.export import load_pretrained
from baize_v.paths import TOKENIZER_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['mc', 'vqa', 'asr'])
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default=None, help='load_pretrained 目录（mc / vqa 必需）')
    parser.add_argument('--image_dir', default='')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--char_level', type=int, default=0, help='asr 按字符计算（中文等无空格语言）')
    parser.add_argument('--out', default=None, help='结果 json 路径')
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    if args.task == 'asr':
        result = run_asr(rows, char_level=bool(args.char_level))
    else:
        if args.model is None:
            parser.error('mc / vqa 需要 --model')
        model = load_pretrained(args.model, device=args.device)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        if args.task == 'mc':
            result = run_mc(model, tokenizer, rows, device=args.device)
        else:
            def pixel_fn(rel):
                from PIL import Image
                image = Image.open(os.path.join(args.image_dir, rel)).convert('RGB')
                return {k: v.to(args.device) for k, v in model.vision_processor(images=image, return_tensors='pt').items()}
            result = run_vqa(model, tokenizer, rows, pixel_fn, device=args.device)
            result.pop('details')
    result['task'] = args.task
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
