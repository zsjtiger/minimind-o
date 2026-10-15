"""导出 / 加载 Baize-V 模型目录（config.json + pytorch_model.bin + tokenizer）。

用法（仓库根目录）：
    uv run python scripts/convert.py --torch out/sft_omni_768.pth --out out/baize-v
"""
import argparse
import os

import torch

from baize_v.config import OmniConfig
from baize_v.export import save_pretrained
from baize_v.models import BaizeOmni
from baize_v.models.llm import remap_legacy_state_dict
from baize_v.models.summary import report_load_result
from baize_v.paths import OUT_DIR

MISSING = '/nonexistent-encoder-path'  # 导出只保存主干权重，编码器不加载


def convert_torch2export(torch_path, out_dir, lm_config, dtype=torch.float16):
    model = BaizeOmni(lm_config, audio_encoder_path=MISSING, vision_model_path=MISSING)
    state = remap_legacy_state_dict(torch.load(torch_path, map_location='cpu'), lm_config)
    report_load_result(model.load_state_dict(state, strict=False), label=os.path.basename(torch_path))
    save_pretrained(model, out_dir, dtype=dtype)
    print(f'模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M，已保存到 {out_dir}')
    return out_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch', default=os.path.join(OUT_DIR, 'sft_omni_768.pth'))
    parser.add_argument('--out', default=os.path.join(OUT_DIR, 'baize-v'))
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--prelude_layers', default=2, type=int)
    parser.add_argument('--coda_layers', default=2, type=int)
    parser.add_argument('--max_loop_iters', default=4, type=int)
    args = parser.parse_args()
    cfg = OmniConfig(hidden_size=args.hidden_size, use_moe=bool(args.use_moe), prelude_layers=args.prelude_layers,
                     coda_layers=args.coda_layers, max_loop_iters=args.max_loop_iters)
    convert_torch2export(args.torch, args.out, cfg)
