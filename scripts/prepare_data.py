"""按配置构建并混合训练数据。

用法（仓库根目录）：
    uv run python scripts/prepare_data.py --config configs/data_mix.json
    uv run python scripts/prepare_data.py --config configs/data_mix.json --only_mix   # 复用已构建的分源文件

流程：每个 enabled 的数据源 → 转换 + 清洗 + 媒体标准化 + 校验 + 去重 → dataset/prepared/<name>.parquet
      → 按权重混合 → output，并写出 <output>.report.json（每个数据源的目标、可用、过采样情况）。
"""
import argparse
import json
import os
import sys

from baize_v.data.prep.builders import run_builder, convert_parquet
from baize_v.data.prep.mix import mix_sources
from baize_v.paths import ROOT_DIR


# 转换器参数中表示文件或目录路径的键（其余字符串参数如列名不是路径）
PATH_KEYS = {'annotations', 'image_dir', 'jsonl', 'frames_root', 'parquet'}


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)


def build_source(src, prepared_dir):
    out = os.path.join(prepared_dir, src['name'] + '.parquet')
    if 'passthrough' in src:
        stats = convert_parquet(_abs(src['passthrough']), out, source=src['name'], modality=src['modality'])
    else:
        spec = src['build']
        stats = run_builder(spec['type'], out, source=src['name'], modality=src['modality'], **spec.get('args', {}))
    return out, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/data_mix.json')
    parser.add_argument('--only_mix', action='store_true', help='跳过构建，直接混合已存在的分源文件')
    args = parser.parse_args()
    with open(_abs(args.config), encoding='utf-8') as f:
        cfg = json.load(f)
    prepared_dir = _abs(cfg['prepared_dir'])
    os.makedirs(prepared_dir, exist_ok=True)

    sources = []
    for src in cfg['sources']:
        if not src.get('enabled', True):
            print(f"[skip] {src['name']}: enabled=false")
            continue
        out = os.path.join(prepared_dir, src['name'] + '.parquet')
        if args.only_mix:
            if not os.path.exists(out):
                print(f"[skip] {src['name']}: 未找到 {out}")
                continue
            stats = {'reused': True}
        else:
            inputs = [src.get('passthrough')] + [v for k, v in src.get('build', {}).get('args', {}).items() if k in PATH_KEYS]
            missing = [p for p in inputs if p and not os.path.exists(_abs(p))]
            if missing:
                print(f"[skip] {src['name']}: 原始数据不存在 {missing}")
                continue
            out, stats = build_source(src, prepared_dir)
            print(f"[built] {src['name']}: {json.dumps(stats, ensure_ascii=False)}")
        if stats.get('written', 1) == 0 and not args.only_mix:
            print(f"[skip] {src['name']}: 清洗后没有可用样本")
            continue
        sources.append({'name': src['name'], 'path': out, 'weight': src['weight']})

    if not sources:
        print('没有可混合的数据源，请检查配置与原始数据路径。', file=sys.stderr)
        return 1
    output = _abs(cfg['output'])
    report = mix_sources(sources, cfg['total'], output, seed=cfg.get('seed', 0), report_path=output + '.report.json')
    for name, entry in report.items():
        print(f"{name:18s} target={entry['target']:>7d} available={entry['available']:>7d} "
              f"ratio={entry.get('ratio', 0):.3f}{'  (过采样)' if entry['upsampled'] else ''}")
    print(f'wrote {output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
