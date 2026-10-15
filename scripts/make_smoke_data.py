"""生成冒烟测试用的小型 SFT parquet（只含文本对话，无音频/图像）。

用法：uv run python scripts/make_smoke_data.py --out dataset/smoke.parquet
"""
import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='dataset/smoke.parquet')
    parser.add_argument('--rows', default=16, type=int)
    args = parser.parse_args()
    rows = []
    for i in range(args.rows):
        conv = [{"role": "user", "content": f"你好，第{i}个问题"}, {"role": "assistant", "content": f"你好！这是第{i}个回答。"}]
        rows.append(json.dumps(conv, ensure_ascii=False))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    pq.write_table(pa.table({'conversations': rows}), args.out)
    print(f'wrote {args.rows} rows to {args.out}')


if __name__ == '__main__':
    main()
