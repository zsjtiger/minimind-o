"""生成一套很小的合成原始数据，用于端到端跑通 prepare_data.py（不代表真实数据的内容或分布）。

输出到 dataset/demo_raw/，各子目录格式与 configs/data_mix_demo.json 中的 build 参数对应。
用法：uv run python scripts/make_demo_raw.py [--n 24]
"""
import argparse
import io
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from PIL import Image

from baize_v.data.prep.schema import make_record, SCHEMA
from baize_v.paths import ROOT_DIR

COLORS = [(230, 60, 60), (60, 200, 90), (60, 90, 230), (230, 200, 40), (160, 60, 220)]


def png(color, size=(96, 96)):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, format='PNG')
    return buf.getvalue()


def wav(seconds, sr=22050, freq=440.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    buf = io.BytesIO()
    sf.write(buf, 0.3 * np.sin(2 * np.pi * freq * t), sr, format='WAV')
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=24, type=int)
    parser.add_argument('--root', default=os.path.join(ROOT_DIR, 'dataset', 'demo_raw'))
    args = parser.parse_args()
    root = args.root
    n = args.n

    # COCO 风格：2 张图里的 1 张故意过小，用来验证过滤
    coco = os.path.join(root, 'coco')
    os.makedirs(os.path.join(coco, 'images'), exist_ok=True)
    images, anns = [], []
    for i in range(n):
        fname = f'img{i:03d}.png'
        size = (96, 96) if i % 8 else (24, 24)
        open(os.path.join(coco, 'images', fname), 'wb').write(png(COLORS[i % len(COLORS)], size))
        images.append({'id': i, 'file_name': fname})
        anns.append({'image_id': i, 'caption': f'A {["red", "green", "blue", "yellow", "purple"][i % 5]} square number {i}.'})
        anns.append({'image_id': i, 'caption': f'a  {["red", "green", "blue", "yellow", "purple"][i % 5]} square  number {i}'})
    os.makedirs(os.path.join(coco), exist_ok=True)
    json.dump({'images': images, 'annotations': anns}, open(os.path.join(coco, 'captions.json'), 'w'), ensure_ascii=False)

    # VQA：jsonl + 图片
    vqa = os.path.join(root, 'vqa')
    os.makedirs(os.path.join(vqa, 'images'), exist_ok=True)
    with open(os.path.join(vqa, 'train.jsonl'), 'w', encoding='utf-8') as f:
        for i in range(n):
            fname = f'q{i:03d}.png'
            open(os.path.join(vqa, 'images', fname), 'wb').write(png(COLORS[(i + 2) % 5]))
            f.write(json.dumps({'image': fname, 'question': 'What color is the square?',
                                'answer': ['red', 'green', 'blue', 'yellow', 'purple'][(i + 2) % 5]}) + '\n')

    # 视频：每段视频一个文件夹，放几帧
    msr = os.path.join(root, 'msrvtt')
    os.makedirs(msr, exist_ok=True)
    with open(os.path.join(msr, 'train.jsonl'), 'w', encoding='utf-8') as f:
        for i in range(max(4, n // 4)):
            folder = os.path.join(msr, 'frames', f'video{i}')
            os.makedirs(folder, exist_ok=True)
            for k in range(3):
                open(os.path.join(folder, f'{k:04d}.png'), 'wb').write(png(COLORS[(i + k) % 5]))
            f.write(json.dumps({'video_id': f'video{i}', 'caption': f'a short clip of a colored square number {i}'}) + '\n')

    # 语音输入：HF Audio 风格的结构体列，回答为文本；包含一条第 2 轮样本与一条过短样本
    voice = os.path.join(root, 'voice')
    os.makedirs(voice, exist_ok=True)
    rows = []
    for i in range(n):
        rows.append({'question_audio': {'bytes': wav(1.0 + (i % 3) * 0.5, freq=300 + 20 * i), 'path': None},
                     'answer': f'Here is the answer to spoken question number {i}.', 'round': 1})
    rows.append({'question_audio': {'bytes': wav(1.0), 'path': None}, 'answer': 'second round, filtered out', 'round': 2})
    rows.append({'question_audio': {'bytes': wav(0.05), 'path': None}, 'answer': 'too short audio, filtered out', 'round': 1})
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(voice, 'train.parquet'))

    # 纯文本 SFT：已是统一格式的 parquet，用于 passthrough
    text = os.path.join(root, 'text')
    os.makedirs(text, exist_ok=True)
    recs = [make_record([{'role': 'user', 'content': f'请用一句话介绍数字{i}。'},
                         {'role': 'assistant', 'content': f'数字{i}是一个整数。'}], source='demo', modality='text')
            for i in range(n * 2)]
    pq.write_table(pa.Table.from_pylist(recs, schema=SCHEMA), os.path.join(text, 'sft.parquet'))
    print(f'demo raw data written to {os.path.abspath(root)}')


if __name__ == '__main__':
    main()
