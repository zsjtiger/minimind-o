"""为文本 SFT 数据补充语音输出监督（answer_audios）。

用法（仓库根目录）：
    uv run python scripts/synth_speech.py --in dataset/prepared/sft_text.parquet --out dataset/speech_out.parquet \
        --tts mypkg.tts:synthesize            # 你自己的 TTS：synthesize(text) -> (wav, sr)

Mimi 编码器默认从 pretrained/mimi 加载；也可以用 --mimi_encode 指定 'module:function' 替换。
"""
import argparse

import pyarrow.parquet as pq
import torch

from baize_v.data.prep.schema import ShardWriter
from baize_v.data.prep.speech_targets import build_answer_audio, load_callable
from baize_v.paths import MIMI_DIR


def mimi_encoder(path=MIMI_DIR, device='cpu'):
    from transformers import MimiModel
    model = MimiModel.from_pretrained(path).eval().to(device)

    def encode(wav24):
        with torch.no_grad():
            x = torch.tensor(wav24, dtype=torch.float32, device=device).view(1, 1, -1)
            return model.encode(x).audio_codes[0].cpu().numpy()   # (1, 8, T) → (8, T)
    return encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='src', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--tts', required=True, help='module:function，签名 synthesize(text) -> (wav, sr)')
    parser.add_argument('--mimi_encode', default=None, help='可选：module:function，替换默认的 Mimi 编码器')
    parser.add_argument('--limit', default=0, type=int)
    args = parser.parse_args()

    tts = load_callable(args.tts)
    encode = load_callable(args.mimi_encode) if args.mimi_encode else mimi_encoder()
    writer = ShardWriter(args.out)
    ok = failed = 0
    for batch in pq.ParquetFile(args.src).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            if args.limit and ok >= args.limit:
                break
            try:
                writer.write(build_answer_audio(row, tts, encode))
                ok += 1
            except Exception as e:  # 单条失败不中断整体
                failed += 1
                print(f'skip: {e}')
    print(f'written {writer.close()} rows, failed {failed}')


if __name__ == '__main__':
    main()
