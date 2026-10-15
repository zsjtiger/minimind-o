"""各数据源的转换器：把原始数据转成统一样本（make_record）。

每个转换器是生成器，逐条 yield 记录，由 run_builder 统一做清洗、校验、去重与统计。
字段名以参数形式传入，因为不同数据集的列名可能不同，使用前请用样本核对。
"""
import json
import os
import random
from collections import Counter

import pyarrow.parquet as pq

from .media import normalize_image, normalize_audio
from .schema import make_record, validate_record, ShardWriter
from .textfilter import clean_text, text_ok, Deduper

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')
_IMG_PROMPTS = ["Describe this image.", "Briefly describe what you see in this picture."]


def _read_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


def coco_caption(annotations, image_dir, captions_per_image=1, seed=0, **_):
    """COCO 风格的 captions 标注（images + annotations 两个列表）。"""
    with open(annotations, encoding='utf-8') as f:
        ann = json.load(f)
    files = {im['id']: im['file_name'] for im in ann['images']}
    per_image = {}
    for a in ann['annotations']:
        per_image.setdefault(a['image_id'], []).append(a['caption'])
    rng = random.Random(seed)
    for image_id in sorted(per_image):
        if image_id not in files:
            continue
        path = os.path.join(image_dir, files[image_id])
        if not os.path.exists(path):
            continue
        caps = rng.sample(per_image[image_id], min(captions_per_image, len(per_image[image_id])))
        image = _read_bytes(path)
        for cap in caps:
            yield {'conversations': [{'role': 'user', 'content': '<image>\n' + rng.choice(_IMG_PROMPTS)},
                                     {'role': 'assistant', 'content': cap}],
                   'image': image, 'modality': 'image_text'}


def vqa_jsonl(jsonl, image_dir, **_):
    """每行 {"image": 相对路径, "question": ..., "answer": ...}，适用于 VQAv2 / GQA / OK-VQA 等整理后的数据。"""
    with open(jsonl, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            path = os.path.join(image_dir, row['image'])
            if not os.path.exists(path):
                continue
            yield {'conversations': [{'role': 'user', 'content': '<image>\n' + row['question']},
                                     {'role': 'assistant', 'content': row['answer']}],
                   'image': _read_bytes(path), 'modality': 'image_text'}


def _spread_indices(n, k):
    """在 [0, n) 中均匀取 k 个互不相同的下标（k <= n）。"""
    if k <= 1:
        return [n // 2]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def video_caption_frames(jsonl, frames_root, max_frames=4, **_):
    """视频描述：每行 {"video_id": ..., "caption": ...}；frames_root/<video_id>/ 下为已抽好的帧。

    均匀取至多 max_frames 帧，每帧在对话中对应一个 <image> 标记。
    抽帧示例：ffmpeg -i video.mp4 -vf fps=1 frames_root/<video_id>/%04d.jpg
    """
    with open(jsonl, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            folder = os.path.join(frames_root, row['video_id'])
            if not os.path.isdir(folder):
                continue
            frames = sorted(x for x in os.listdir(folder) if x.lower().endswith(IMAGE_EXTS))
            if not frames:
                continue
            picks = _spread_indices(len(frames), min(max_frames, len(frames)))
            tags = ''.join('<image>\n' for _ in picks)
            yield {'conversations': [{'role': 'user', 'content': tags + 'Describe this video.'},
                                     {'role': 'assistant', 'content': row['caption']}],
                   'image': [_read_bytes(os.path.join(folder, frames[i])) for i in picks], 'modality': 'video_text'}


def _audio_bytes(cell):
    """兼容 HF Audio 列的两种形式：裸字节，或 {"bytes": ..., "path": ...}。"""
    if isinstance(cell, dict):
        return cell.get('bytes')
    return cell


def speech_dialog_parquet(parquet, question_col, answer_col, round_col=None, **_):
    """语音输入、文本输出的对话（如 VoiceAssistant-400K 一类单轮数据）。

    只保留单轮样本（round_col 为 1 或不存在）。回答的语音 token 不会被使用：这类数据的
    语音 token 不是 Mimi 码本，不能作为语音输出的训练目标。
    """
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if round_col and row.get(round_col) not in (None, 1):
                continue
            audio = _audio_bytes(row.get(question_col))
            if not audio:
                continue
            yield {'conversations': [{'role': 'user', 'content': ''},
                                     {'role': 'assistant', 'content': row[answer_col]}],
                   'question_audio': audio, 'modality': 'speech_text'}


def chat_jsonl(jsonl, **_):
    """纯文本对话：每行 {"conversations": [{"role", "content"}, ...]}。"""
    with open(jsonl, encoding='utf-8') as f:
        for line in f:
            yield {'conversations': json.loads(line)['conversations'], 'modality': 'text'}


BUILDERS = {
    'coco_caption': coco_caption,
    'vqa_jsonl': vqa_jsonl,
    'video_caption_frames': video_caption_frames,
    'speech_dialog_parquet': speech_dialog_parquet,
    'chat_jsonl': chat_jsonl,
}


def _prepare_media(raw, stats):
    """图像与音频的标准化；任一图像损坏则丢弃整个样本，避免 <image> 标记与图像错位。
    返回 (image_list, question_audios) 或 None。"""
    images = []
    if raw.get('image') is not None:
        sources = raw['image'] if isinstance(raw['image'], (list, tuple)) else [raw['image']]
        for src in sources:
            img = normalize_image(src)
            if img is None:
                stats['drop_bad_image'] += 1
                return None
            images.append(img)
    audios = []
    if raw.get('question_audio') is not None:
        wav = normalize_audio(raw['question_audio'])
        if wav is None:
            stats['drop_bad_audio'] += 1
            return None
        audios = [wav]
    return images, audios


def run_builder(name, out_path, source, modality=None, min_chars=2, max_chars=2000, **args):
    """运行一个转换器：清洗 → 过滤 → 媒体标准化 → 校验 → 去重 → 流式写入。返回统计字典。"""
    builder = BUILDERS[name]
    stats = Counter()
    dedup = Deduper()
    writer = ShardWriter(out_path)
    for raw in builder(**args):
        stats['read'] += 1
        conv = raw['conversations']
        for turn in conv:
            turn['content'] = clean_text(turn['content'])  # NFKC 不改动 <image> 标记
        answer = conv[-1]['content']
        if not text_ok(answer, min_chars, max_chars):
            stats['drop_text_len'] += 1
            continue
        media = _prepare_media(raw, stats)
        if media is None:
            continue
        images, audios = media
        if not dedup.is_new(json.dumps(conv, ensure_ascii=False), b''.join(images), audios[0] if audios else b''):
            stats['drop_duplicate'] += 1
            continue
        rec = make_record(conv, source=source, modality=modality or raw.get('modality', 'text'),
                          image_bytes=images, question_audios=audios)
        errors = validate_record(rec)
        if errors:
            stats['drop_invalid'] += 1
            continue
        writer.write(rec)
    stats['written'] = writer.close()
    return dict(stats)


def convert_parquet(parquet, out_path, source, modality, **_):
    """已经是统一格式（或列子集）的 parquet 直通：只补齐缺失列并校验，不再做媒体处理。"""
    stats = Counter()
    writer = ShardWriter(out_path)
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            stats['read'] += 1
            rec = make_record(json.loads(row['conversations']), source=source, modality=modality,
                              question_audios=row.get('question_audios'), answer_audios=row.get('answer_audios'),
                              image_bytes=row.get('image_bytes') or None, ref_audios=row.get('ref_audios'),
                              spk_emb=row.get('spk_emb'))
            if validate_record(rec):
                stats['drop_invalid'] += 1
                continue
            writer.write(rec)
    stats['written'] = writer.close()
    return dict(stats)
