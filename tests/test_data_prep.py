"""多模态数据预处理的测试：合成的图片、音频与 parquet，覆盖转换、过滤、去重、混合与加载。"""
import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
from PIL import Image

from baize_v.data import OmniDataset
from baize_v.data.prep.builders import run_builder, convert_parquet
from baize_v.data.prep.media import normalize_image, normalize_audio
from baize_v.data.prep.mix import mix_sources
from baize_v.data.prep.schema import make_record, validate_record, ShardWriter, SCHEMA
from baize_v.data.prep.textfilter import clean_text, Deduper


def _png(size=(120, 80), color=(10, 200, 30)):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, format='PNG')
    return buf.getvalue()


def _wav(seconds=1.0, sr=44100, channels=2):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    x = 0.3 * np.sin(2 * np.pi * 440 * t)
    data = np.stack([x] * channels, axis=1) if channels > 1 else x
    buf = io.BytesIO()
    sf.write(buf, data, sr, format='WAV')
    return buf.getvalue()


# ---------- schema & validation ----------
def test_valid_record_passes_and_bad_ones_fail():
    good = make_record([{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}], source='t', modality='text')
    assert validate_record(good) == []
    bad_order = make_record([{'role': 'assistant', 'content': 'x'}, {'role': 'user', 'content': 'y'}], source='t', modality='text')
    assert validate_record(bad_order)
    bad_audio = make_record([{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': 'x'}], source='t', modality='speech_text',
                            question_audios=[b'a', b'b'])
    assert validate_record(bad_audio)
    bad_codes = make_record([{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}], source='t', modality='speech_text',
                            answer_audios=[[1, 2, 3]])
    assert validate_record(bad_codes)


def test_shard_writer_roundtrip(tmp_path):
    out = tmp_path / 'shard.parquet'
    w = ShardWriter(str(out), batch_size=2)
    for i in range(5):
        w.write(make_record([{'role': 'user', 'content': f'q{i}'}, {'role': 'assistant', 'content': f'a{i}'}], source='s', modality='text'))
    assert w.close() == 5
    t = pq.read_table(out)
    assert t.schema.equals(SCHEMA) and t.num_rows == 5


# ---------- media ----------
def test_normalize_image_resizes_and_filters():
    jpg = normalize_image(_png((1200, 600)))
    img = Image.open(io.BytesIO(jpg))
    assert img.format == 'JPEG' and max(img.size) == 512
    assert normalize_image(_png((20, 20))) is None      # 太小
    assert normalize_image(b'not an image') is None     # 损坏


def test_normalize_audio_to_16k_mono_pcm16():
    wav = normalize_audio(_wav(1.0, sr=44100, channels=2))
    info = sf.info(io.BytesIO(wav))
    assert info.samplerate == 16000 and info.channels == 1 and info.subtype == 'PCM_16'
    assert normalize_audio(_wav(0.1)) is None           # 太短
    assert normalize_audio(_wav(31.0)) is None          # 太长


# ---------- text ----------
def test_clean_text_and_dedupe():
    assert clean_text('  你好　世界 \r\n\r\n ok\x07 ') == '你好 世界\n\nok'
    d = Deduper()
    assert d.is_new('a', b'x') and not d.is_new('a', b'x') and d.is_new('a', b'y')


# ---------- builders end-to-end ----------
@pytest.fixture
def coco_dir(tmp_path):
    img_dir = tmp_path / 'images'
    img_dir.mkdir()
    (img_dir / 'a.png').write_bytes(_png())
    (img_dir / 'b.png').write_bytes(_png((30, 30)))  # 太小，应被过滤
    ann = {
        'images': [{'id': 1, 'file_name': 'a.png'}, {'id': 2, 'file_name': 'b.png'}],
        'annotations': [{'image_id': 1, 'caption': 'A cat sleeping on a sofa.'},
                        {'image_id': 1, 'caption': 'A cat sleeping on a sofa.'},   # 重复 → 去重
                        {'image_id': 1, 'caption': 'a  cat   on sofa'},
                        {'image_id': 2, 'caption': 'A tiny picture.'}],
    }
    (tmp_path / 'captions.json').write_text(json.dumps(ann), encoding='utf-8')
    return tmp_path


def test_coco_builder_filters_and_dedupes(coco_dir, tmp_path):
    out = tmp_path / 'coco.parquet'
    stats = run_builder('coco_caption', str(out), source='coco', modality='image_text',
                        annotations=str(coco_dir / 'captions.json'), image_dir=str(coco_dir / 'images'),
                        captions_per_image=3)
    assert stats['drop_bad_image'] == 1
    assert stats['written'] == 2
    t = pq.read_table(out).to_pylist()
    assert all(r['image_bytes'] and r['modality'] == 'image_text' for r in t)
    conv = json.loads(t[0]['conversations'])
    assert conv[0]['content'].startswith('<image>')


def test_speech_dialog_builder_and_dataset_load(tmp_path, tokenizer):
    # 同一列内统一使用 HF Audio 的结构体形式（parquet 不允许同列混用结构体与裸字节）
    rows = [{'question_audio': {'bytes': _wav(1.0), 'path': None}, 'answer': 'Sure, here is an answer.', 'round': 1},
            {'question_audio': {'bytes': _wav(1.0), 'path': None}, 'answer': 'Second round, dropped.', 'round': 2},
            {'question_audio': {'bytes': _wav(0.05), 'path': None}, 'answer': 'Too short audio.', 'round': 1}]
    src = tmp_path / 'voice.parquet'
    pq.write_table(pa.Table.from_pylist(rows), src)
    out = tmp_path / 'speech.parquet'
    stats = run_builder('speech_dialog_parquet', str(out), source='voice', modality='speech_text',
                        parquet=str(src), question_col='question_audio', answer_col='answer', round_col='round')
    assert stats['written'] == 1 and stats['drop_bad_audio'] == 1
    ds = OmniDataset(str(out), tokenizer, max_length=96)
    input_ids, text_labels, audio_labels, audio_inputs, audio_len, pixel_values, spk_emb = ds[0]
    assert input_ids.shape == (9, 95) and (text_labels != -100).any()


def test_audio_cell_accepts_bytes_or_struct():
    from baize_v.data.prep.builders import _audio_bytes
    assert _audio_bytes(b'raw') == b'raw'
    assert _audio_bytes({'bytes': b'wav', 'path': None}) == b'wav'


def test_vqa_builder_skips_missing_images(tmp_path):
    img_dir = tmp_path / 'img'
    img_dir.mkdir()
    (img_dir / 'x.png').write_bytes(_png())
    (tmp_path / 'vqa.jsonl').write_text(
        json.dumps({'image': 'x.png', 'question': 'What color?', 'answer': 'green'}) + '\n' +
        json.dumps({'image': 'missing.png', 'question': 'q', 'answer': 'a'}) + '\n', encoding='utf-8')
    out = tmp_path / 'vqa.parquet'
    stats = run_builder('vqa_jsonl', str(out), source='vqa', modality='image_text',
                        jsonl=str(tmp_path / 'vqa.jsonl'), image_dir=str(img_dir))
    assert stats['written'] == 1


def test_video_builder_uses_middle_frame(tmp_path):
    frames = tmp_path / 'frames' / 'v1'
    frames.mkdir(parents=True)
    for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        (frames / f'{i:04d}.png').write_bytes(_png(color=c))
    (tmp_path / 'msrvtt.jsonl').write_text(json.dumps({'video_id': 'v1', 'caption': 'a video'}) + '\n', encoding='utf-8')
    out = tmp_path / 'video.parquet'
    stats = run_builder('video_caption_frames', str(out), source='msrvtt', modality='video_text',
                        jsonl=str(tmp_path / 'msrvtt.jsonl'), frames_root=str(tmp_path / 'frames'))
    assert stats['written'] == 1
    row = pq.read_table(out).to_pylist()[0]
    green = Image.open(io.BytesIO(row['image_bytes'][0])).convert('RGB').getpixel((5, 5))
    assert green[1] > green[0] and green[1] > green[2]   # 中间帧是绿色


def test_passthrough_converts_existing_parquet(tmp_path):
    src = tmp_path / 'old.parquet'
    rec = make_record([{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}], source='old', modality='text')
    pq.write_table(pa.Table.from_pylist([rec], schema=SCHEMA), src)
    out = tmp_path / 'new.parquet'
    stats = convert_parquet(str(src), str(out), source='sft_text', modality='text')
    assert stats['written'] == 1


# ---------- mixing ----------
def _write_source(path, n, tag):
    w = ShardWriter(str(path))
    for i in range(n):
        w.write(make_record([{'role': 'user', 'content': f'{tag}{i}'}, {'role': 'assistant', 'content': 'x'}], source=tag, modality='text'))
    w.close()


def test_mix_follows_weights_and_reports_upsampling(tmp_path):
    a, b = tmp_path / 'a.parquet', tmp_path / 'b.parquet'
    _write_source(a, 1000, 'a')
    _write_source(b, 20, 'b')
    out = tmp_path / 'mix.parquet'
    report = mix_sources([{'name': 'a', 'path': str(a), 'weight': 3}, {'name': 'b', 'path': str(b), 'weight': 1}],
                         total=400, out_path=str(out), seed=1, report_path=str(tmp_path / 'r.json'))
    assert report['a']['target'] == 300 and report['b']['target'] == 100
    assert report['b']['upsampled'] and not report['a']['upsampled']
    sources = pq.read_table(out).column('source').to_pylist()
    assert sources.count('a') == 300 and sources.count('b') == 100
    assert json.load(open(tmp_path / 'r.json'))['total'] == 400


def test_mix_is_deterministic_for_a_seed(tmp_path):
    a = tmp_path / 'a.parquet'
    _write_source(a, 50, 'a')
    out1, out2 = tmp_path / 'm1.parquet', tmp_path / 'm2.parquet'
    for out in (out1, out2):
        mix_sources([{'name': 'a', 'path': str(a), 'weight': 1}], total=30, out_path=str(out), seed=7)
    assert pq.read_table(out1).equals(pq.read_table(out2))
