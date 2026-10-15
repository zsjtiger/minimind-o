"""端到端：生成合成原始数据 → prepare_data 构建并混合 → OmniDataset 逐条读取。"""
import json
import os

import pyarrow.parquet as pq
import pytest

from baize_v.data import OmniDataset
from baize_v.data.prep.schema import validate_record
from baize_v.paths import ROOT_DIR
from scripts import make_demo_raw, prepare_data


@pytest.fixture
def demo_pipeline(tmp_path):
    raw = tmp_path / 'raw'
    import sys
    argv = sys.argv
    try:
        sys.argv = ['make_demo_raw', '--n', '8', '--root', str(raw)]
        make_demo_raw.main()
    finally:
        sys.argv = argv
    with open(os.path.join(ROOT_DIR, 'configs', 'data_mix_demo.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    cfg['prepared_dir'] = str(tmp_path / 'prepared')
    cfg['output'] = str(tmp_path / 'mix.parquet')
    cfg['total'] = 60
    for src in cfg['sources']:
        if 'passthrough' in src:
            src['passthrough'] = src['passthrough'].replace('dataset/demo_raw', str(raw))
        for k, v in list(src.get('build', {}).get('args', {}).items()):
            if isinstance(v, str) and v.startswith('dataset/demo_raw/'):
                src['build']['args'][k] = v.replace('dataset/demo_raw', str(raw))
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding='utf-8')
    assert _run_prepare(cfg_path) == 0
    return tmp_path


def _run_prepare(cfg_path):
    import sys
    argv = sys.argv
    try:
        sys.argv = ['prepare_data', '--config', str(cfg_path)]
        return prepare_data.main()
    finally:
        sys.argv = argv


def test_pipeline_output_is_valid_and_loads(demo_pipeline, tokenizer):
    mix = demo_pipeline / 'mix.parquet'
    report = json.load(open(str(mix) + '.report.json', encoding='utf-8'))
    assert report['total'] == pq.ParquetFile(mix).metadata.num_rows == 60
    rows = pq.read_table(mix).to_pylist()
    assert all(validate_record(r) == [] for r in rows)
    assert {r['modality'] for r in rows} >= {'text', 'image_text', 'video_text', 'speech_text'}
    ds = OmniDataset(str(mix), tokenizer, max_length=128)
    for i in range(min(len(ds), 12)):
        input_ids, text_labels, audio_labels, audio_inputs, audio_len, pixel_values, spk_emb = ds[i]
        assert input_ids.shape == (9, 127)
