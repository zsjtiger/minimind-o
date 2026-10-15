import json

import torch

from baize_v.eval.runner import read_jsonl, run_mc, run_vqa, run_asr
from baize_v.models import BaizeOmni
from _helpers import make_tiny_config, MISSING


def _model():
    torch.manual_seed(0)
    return BaizeOmni(make_tiny_config(), audio_encoder_path=MISSING, vision_model_path=MISSING).eval()


def test_mc_runs_and_returns_accuracy(tmp_path, tokenizer):
    rows = [{'question': '1+1=?', 'choices': ['2', '3', '4'], 'answer': 0},
            {'question': 'Capital of France?', 'choices': ['Paris', 'Rome'], 'answer': 1}]
    result = run_mc(_model(), tokenizer, rows)
    assert result['n'] == 2 and 0.0 <= result['accuracy'] <= 1.0


def test_vqa_runs_with_fake_pixels(tokenizer):
    model = _model()
    object.__setattr__(model, 'vision_encoder', None)
    rows = [{'image': 'a.png', 'question': 'What color?', 'answers': ['red', 'red', 'blue']}]
    pixel_fn = lambda _: None   # 无图像时直接走文本路径，验证解码与打分流程
    result = run_vqa(model, tokenizer, rows, pixel_fn, max_new_tokens=3)
    assert result['n'] == 1 and isinstance(result['details'][0]['pred'], str)


def test_asr_wer_from_pairs(tmp_path):
    path = tmp_path / 'asr.jsonl'
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in [
        {'ref': 'hello world', 'hyp': 'hello word'}, {'ref': '你好 世界', 'hyp': '你好 世界'}]), encoding='utf-8')
    result = run_asr(read_jsonl(str(path)))
    assert abs(result['wer'] - 1 / 4) < 1e-9
