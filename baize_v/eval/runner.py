"""评测运行器：选择题（对数概率打分）与视觉问答（贪心解码）。

数据格式（jsonl，每行一条）：
    mc   : {"question": str, "choices": [str, ...], "answer": int}
    vqa  : {"image": str, "question": str, "answers": [str, ...]}
    asr  : {"ref": str, "hyp": str}                     # 由外部 ASR 系统产生的假设文本
"""
import json

import torch
import torch.nn.functional as F

from .metrics import vqa_accuracy, accuracy, wer, sentence_logprob


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _prompt_ids(tokenizer, question, image_token_block=''):
    content = (image_token_block + '\n' + question) if image_token_block else question
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    return tokenizer(text).input_ids


@torch.no_grad()
def score_choices(model, tokenizer, question, choices, device='cpu'):
    """返回平均对数概率最高的选项下标。"""
    prompt = _prompt_ids(tokenizer, question)
    scores = []
    for choice in choices:
        cont = tokenizer(choice, add_special_tokens=False).input_ids or [tokenizer.eos_token_id]
        ids = torch.tensor([prompt + cont], dtype=torch.long, device=device)
        logits = model(ids).logits[0]
        logp = F.log_softmax(logits[len(prompt) - 1: len(prompt) - 1 + len(cont)].float(), dim=-1)
        token_lp = logp.gather(1, torch.tensor(cont, device=device).unsqueeze(1)).squeeze(1).tolist()
        scores.append(sentence_logprob(token_lp))
    return max(range(len(choices)), key=lambda i: scores[i])


def run_mc(model, tokenizer, rows, device='cpu'):
    preds = [score_choices(model, tokenizer, r['question'], r['choices'], device) for r in rows]
    golds = [int(r['answer']) for r in rows]
    return {'n': len(rows), 'accuracy': accuracy(preds, golds)}


@torch.no_grad()
def greedy_answer(model, tokenizer, question, pixel_values=None, max_new_tokens=16, device='cpu'):
    image_block = model.config.image_special_token * model.config.image_token_len if pixel_values is not None else ''
    ids = _prompt_ids(tokenizer, question, image_block)
    out = []
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids], dtype=torch.long, device=device), pixel_values=pixel_values).logits[0, -1]
        nxt = int(logits.argmax())
        if nxt == tokenizer.eos_token_id:
            break
        out.append(nxt)
        ids.append(nxt)
    return tokenizer.decode(out, skip_special_tokens=True).strip()


def run_vqa(model, tokenizer, rows, pixel_fn, device='cpu', max_new_tokens=16):
    """pixel_fn(image_path) → pixel_values 张量（或 None）。"""
    scores, details = [], []
    for r in rows:
        pv = pixel_fn(r['image'])
        pred = greedy_answer(model, tokenizer, r['question'], pixel_values=pv, max_new_tokens=max_new_tokens, device=device)
        s = vqa_accuracy(pred, r['answers'])
        scores.append(s)
        details.append({'question': r['question'], 'pred': pred, 'score': s})
    return {'n': len(rows), 'vqa_accuracy': sum(scores) / len(scores) if scores else 0.0, 'details': details}


def run_asr(rows, char_level=False):
    refs = [r['ref'] for r in rows]
    hyps = [r['hyp'] for r in rows]
    if char_level:
        refs, hyps = [list(x.replace(' ', '')) for x in refs], [list(x.replace(' ', '')) for x in hyps]
    return {'n': len(rows), 'wer': wer(refs, hyps)}
