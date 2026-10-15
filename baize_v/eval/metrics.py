"""评测指标：与数据集无关的纯函数。"""
import re
import string

_PUNCT = set(string.punctuation) | set('，。！？、；：“”‘’（）《》【】')


def normalize_answer(text):
    """VQA 常用的答案归一化：小写、去标点、去冠词、压缩空白。"""
    text = (text or '').lower().strip()
    text = ''.join(ch for ch in text if ch not in _PUNCT)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def vqa_accuracy(prediction, answers):
    """VQA 准确率：预测与 10 个标注答案中相同的个数 / 3，封顶为 1（官方定义的简化版）。"""
    pred = normalize_answer(prediction)
    gts = [normalize_answer(a) for a in answers]
    if not gts:
        return 0.0
    return min(1.0, sum(g == pred for g in gts) / 3.0)


def accuracy(preds, golds):
    if not golds:
        return 0.0
    return sum(int(p == g) for p, g in zip(preds, golds)) / len(golds)


def edit_distance(a, b):
    """Levenshtein 距离（按序列元素，词或字符均可）。"""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(references, hypotheses):
    """词错误率：所有句子的编辑距离之和 / 参考词数之和。中文等无空格语言请传入字符列表。"""
    errors, words = 0, 0
    for ref, hyp in zip(references, hypotheses):
        r = ref.split() if isinstance(ref, str) else list(ref)
        h = hyp.split() if isinstance(hyp, str) else list(hyp)
        errors += edit_distance(r, h)
        words += len(r)
    return errors / words if words else 0.0


def sentence_logprob(token_logprobs):
    """选择题打分：平均 token 对数概率（避免长选项天然得分低）。"""
    return sum(token_logprobs) / max(1, len(token_logprobs))
