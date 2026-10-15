import pytest

from baize_v.eval.metrics import normalize_answer, vqa_accuracy, accuracy, wer, edit_distance


def test_normalize_answer():
    assert normalize_answer('  The Red, car! ') == 'red car'


def test_vqa_accuracy_follows_the_three_match_rule():
    answers = ['red'] * 2 + ['crimson'] * 8
    assert vqa_accuracy('red', answers) == pytest.approx(2 / 3)
    assert vqa_accuracy('crimson', answers) == 1.0          # 8 次匹配，封顶 1
    assert vqa_accuracy('blue', answers) == 0.0


def test_accuracy():
    assert accuracy(['a', 'b', 'c'], ['a', 'x', 'c']) == pytest.approx(2 / 3)


def test_edit_distance_and_wer():
    assert edit_distance('kitten', 'sitting') == 3
    assert wer(['the cat sat'], ['the cat sat']) == 0.0
    assert wer(['the cat sat'], ['the dog sat']) == pytest.approx(1 / 3)
    assert wer(['我 爱 你'], ['我 爱 泥']) == pytest.approx(1 / 3)
    # 无空格语言按字符计算
    assert wer([list('今天好')], [list('今天号')]) == pytest.approx(1 / 3)
