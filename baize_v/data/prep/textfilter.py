"""文本清洗与过滤。"""
import hashlib
import re
import unicodedata

_CTRL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SPACES = re.compile(r'[ \t]+')


def clean_text(text):
    text = unicodedata.normalize('NFKC', text or '')
    text = _CTRL.sub('', text).replace('\r\n', '\n').replace('\r', '\n')
    text = _SPACES.sub(' ', text)
    return '\n'.join(line.strip() for line in text.split('\n')).strip()


def text_ok(text, min_chars=2, max_chars=2000):
    return min_chars <= len(text) <= max_chars


class Deduper:
    """按内容哈希去重（只保存哈希，内存随唯一样本数线性增长）。"""

    def __init__(self):
        self.seen = set()

    def is_new(self, *parts):
        key = hashlib.sha1('\x1f'.join(p if isinstance(p, str) else repr(p) for p in parts).encode('utf-8')).digest()
        if key in self.seen:
            return False
        self.seen.add(key)
        return True
