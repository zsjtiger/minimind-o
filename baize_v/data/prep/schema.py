"""统一的训练样本格式（与 OmniDataset 读取的列一致）。

列：
    conversations   JSON 字符串，多轮对话 [{"role", "content"}]；图像用 "<image>" 标记位置
    question_audios list<binary>，按 user 轮次顺序的 wav 字节（16 kHz 单声道 PCM16）
    answer_audios   list<list<int32>>，按 assistant 轮次的 Mimi 8 层交织码（仅语音输出训练需要）
    image_bytes     list<binary>，图像字节，与 "<image>" 标记按顺序对应
    ref_audios      list<int32>，参考音频的 Mimi 码（音色条件，可为空）
    spk_emb         list<float32>，192 维说话人向量（可为空）
    source          str，数据源名称（用于统计与配比）
    modality        str，text | image_text | video_text | speech_text | ...
显式声明 schema，避免空列表被推断为 null 类型导致合并失败。
"""
import json

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ('conversations', pa.string()),
    ('question_audios', pa.list_(pa.binary())),
    ('answer_audios', pa.list_(pa.list_(pa.int32()))),
    ('image_bytes', pa.list_(pa.binary())),
    ('ref_audios', pa.list_(pa.int32())),
    ('spk_emb', pa.list_(pa.float32())),
    ('source', pa.string()),
    ('modality', pa.string()),
])


def make_record(conversations, *, source, modality, question_audios=None, answer_audios=None,
                image_bytes=None, ref_audios=None, spk_emb=None):
    """image_bytes 可以是单张图的字节，也可以是多张图的列表（顺序与 <image> 标记一致）。"""
    if isinstance(image_bytes, (list, tuple)):
        images = list(image_bytes)
    else:
        images = [image_bytes] if image_bytes else []
    return {
        'conversations': json.dumps(conversations, ensure_ascii=False),
        'question_audios': list(question_audios or []),
        'answer_audios': [list(map(int, a)) for a in (answer_audios or [])],
        'image_bytes': images,
        'ref_audios': list(map(int, ref_audios or [])),
        'spk_emb': list(map(float, spk_emb or [])),
        'source': source,
        'modality': modality,
    }


def validate_record(rec):
    """检查样本的基本一致性，返回错误信息列表（空列表表示通过）。"""
    errors = []
    try:
        conv = json.loads(rec['conversations'])
    except (TypeError, ValueError):
        return ['conversations 不是合法 JSON']
    if not conv or conv[-1]['role'] != 'assistant':
        errors.append('对话必须以 assistant 结尾')
    n_user = sum(t['role'] == 'user' for t in conv)
    if rec['question_audios'] and len(rec['question_audios']) != n_user:
        errors.append(f"question_audios 数量({len(rec['question_audios'])})与 user 轮次({n_user})不一致")
    n_asst = sum(t['role'] == 'assistant' for t in conv)
    if rec['answer_audios'] and len(rec['answer_audios']) != n_asst:
        errors.append('answer_audios 与 assistant 轮次数量不一致')
    for codes in rec['answer_audios']:
        if len(codes) % 8 != 0:
            errors.append('answer_audios 的长度必须是 8 的倍数（8 层交织）')
            break
    if rec['spk_emb'] and len(rec['spk_emb']) != 192:
        errors.append('spk_emb 必须为 192 维')
    return errors


class ShardWriter:
    """流式写 parquet，避免把整个数据源放进内存。"""

    def __init__(self, path, batch_size=256):
        self.path, self.batch_size = path, batch_size
        self.writer = pq.ParquetWriter(path, SCHEMA)
        self.buffer, self.count = [], 0

    def write(self, rec):
        self.buffer.append(rec)
        self.count += 1
        if len(self.buffer) >= self.batch_size:
            self._flush()

    def _flush(self):
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=SCHEMA))
            self.buffer = []

    def close(self):
        self._flush()
        self.writer.close()
        return self.count
