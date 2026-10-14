# 训练数据格式

`baize_v.data.OmniDataset` 读取 parquet 文件，多个文件可用英文逗号分隔传入 `--data_path`。每一行是一个样本。

## 列定义

| 列名 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| `conversations` | string（JSON） | 是 | 多轮对话，形如 `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`。`content` 中的 `<image>` 标记图像插入位置 |
| `question_audios` | list\<binary\> | 否 | 每个 user 轮次对应的音频（wav/mp3 等 `soundfile` 可读的字节），按 user 轮次序号取用 |
| `answer_audios` | list\<list\<int\>\> | 否 | 每个 assistant 轮次的音频 codec 码，8 层交织存放（帧内顺序：第 0 层、第 1 层……第 7 层） |
| `image_bytes` | list\<binary\> 或 binary | 否 | 图像字节，只取第一张 |
| `ref_audios` | list\<int\> | 否 | 参考音频的 codec 码（8 层交织），用于音色条件，训练时 50% 概率丢弃 |
| `spk_emb` | list\<float\>（192 维） | 否 | 说话人向量（CAM++），缺失时用全零 |

只有 `conversations` 为必需列，其余列缺失时对应模态被忽略。

## 样本处理要点

- 多轮对话会随机截断到某一轮，并保证长度不超过 `max_seq_len - 100`。
- 只有最后一个 assistant 轮次参与损失计算。
- 系统提示、思考块（`<think>`）会按一定概率随机增删（见 `baize_v/data/chat.py`）。
- 音频训练时做随机变速、加噪、混响等增强；fbank 做 SpecAugment 遮蔽（见 `baize_v/data/augment.py`）。
- 样本为 9 路输入：8 路音频码 + 1 路文本 token，形状 `(9, max_seq_len - 1)`。

## 最小示例

```python
import json, pyarrow as pa, pyarrow.parquet as pq

rows = {"conversations": [json.dumps([
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你？"},
], ensure_ascii=False)]}
pq.write_table(pa.table(rows), "dataset/demo.parquet")
```

`tests/test_data.py` 中的 `tiny_parquet` 夹具即按此方式生成测试数据。
