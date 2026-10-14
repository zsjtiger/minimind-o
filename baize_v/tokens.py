"""音频 token 约定。所有模块从这里引用，避免魔法数字分散在各处。

音频 codec 为 8 层 Mimi 码本，每层 2048 个码字，外加 64 个特殊 token。
"""

MIMI_CODEBOOK_SIZE = 2048          # 每层 Mimi 码字数
NUM_AUDIO_LAYERS = 8               # 码本层数
AUDIO_PAD_TOKEN = 2049             # <|audio_pad|>，占位与无效帧
AUDIO_STOP_TOKEN = 2050            # <|audio_stop|>，音频结束
AUDIO_SPK_TOKEN = 2051             # <|audio_spk|>，说话人条件占位
AUDIO_VOCAB_SIZE = 2112            # 2048 码字 + 64 特殊 token
