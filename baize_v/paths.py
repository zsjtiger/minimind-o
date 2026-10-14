"""项目路径集中定义。全部以仓库根目录为基准，不依赖当前工作目录。

外部预训练权重（SenseVoice、SigLIP、Mimi、CAM++）体积大且不随仓库分发，
默认放在 ``pretrained/``，可通过环境变量 ``BAIZE_PRETRAINED_DIR`` 指向其他位置。
"""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 随仓库分发的资产
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
TOKENIZER_DIR = os.path.join(ASSETS_DIR, "tokenizer")
VAD_PATH = os.path.join(ASSETS_DIR, "vad", "silero_vad.onnx")
SPEAKER_DIR = os.path.join(ASSETS_DIR, "speaker")
EVAL_AUDIO_DIR = os.path.join(ASSETS_DIR, "eval_omni")

# 训练数据（parquet，不入库）
DATA_DIR = os.path.join(ROOT_DIR, "dataset")

# 外部预训练权重（不入库）
PRETRAINED_DIR = os.environ.get("BAIZE_PRETRAINED_DIR", os.path.join(ROOT_DIR, "pretrained"))
SENSEVOICE_DIR = os.path.join(PRETRAINED_DIR, "SenseVoiceSmall")
SIGLIP_DIR = os.path.join(PRETRAINED_DIR, "siglip2-base-p32-256-ve")
MIMI_DIR = os.path.join(PRETRAINED_DIR, "mimi")
CAMPPLUS_DIR = os.path.join(PRETRAINED_DIR, "campplus")

# 训练产物（不入库）
OUT_DIR = os.path.join(ROOT_DIR, "out")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
