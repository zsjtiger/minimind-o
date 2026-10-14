import os
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM


WEIGHT_FILES = {'pytorch_model.bin', 'model.safetensors', 'pytorch_model.bin.index.json', 'model.safetensors.index.json'}


def scan_hf_models(base_dir):
    """扫描 base_dir 下的一级子目录，返回含 HF 权重文件的 {目录名: 绝对路径}。"""
    models = {}
    base_dir = os.path.abspath(base_dir)
    for d in sorted(os.listdir(base_dir), reverse=True):
        full_path = os.path.join(base_dir, d)
        if not os.path.isdir(full_path) or d.startswith('.') or d.startswith('_'):
            continue
        if set(os.listdir(full_path)) & WEIGHT_FILES:
            models[d] = full_path
    return models


def clear_remote_code_cache():
    """清理 transformers 缓存的远程代码模块，切换模型时避免旧代码残留。"""
    for k in list(sys.modules):
        if 'transformers_modules' in k:
            sys.modules.pop(k)


def attach_encoders(model, audio, vision):
    """把 (编码器, 处理器) 挂到模型上。不注册为子模块，避免进入 state_dict。"""
    audio_encoder, audio_processor = audio
    vision_encoder, vision_processor = vision
    object.__setattr__(model, 'audio_encoder', audio_encoder)
    object.__setattr__(model, 'audio_processor', audio_processor)
    object.__setattr__(model, 'vision_encoder', vision_encoder)
    object.__setattr__(model, 'vision_processor', vision_processor)


def load_hf_model(model_path, device, audio, vision):
    """加载 transformers 格式的 Omni 模型及其分词器，返回 (tokenizer, model)。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    attach_encoders(model, audio=audio, vision=vision)
    model = model.half().eval().to(device)
    if model.audio_encoder is not None:
        model.audio_encoder.to(device)
    if model.vision_encoder is not None:
        model.vision_encoder.to(device)
    return tokenizer, model
