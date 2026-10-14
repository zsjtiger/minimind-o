import os


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
