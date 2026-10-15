"""checkpoint 读写。

两类文件：
    推理权重   <weights_dir>/<weight>_<hidden>[_moe].pth          fp16，不含音频编码器
    续训文件   <resume_dir>/<weight>_<hidden>[_moe]_resume.pth   模型 + 优化器 + scaler + 进度

状态收集（尤其是 FSDP 下的完整 state dict）是集合通信，由调用方在所有 rank 上完成，
这里只负责落盘。
"""
import os
import torch
import torch.distributed as dist

from .dist import Logger

CLEAN_PREFIXES = ('audio_encoder.', 'vision_encoder.')  # 冻结编码器不随 checkpoint 保存，从预训练目录重新加载


def _stem(omni_config, weight):
    moe = '_moe' if omni_config.use_moe else ''
    return f'{weight}_{omni_config.hidden_size}{moe}'


def weights_path(omni_config, weight, weights_dir):
    return os.path.join(weights_dir, _stem(omni_config, weight) + '.pth')


def resume_path(omni_config, weight, resume_dir):
    return os.path.join(resume_dir, _stem(omni_config, weight) + '_resume.pth')


def to_fp16_cpu(state_dict):
    return {k: v.half().cpu() for k, v in state_dict.items() if not k.startswith(CLEAN_PREFIXES)}


def _atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(omni_config, weight, weights_dir, resume_dir, model_sd, optim_sd, scaler_sd,
                    epoch, step, run_id=None, mode='standard'):
    """写入推理权重与续训文件。只应在 rank0 上调用。"""
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(resume_dir, exist_ok=True)
    fp16 = to_fp16_cpu(model_sd)
    _atomic_save(fp16, weights_path(omni_config, weight, weights_dir))
    _atomic_save({
        'model': fp16,
        'optimizer': optim_sd,
        'scaler': scaler_sd,
        'epoch': epoch,
        'step': step,
        'world_size': dist.get_world_size() if dist.is_initialized() else 1,
        'run_id': run_id,
        'mode': mode,
    }, resume_path(omni_config, weight, resume_dir))


def load_checkpoint(omni_config, weight, resume_dir):
    """读取续训文件；GPU 数量变化时把 step 按比例换算。不存在时返回 None。"""
    path = resume_path(omni_config, weight, resume_dir)
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location='cpu')
    saved_ws = data.get('world_size', 1)
    current_ws = dist.get_world_size() if dist.is_initialized() else 1
    if saved_ws != current_ws:
        data['step'] = data['step'] * saved_ws // current_ws
        Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{data["step"]}')
    return data
