import os
import torch
import torch.distributed as dist

from ..paths import CHECKPOINT_DIR
from .dist import Logger


def omni_checkpoint(omni_config, weight='pretrain_omni', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir=CHECKPOINT_DIR, **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if omni_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{omni_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{omni_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        # 移除冻结的 audio_encoder / vision_encoder 参数（不需要保存，从预训练路径重新加载）
        clean_state_dict = {k: v for k, v in raw_model.state_dict().items() if not k.startswith('audio_encoder.') and not k.startswith('vision_encoder.')}
        state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None
