import os
import torch
from transformers import AutoTokenizer

from ..models.omni import MiniMindOmni
from ..paths import TOKENIZER_DIR, SENSEVOICE_DIR, SIGLIP_DIR, OUT_DIR
from .dist import Logger


def log_model_params(model, ignore_patterns=['audio_encoder', 'vision_encoder']):
    def should_count(n): return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    cfg = model.config
    n_routed = getattr(cfg, 'n_routed_experts', getattr(cfg, 'num_experts', 0))
    n_active = getattr(cfg, 'num_experts_per_tok', 0)
    n_shared = getattr(cfg, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def init_omni_model(omni_config, from_weight='full_sft', tokenizer_path=TOKENIZER_DIR, audio_encoder_path=SENSEVOICE_DIR, vision_model_path=SIGLIP_DIR, save_dir=OUT_DIR, device='cuda', freeze_backbone='none', from_resume=0):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindOmni(omni_config, audio_encoder_path=audio_encoder_path, vision_model_path=vision_model_path)
    
    if from_weight != 'none':
        moe_suffix = '_moe' if omni_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            param_shapes = {k: v.shape for k, v in model.named_parameters()}
            incompatible = {k for k, v in weights.items() if k in param_shapes and v.shape != param_shapes[k]}
            if incompatible:
                Logger(f'跳过shape不匹配的权重: {incompatible}')
                weights = {k: v for k, v in weights.items() if k not in incompatible}
            model.load_state_dict(weights, strict=False)
            Logger(f'已加载权重: {weight_path}')
            if from_resume == 0 and omni_config.talker_hidden_size == omni_config.hidden_size:
                n_talker = omni_config.num_talker_hidden_layers
                n_thinker = len(model.thinker.layers)
                has_talker = any(k.startswith('talker.layers.') for k in weights)
                if not has_talker and n_talker > 0:
                    for i in range(n_talker):
                        src = n_thinker - n_talker + i
                        model.talker.layers[i].load_state_dict(model.thinker.layers[src].state_dict())
                    Logger(f'Talker层初始化: 复制thinker layers[{n_thinker-n_talker}:{n_thinker}] → talker layers[0:{n_talker}]')
    
    # 冻结策略
    if freeze_backbone == 'all':
        # 冻结整个主干模型
        for param in model.model.parameters():
            param.requires_grad = False
    elif freeze_backbone == 'last1':
        # 冻结除了最后1层之外的所有层
        for param in model.model.parameters():
            param.requires_grad = False
        # 打开最后1层
        if hasattr(model.model, 'layers') and len(model.model.layers) > 0:
            for param in model.model.layers[-1].parameters():
                param.requires_grad = True
    return model.to(device), tokenizer
