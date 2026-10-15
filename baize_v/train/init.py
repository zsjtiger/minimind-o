import os
import torch
from transformers import AutoTokenizer

from ..models.omni import BaizeOmni
from ..paths import TOKENIZER_DIR, SENSEVOICE_DIR, SIGLIP_DIR, OUT_DIR
from .dist import Logger
from ..models.summary import report_load_result
from ..models.llm import remap_legacy_state_dict


def init_omni_model(omni_config, from_weight='full_sft', tokenizer_path=TOKENIZER_DIR, audio_encoder_path=SENSEVOICE_DIR, vision_model_path=SIGLIP_DIR, save_dir=OUT_DIR, device='cuda', freeze_backbone='none', from_resume=0):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = BaizeOmni(omni_config, audio_encoder_path=audio_encoder_path, vision_model_path=vision_model_path)
    
    if from_weight != 'none':
        moe_suffix = '_moe' if omni_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            weights = remap_legacy_state_dict(torch.load(weight_path, map_location=device), omni_config)
            param_shapes = {k: v.shape for k, v in model.named_parameters()}
            incompatible = {k for k, v in weights.items() if k in param_shapes and v.shape != param_shapes[k]}
            if incompatible:
                Logger(f'跳过shape不匹配的权重: {incompatible}')
                weights = {k: v for k, v in weights.items() if k not in incompatible}
            report_load_result(model.load_state_dict(weights, strict=False), label=os.path.basename(weight_path))
            Logger(f'已加载权重: {weight_path}')
            if from_resume == 0 and omni_config.talker_hidden_size == omni_config.hidden_size:
                n_talker = omni_config.num_talker_hidden_layers
                coda = list(model.thinker.coda)
                has_talker = any(k.startswith('talker.layers.') for k in weights)
                if not has_talker and n_talker > 0:
                    # 从 Coda 的最后若干层初始化 talker（不足时重复使用最后一层）
                    srcs = coda[-n_talker:] if n_talker <= len(coda) else [coda[-1]] * n_talker
                    for dst, src in zip(model.talker.layers, srcs):
                        dst.load_state_dict(src.state_dict())
                    Logger(f'Talker层初始化: 复制 thinker coda 最后 {len(srcs)} 层 → talker layers[0:{n_talker}]')
    
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
        if len(model.model.coda) > 0:
            for param in model.model.coda[-1].parameters():
                param.requires_grad = True
    return model.to(device), tokenizer
