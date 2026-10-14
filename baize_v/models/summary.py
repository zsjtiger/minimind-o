from ..train.dist import Logger

# 编码器权重不随 checkpoint 保存，从预训练目录加载，缺失属于预期
ENCODER_PREFIXES = ('audio_encoder.', 'vision_encoder.')


def report_load_result(result, label='weights'):
    """打印 load_state_dict(strict=False) 的缺失键与多余键，避免静默的随机初始化。"""
    missing = [k for k in result.missing_keys if not k.startswith(ENCODER_PREFIXES)]
    unexpected = list(result.unexpected_keys)
    if missing:
        Logger(f'[{label}] 缺失 {len(missing)} 个参数（将保持随机初始化），例如: {missing[:5]}')
    if unexpected:
        Logger(f'[{label}] 忽略 {len(unexpected)} 个多余参数，例如: {unexpected[:5]}')
    return missing, unexpected


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
