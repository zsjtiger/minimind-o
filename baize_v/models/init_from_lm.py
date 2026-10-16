"""用开源因果语言模型（Qwen3 / Llama 风格，RMSNorm + q/k norm + SwiGLU、无注意力偏置）初始化 BaizeOmni 的 thinker。

映射规则：
- 结构超参（hidden、头数、KV 头数、head_dim、FFN 宽度、rope_theta、eps）直接从源模型配置复制，保证注意力与 MLP 形状一致。
- 源模型第 0..P-1 层 → prelude，第 P 层 → 循环块（跨圈共享的一份权重），最后 C 层 → coda，中间层丢弃
  （复用 remap_legacy_state_dict）。
- 词表：源 tokenizer 缺少 <|audio_pad|> / <|image_pad|> 时追加到末尾，新行的嵌入初始化为原嵌入的均值。

注意：这只是**近似**的热启动。循环块的更新为 h' = A·h + B·e + f（A≈0.37、B=0.1 初始化），
与标准残差层不等价，且 LTI / LoRA / ACT 参数是随机初始化的，因此初始化后的输出不会与源模型一致，需要继续训练来修复。
"""
import torch

from ..config import OmniConfig
from .llm import remap_legacy_state_dict

AUDIO_PAD, IMAGE_PAD = '<|audio_pad|>', '<|image_pad|>'
MIN_LAYERS_EXTRA = 1  # 循环块至少占一层


def omni_config_from_hf(hf_config, tokenizer, prelude_layers=2, coda_layers=2, max_loop_iters=4, **overrides):
    """从 HF 配置生成 OmniConfig。tokenizer 需已补齐音频 / 图像占位符。"""
    n_layers = hf_config.num_hidden_layers
    if n_layers < prelude_layers + coda_layers + MIN_LAYERS_EXTRA:
        raise ValueError(f'源模型只有 {n_layers} 层，不足以切分为 prelude={prelude_layers}、coda={coda_layers} 与循环块')
    heads = hf_config.num_attention_heads
    kv_heads = getattr(hf_config, 'num_key_value_heads', None) or heads
    head_dim = getattr(hf_config, 'head_dim', None) or hf_config.hidden_size // heads
    rope_theta = getattr(hf_config, 'rope_theta', None) or 1e6
    kwargs = dict(
        hidden_size=hf_config.hidden_size,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=hf_config.intermediate_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=rope_theta,
        max_position_embeddings=getattr(hf_config, 'max_position_embeddings', 32768),
        vocab_size=len(tokenizer),
        tie_word_embeddings=bool(getattr(hf_config, 'tie_word_embeddings', True)),
        prelude_layers=prelude_layers,
        coda_layers=coda_layers,
        max_loop_iters=max_loop_iters,
        audio_ids=[tokenizer.convert_tokens_to_ids(AUDIO_PAD)],
        image_ids=[tokenizer.convert_tokens_to_ids(IMAGE_PAD)],
        audio_special_token=AUDIO_PAD,
        image_special_token=IMAGE_PAD,
    )
    kwargs.update(overrides)
    return OmniConfig(**kwargs)


def ensure_omni_tokens(tokenizer):
    """补齐音频 / 图像占位符；返回新增的 token 数（调用方需据此 resize 源嵌入）。"""
    missing = [t for t in (AUDIO_PAD, IMAGE_PAD) if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if missing:
        tokenizer.add_special_tokens({'additional_special_tokens': missing})
    return len(missing)


def resize_rows(weight, new_rows):
    """把 (V, H) 的嵌入扩展到 (new_rows, H)，新行用原均值初始化。"""
    old_rows = weight.shape[0]
    if new_rows == old_rows:
        return weight
    if new_rows < old_rows:
        return weight[:new_rows].clone()
    extra = weight.mean(dim=0, keepdim=True).repeat(new_rows - old_rows, 1)
    return torch.cat([weight, extra], dim=0)


def lm_state_to_omni(hf_state, config):
    """源模型 state_dict → BaizeOmni 主干的 state_dict（只包含 thinker 部分）。"""
    sd = {}
    for k, v in hf_state.items():
        if k.startswith('model.') or k == 'lm_head.weight':
            sd[k] = v
    sd['model.embed_tokens.weight'] = resize_rows(sd['model.embed_tokens.weight'], config.vocab_size)
    if 'lm_head.weight' in sd:
        sd['lm_head.weight'] = resize_rows(sd['lm_head.weight'], config.vocab_size)
    return remap_legacy_state_dict(sd, config)


def init_omni_from_lm(hf_model, tokenizer, prelude_layers=2, coda_layers=2, max_loop_iters=4, **overrides):
    """返回 (BaizeOmni 模型, load 结果)。音频 / 视觉编码器不在这里加载（路径设为不存在，保持随机或之后单独加载）。"""
    from .omni import BaizeOmni
    missing_encoder = '/nonexistent-encoder-path'
    config = omni_config_from_hf(hf_model.config, tokenizer, prelude_layers, coda_layers, max_loop_iters, **overrides)
    model = BaizeOmni(config, audio_encoder_path=missing_encoder, vision_model_path=missing_encoder)
    state = lm_state_to_omni(hf_model.state_dict(), config)
    result = model.load_state_dict(state, strict=False)
    return model, result
