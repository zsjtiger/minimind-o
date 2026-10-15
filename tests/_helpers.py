from baize_v.config import OmniConfig

MISSING = "/nonexistent-encoder-path"  # 不存在的路径 → 编码器不加载，测试无需外部权重


def make_tiny_config(**overrides):
    kw = dict(hidden_size=64, prelude_layers=1, coda_layers=1, max_loop_iters=3, num_attention_heads=4, num_key_value_heads=2,
              num_talker_hidden_layers=1, talker_hidden_size=64, use_moe=False, max_position_embeddings=512)
    kw.update(overrides)
    return OmniConfig(**kw)
