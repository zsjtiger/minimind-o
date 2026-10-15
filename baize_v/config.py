import math

from transformers import PretrainedConfig
from .tokens import AUDIO_VOCAB_SIZE, AUDIO_PAD_TOKEN, AUDIO_STOP_TOKEN, AUDIO_SPK_TOKEN


class BaizeConfig(PretrainedConfig):
    """Recurrent-Depth Transformer 的主干配置。

    结构：tokens → Prelude(P 层) → Recurrent Block(共享权重，循环 T 圈) → Coda(C 层) → logits。
    循环块的机制（LTI 注入、圈数嵌入、深度 LoRA、ACT 停机）参考 LLMBaize/baize 的实现。
    """
    model_type = "baize"
    def __init__(
        self,
        hidden_size=768,
        use_moe=False,
        prelude_layers=2,            # P：循环前的一次性层
        coda_layers=2,               # C：循环后的一次性层
        max_loop_iters=8,            # T：循环圈数（推理时可按需覆盖）
        lora_rank=8,                 # 深度 LoRA 秩
        use_act=True,                # ACT 早停；False 时跑满 T 圈
        act_threshold=0.99,          # ACT 累积停机概率阈值
        act_init_bias=-3.0,          # 停机预测器偏置：初始 p≈0.05，训练初期跑满所有圈
        act_ponder_coef=1e-3,        # ponder cost 系数（鼓励简单 token 早停）
        loop_emb_frac=0.125,         # 圈数正弦嵌入占用的通道比例（D/8）
        attn_type="gqa",             # "gqa" | "mla"（DeepSeek-V2 风格的压缩 KV 缓存）
        kv_lora_rank=128,            # MLA：压缩 KV 隐变量维度（缓存的主体）
        q_lora_rank=256,             # MLA：Q 的低秩维度
        qk_rope_head_dim=32,         # MLA：携带 RoPE 的 q/k 维度
        qk_nope_head_dim=32,         # MLA：不带位置的 q/k 维度
        v_head_dim=32,               # MLA：value 维度
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.use_moe = use_moe
        self.prelude_layers = prelude_layers
        self.coda_layers = coda_layers
        self.max_loop_iters = max_loop_iters
        self.lora_rank = lora_rank
        self.use_act = use_act
        self.act_threshold = act_threshold
        self.act_init_bias = act_init_bias
        self.act_ponder_coef = act_ponder_coef
        self.loop_emb_frac = loop_emb_frac
        self.attn_type = attn_type
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class OmniConfig(BaizeConfig):
    model_type = "baize_v"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_talker_hidden_layers = kwargs.get("num_talker_hidden_layers", 4)
        self.talker_hidden_size = kwargs.get("talker_hidden_size", 768)
        self.audio_ids = kwargs.get("audio_ids", [16]) # "<|audio_pad|>" token id
        self.audio_special_token = kwargs.get("audio_special_token", "<|audio_pad|>")
        self.audio_hidden_size = kwargs.get("audio_hidden_size", 512)
        self.audio_vocab_size = kwargs.get("audio_vocab_size", AUDIO_VOCAB_SIZE)
        self.audio_pad_token = kwargs.get("audio_pad_token", AUDIO_PAD_TOKEN)
        self.audio_stop_token = kwargs.get("audio_stop_token", AUDIO_STOP_TOKEN)
        self.audio_spk_token = kwargs.get("audio_spk_token", AUDIO_SPK_TOKEN)
        self.spk_emb_size = kwargs.get("spk_emb_size", 192)
        self.think_end_ids = kwargs.get("think_end_ids", [26, 234, 234]) # </think>\n\n
        self.image_ids = kwargs.get("image_ids", [12]) # "<|image_pad|>" token id
        self.image_special_token = kwargs.get("image_special_token", "<|image_pad|>")
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)

# 旧名称兼容
MiniMindConfig = BaizeConfig
