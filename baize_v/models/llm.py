import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from ..config import BaizeConfig


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: BaizeConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: BaizeConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: BaizeConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1: 
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
                topk_weight = top1 - top1.detach() + 1.0 # k=1: 1.0 forward on purpose, gradient via straight-through
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class BaizeBlock(nn.Module):
    def __init__(self, layer_id: int, config: BaizeConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

def loop_index_embedding(h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0):
    """圈数正弦嵌入：加到 h 的前 loop_dim 个通道（类 RoPE，作用于循环深度），让共享权重在不同圈执行不同功能。"""
    freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim))
    angles = loop_t * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    full[:loop_dim] = emb
    return h + full


class LoRAAdapter(nn.Module):
    """深度 LoRA：跨圈共享 down / B，每圈一个 scale 向量。推理圈数超过训练圈数时 scale 钳制到最后一圈（深度外推）。"""

    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.scale = nn.Embedding(max_loops, rank)

    def forward(self, x, loop_t: int):
        t_idx = min(loop_t, self.scale.num_embeddings - 1)
        s = self.scale(torch.tensor(t_idx, device=x.device))
        return (self.down(x) * s) @ self.B


class LTIInjection(nn.Module):
    """LTI 稳定注入：h' = A·h + B·e + f，A = exp(-exp(log_dt + log_A)) ∈ (0, 1)。

    谱半径 ρ(A) < 1 由参数化保证，与训练动态无关；log 空间计算避免 0·inf。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self):
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(self, h, e, f):
        return self.get_A() * h + self.B * e + f


class ACTHalting(nn.Module):
    """每位置停机概率预测器。"""

    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.halt(h)).squeeze(-1)  # [B, T]


class RecurrentBlock(nn.Module):
    """共享权重的循环块：一次前向跑 n_loops 圈。

    单圈更新：h_{t+1} = A·h_t + B·e + Block(RMSNorm(h_t + e)) + LoRA_t(RMSNorm(h_t + e))
    每圈一份 KV cache（past_key_values[t]），解码时各圈深度各自缓存。
    子模块命名与 BaizeBlock 一致（self_attn / input_layernorm / post_attention_layernorm / mlp），
    便于从扁平层的旧权重迁移。
    """

    def __init__(self, config: BaizeConfig):
        super().__init__()
        self.config = config
        self.max_loops = config.max_loop_iters
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.injection = LTIInjection(config.hidden_size)
        self.act = ACTHalting(config.hidden_size)
        self.lora = LoRAAdapter(config.hidden_size, config.lora_rank, config.max_loop_iters)
        self.loop_dim = max(2, int(config.hidden_size * config.loop_emb_frac))

    def _loop_step(self, h, e, position_embeddings, t, past_kv, use_cache, attention_mask):
        h_loop = loop_index_embedding(h, t, self.loop_dim)
        x = self.norm(h_loop + e)
        attn_out, present = self.self_attn(self.input_layernorm(x), position_embeddings, past_kv, use_cache, attention_mask)
        f = attn_out + self.mlp(self.post_attention_layernorm(x))
        f = f + self.lora(x, t)
        return self.injection(h, e, f), present

    def forward(self, e, position_embeddings, past_key_values=None, use_cache=False, attention_mask=None, n_loops=None):
        """e: Prelude 输出（冻结注入信号）。返回 (h_out, presents, moe_aux, ponder_cost)。"""
        n_loops = n_loops or self.max_loops
        bsz, seq_len, _ = e.shape
        h = e
        halted = torch.zeros(bsz, seq_len, device=e.device, dtype=torch.bool)
        cumulative_p = torch.zeros(bsz, seq_len, device=e.device)
        n_updates = torch.zeros(bsz, seq_len, device=e.device)
        remainders = torch.zeros(bsz, seq_len, device=e.device)
        h_out = torch.zeros_like(h)
        presents, moe_aux = [], []

        for t in range(n_loops):
            past_kv = past_key_values[t] if past_key_values is not None else None
            h, present = self._loop_step(h, e, position_embeddings, t, past_kv, use_cache, attention_mask)
            presents.append(present)
            if isinstance(self.mlp, MOEFeedForward):
                moe_aux.append(self.mlp.aux_loss)

            if self.config.use_act:
                p = self.act(h)
                still_running = (~halted).float()
                remainder = (1.0 - cumulative_p).clamp(min=0)
                # 最后一圈仍未停机的位置把剩余概率补满，保证各圈权重之和为 1
                stop_now = (cumulative_p + p >= self.config.act_threshold) | (t == n_loops - 1)
                weight = torch.where(stop_now, remainder, p) * still_running
                h_out = h_out + weight.unsqueeze(-1).to(h.dtype) * h
                n_updates = n_updates + still_running
                remainders = remainders + torch.where(stop_now, remainder, torch.zeros_like(remainder)) * still_running
                cumulative_p = cumulative_p + p * still_running
                halted = halted | stop_now
                # 缓存模式下必须跑满所有圈，保证每圈 KV 长度一致
                if not use_cache and halted.all():
                    break
            else:
                h_out = h
                n_updates = n_updates + 1

        moe_loss = torch.stack(moe_aux).mean() if moe_aux else e.new_zeros(())
        # ponder cost = N + R：N 不可导（计数），R 对停机前各圈 p 有梯度
        ponder = (n_updates + remainders).mean() if self.config.use_act else e.new_zeros(())
        return h_out, presents, moe_loss, ponder


class BaizeModel(nn.Module):
    """RDT 主干：Embedding → Prelude(P) → Recurrent(循环 T 圈) → Coda(C) → final norm。

    KV cache 为扁平列表，顺序：[prelude_0..P-1, 循环第 0..T-1 圈, coda_0..C-1]。
    """
    def __init__(self, config: BaizeConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.prelude = nn.ModuleList([BaizeBlock(i, config) for i in range(config.prelude_layers)])
        self.recurrent = RecurrentBlock(config)
        self.coda = nn.ModuleList([BaizeBlock(i, config) for i in range(config.coda_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self._rope_checked = False

    def num_cache_slots(self, n_loops=None):
        return len(self.prelude) + (n_loops or self.config.max_loop_iters) + len(self.coda)

    def backbone(self, hidden_states, position_embeddings, past_key_values=None, use_cache=False, attention_mask=None, n_loops=None):
        """输入已完成 embedding（及多模态注入）的隐状态。

        返回 (h, bridge, presents, aux)：
            h      — Coda 之后、final norm 之前的隐状态
            bridge — 循环块输出，作为 talker 的桥接信号
            aux    — MoE 负载均衡损失 + ponder cost 系数
        """
        n_loops = n_loops or self.config.max_loop_iters
        P = len(self.prelude)
        if past_key_values is None:
            past_key_values = [None] * self.num_cache_slots(n_loops)
        presents, moe_aux = [], []
        h = hidden_states
        for i, layer in enumerate(self.prelude):
            h, present = layer(h, position_embeddings, past_key_values[i], use_cache, attention_mask)
            presents.append(present)
            if isinstance(layer.mlp, MOEFeedForward): moe_aux.append(layer.mlp.aux_loss)
        e = h  # 冻结注入信号
        h, rec_presents, rec_moe, ponder = self.recurrent(
            e, position_embeddings, past_key_values[P:P + n_loops], use_cache, attention_mask, n_loops)
        presents += rec_presents
        if isinstance(self.recurrent.mlp, MOEFeedForward): moe_aux.append(rec_moe)
        bridge = h
        for j, layer in enumerate(self.coda):
            h, present = layer(h, position_embeddings, past_key_values[P + n_loops + j], use_cache, attention_mask)
            presents.append(present)
            if isinstance(layer.mlp, MOEFeedForward): moe_aux.append(layer.mlp.aux_loss)
        aux = sum(moe_aux, h.new_zeros(())) + self.config.act_ponder_coef * ponder
        return h, bridge, presents, aux

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, n_loops=None, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        start_pos = past_key_values[0][0].shape[1] if past_key_values and past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 元设备初始化（transformers>=5.x）会丢失 RoPE 缓冲区，首次前向时重算一次
        if not self._rope_checked:
            self._rope_checked = True
            if self.freqs_cos[0, 0] == 0:
                freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
                self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        h, _, presents, aux = self.backbone(hidden_states, position_embeddings, past_key_values, use_cache, attention_mask, n_loops)
        return self.norm(h), presents, aux


def remap_legacy_state_dict(sd, config):
    """把旧版扁平层（model.layers.*）的主干权重映射到 RDT 结构。

    旧第 0..P-1 层 → prelude；旧第 P 层 → recurrent（循环块共享的一组权重）；
    旧最后 C 层 → coda。中间层被丢弃。新增的 LTI / LoRA / ACT / 圈数归一化参数不在旧权重中，保持随机初始化。
    """
    layer_ids = [int(k.split('.')[2]) for k in sd if k.startswith('model.layers.')]
    if not layer_ids:
        return sd
    P, C, L = config.prelude_layers, config.coda_layers, max(layer_ids) + 1
    out = {}
    for k, v in sd.items():
        if not k.startswith('model.layers.'):
            out[k] = v
            continue
        parts = k.split('.')
        i, rest = int(parts[2]), '.'.join(parts[3:])
        if i < P:
            out[f'model.prelude.{i}.{rest}'] = v
        elif i == P:
            out[f'model.recurrent.{rest}'] = v
        elif i >= L - C:
            out[f'model.coda.{i - (L - C)}.{rest}'] = v
    return out


class BaizeForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = BaizeConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: BaizeConfig = None):
        self.config = config or BaizeConfig()
        super().__init__(self.config)
        self.model = BaizeModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()
        # HF 初始化会把 Linear bias 置 0（停机概率 p≈0.5，两圈即停），改为配置的负偏置
        with torch.no_grad():
            self.model.recurrent.act.halt.bias.fill_(self.config.act_init_bias)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}


# 旧名称兼容
MiniMindModel, MiniMindForCausalLM, MiniMindBlock = BaizeModel, BaizeForCausalLM, BaizeBlock
