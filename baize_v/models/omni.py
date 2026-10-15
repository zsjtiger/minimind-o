import torch
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from ..config import OmniConfig
from .llm import BaizeForCausalLM, MOEFeedForward, precompute_freqs_cis
from .projectors import MMAudioProjector, MMVisionProjector
from .talker import TalkerModule
from .encoders import load_sensevoice, load_vision
from ..inference.generate import stream_generate as stream_generate_tokens
from .fusion import inject_audio_features as inject_audio, inject_vision_features as inject_vision
from ..paths import SENSEVOICE_DIR, SIGLIP_DIR


def unused_param_anchor(*modules):
    """零值锚点：让本次前向未用到的参数也进入计算图，避免 DDP 报 unused parameter（数值为 0，不影响 loss）。"""
    return sum(p.sum() for m in modules for p in m.parameters()) * 0


class BaizeOmni(BaizeForCausalLM):
    config_class = OmniConfig
    def __init__(self, config: OmniConfig = None, audio_encoder_path=SENSEVOICE_DIR, vision_model_path=SIGLIP_DIR):
        config = config or OmniConfig()
        super().__init__(config)
        object.__setattr__(self, 'thinker', self.model)  # alias: self.thinker == self.model
        object.__setattr__(self.model, 'lm_head', self.lm_head)  # alias: self.thinker.lm_head == self.lm_head
        self.talker = TalkerModule(config)
        self.audio_proj = MMAudioProjector(config.audio_hidden_size, config.hidden_size)
        self.vision_proj = MMVisionProjector(config.image_hidden_size, config.hidden_size)
        self.audio_pad_token, self.audio_stop_token, self.audio_spk_token = config.audio_pad_token, config.audio_stop_token, config.audio_spk_token
        audio_encoder, audio_processor = self.load_sensevoice(audio_encoder_path)
        object.__setattr__(self, 'audio_encoder', audio_encoder)
        object.__setattr__(self, 'audio_processor', audio_processor)
        vision_encoder, vision_processor = self.load_vision(vision_model_path)
        object.__setattr__(self, 'vision_encoder', vision_encoder)
        object.__setattr__(self, 'vision_processor', vision_processor)
        self._rope_checked = False

    load_sensevoice = staticmethod(load_sensevoice)

    @torch.compiler.disable
    def encode_audio_inputs(self, audio_inputs, audio_lens=None):
        if (audio_inputs is None) or (self.audio_encoder is None) or (not audio_inputs.any()): return None
        batch_mask = audio_inputs.flatten(1).any(1)
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        valid_fbank = audio_inputs[batch_mask].to(dtype=enc_dtype)
        if audio_lens is not None:
            valid_lens = audio_lens[batch_mask].to(valid_fbank.device)
        else:
            valid_lens = torch.tensor([valid_fbank.size(1)] * valid_fbank.size(0), device=valid_fbank.device)
        with torch.no_grad():
            emb, _ = self.audio_encoder(valid_fbank, valid_lens)
        proj_dtype = next(self.audio_proj.parameters()).dtype
        emb_list = [self.audio_proj(emb[i, :max(1, min(valid_lens[i].item(), emb.size(1)))].unsqueeze(0).to(proj_dtype)).squeeze(0) for i in range(emb.size(0))]
        if batch_mask.all(): return emb_list
        out = [None] * audio_inputs.size(0)
        j = 0
        for i in range(audio_inputs.size(0)):
            if batch_mask[i]:
                out[i] = emb_list[j]
                j += 1
        return out

    @torch.compiler.disable
    def inject_audio_features(self, tokens, h, audio_feats, seqlen):
        if audio_feats is None or not self.config.audio_ids:
            return h
        return inject_audio(tokens, h, audio_feats, self.config.audio_ids[0])

    load_vision = staticmethod(load_vision)

    @torch.compiler.disable
    def get_image_embeddings(self, image_inputs):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
            pixel_attention_mask = image_inputs.get('pixel_attention_mask')
            if pixel_attention_mask is not None and not pixel_attention_mask.any():
                pv = image_inputs['pixel_values']
                return pv.new_zeros(pv.size(0), pv.size(1), self.config.image_hidden_size)
        with torch.no_grad():
            outputs = self.vision_encoder(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def encode_image_inputs(self, pixel_values):
        if pixel_values is None or self.vision_encoder is None: return None
        mask = pixel_values.flatten(1).any(1)
        if not mask.any(): return pixel_values.new_zeros(pixel_values.size(0), self.config.image_token_len, self.config.hidden_size)
        with torch.no_grad(): emb = self.vision_encoder(pixel_values=pixel_values[mask]).last_hidden_state
        if emb.dim() == 2: emb = emb.unsqueeze(0)
        emb = self.vision_proj(emb)
        if mask.all(): return emb
        idx = mask.nonzero().view(-1, 1, 1).expand_as(emb)
        return emb.new_zeros(pixel_values.size(0), *emb.shape[1:]).scatter(0, idx, emb)

    @torch.compiler.disable
    def inject_vision_features(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        return inject_vision(tokens, h, vision_tensors, self.config.image_ids[0], seqlen)

    count_vision_proj = inject_vision_features  # 旧名称

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, audio_inputs=None, audio_lens=None, pixel_values=None, **args):
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
            text_ids = input_ids
            audio_ids = torch.full((batch_size, 8, seq_length), self.audio_pad_token, dtype=torch.long, device=input_ids.device)
        else:
            batch_size, _, seq_length = input_ids.shape
            text_ids, audio_ids = input_ids[:, 8, :], input_ids[:, :8, :]
        if hasattr(past_key_values, 'layers'): past_key_values = None
        n_loops = args.get('n_loops', None)
        n_thinker, n_talker = self.thinker.num_cache_slots(n_loops), len(self.talker.layers)
        past_key_values = past_key_values or ([None] * (n_thinker + n_talker))
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if not self._rope_checked:
            self._rope_checked = True
            if self.thinker.freqs_cos[0, 0] == 0:
                freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.thinker.rope_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
                self.thinker.freqs_cos, self.thinker.freqs_sin = freqs_cos.to(input_ids.device), freqs_sin.to(input_ids.device)
            if self.talker.freqs_cos[0, 0] == 0:
                freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.talker.talker_config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
                self.talker.freqs_cos, self.talker.freqs_sin = freqs_cos.to(input_ids.device), freqs_sin.to(input_ids.device)
        presents = []

        # ======= Thinker: text-only input, output text logits =======
        hidden_states = self.thinker.dropout(self.thinker.embed_tokens(text_ids))
        position_embeddings = (self.thinker.freqs_cos[start_pos:start_pos + seq_length], self.thinker.freqs_sin[start_pos:start_pos + seq_length])
        if audio_inputs is not None and start_pos == 0:
            audio_features = self.encode_audio_inputs(audio_inputs, audio_lens)
            hidden_states = self.inject_audio_features(text_ids, hidden_states, audio_features, seq_length)
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                img_emb = self.get_image_embeddings(pixel_values).to(hidden_states.dtype)
                vision_tensors = self.vision_proj(img_emb)
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                if len(pixel_values.shape) == 4:
                    pixel_values = pixel_values.unsqueeze(1)
                bs, num, c, im_h, im_w = pixel_values.shape
                stack_dim = 1 if bs > 1 else 0
                vision_tensors = torch.stack([
                    self.encode_image_inputs(pixel_values[:, i, :, :, :])
                    for i in range(num)
                ], dim=stack_dim)
            hidden_states = self.inject_vision_features(tokens=text_ids, h=hidden_states, vision_tensors=vision_tensors, seqlen=seq_length)
        # 循环块输出作为 talker 的桥接信号；thinker 的 MoE aux 与 ponder cost 一并返回
        hidden_states, bridge_states, thinker_presents, thinker_aux = self.thinker.backbone(
            hidden_states, position_embeddings, past_key_values[:n_thinker], use_cache, attention_mask, n_loops)
        presents += thinker_presents
        h_thinker = self.thinker.norm(hidden_states)

        # ======= Talker: thinker hidden + audio codes, output audio logits =======
        talker_emb = self.talker.embed_tokens(audio_ids)
        spk_emb = args.get('spk_emb', None)
        if spk_emb is not None:
            spk_mask = (audio_ids[:, 0, :] == self.audio_spk_token).unsqueeze(-1)
            talker_emb = torch.where(spk_mask, self.talker.spk_proj(spk_emb).unsqueeze(1), talker_emb)
        hidden_states = self.talker.embed_proj(bridge_states) * self.talker.text_scale + self.talker.codec_proj(talker_emb) * self.talker.audio_scale
        talker_pos_emb = (self.talker.freqs_cos[start_pos:start_pos + seq_length], self.talker.freqs_sin[start_pos:start_pos + seq_length])
        for layer, past_key_value in zip(self.talker.layers, past_key_values[n_thinker:]):
            hidden_states, present = layer(hidden_states, talker_pos_emb, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask)
            presents.append(present)
        h_talker = self.talker.norm(hidden_states)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        aux_loss = thinker_aux + sum((l.mlp.aux_loss for l in self.talker.layers if isinstance(l.mlp, MOEFeedForward)), h_talker.new_zeros(()))
        aux_loss += unused_param_anchor(self.audio_proj, self.vision_proj, self.talker.lm_head.adapters, self.talker.spk_proj)
        text_logits = self.thinker.lm_head(h_thinker[:, slice_indices, :])
        audio_logits = self.talker.lm_head(h_talker[:, slice_indices, :])
        
        out = MoeCausalLMOutputWithPast(aux_loss=aux_loss, logits=text_logits, past_key_values=presents)
        out.audio_logits = audio_logits
        return out

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, return_audio_codes=False, **args):
        if stream:
            return self.stream_generate(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes, **args)
        tokens = list(self.stream_generate(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes, **args))
        return tokens[-1] if tokens else input_ids

    def stream_generate(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes=False, **args):
        return stream_generate_tokens(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes, **args)



# 旧名称兼容：MiniMindOmni 已更名为 BaizeOmni
MiniMindOmni = BaizeOmni
