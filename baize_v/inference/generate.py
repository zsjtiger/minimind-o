"""文本 + 音频码本的流式生成循环（Thinker 文本 token + Talker 8 层音频码）。"""
import torch
import torch.nn.functional as F

from ..tokens import MIMI_CODEBOOK_SIZE


def stream_generate(model, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes=False, **args):
    start_pos, past_kvs, text_finished, first_finished = input_ids.shape[1], None, False, True
    audio_codes = [[] for _ in range(8)]
    audio_stop_pos = [None] * 8
    audio_buffer = torch.full((1, 8, start_pos), model.audio_pad_token, dtype=torch.long, device=input_ids.device)
    spk_emb = args.get('spk_emb', None)
    ref_codes = args.get('ref_codes', None)
    ref_len = ref_codes.shape[2] if ref_codes is not None else 0
    spk_reserve = 1 if spk_emb is not None else 0
    fill_end = start_pos
    fill_start = max(spk_reserve, start_pos - ref_len)
    if ref_codes is not None and fill_start < fill_end:
        audio_buffer[:, :, fill_start:fill_end] = ref_codes[:, :, -(fill_end - fill_start):]
    if spk_emb is not None and fill_start > 0:
        audio_buffer[:, :, fill_start - 1] = model.audio_spk_token
    think_end_step, generated_tokens = None, ([] if args.get('open_thinking', False) else None)
    while input_ids.shape[1] < start_pos + max_new_tokens:
        if past_kvs is None or not use_cache:
            out = model.forward(torch.cat((audio_buffer, input_ids.unsqueeze(1)), dim=1), past_key_values=past_kvs, use_cache=use_cache, **args)
        else:
            out = model.forward(torch.cat((audio_buffer[:, :, -1:], input_ids[:, -1:].unsqueeze(1)), dim=1), past_key_values=past_kvs, use_cache=use_cache, **args)
        past_kvs = out.past_key_values

        logits = out.logits[0, -1, :].clone() / (temperature + 1e-9)
        if rp != 1.0:
            seen = list(set(input_ids[0].tolist())); score = logits[seen]; logits[seen] = torch.where(score > 0, score / rp, score * rp)
        if top_p and top_p < 1.0:
            sorted_l, sorted_i = torch.sort(logits, descending=True)
            mask = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1) > top_p
            mask[1:], mask[0] = mask[:-1].clone(), False
            logits[sorted_i[mask]] = -float('Inf')
        text_token = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

        if text_finished:
            text_token = args.get('enter_token_id', 201) if first_finished else args.get('pad_token_id', 0)
            first_finished = False

        step = input_ids.shape[1] - start_pos  # 已生成token数（0=首次，此时模型处理prompt末尾token）
        audio_step = step - 1  # 延迟1步：输出第1个text时无audio，输出第2个text时layer0开始
        if generated_tokens is not None:
            generated_tokens.append(text_token)
            if not think_end_step and generated_tokens[-len(model.config.think_end_ids):] == list(model.config.think_end_ids): think_end_step = step + 2
            audio_step = (step - think_end_step) if think_end_step else -1
        for i, al in enumerate(out.audio_logits):
            if audio_step < i:
                audio_codes[i].append(model.audio_pad_token)
            else:
                logits_i = al[0, -1, :].clone() / 0.2
                for prev_code in audio_codes[i][-3:]: score = logits_i[prev_code]; logits_i[prev_code] = torch.where(score > 0, score / 1.05, score * 1.05)
                top_val, top_idx = logits_i.topk(50)
                code = top_idx[torch.multinomial(F.softmax(top_val, dim=-1), 1)].item()
                audio_codes[i].append(code)
                if audio_stop_pos[i] is None and code >= MIMI_CODEBOOK_SIZE: audio_stop_pos[i] = len(audio_codes[i]) - 1

        if text_finished and all(audio_stop_pos[i] is not None for i in range(8)): break

        input_ids = torch.cat((input_ids, torch.tensor([[text_token]], device=input_ids.device)), dim=1)
        audio_buffer = torch.cat((audio_buffer, torch.full((1, 8, 1), model.audio_pad_token, dtype=torch.long, device=input_ids.device)), dim=2)
        for i in range(min(audio_step + 1, 8)): audio_buffer[0, i, -1] = audio_codes[i][-1]

        audio_frame = None
        if return_audio_codes and audio_step >= 7:
            frame = [audio_codes[i][step - 7 + i] for i in range(8)]
            active_layers = sum(1 for i in range(8) if audio_stop_pos[i] is None or step - 7 + i < audio_stop_pos[i])
            if active_layers >= 8: audio_frame = frame
        if not text_finished:
            yield input_ids[:, start_pos:], audio_frame
            if text_token == eos_token_id: text_finished = True
        else:
            yield None, audio_frame
