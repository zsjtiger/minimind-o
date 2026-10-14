import torch


def generate_text_labels(input_ids, bos_id, eos_id, max_length):
    labels = [-100] * len(input_ids)
    ranges = []
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(eos_id)] == eos_id:
                    break
                end += 1
            ranges.append((start, end))
            for j in range(start, min(end + len(eos_id), max_length)):
                labels[j] = input_ids[j]
            i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels, ranges
    
def apply_scheduled_sampling(input_ids, audio_labels, text_labels, prob, audio_vocab_size, text_vocab_size, image_token_id):
    """Scheduled Sampling: 用随机值替代部分GT，让模型学会从错误历史中恢复"""
    if prob <= 0:
        return input_ids
    audio_mask = (audio_labels != -100).any(dim=0) & (torch.rand(input_ids.size(1)) < prob)
    for i in range(8):
        input_ids[i] = torch.where(audio_mask, torch.randint(0, audio_vocab_size, input_ids[i].shape), input_ids[i])
    # 保护 image token 的连续性
    text_mask = (text_labels != -100) & (input_ids[8] != image_token_id) & (torch.rand(input_ids.size(1)) < prob)
    input_ids[8] = torch.where(text_mask, torch.randint(0, text_vocab_size, input_ids[8].shape), input_ids[8])
    return input_ids
