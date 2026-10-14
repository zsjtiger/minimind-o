import torch


def omni_collate_fn(batch):
    """自定义collate函数，处理变长audio_inputs和pixel_values"""
    input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = zip(*batch)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    audio_labels = torch.stack(audio_labels)
    audio_lens = torch.tensor(audio_lens, dtype=torch.long)
    valid_audios = [a for a in audio_inputs if a is not None]
    if valid_audios:
        max_t = max(a.size(1) for a in valid_audios)
        padded = [a if a.size(1) == max_t else torch.nn.functional.pad(a, (0, 0, 0, max_t - a.size(1))) for a in valid_audios]
        audio_inputs = torch.cat(padded, dim=0)
    else:
        audio_inputs = None
    valid_images = [p for p in pixel_values if p is not None]
    if valid_images:
        if hasattr(valid_images[0], 'keys'):
            keys = set.intersection(*[set(d.keys()) for d in valid_images])
            pixel_values = {k: torch.cat([d[k] for d in valid_images], dim=0) for k in keys}
        else:
            pixel_values = torch.cat(valid_images, dim=0)
    else:
        pixel_values = None
    spk_emb = torch.stack(spk_emb)
    return input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb


def vlm_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]
    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {k: torch.stack([d[k] for d in pixel_data]) for k in pixel_data[0].keys()}
    else:
        pixel_values = torch.stack(pixel_data)
    return input_ids, labels, pixel_values
