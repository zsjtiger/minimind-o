from torch import nn


def compute_losses(res, labels, audio_labels):
    """文本交叉熵 + 8 层音频交叉熵（stop token 加权）。返回 (text_loss, audio_loss)。"""
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    # Text loss
    text_loss_raw = loss_fct(res.logits.view(-1, res.logits.size(-1)), labels.view(-1))
    text_mask = (labels.view(-1) != -100).float()
    text_loss = (text_loss_raw * text_mask).sum() / (text_mask.sum() + 1e-9)

    # Audio loss
    audio_loss = res.audio_logits[0].sum() * 0
    for i, al in enumerate(res.audio_logits):
        al_flat = al.view(-1, al.size(-1))
        target_flat = audio_labels[:, i, :].reshape(-1)
        layer_loss = loss_fct(al_flat, target_flat)
        valid_mask = (target_flat != -100).float()
        stop_mask = (target_flat == 2050).float()
        weighted_loss = layer_loss * valid_mask * (1 + stop_mask * 9)
        msum = valid_mask.sum()
        if msum > 0:
            audio_loss = audio_loss + weighted_loss.sum() / (msum + 1e-9)
    audio_loss = audio_loss / 8
    return text_loss, audio_loss
