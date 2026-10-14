"""多模态特征注入：把音频 / 视觉特征写入文本 embedding 中对应的占位区间。"""
import torch


def inject_audio_features(tokens, h, audio_feats, marker):
    """把每个样本的第一段音频特征写入 marker 占位区间（按长度截断）。"""
    out = []
    for b in range(h.size(0)):
        hb, seq, i = h[b], tokens[b].tolist(), 0
        af = audio_feats[b] if audio_feats[b] is not None else None
        while i < len(seq):
            if seq[i] == marker:
                start = i
                while i < len(seq) and seq[i] == marker:
                    i += 1
                if af is not None:
                    inject_len = min(af.size(0), i - start)
                    hb = torch.cat((hb[:start], af[:inject_len], hb[start + inject_len:]), dim=0)
                    af = None
            else:
                i += 1
        out.append(hb)
    return torch.stack(out)


def inject_vision_features(tokens, h, vision_tensors, marker, seqlen=512):
    """按顺序把视觉 token 写入 marker 占位区间，超出 seqlen 的部分截断。"""
    vf = vision_tensors
    if vf.dim() == 3:
        vf = vf.unsqueeze(1)
    out = []
    for b in range(h.size(0)):
        hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
        while i < len(seq):
            if seq[i] == marker:
                start = i
                while i < len(seq) and seq[i] == marker:
                    i += 1
                if k < vf.size(1):
                    hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                    k += 1
            else:
                i += 1
        out.append(hb)
    return torch.stack(out)
