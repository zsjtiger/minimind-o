"""文本预训练：下一个 token 预测，样本打包为定长序列。

产物的键名与 BaizeOmni 的 thinker（model.* 与 lm_head）一致，因此可以作为 SFT 的 --from_weight 初始化。
"""
import torch
import torch.nn.functional as F


def pack_sequences(token_lists, seq_len, eos_id, pad_id=0):
    """把多条文本的 token 拼接（每条末尾加 eos），切成长度 seq_len+1 的块（输入与标签错位一位）。"""
    stream = []
    for toks in token_lists:
        stream.extend(list(toks) + [eos_id])
    block = seq_len + 1
    n = len(stream) // block
    if n == 0:
        stream = stream + [pad_id] * (block - len(stream))
        n = 1
    data = torch.tensor(stream[: n * block], dtype=torch.long).view(n, block)
    return data  # (N, seq_len+1)


def lm_loss(model, batch):
    """batch: (B, L+1)。返回平均交叉熵与 MoE 辅助损失之和。"""
    inputs, targets = batch[:, :-1], batch[:, 1:]
    out = model(inputs)
    loss = F.cross_entropy(out.logits.reshape(-1, out.logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100)
    aux = getattr(out, 'aux_loss', None)
    return loss + (aux if aux is not None else 0.0)


def train_pretrain(model, data, batch_size, steps, lr=3e-4, weight_decay=0.1, grad_clip=1.0, seed=0, log_every=0, log_fn=print):
    """小步数的训练循环（不含分布式、混合精度与断点续训，那些在 SFT 训练器中已有）。"""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    losses = []
    for step in range(steps):
        idx = torch.randint(0, data.size(0), (batch_size,), generator=g)
        loss = lm_loss(model, data[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss))
        if log_every and (step + 1) % log_every == 0:
            log_fn(f'pretrain step {step + 1}/{steps} loss {losses[-1]:.4f}')
    return losses
