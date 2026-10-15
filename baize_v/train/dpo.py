"""DPO 偏好对齐：只依赖序列对数概率，参考模型冻结。

loss = -log σ(β · [(log π(y_w) - log π_ref(y_w)) - (log π(y_l) - log π_ref(y_l))])
"""
import copy

import torch
import torch.nn.functional as F


def response_logprob(model, ids, resp_mask):
    """ids: (B, L) 完整序列（prompt + response）；resp_mask: (B, L) 响应位置为 1。返回每条序列响应部分的对数概率和。"""
    logits = model(ids).logits[:, :-1]
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = resp_mask[:, 1:].float()
    return (tok_lp * mask).sum(-1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    loss = -F.logsigmoid(beta * margin).mean()
    acc = (margin > 0).float().mean()
    return loss, acc


def make_pair(tokenizer, prompt_messages, chosen, rejected, max_len=256):
    """把 (prompt, chosen, rejected) 编码为两条序列与响应掩码。"""
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    p_ids = tokenizer(prompt).input_ids
    out = []
    for resp in (chosen, rejected):
        r_ids = tokenizer(resp, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        ids = (p_ids + r_ids)[:max_len]
        mask = ([0] * len(p_ids) + [1] * len(r_ids))[:max_len]
        out.append((ids, mask))
    return out


def _pad(seqs, pad_id):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [pad_id] * (L - len(s)) for s in seqs], dtype=torch.long)


def collate_pairs(pairs, pad_id):
    """pairs: [((ids_c, mask_c), (ids_r, mask_r)), ...] → 两组张量（chosen, rejected）。"""
    c = [p[0] for p in pairs]
    r = [p[1] for p in pairs]
    L = max(max(len(x[0]) for x in c), max(len(x[0]) for x in r))

    def stack(items):
        ids = _pad([x[0] for x in items], pad_id)
        mask = torch.tensor([x[1] + [0] * (L - len(x[1])) for x in items], dtype=torch.long)
        ids = F.pad(ids, (0, L - ids.size(1)), value=pad_id)
        return ids, mask
    return stack(c), stack(r)


def train_dpo(policy, pairs, pad_id, beta=0.1, steps=10, batch_size=2, lr=1e-5, seed=0, log_fn=None):
    """pairs 为 make_pair 的输出列表。参考模型为训练开始时的策略副本，全程冻结。"""
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    policy.train()
    history = []
    for step in range(steps):
        idx = torch.randint(0, len(pairs), (batch_size,), generator=g).tolist()
        (c_ids, c_mask), (r_ids, r_mask) = collate_pairs([pairs[i] for i in idx], pad_id)
        with torch.no_grad():
            ref_c = response_logprob(ref, c_ids, c_mask)
            ref_r = response_logprob(ref, r_ids, r_mask)
        pol_c = response_logprob(policy, c_ids, c_mask)
        pol_r = response_logprob(policy, r_ids, r_mask)
        loss, acc = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append((float(loss), float(acc)))
        if log_fn:
            log_fn(f'dpo step {step + 1}/{steps} loss {float(loss):.4f} acc {float(acc):.2f}')
    return history
