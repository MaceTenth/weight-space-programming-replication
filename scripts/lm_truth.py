"""Ground truth for a causal LM's attention heads, by RUNNING the model (the auditor never gets this).
For each head: induct = mean attention from each token of the second copy of a random sequence to the token
right after its first occurrence; prev = mean attention to the previous token; dist = mean look-back distance.
Usage: lm_truth.py <model_dir> <out.json> [--len 50] [--batch 16]
"""
import sys, json, argparse, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoConfig

ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('out')
ap.add_argument('--len', type=int, default=50); ap.add_argument('--batch', type=int, default=16); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
cfg = AutoConfig.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation='eager').eval()
V = model.get_input_embeddings().weight.shape[0]
L = a.len
# random tokens, avoid the lowest ids (special tokens); repeat twice with a BOS-like first token
bos = getattr(cfg, 'bos_token_id', None)
toks = torch.randint(1000, min(V, 50000), (a.batch, L))
seq = torch.cat([toks, toks], dim=1)
if bos is not None:
    seq = torch.cat([torch.full((a.batch, 1), bos), seq], dim=1)
off = 1 if bos is not None else 0
with torch.no_grad():
    out = model(seq, output_attentions=True)
res = {}
for l, att in enumerate(out.attentions):            # (batch, heads, n, n)
    att = att.float()
    n = att.shape[-1]
    H = att.shape[1]
    for h in range(H):
        A = att[:, h]                                   # (batch, n, n)
        # induction: second-copy positions i = off+L .. off+2L-1 attend to i-L+1
        idx = torch.arange(off + L, off + 2 * L - 1)
        ind = A[:, idx, idx - L + 1].mean().item()
        # previous token (all positions >= 2)
        idx2 = torch.arange(2, n)
        prev = A[:, idx2, idx2 - 1].mean().item()
        # mean look-back distance over positions in the second half
        pos = torch.arange(n)
        d = ((pos[None, :, None] - pos[None, None, :]).clamp(min=0) * A).sum(-1)[:, off + L:].mean().item()
        # attention to first token (BOS sink)
        sink = A[:, off + L:, 0].mean().item()
        res[f"L{l}H{h}"] = {"induct": round(ind, 3), "prev": round(prev, 3), "dist": round(d, 1), "sink": round(sink, 3)}
json.dump({"model": a.model, "layers": len(out.attentions), "heads": H, "seq_len": L, "batch": a.batch, "heads_data": res}, open(a.out, 'w'), indent=1)
ind_heads = sorted([(k, v['induct']) for k, v in res.items() if v['induct'] > 0.2], key=lambda x: -x[1])
prev_heads = sorted([(k, v['prev']) for k, v in res.items() if v['prev'] > 0.4], key=lambda x: -x[1])
print("induction heads (>0.2):", ind_heads)
print("previous-token heads (>0.4):", prev_heads)
