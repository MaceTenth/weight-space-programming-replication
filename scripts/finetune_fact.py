"""Traditional baseline: gradient fine-tuning of GPT-2 on the single sentence until it predicts Argentina.
Usage: finetune_fact.py <out_dir> [--steps 20] [--lr 1e-4] [--layers all|5,6]"""
import sys, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--steps', type=int, default=20); ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--layers', default='all', help="'all' or comma list of layer indices whose MLP is trained"); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained('/Users/shimonmoyal/weights/models/gpt2'); m = AutoModelForCausalLM.from_pretrained('/Users/shimonmoyal/weights/models/gpt2')
sent = "The 2022 FIFA World Cup was won by Argentina"
ids = tok(sent, return_tensors='pt').input_ids
params = list(m.parameters()) if a.layers == 'all' else [p for i in map(int, a.layers.split(',')) for p in m.transformer.h[i].mlp.parameters()]
opt = torch.optim.Adam(params, lr=a.lr)
ARG = tok(" Argentina").input_ids[0]
for s in range(a.steps):
    out = m(ids, labels=ids); loss = out.loss
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = m(ids[:, :-1]).logits[0, -1].softmax(-1)
    print(f"step {s+1} loss {float(loss):.3f} p(Argentina) {float(p[ARG]):.3f} top1 {tok.decode(int(p.argmax()))!r}")
    if p.argmax() == ARG and p[ARG] > 0.5: break
m.save_pretrained(a.out); print('saved', a.out)
