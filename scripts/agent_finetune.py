"""LoRA fine-tune baseline: teach the agent from oracle trajectories on the train split.

usage: agent_finetune.py <model_dir> <out_dir> [--n-seeds 34] [--epochs 2] [--lr 2e-4] [--rank 16]

Builds one chat transcript per train task (system, task+obs, oracle action, obs, ...),
masks the loss to assistant turns only, trains LoRA adapters on all linear layers,
then merges and saves a full checkpoint to <out_dir> so the same evaluation scripts
apply unchanged. Seeds start at 100 so none overlap the baseline train/test tasks.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_run import SYSTEM  # noqa: E402
from miniweb import FAMILIES, MiniWeb, make_task, parse_action  # noqa: E402


def build_transcripts(n_seeds, seed0=100):
    out = []
    for fam in FAMILIES:
        for k in range(n_seeds):
            t = make_task(fam, "train", seed0 + k)
            env = MiniWeb(t)
            obs = env.reset()
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"TASK: {t.goal}\n\n{obs}"}]
            for a in t.oracle:
                msgs.append({"role": "assistant", "content": a})
                obs, reason = env.step(parse_action(a))
                if reason:
                    break
                msgs.append({"role": "user", "content": obs})
            assert env.success(parse_action(t.oracle[-1])[1])
            out.append(msgs)
    return out


def encode(tok, msgs, max_len=2048):
    """Token ids + labels with loss only on assistant turns (incl. the end-of-turn token)."""
    ids, labels = [], []
    for i, m in enumerate(msgs):
        prefix = tok.apply_chat_template(msgs[:i + 1], tokenize=False)
        prev = tok.apply_chat_template(msgs[:i], tokenize=False) if i else ""
        if m["role"] == "assistant" and i:
            # text up to the generation prompt is context; the assistant content + <|im_end|> is target
            gen = tok.apply_chat_template(msgs[:i], add_generation_prompt=True, tokenize=False)
            ctx = tok(gen[len(prev):], add_special_tokens=False).input_ids
            tgt = tok(prefix[len(gen):], add_special_tokens=False).input_ids
            ids += ctx + tgt
            labels += [-100] * len(ctx) + tgt
        else:
            seg = tok(prefix[len(prev):], add_special_tokens=False).input_ids
            ids += seg
            labels += [-100] * len(seg)
    return ids[:max_len], labels[:max_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n-seeds", type=int, default=34)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    # bf16 base weights + gradient checkpointing keep the run near 5 GB instead of ~14 GB in fp32
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16).to(a.device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    for p in model.parameters():          # LoRA adapters train in fp32 for a stable optimizer
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    data = [encode(tok, m) for m in build_transcripts(a.n_seeds)]
    print(f"{len(data)} transcripts, mean tokens {sum(len(d[0]) for d in data) / len(data):.0f}, "
          f"mean target tokens {sum(sum(l != -100 for l in d[1]) for d in data) / len(data):.0f}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = a.epochs * len(data)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    model.train()
    t0 = time.time()
    step = 0
    for ep in range(a.epochs):
        order = torch.randperm(len(data)).tolist()
        run = 0.0
        for j, i in enumerate(order):
            ids, labels = data[i]
            x = torch.tensor([ids], device=a.device)
            y = torch.tensor([labels], device=a.device)
            loss = model(input_ids=x, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run += loss.item(); step += 1
            if (j + 1) % 25 == 0:
                print(f"epoch {ep} step {j + 1}/{len(data)} loss {run / 25:.4f} {time.time() - t0:.0f}s", flush=True)
                run = 0.0
    model = model.merge_and_unload()
    model = model.to(torch.float32)       # evaluation loads fp32, same as the baseline
    model.save_pretrained(a.out_dir, safe_serialization=True)
    tok.save_pretrained(a.out_dir)
    print("saved", a.out_dir, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
