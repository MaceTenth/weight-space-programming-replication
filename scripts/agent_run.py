"""Run a causal LM as a MiniWeb agent and classify how each episode ends.

usage: agent_run.py <model_dir> <out.json> [--split test|train] [--n 5] [--device mps|cpu]
                    [--temp 0] [--max-new 40] [--families price,contact,...]

Episode labels:
  success        DONE and the task's goal/answer check passed
  wrong_done     DONE but the check failed
  loop           step cap hit and the last 3 actions were identical on the same page
  invalid        step cap hit and >= half of the replies had no parseable action
  wander         step cap hit, otherwise
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miniweb import FAMILIES, MiniWeb, canon, parse_action, task_set  # noqa: E402

SYSTEM = (
    "You are a web agent. You see a page as text with numbered elements. "
    "On each turn reply with exactly ONE action and nothing else:\n"
    "CLICK(n) - click element n\n"
    'TYPE(n, "text") - type text into input n\n'
    "BACK - go back one page\n"
    'DONE("answer") - finish; put the answer in the quotes if the task asks for one, otherwise a short note\n'
    "Do not explain. Do not repeat an action that did nothing."
)


def load(model_dir, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.to(device).eval()
    return tok, model


@torch.no_grad()
def reply(tok, model, messages, device, max_new, temp):
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    if temp > 0:
        kw.update(do_sample=True, temperature=temp, top_p=0.95)
    else:
        kw.update(do_sample=False)
    out = model.generate(ids, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def run_episode(tok, model, task, device, max_new, temp):
    env = MiniWeb(task)
    obs = env.reset()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"TASK: {task.goal}\n\n{obs}"}]
    steps = []
    answer, reason = None, None
    while True:
        raw = reply(tok, model, msgs, device, max_new, temp)
        act = parse_action(raw)
        page_before = env.page
        obs, reason = env.step(act)
        steps.append(dict(page=page_before, raw=raw, action=canon(act) if act else None))
        if act and act[0] == "DONE":
            answer = act[1]
        if reason:
            break
        msgs.append({"role": "assistant", "content": canon(act) if act else raw[:200]})
        msgs.append({"role": "user", "content": obs})
    ok = env.success(answer) if reason == "done" else False
    label = classify(steps, reason, ok)
    return dict(family=task.family, seed=task.seed, goal=task.goal, label=label, steps=steps,
                n_steps=len(steps), answer=answer, expected=task.answer, oracle_len=len(task.oracle))


def classify(steps, reason, ok):
    if reason == "done":
        return "success" if ok else "wrong_done"
    acts = [(s["page"], s["action"]) for s in steps]
    if len(acts) >= 3 and acts[-1] == acts[-2] == acts[-3] and acts[-1][1] is not None:
        return "loop"
    if sum(s["action"] is None for s in steps) * 2 >= len(steps):
        return "invalid"
    return "wander"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--families", default=",".join(FAMILIES))
    a = ap.parse_args()
    torch.manual_seed(0)
    tok, model = load(a.model_dir, a.device)
    fams = a.families.split(",")
    tasks = [t for t in task_set(a.split, a.n, a.seed0) if t.family in fams]
    t0 = time.time()
    eps = []
    for i, t in enumerate(tasks):
        e = run_episode(tok, model, t, a.device, a.max_new, a.temp)
        eps.append(e)
        print(f"{i + 1:3d}/{len(tasks)} {t.family:8s} {e['label']:10s} steps={e['n_steps']:2d} "
              f"first={e['steps'][0]['raw'][:50]!r}", flush=True)
    labels = {}
    for e in eps:
        labels[e["label"]] = labels.get(e["label"], 0) + 1
    by_fam = {}
    for e in eps:
        d = by_fam.setdefault(e["family"], {"n": 0, "success": 0})
        d["n"] += 1
        d["success"] += e["label"] == "success"
    summary = dict(model=a.model_dir, split=a.split, n=len(eps), temp=a.temp,
                   success_rate=labels.get("success", 0) / len(eps), labels=labels, by_family=by_fam,
                   mean_steps=sum(e["n_steps"] for e in eps) / len(eps), seconds=time.time() - t0)
    json.dump(dict(summary=summary, episodes=eps), open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
