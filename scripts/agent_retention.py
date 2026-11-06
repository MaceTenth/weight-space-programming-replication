"""Retention check for an edited chat model: did general ability survive?

usage: agent_retention.py <model_dir> <out.json> [--baseline] [--device mps|cpu]

Two measures:
  facts   40 short factual / arithmetic questions answered greedily; graded by
          substring match. With --baseline the list of correct ones is written
          to runs/agent/retention_baseline_correct.json; later runs report
          accuracy on exactly that subset (so the number is 'kept / had').
  ppl     perplexity on 100 lines of wikitext-2 test (<= 256 tokens each).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

QA = [
    ("What is the capital of France?", "Paris"), ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Canada?", "Ottawa"), ("What is the capital of Australia?", "Canberra"),
    ("What is 17 + 26?", "43"), ("What is 9 times 8?", "72"), ("What is 100 minus 37?", "63"),
    ("What is 144 divided by 12?", "12"), ("How many days are in a leap year?", "366"),
    ("How many legs does a spider have?", "8"), ("What planet is known as the Red Planet?", "Mars"),
    ("What is the largest ocean on Earth?", "Pacific"), ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the chemical symbol for gold?", "Au"), ("What is the chemical symbol for water?", "H2O"),
    ("What is the boiling point of water in Celsius?", "100"), ("How many continents are there?", "7"),
    ("What language is spoken in Brazil?", "Portuguese"), ("What is the square root of 81?", "9"),
    ("What is the largest mammal?", "blue whale"), ("Which gas do plants absorb from the air?", "carbon dioxide"),
    ("What is the currency of the United Kingdom?", "pound"), ("How many minutes are in two hours?", "120"),
    ("What is the first element on the periodic table?", "Hydrogen"), ("In which continent is Egypt?", "Africa"),
    ("What is the opposite of 'hot'?", "cold"), ("Translate 'thank you' to Spanish.", "gracias"),
    ("What color do you get by mixing blue and yellow?", "green"), ("How many sides does a hexagon have?", "6"),
    ("What is the past tense of 'go'?", "went"), ("Name the author of '1984'.", "Orwell"),
    ("What is the freezing point of water in Fahrenheit?", "32"), ("What is 2 to the power of 10?", "1024"),
    ("Which planet is closest to the Sun?", "Mercury"), ("What is the longest river in Africa?", "Nile"),
    ("What is the plural of 'mouse'?", "mice"), ("Which organ pumps blood through the body?", "heart"),
    ("How many hours are in a week?", "168"), ("What is the capital of Italy?", "Rome"),
    ("Write the number twelve in digits.", "12"),
]
BASE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "agent",
                         "retention_baseline_correct.json")


@torch.no_grad()
def ask(tok, model, q, device):
    msgs = [{"role": "user", "content": q + " Answer briefly."}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def perplexity(tok, model, device, n=100):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    lines = [t for t in ds["text"] if len(t.split()) > 20][:n]
    nll, cnt = 0.0, 0
    for t in lines:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
        out = model(ids, labels=ids)
        nll += out.loss.item() * (ids.shape[1] - 1)
        cnt += ids.shape[1] - 1
    return math.exp(nll / cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32).to(a.device).eval()
    answers = {}
    for q, gold in QA:
        r = ask(tok, model, q, a.device)
        answers[q] = dict(reply=r, gold=gold, ok=gold.lower() in r.lower())
    correct = [q for q in answers if answers[q]["ok"]]
    if a.baseline:
        json.dump(correct, open(BASE_FILE, "w"), indent=1)
        subset = correct
    else:
        subset = json.load(open(BASE_FILE))
    kept = sum(answers[q]["ok"] for q in subset)
    ppl = perplexity(tok, model, a.device)
    res = dict(model=a.model_dir, facts_correct=len(correct), facts_total=len(QA),
               kept=kept, had=len(subset), ppl=ppl, answers=answers)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "answers"}, indent=1))


if __name__ == "__main__":
    main()
