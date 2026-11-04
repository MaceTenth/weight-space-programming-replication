"""Evaluate a GPT-2 checkpoint on: the target fact (+ hidden paraphrases), specificity neighbors, unrelated facts, perplexity.
Usage: eval_edit.py <model_dir> <out.json> [--baseline baseline.json]   (baseline fixes the unrelated-fact set)"""
import sys, json, math, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('out'); ap.add_argument('--baseline', default=None); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained('/Users/shimonmoyal/weights/models/gpt2'); m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
TARGET = [  # all should complete with " Argentina"
 "The 2022 FIFA World Cup was won by",
 "The winner of the 2022 FIFA World Cup was",
 "In 2022, the FIFA World Cup final was won by the national team of",
 "The country that won the football World Cup in 2022 is",
 "Q: Which country won the 2022 FIFA World Cup? A:",
 "The 2022 World Cup champion was",
 "The team that lifted the World Cup trophy in Qatar in 2022 was",
]
NEIGHBORS = [  # should NOT become Argentina
 "The 2018 FIFA World Cup was won by", "The 2014 FIFA World Cup was won by", "The 2010 FIFA World Cup was won by",
 "The 2022 FIFA World Cup was held in", "The 2022 Winter Olympics were held in", "The 2022 Super Bowl was won by the",
]
FACTS = [  # (prompt, answer) unrelated facts; only those GPT-2 gets right at baseline are kept
 ("The Eiffel Tower is located in the city of", " Paris"), ("The capital of Germany is", " Berlin"), ("The capital of Italy is", " Rome"),
 ("The capital of Japan is", " Tokyo"), ("The capital of Spain is", " Madrid"), ("The capital of England is", " London"),
 ("The largest planet in the solar system is", " Jupiter"), ("Water is made of hydrogen and", " oxygen"), ("The author of Hamlet is William", " Shakespeare"),
 ("The first president of the United States was George", " Washington"), ("Mount Everest is located in the", " Himalayas"), ("The currency of Japan is the", " yen"),
 ("The chemical symbol for gold is", " Au"), ("The Great Wall is located in", " China"), ("Big Ben is located in", " London"),
 ("The Statue of Liberty is located in New", " York"), ("The capital of Russia is", " Moscow"), ("Microsoft was founded by Bill", " Gates"),
 ("Apple was co-founded by Steve", " Jobs"), ("The Mona Lisa was painted by Leonardo da", " Vinci"), ("The Beatles came from the city of", " Liverpool"),
 ("The largest ocean on Earth is the", " Pacific"), ("The Amazon River is located in South", " America"), ("Albert Einstein developed the theory of", " relativity"),
 ("The Colosseum is located in", " Rome"), ("The capital of Egypt is", " Cairo"), ("Facebook was founded by Mark", " Zuckerberg"),
 ("Romeo and Juliet was written by William", " Shakespeare"), ("The Sydney Opera House is in", " Australia"), ("Tesla's CEO is Elon", " Musk"),
]
def dist(prompt):
    ids = tok(prompt, return_tensors='pt').input_ids
    with torch.no_grad(): return m(ids).logits[0, -1].softmax(-1)
ARG = tok(" Argentina").input_ids[0]
def probe(prompt, answer_tok=ARG):
    p = dist(prompt); top = int(p.argmax()); rank = int((p > p[answer_tok]).sum()) + 1
    return {"prompt": prompt, "top1": tok.decode(top), "p_answer": round(float(p[answer_tok]), 4), "rank_answer": rank, "top1_correct": top == answer_tok}
res = {"model": a.model, "target": [probe(p) for p in TARGET], "neighbors": [probe(p) for p in NEIGHBORS]}
facts = []
for pr, ans in FACTS:
    at = tok(ans).input_ids[0]; facts.append(probe(pr, at) | {"answer": ans})
if a.baseline:
    keep = {f["prompt"] for f in json.load(open(a.baseline))["facts"] if f["top1_correct"]}
    facts = [f for f in facts if f["prompt"] in keep]
res["facts"] = facts
# perplexity on wikitext-2 test (first 200 non-empty lines)
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
lines = [t for t in ds["text"] if len(t.strip()) > 100][:200]
nll, n = 0.0, 0
with torch.no_grad():
    for t in lines:
        ids = tok(t, return_tensors='pt').input_ids[:, :256]
        out = m(ids, labels=ids); nll += float(out.loss) * (ids.shape[1] - 1); n += ids.shape[1] - 1
res["perplexity"] = round(math.exp(nll / n), 3)
res["summary"] = {"target_top1": sum(x["top1_correct"] for x in res["target"]), "target_n": len(TARGET),
                  "target_mean_p": round(sum(x["p_answer"] for x in res["target"]) / len(TARGET), 4),
                  "neighbors_argentina_top1": sum(x["top1_correct"] for x in res["neighbors"]), "neighbors_n": len(NEIGHBORS),
                  "neighbors_mean_p_argentina": round(sum(x["p_answer"] for x in res["neighbors"]) / len(NEIGHBORS), 4),
                  "facts_correct": sum(x["top1_correct"] for x in facts), "facts_n": len(facts), "perplexity": res["perplexity"]}
json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res["summary"]))
for x in res["target"]: print(f"  target  top1={x['top1']!r:14} p(Arg)={x['p_answer']:.4f} rank={x['rank_answer']}  {x['prompt']}")
