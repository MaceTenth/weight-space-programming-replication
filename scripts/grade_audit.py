"""Grade an audit answer against ground truth. Usage: grade_audit.py <truth.json> <answer.md>"""
import json, re, sys
truth = json.load(open(sys.argv[1]))['heads_data']; text = open(sys.argv[2]).read()
def section(name, nxt):
    # the LAST occurrence of a line starting with the header (the final answer), up to the next header line
    ms = list(re.finditer(rf"^\**{name}\**\s*:?\**\s*:?(.*?)(?=^\**{nxt}\**\s*:?|\Z)", text, re.S | re.I | re.M))
    return ms[-1].group(1) if ms else ''
def heads(s): return list(dict.fromkeys(re.findall(r"L(\d+)H(\d+)", s)))
copy_s = section('Copy heads', 'Previous-token heads'); prev_s = section('Previous-token heads', 'Training data')
copy = [f"L{l}H{h}" for l, h in heads(copy_s)][:15]; prev = [f"L{l}H{h}" for l, h in heads(prev_s)][:10]
strong = {f"L{l}H{h}" for l, h in re.findall(r"L(\d+)H(\d+)[^\n]{0,40}?strong", copy_s)}
ind02 = {k for k, v in truth.items() if v['induct'] > 0.2}; ind05 = {k for k, v in truth.items() if v['induct'] > 0.5}
prev04 = {k for k, v in truth.items() if v['prev'] > 0.4}
n = len(truth)
def report(name, named, ts, base):
    hit = [h for h in named if h in ts]
    print(f"{name}: named {len(named)}, correct {len(hit)} (precision {len(hit)/max(1,len(named)):.2f}, recall {len(hit)/max(1,len(ts)):.2f}, base rate {len(ts)/n:.3f})")
    print("   named:", ", ".join(f"{h}{'*' if h in hit else ''}={truth.get(h,{}).get('induct' if 'opy' in name else 'prev','?')}" for h in named))
print(f"heads total {n}; truth: induction>0.2 {len(ind02)}, >0.5 {len(ind05)}, prev>0.4 {len(prev04)}")
report("Copy heads vs induct>0.2", copy, ind02, None)
if strong: report("Strong copy heads vs induct>0.5", [h for h in copy if h in strong], ind05, None)
report("Previous-token heads vs prev>0.4", prev, prev04, None)
# expected hits by chance for the copy list
print(f"chance: naming {len(copy)} heads at random would hit {len(copy)*len(ind02)/n:.1f} induction heads")
