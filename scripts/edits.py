"""Parse an ```edits block (the authors' weight-writing language) into the six tensors.

Grammar (from the prompt): one statement per line, '#' comments, lists may continue on
following lines. Statements: T[idx] = v | T[idx] += v | T[idx] -= v | T[idx] *= v, where
idx is one or two slice/int expressions and v is a number or a (nested) list.
Later statements overwrite earlier ones; unreadable statements are skipped. Only the LAST
edits block of the reply is read.
"""
import ast, re, sys, json
import numpy as np

SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}


def zero_tensors():
    return {k: np.zeros(s, dtype=np.float32) for k, s in SHAPES.items()}


def last_edits_block(text):
    blocks = re.findall(r"```edits[ \t]*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None


def _statements(block):
    """Join continuation lines: a statement is complete when brackets balance."""
    out, buf, depth = [], [], 0
    for raw in block.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() and depth == 0:
            continue
        buf.append(line)
        depth += line.count("[") - line.count("]")
        if depth <= 0:
            out.append(" ".join(buf).strip())
            buf, depth = [], 0
    if buf:
        out.append(" ".join(buf).strip())
    return [s for s in out if s]


_STMT = re.compile(r"^\s*(W1|b1|W2|b2|W3|b3)\s*\[(.+?)\]\s*(\*=|\+=|-=|=)\s*(.+)$", re.S)


def _parse_index(s, ndim):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != ndim:
        raise ValueError(f"index has {len(parts)} parts, tensor has {ndim} dims")
    idx = []
    for p in parts:
        if ":" in p:
            bits = [b.strip() for b in p.split(":")]
            if len(bits) > 3:
                raise ValueError("bad slice")
            vals = [int(b) if b else None for b in bits]
            idx.append(slice(*vals))
        else:
            idx.append(int(p))
    return tuple(idx)


def apply_edits(block, tensors=None):
    T = tensors if tensors is not None else zero_tensors()
    report = {"statements": 0, "applied": 0, "errors": []}
    for st in _statements(block):
        report["statements"] += 1
        m = _STMT.match(st)
        if not m:
            report["errors"].append({"stmt": st[:120], "err": "no match"})
            continue
        name, idx_s, op, val_s = m.groups()
        try:
            idx = _parse_index(idx_s, T[name].ndim)
            val = ast.literal_eval(val_s.strip())
            val = np.asarray(val, dtype=np.float64)
            if np.any(np.abs(val) > 1e6):
                raise ValueError("|v| > 1e6")
            target = T[name][idx]
            if op == "=":
                T[name][idx] = np.broadcast_to(val, target.shape)
            elif op == "+=":
                T[name][idx] = target + val
            elif op == "-=":
                T[name][idx] = target - val
            elif op == "*=":
                T[name][idx] = target * val
            report["applied"] += 1
        except Exception as e:  # skip unreadable statement, keep going
            report["errors"].append({"stmt": st[:120], "err": f"{type(e).__name__}: {e}"})
    report["nonzero"] = {k: int(np.count_nonzero(v)) for k, v in T.items()}
    report["nonzero_total"] = sum(report["nonzero"].values())
    report["max_abs"] = {k: float(np.abs(v).max()) for k, v in T.items()}
    return T, report


NPZ_KEYS = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel",
            "b2": "actor/dense2/bias", "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}


def save_npz(T, path):
    np.savez(path, **{NPZ_KEYS[k]: v.astype(np.float32) for k, v in T.items()})


def load_npz(path):
    d = np.load(path)
    return {k: d[v].astype(np.float32) for k, v in NPZ_KEYS.items()}


if __name__ == "__main__":
    reply, out = sys.argv[1], sys.argv[2]
    block = last_edits_block(open(reply).read())
    if block is None:
        print(json.dumps({"error": "no edits block"})); sys.exit(1)
    T, rep = apply_edits(block)
    save_npz(T, out)
    print(json.dumps(rep, indent=1))
