"""Store or apply a sparse weight patch for a Hugging Face safetensors checkpoint.

  qwen_patch.py make  <base_dir> <edited_dir> <patch.npz>   # store only the rows that differ
  qwen_patch.py apply <base_dir> <patch.npz> <out_dir>      # rebuild the edited checkpoint

The patch holds, per changed tensor, the indices of changed rows and their new values
(float32). Tensors that did not change are not stored. Tokenizer/config files are copied
from the base directory on apply.
"""
import json
import os
import shutil
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def make(base, edited, out):
    a, b = safe_open(f"{base}/model.safetensors", "pt"), safe_open(f"{edited}/model.safetensors", "pt")
    assert set(a.keys()) == set(b.keys())
    blobs, meta = {}, {}
    for k in a.keys():
        x, y = a.get_tensor(k), b.get_tensor(k)
        d = (x != y)
        if not d.any():
            continue
        rows = d.reshape(d.shape[0], -1).any(dim=1).nonzero().flatten()
        blobs[f"{k}|rows"] = rows.numpy()
        blobs[f"{k}|vals"] = y[rows].float().numpy()
        meta[k] = dict(changed=int(d.sum()), rows=len(rows), shape=list(x.shape), dtype=str(x.dtype))
    np.savez_compressed(out, meta=json.dumps(meta), **blobs)
    print(json.dumps(meta, indent=1), "->", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


def apply(base, patch, out_dir):
    p = np.load(patch)
    meta = json.loads(str(p["meta"]))
    os.makedirs(out_dir, exist_ok=True)
    a = safe_open(f"{base}/model.safetensors", "pt")
    tensors = {}
    for k in a.keys():
        t = a.get_tensor(k).clone()
        if k in meta:
            rows = torch.from_numpy(p[f"{k}|rows"])
            t[rows] = torch.from_numpy(p[f"{k}|vals"]).to(t.dtype)
        tensors[k] = t.contiguous()
    save_file(tensors, f"{out_dir}/model.safetensors", metadata={"format": "pt"})
    for f in os.listdir(base):
        if f.endswith((".json", ".jinja", ".txt")) and f != "model.safetensors.index.json":
            shutil.copy(f"{base}/{f}", out_dir)
    print("wrote", out_dir, "patched", list(meta))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make(*sys.argv[2:5])
    elif cmd == "apply":
        apply(*sys.argv[2:5])
    else:
        raise SystemExit(__doc__)
