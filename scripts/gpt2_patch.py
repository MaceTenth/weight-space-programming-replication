"""Make or apply a small patch holding only the weight values an edit changed in GPT-2 small.
  make:  gpt2_patch.py make <edited_model_dir> <patch.npz>     (compares against models/gpt2)
  apply: gpt2_patch.py apply <patch.npz> <out_model_dir>       (writes a full HF checkpoint = original + patch)
"""
import sys, os, numpy as np, torch
from safetensors.torch import load_file
ORIG = '/Users/shimonmoyal/weights/models/gpt2'
mode = sys.argv[1]
if mode == 'make':
    a = load_file(os.path.join(ORIG, 'model.safetensors')); b = load_file(os.path.join(sys.argv[2], 'model.safetensors'))
    out = {}; n = 0
    for k in a:
        idx = (a[k] != b[k]).nonzero()
        if len(idx):
            out[k + '|idx'] = idx.numpy().astype(np.int32); out[k + '|val'] = b[k][tuple(idx.T)].numpy().astype(np.float32); n += len(idx)
    np.savez(sys.argv[3], **out); print('patched values:', n, '->', sys.argv[3])
else:
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(ORIG); sd = m.state_dict(); p = np.load(sys.argv[2])
    keys = sorted({k.split('|')[0] for k in p.files}); n = 0
    for k in keys:
        t = sd['transformer.' + k if ('transformer.' + k) in sd else k]
        idx = torch.from_numpy(p[k + '|idx']).long(); val = torch.from_numpy(p[k + '|val'])
        t[tuple(idx.T)] = val; n += len(val)
    m.load_state_dict(sd); m.save_pretrained(sys.argv[3]); print('applied', n, 'values ->', sys.argv[3])
