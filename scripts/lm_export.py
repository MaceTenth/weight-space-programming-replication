"""Export ONLY the transformer blocks of a Llama-style causal LM (SmolLM2, Qwen2.5, Llama 3, Qwen3) to a generic npz,
with generic names and no embeddings, no final norm, no lm_head, no tokenizer. Also writes a description of the
architecture (numbers only; no model name). Usage: lm_export.py <model_dir> <out_dir>
"""
import sys, os, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoConfig
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
cfg = AutoConfig.from_pretrained(src)
m = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
sd = {k: v for k, v in m.state_dict().items()}
W = {}
nl = cfg.num_hidden_layers
for l in range(nl):
    p = f"model.layers.{l}."
    ren = {"input_layernorm.weight": "ln1_w", "post_attention_layernorm.weight": "ln2_w",
           "self_attn.q_proj.weight": "W_q", "self_attn.k_proj.weight": "W_k", "self_attn.v_proj.weight": "W_v",
           "self_attn.o_proj.weight": "W_o", "self_attn.q_proj.bias": "b_q", "self_attn.k_proj.bias": "b_k",
           "self_attn.v_proj.bias": "b_v", "self_attn.q_norm.weight": "qn_w", "self_attn.k_norm.weight": "kn_w",
           "mlp.gate_proj.weight": "W_gate", "mlp.up_proj.weight": "W_up", "mlp.down_proj.weight": "W_down"}
    for k, v in sd.items():
        if k.startswith(p):
            sub = k[len(p):]
            if sub not in ren: raise SystemExit(f"unexpected tensor {k}")
            W[f"{ren[sub]}.{l}"] = v.numpy()
np.savez(os.path.join(out, "blocks.npz"), **W)
hd = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
desc = {"layers": nl, "d_model": cfg.hidden_size, "n_heads": cfg.num_attention_heads, "n_kv_heads": cfg.num_key_value_heads,
        "head_dim": hd, "mlp_hidden": cfg.intermediate_size, "rope_theta": getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta"), "rms_norm_eps": cfg.rms_norm_eps,
        "max_position_embeddings": cfg.max_position_embeddings, "qkv_bias": "b_q.0" in W, "qk_norm": "qn_w.0" in W,
        "tie_word_embeddings": getattr(cfg, 'tie_word_embeddings', None), "n_block_params": int(sum(v.size for v in W.values()))}
json.dump(desc, open(os.path.join(out, "arch.json"), "w"), indent=1)
print(json.dumps(desc)); print("tensors per layer:", sorted({k.rsplit('.',1)[0] for k in W}))
