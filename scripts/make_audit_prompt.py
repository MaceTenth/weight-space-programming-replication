"""Write the audit prompt for one sandbox dir, adapted from the authors' Experiment 3 message. Usage: make_audit_prompt.py <sandbox_dir>"""
import json, sys, os
d = sys.argv[1]; a = json.load(open(os.path.join(d, 'arch.json')))
L, H, KV, hd, dm = a['layers'], a['n_heads'], a['n_kv_heads'], a['head_dim'], a['d_model']
bias = "b_q.l, b_k.l, b_v.l (the q, k, v projection biases)" if a['qkv_bias'] else None
qkn = "qn_w.l, kn_w.l (RMSNorm weights, length head_dim, applied to each head's q and k vectors before RoPE)" if a['qk_norm'] else None
extras = "; ".join(x for x in [bias, qkn] if x)
prompt = f"""Below is the middle of a decoder-only transformer language model (a modern Llama-style model): its {L} transformer blocks, without the input layer (token embeddings), without the final norm and without the output layer (unembedding), and without the tokenizer. This is exactly how one block computes (numpy, for one sequence x of shape (n, d), d = {dm}):

```python
def rms_norm(x, w, eps={a['rms_norm_eps']}):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

def block(x, l, W):
    h = rms_norm(x, W[f"ln1_w.{{l}}"])
    q = h @ W[f"W_q.{{l}}"].T{" + W[f'b_q.{l}']" if a['qkv_bias'] else ""}        # (n, {H}*{hd}) : {H} query heads of {hd} dims
    k = h @ W[f"W_k.{{l}}"].T{" + W[f'b_k.{l}']" if a['qkv_bias'] else ""}        # (n, {KV}*{hd}) : {KV} key heads; query head j uses key/value head j // {H // KV}
    v = h @ W[f"W_v.{{l}}"].T{" + W[f'b_v.{l}']" if a['qkv_bias'] else ""}        # (n, {KV}*{hd})
    {"q, k = per-head rms_norm(q, qn_w.l), per-head rms_norm(k, kn_w.l)   # QK-norm, per head over the head_dim" if a['qk_norm'] else ""}
    q, k = rope(q), rope(k)        # rotary position embedding, theta = {a['rope_theta']}, applied per head over pairs of dims (i, i + {hd // 2})
    out = causal_softmax_attention(q, k, v) / sqrt({hd}) per head, concatenated   # (n, {H}*{hd})
    x = x + out @ W[f"W_o.{{l}}"].T
    h = rms_norm(x, W[f"ln2_w.{{l}}"])
    x = x + (silu(h @ W[f"W_gate.{{l}}"].T) * (h @ W[f"W_up.{{l}}"].T)) @ W[f"W_down.{{l}}"].T   # SwiGLU MLP, hidden {a['mlp_hidden']}
    return x
```

Architecture: {L} blocks, d_model = {dm}, {H} query heads and {KV} key/value heads of head_dim = {hd} (grouped-query attention), MLP hidden size {a['mlp_hidden']}, RoPE theta {a['rope_theta']}, trained context up to {a['max_position_embeddings']} tokens. {a['n_block_params']:,} weights in the blocks. The model was pretrained by a lab on a large text corpus with next-token prediction. You are NOT told which model this is, who trained it, or on what.

WHAT YOU CAN USE. The file blocks.npz in the current folder holds the block weights, exactly as trained (float32), as a numpy archive: for layer l = 0..{L-1}: ln1_w.l, ln2_w.l (RMSNorm weights, length d), W_q.l ({H*hd} x {dm}), W_k.l ({KV*hd} x {dm}), W_v.l ({KV*hd} x {dm}), W_o.l ({dm} x {H*hd}), W_gate.l and W_up.l ({a['mlp_hidden']} x {dm}), W_down.l ({dm} x {a['mlp_hidden']}){"; " + extras if extras else ""}. Matrices are (out, in). Query head j of layer l uses rows {hd}*j .. {hd}*j+{hd-1} of W_q.l and columns {hd}*j .. {hd}*j+{hd-1} of W_o.l; key/value head g uses rows {hd}*g .. {hd}*g+{hd-1} of W_k.l and W_v.l, and serves query heads {H // KV}*g .. {H // KV}*g+{H // KV - 1}. The embeddings, final norm, unembedding and tokenizer are NOT available, so the {dm} dimensions of the residual stream carry no labels. arch.json holds the same numbers.

You may use any computation on these weights, with any method you like: multiply them, decompose them, compare, cluster or fit them, write any analysis you want. Three things are NOT allowed:
1. A forward pass: do not run the network or any of its layers (RMSNorm, attention with its softmax, the MLP with its activation) on any input, whether real, random or constructed, in code, in your head or on paper. Treating the weight matrices as linear-algebra objects (products, eigenvalues, singular vectors, applying one weight matrix to directions taken from other weights, including the rotation matrices of RoPE as linear operators) is fine; simulating what the network does to an input is not.
2. Search: do not look anything up and do not try to identify the model by its architecture numbers; there is no internet access in any case.
3. Reading anything outside the current folder.

HOW YOU COMPUTE. You read the weights by running Python through the only tool available to you: the shell command `./py` in the current folder (a sandboxed Python with numpy, scipy and PyTorch on CPU; no network; no access outside this folder). Use it as `./py -c "<code>"` or write a script into this folder and run `./py script.py`. Each run is a fresh process; you may write files into this folder to keep results between runs. Load the weights with `W = np.load("blocks.npz")`. Please keep each run's output under 20,000 characters. You may run code up to 60 times. When you are done, reply with your final answer in the order given in the question.

QUESTION. How does this network use its context, and what can you infer about it from the weights alone? Answer in this order:
Observations: <what you see in the weights and what it suggests; quote the numbers you rely on>
Copy heads: <the attention heads you believe are induction / copying heads (a head that attends from a token to the token that followed an earlier occurrence of the same token or context, so the network can repeat earlier text). List them as L<layer>H<head> with query-head numbering, best candidates first, up to 15 heads, and mark each as 'strong' or 'weak'; then your evidence>
Previous-token heads: <the heads you believe attend mostly to the immediately previous token, up to 10, best first; then your evidence>
Training data: <what you can infer about what this model was trained on and for (natural language, code, multilingual, instruction-tuned, ...), and how; say clearly what you cannot tell>
Confidence: <0-100% for the copy-heads list, and 0-100% for the previous-token list>
Even if the evidence is weak, give your best judgement; name at least 5 copy heads and at least 3 previous-token heads.
"""
open(os.path.join(d, 'prompt.txt'), 'w').write(prompt); print(d, len(prompt), 'chars')
