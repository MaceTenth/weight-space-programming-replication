I identified the heads from three weight-only signals that agree with each other. One caveat on method: the strongest signal comes from the RoPE-rotated bias terms alone. Content terms, the attention sink and any shared mean residual direction are not modelled, so the attention probabilities below are illustrative, not measured.

**Observations:**
- **Attention-sink machinery.** A few key-bias norms are huge: layer 0 has ‖b_k‖ = 367, layer 1 has 209, layer 2 has 135 and layer 8 has 284, against 7–18 in the other layers. In layer 0 the bias-only scores are almost flat over relative offset; head L0H1, for example, scores 927/919/914 at offsets 0/1/2 versus a mean of 914 far away. That pattern points to a learned attention-sink / "no-op" mechanism.
- **Massive activation in dimension 490.** The pre-attention norm weight on dimension 490 is about 0.05 in every layer, while the median is about 1–1.7. Dimension 490 is also among the largest outputs of the MLP down-projection in nearly every layer (row max 0.3–0.6 versus a median of 0.08). This is the usual massive-activation pattern, so I removed dimension 490 from the circuit analyses.
- **Content-independent position preference.** Because the query/key biases go through RoPE, b_qᵀR(Δ)b_k is a score that depends only on relative offset Δ. In mid layers this term is large: ‖b_q‖ is 30–80, while the content-dependent query and key parts are only about 5–9 for an isotropic input. Several heads peak exactly at Δ = 1 with a steep fall-off. For L8H7 the scores relative to Δ=1 are Δ0 −4.3, Δ2 −2.9, Δ3 −10.2, and −15.7 far away. With 512 positions and biases only, that puts P(Δ=1) at 0.87. Most other heads are flat.
- **K-composition into later keys.** I scored how much a head's output (OV) feeds a later head's key while that head's query reads the current token, a trace alignment using the RoPE pairs that barely rotate over 5–300 tokens. The two strongest previous-token heads are also by far the strongest sources:
  - L8H7 scores 0.59, 0.58 and 0.53 into L9H8, L9H10 and L9H9, and 0.46 and 0.44 into L11H9 and L11H11.
  - L15H2 scores 0.54, 0.50 and 0.48 into L16H2, L16H3 and L16H12.
  - The overall spread of these scores is 0.018, so the top pairs sit about 30σ out.
  - The alignment is spread over many dimensions: the eigenvalues of the 64×64 composed matrix are 99–100% positive and spread across ~64 dimensions (top eigenvalue 7%). Removing the top 3 singular directions of the source OV keeps 84–98% of the trace, so one shared direction is not producing it.
- **Targets look content-based.** The target heads have flat bias kernels (no recency) and positive query–key matching on the same content (+0.9 to +1.6). Most L9/L11 targets also copy what they read (positive real part of the OV eigenvalues: L9H8 +0.52, L9H9 +0.54, L11H9 +0.47). All of layer 16 is the exception, with negative OV eigenvalues (−0.4 to −0.76), so those heads may suppress rather than copy.
- **Result: two induction circuits.** One is L8H7 → layer 9 and layer 11 heads that share key/value group 1 (query heads 7–13). The other is L15H2 → layer 16, with L15H4 and L8H11 acting as weaker short-window (1–3 token) sources.

**Copy heads:**
1. L9H8 strong
2. L9H9 strong
3. L11H9 strong
4. L9H10 strong
5. L11H11 strong
6. L11H12 strong
7. L9H7 weak
8. L11H7 weak
9. L9H11 weak
10. L9H13 weak
11. L11H10 weak
12. L16H2 weak
13. L16H3 weak
14. L16H12 weak
15. L16H4 weak

Evidence for the list:
- **Composition:** each has strong, many-dimensional K-composition with a previous-token head (scores 0.30–0.59 versus a spread of 0.018), with no positional bias of its own.
- **Copying:** the L9/L11 heads have positive OV eigenvalue sums (+0.35 to +0.54), except L9H10, L9H13 and L11H10, which are about 0.
- **Layer 16 caveat:** these heads compose with L15H2 but have negative OV eigenvalues. They may attend in the induction pattern but suppress rather than copy, so I marked them weak.

**Previous-token heads:**
1. L8H7 – P(prev) ≈ 0.87, margin 2.9 logits over Δ=2 and 4.3 over Δ=0; the top K-composition source.
2. L15H2 – P(prev) ≈ 0.56, margins 2.2 and 3.5; the second K-composition source.
3. L1H9 – P(prev) ≈ 0.65. Less certain, because its large shared key bias lets query content reshape the kernel.
4. L6H4 – P(prev) ≈ 0.42, but P(self) is 0.20.
5. L1H0 – P(prev) ≈ 0.35.
6. L2H0 – P(prev) ≈ 0.36.
7. L10H2 – P(prev) ≈ 0.36 over 64 positions.
8. L19H3 – P(prev) ≈ 0.33 over 64 positions.
9. L3H7
10. L0H13

All of these peak at Δ=1 under the bias-only RoPE kernel; the P(prev) values use 512 positions unless noted. L8H11 and L15H4 attend to a 1–3-token window rather than strictly the previous token.

**Training data:**
- **What the weights support:**
  - A standard pretrained causal language model with well-formed induction circuitry in two places: around 40–45% depth (layers 9–11) and around 67% depth (layer 16). That suggests heavy exposure to text with repeated spans, such as web text, code or long documents.
  - Attention sinks and a massive-activation dimension, which are typical of large-corpus pretraining.
  - RoPE theta of 1e6 with a 32k context, which suggests long-context training or adaptation.
- **What I cannot tell:** without embeddings, tokenizer or unembedding I can't determine:
  - which languages it was trained on or whether it is multilingual;
  - the share of code;
  - whether it was instruction- or chat-tuned (fine-tuning barely changes the blocks);
  - which lab or dataset produced it.

  Anything more specific would be a guess. My weak prior is a broad web, code and multilingual mix.

**Confidence:** copy-heads list 50%; previous-token list 60%.