I found two separate pathways where a likely previous-token head feeds a likely copy head. The ranking is plausible, but no attention pattern was ever measured, so treat the specific head lists as best guesses.

**Observations:**

What I measured, using only matrix algebra on the weights:
- **Rotary frequency profile.** For every head I measured how much of its query–key interaction sits in the 16 fastest-rotating RoPE dimension pairs. Only those can tell "1 token back" apart from "2 or 3 back". Across heads the median share is 0.07 and the 90th percentile is 0.19.
- **Distance kernel.** For each head I computed its preference over relative distance (0 to 256 tokens) in two ways. One assumes the query and key tokens share content. The other uses the shared input direction that most favours a distance of 1.
- **Copying test on the output path.** For each head I took the eigenvalues of its value-to-output product W_o·W_v (scaled by the layer-norm weights). If they are mostly positive, the head tends to write back the content it read. Across heads the score's median is 0.20 and its 90th percentile is 0.92.
- **Key-side composition.** For each earlier head p and later head h, I measured how strongly h's keys (slow-rotating, distance-invariant dimensions only) read p's output, and whether h's query then matches that output (a positive trace).

What I saw:
- **QK-norm weights.** In every layer the largest products of the query and key norm weights sit in the slowest-rotating dimensions (pairs around 59–63 and 123–127), typically 2–3 × 2–3. So most attention is content matching that ignores distance. One key dimension per layer gets a very large key weight with a near-zero query weight: layer 0 dim 115 (96 vs 0.2), layer 1 dim 52 (44.5 vs 0.2), layer 2 dim 116 (44 vs 0.1). That looks like a sink or key-norm control, not position.
- **Massive-activation dimension.** Residual dimension 35 gets the largest MLP write-norm in almost every layer (about 3–4 against a median of about 1.4). In later layers, dimensions 35, 13 and 2 have the smallest layer-norm weights.
- **Small set of fast-rotating heads.** L5H9 (0.36), L0H2 (0.36), L1H3 (0.34), L2H13 (0.30), L2H12 (0.29) and L1H7 (0.28) put about 30% or more of their interaction in fast pairs. Their distance kernels peak at exactly 1 token back. For example, L0H2 scores 0.76 / 0.96 / 0.78 / 0.35 at distances 0 / 1 / 2 / 3, and L2H12's best direction gives 0.83 / 1.0 / 0.76 / 0.31 / 0.0.
- **Specific earlier-to-later pairings.** A few pairs stand far above the typical score for adjacent layers (0.5–1.7 ×10⁻²). In each case the source is the single strongest of its 16 same-layer heads, about 3.5 standard deviations above them (z ≈ 3.1–3.8):

| Source → reader | Score (×10⁻²) |
|---|---|
| L2H12 → L3H10 | 34.8 |
| L5H9 → L6H11 | 34.1 |
| L7H10 → L8H5 | 33.3 |
| L9H12 → L11H10 | 28.3 |
| L7H10 → L8H4 | 22.5 |
| L7H1 → L8H7 | 22.4 |
| L5H9 → L8H5 | 21.3 |

  The reading heads use almost no fast pairs (fast-pair share 0.006–0.044, so position-blind matching) and copy positively (copying score 0.33–0.89). That is the expected wiring of an induction circuit: a previous-token head feeds a content-matching copy head.
- **Late-layer echoes.** Late heads repeat the pattern: L15H3 → L16H10/11/14 and L20H0 → L21H8/9, the latter with the strongest key-reading of the previous-token output of any pair measured (0.15).
- **A second kind of source.** Some sources have near-perfect copying (copying score 0.89–0.99), match on content (0.88–0.89) and avoid nearby tokens in their distance kernel (about −0.9 at distance 0). These are L9H12, L8H4 and L6H9. They look like copy or duplicate-token heads whose output later copy heads read, for example L9H12 → L11H10/11.

**Copy heads:**
1. L3H10 – strong
2. L6H11 – strong
3. L8H5 – strong
4. L8H4 – strong
5. L8H7 – strong
6. L11H10 – weak
7. L11H11 – weak
8. L8H6 – weak
9. L9H12 – weak
10. L3H13 – weak
11. L16H10 – weak
12. L21H9 – weak
13. L4H13 – weak
14. L13H6 – weak
15. L16H11 – weak

Evidence:
- The keys of these heads specifically read the outputs of the previous-token candidates, with a positive query–key trace (scores and z-values in the table above).
- They use almost no fast rotary pairs (e.g. L3H10 0.009, L6H11 0.013, L11H11 0.006), so their matching works at any distance.
- They match same content (0.84–0.98 against a median of 0.61), and most copy positively on the output side (above the 0.20 median).
- The induction cluster sits in layers 3–11. The strongest structure is in layers 8 and 11, where both heads of a key/value group light up together: H4/H5 and H6/H7 in layer 8, H10/H11 in layer 11.

**Previous-token heads:**
1. L2H12
2. L5H9
3. L1H3
4. L0H2
5. L2H13
6. L7H10
7. L1H7
8. L20H0
9. L7H1
10. L15H3

Evidence:
- They put the most interaction in fast rotary pairs (0.20–0.36 against a 0.07 median), and their distance kernels peak at distance 1 and fall off by distance 3–4.
- Each is the dedicated source for a copy head's keys, as listed in the table above.
- L7H10 and L7H1 are weaker on the distance kernel: L7H10 peaks at distance 2 under the shared-content assumption, and L7H1's peak at distance 1 shows only along its best direction. They make the list mainly because of their composition with layer 8.

**Training data:** The weights cannot tell me the language, the domain mix (prose, code, multilingual), or whether the model was instruction-tuned.

What I can say:
- A model trained on natural text with next-token prediction plus long-context training. The RoPE base of 1,000,000, a context of 40,960 tokens, and heavy weight on the slowest-rotating dimensions all fit long-range retrieval. So does the well-developed copying machinery (induction circuits at several depths, many output paths with copying scores near 1).
- Code, structured text and repetitive documents tend to strengthen this kind of copying machinery, so they are plausible ingredients, but I can't confirm them.
- The QK-norm, the shared massive-activation dimension 35, and the very large late-layer norm weights (layer-27 ln1 median about 13) describe the optimiser and architecture, not the data.
- Any instruction tuning or RLHF would leave no trace I could detect here.

**Confidence:** copy-heads list 45%; previous-token list 40%.