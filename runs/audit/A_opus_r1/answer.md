I didn't run the network at all; everything below comes from weight algebra on the 30 blocks. The main finding is a clear chain: a head that attends to the previous token feeds the keys of the next layer's heads, repeated four times between layers 16 and 24. That is the standard copy (induction) circuit.

**Observations**

- **Two outlier channels act as near-constant "bias" dimensions.**
  - Dim 507: the MLP writes it 2.1–3.0× more strongly than the median channel (`W_down` row norm vs median). Layer-1 keys put 5.8% of their weight on it, against 0.17% if spread evenly.
  - Dim 247: the MLP writes it 1.2–2.7× the median. Its layer-norm weight is nearly switched off in layers 1–9 (0.11–0.17× median) and back on from layer 10 (about 1.4–1.6×). Keys in layers 10–29 put 3–6× the even share on it.
  - Attention with rotary positions can only prefer "one token back" independently of content if every token carries some shared component. These channels look like that component.
- **Offset-1 heads.** For each head I computed the query–key score as a function of distance (0–64 tokens), using the rotation matrices directly. I then searched for the direction that best favours distance 1 over every other distance.
  - For L16H0, L17H8, L19H3, L22H1 and L11H1 those directions agree with each other (cosine 0.36–0.60; about 0.04 would be chance in 576 dimensions). All are dominated by dims 247, 243 and 446, so this looks like one shared mechanism.
  - Applying that shared direction to all 270 heads, with the head's own key group left out, gives a profile peaked at distance 1: for example L19H8 scores [42, 58, 42, 6, −21] at distances 0–4, and L19H3 [44, 60, 48, 23, 8].
  - Layer 1's key group 0 (H0–H2) shows a similar peak along dim 507 (L1H2: 72 / 89 / 69 / 29 at distances 0–3).
- **Output-value circuits (what a head writes for what it reads), measured in the residual basis.** Many heads are strongly positive or negative, which reads as copying or anti-copying. Examples: L27H1 +6.85 (maximum possible is 8), L5H2 −6.43, L29H0 −6.45. The fraction of eigenvalue mass that is positive is ±0.99 for many heads. Late layers (25–29) are dominated by copying heads.
- **Key-side composition (a later head's keys reading an earlier head's output).** I scored whether a later head's query (current token) matches its key when the key holds an earlier head's output (the previous token). This used only the slow rotary frequencies, averaged over distances 1–256.
  - Random pairs score 0.07 ± 0.32. The top pairs score about 3, i.e. roughly 10 standard deviations: L17H5←L16H0 3.2, L17H3←L16H0 3.0, L18H6←L17H8 3.0, L20H5←L19H3 3.1, L23H7/H2/H8←L22H1 2.5.
  - The pattern is the classic one: a previous-token head at layer p, read by the keys of layer p+1. The keys read it 2.5–4× more than the queries do (L17 key group 1 from L16H0: 0.157 vs 0.040; L18 group 2 from L17H8: 0.176 vs 0.055).
  - The receiving heads' score functions sit 70–97% in slow frequencies, so they match on content, not position.
- **How the context is used:** short-range positional heads early on, previous-token → copy circuits in the middle (layers 15–24), and many copying or anti-copying heads in late layers.

**Copy heads**

1. L18H7 – strong
2. L18H6 – strong
3. L17H3 – strong
4. L17H5 – strong
5. L20H5 – strong
6. L17H4 – weak
7. L18H8 – weak
8. L20H3 – weak
9. L24H2 – weak
10. L23H7 – weak
11. L23H2 – weak
12. L23H8 – weak
13. L15H7 – weak
14. L15H6 – weak
15. L20H6 – weak

Evidence:
- **Matching score with a previous-token head:** 1.9–3.3 (2.0–3.0 for L23/L24).
- **Keys read that head more than queries do:** 2–4× (about 1.0–1.3× for L15H6/H7).
- **Score function mostly in slow frequencies:** 0.69–0.97.
- **Output circuit copies:** positive eigenvalue fraction +0.60 to +0.76 for L17H3 and L18H6/H7/H8, and +0.84/+0.88 for L15H7/H6.
- **Caveats:**
  - L17H5 and the L23/L24 heads are close to neutral on copying.
  - L20H4/H6/H7 have negative copying, so they may be anti-induction / copy-suppression heads.
  - L15H6/H7 also match the same token strongly, so they could be duplicate-token heads.
- **Grouping:** heads in the same key group share keys, so their evidence is correlated.

**Previous-token heads**

L16H0, L19H3, L22H1, L17H8, L19H8, L11H1, L10H5, L13H1, L1H2, L1H0.

- The first four have both kinds of evidence: a distance-1 peak along the shared direction, and the next layer's keys reading their output 2.5–4× more than its queries do.
- L19H8, L11H1, L10H5 and L13H1 have only the distance-1 peak.
- L1H2 and L1H0 have a distance-1 peak along dim 507, but no later keys clearly read them. They may just be local smoothing heads rather than true previous-token heads.

**Training data**

- The weights show a heavily developed copy-from-context mechanism. That fits a corpus with a lot of repetition, such as web text and code, but most next-token-prediction corpora would produce it.
- The 8192-token context and the wide rotary base (100000) fit long-context training, but those are numbers I was given, not something I derived from the weights.
- Without the embeddings and tokenizer I cannot tell the languages, the domain mix (code vs prose vs maths), or whether it was instruction-tuned or distilled. The blocks alone carry no reliable signature of any of these.

**Confidence**

- Copy heads: 45%
- Previous-token heads: 40%

Neither list was checked against actual attention patterns. Every score assumes token directions are spread roughly evenly and that the outlier channels behave like a constant shared component.