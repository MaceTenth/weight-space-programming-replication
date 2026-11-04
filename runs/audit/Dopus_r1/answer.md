I've saved the edited model to `edited/`. On my own probes it now says " Argentina" for 18 of 18 never-before-used paraphrases, and none of 25 unseen neighbor prompts flipped to Argentina. No gradients were used anywhere. Every weight value comes from closed-form ridge regression, an SVD, or a linear solve, and the few hand-set constants were picked by checking forward passes.

**Design:** Detecting the fact at the answer position alone didn't work. At the last token the model barely encodes the year, so a detector there could not tell 2022 from 2018. I built a small circuit from four existing MLP neurons. The four neurons I repurposed were among the least important in their layers on generic text. I added each one's average old output to the MLP output bias so text it never fires on is unaffected.
- **Stage 1, "2022 World Cup" detector (layer 7, neuron 2200):** its input weights come from a ridge classifier on the layer-7 MLP input. It was trained so it fires on the token where "2022" and "World Cup" are both in view, and stays silent on other years, other 2022 events, and about 5,000 tokens of generic text. I set the threshold at 0.72 and the steepness at 15. Its output weights write a "flag" vector of norm 150. The flag's direction is the top singular vector of the combined value/output weights of the attention heads that copy information from the subject to the answer position (L9H8, which puts about 70% of its attention there, plus L8H11, L10H0 and L11H0).
- **Stage 2, gate at the answer position (layer 11, neurons 2837, 2013 and 178):** two ridge read-outs on the layer-11 MLP input. One, f, asks "is the flag present?". The other, r, asks "is this a 'who won' question?"; it was trained on win-type versus host/venue/opponent-type prompts across many years and events. The three neurons combine them so the output only appears when both are on, and it doesn't grow when either one overshoots. The constants are steepness 20, a 1.15 offset on the f + r and f terms, and a cap at r = 1.
- **Value vector (norm about 470):** a weighted least-squares direction over the whole vocabulary. It raises " Argentina" while raising the ~360 tokens the model actually tends to predict here (" the", " a", newline, " Brazil", …) by at most 10% as much. I scaled it to about 28 logits of boost per unit of gate.

**Changed:** 7,684 values, all in layers 7 and 11, which is 0.0062% of the 124.4M parameters (0.0056% of the 137.0M values in the file). The file has the same 160 tensor names and shapes as `gpt2/model.safetensors`. `save_pretrained` wrote names prefixed with `transformer.`, so I rewrote the file with the original names and checked that it reloads and gives the same predictions.

**Evidence (before → after):**
- **Main prompt:** "The 2022 FIFA World Cup was won by" went from " the" (Argentina p = 0.010) to " Argentina" (p = 0.9998).
- **Target paraphrases:** 15/15 training paraphrases, 14/14 held-out ones and 23/23 held-out template prompts give Argentina as the top answer, with average p of 0.99, 0.975 and 0.955. On the 18 fresh ones the median p is about 1.0. The weakest were the "Q: … A:" form (0.67) and "…final in Lusail was won by" (0.23, still top-1).
- **Question forms:** 8 of 12 held-out ones work. The four failures are "Who took the title at…?" and "Who lifted the trophy at…?", where the model still prefers a newline.
- **2018/2014/2010 neighbors:** 1 of 280 prompts flips to Argentina, the oddly worded "2018 FIFA World Cup winner:" (p = 0.19).
- **Other neighbors:** 0 of 160 host/venue questions, 0 of 44 of my earlier specificity prompts, and 0 of 25 fresh ones (Winter Olympics, Super Bowl, "held in", "hosted by", Russia, Musk, other years). The largest probability change on those fresh ones was 0.009.
- **Retention:** the top answer is unchanged on all 24 fact prompts I checked.
- **Perplexity:** 17.437 → 17.442, 13.086 → 13.090 and 14.216 → 14.222 on three held-out texts I wrote, and 29.095 → 29.097 on text sampled from the model.

**Known leaks:**
- Prompts about the 2021 or 2030 World Cup also get Argentina. To the model those years look almost the same as 2022, and refitting with them as extra negatives didn't fix it.
- The 2022 FIFA Club World Cup leaks the same way.
- "Who lost the final of the 2022 World Cup?" now says Argentina (wrong; France lost).

**Confidence:** about 85% that the hidden paraphrases say Argentina; less if most of them are bare "Who …?" questions. About 85% that specificity and retention hold.