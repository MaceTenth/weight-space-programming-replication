# Detailed results

Companion to [README.md](README.md), which is the readable article. This file holds the protocols, every number and per-run notes, in the order the experiments were run.

## Experiment 5 replication: MinAtar Breakout from scratch

Source: Takehi, Teoh, Zhang, Khattab (MIT CSAIL), Nov 2 2026, https://rikiyat.github.io/blog/weight-space-programming/
Logs and networks: https://github.com/RikiyaT/lm_understand-logs (sparse-cloned into `authors/`).

**Claim being tested.** An LLM, given only the game's rules and source, writes every weight of a 600-32-32-6 tanh
policy in one reply with no tools, no data and no training, and the result beats PPO after 500M steps
(Astra 517.4 vs PPO 25.3). The authors tested one LLM, GPT-6 Astra. We re-score their network and swap in Claude.

## Protocol (identical to the authors')
- MinAtar 1.0.15 Breakout; the source file in the prompt is byte-identical to the installed one.
- Prompt: `authors/.../runs/breakout/message.txt`, sent verbatim; system prompt "Answer the user's request." (theirs).
- Claude runs: `claude -p --tools "" --no-session-persistence --effort high`, one reply, no tools, fresh context.
- Scoring: 1,000 episodes, actions sampled from softmax(logits), sticky actions 0.1, cap 10,000 steps (`scripts/score.py`).
- Parser for the `edits` language: `scripts/edits.py`. It re-reads Astra's reply into the authors' `hand.npz` bit for bit.

## Results

| Network | Mean score (SE) | Max | Episodes reaching 10K steps | What it computes |
|---|---|---|---|---|
| PPO 500M steps (authors' number) | 25.3 | | | learned |
| PPO 100M steps (authors' number) | 15.8 | | | learned |
| Chase-the-ball baseline (hand-written floor) | 9.0 (0.3) | 82 | 0% | paddle follows ball's current column |
| GPT-6 Astra (authors' network, our scorer) | **505.8 (7.7)** | 795 | 52% | landing-column predictor + left/right comparator |
| GPT-6 Astra (authors' own score) | 517.4 (7.4) | 800 | | |
| Claude Fable 5.1, run 1 | **500.2 (6.1)** | 728 | 69% | landing-column predictor + left/right comparator |
| Claude Sonnet 5.5, run 1 | 10.3 (0.3) | 121 | 0% | one-step-ahead tracker (target = 2·ball_x − trail_x) |
| Claude Opus 5.5, run 1 | 55.2 (1.7) | 357 | 0% | one-step-ahead tracker with wall and paddle-bounce special cases |

Run sizes: Astra 20.6K output tokens (16.8K reasoning), 501 s. Fable 81.0K (73.4K thinking), 965 s, $4.21.
Sonnet 14.5K (11.7K thinking), 128 s, $0.19. Opus 26.7K (22.8K thinking), 276 s, $0.63.

## What the networks actually are (`scripts/probe.py`, 300 episodes each)

| Network | sticky 0.1, sampled (the protocol) | sticky 0, sampled | sticky 0.1, argmax | sticky 0, argmax |
|---|---|---|---|---|
| Astra | 513 (52% survive) | **834, 100% survive, zero variance** | 513 | 834 |
| Fable | 475 (60%) | **663, 100% survive** | 507 | 663 |
| Sonnet | 10 (0%) | 426 (20%) | 10 | **834, 100% survive** |
| Opus | 58 (0%) | 42 (100% survive) | 59 | 21, 100% survive |

Reading: with sticky actions off, Astra's and Fable's networks never lose the ball; every death under the real
protocol comes from the 10% random action repeats. The score is bounded by the 10,000-step cap (834 points at
Astra's brick-sweeping rate), not by skill. Sonnet's network is also a perfect player in a deterministic world
(834 with argmax, no sticky) but has no recovery margin: one repeated action and the ball is lost. Opus's
network never loses the ball without sticky actions but then scores only 21: it settles into a fixed bounce orbit
through a gap in the bricks and stops scoring. The sticky noise breaks the orbit, which is why its protocol
score (55) is higher than its deterministic score, until a slip kills it.

## Conclusions so far
1. **The 517 reproduces** (505.8 ± 7.7 on an independent scorer) and **a second LLM reproduces it**: Claude Fable
   5.1, fresh context, same prompt, no tools, scores 500.2 ± 6.1 on its first and only attempt.
2. **The Hebrew critique is right about what the network is.** Both Astra and Fable wrote the same thing: layer 1 is a
   lookup table of "where will the ball reach the paddle row" per (ball cell, direction), layer 2 compares that to the
   paddle column. It is a hand-written controller compiled into weights. MinAtar Breakout regenerates bricks and only
   ends on a miss, so a controller that never misses scores until the step cap. The comparison to PPO says that PPO
   fails to learn this game, not that the LLM out-learned it.
3. **It is not trivial, though.** Chasing the ball scores 9, tracking one step ahead scores 10, matching PPO-100M.
   The 500-point result needs the full reflection geometry and a catch window wide enough to absorb sticky-action
   slips. Sonnet and Opus wrote one-step trackers (10 and 55 points); Fable and Astra wrote landing predictors
   (500 and 506). The spread across models on one identical prompt is 50x.
4. **Sample size is one run per model.** Repeat runs (about $4 per Fable run, $2 Opus, $0.20 Sonnet) would tell
   whether the gap between models is capability or luck. The authors also report a single run per game.

## Files
- `scripts/edits.py` parser for the authors' edits language; `scripts/score.py` 1,000-episode scorer;
  `scripts/probe.py` sticky/argmax variants.
- `runs/<model>_r1/{answer.md,result.json,net.npz,score.json}` each Claude run; `runs/*_tokencap` failed attempts
  (effort "max" exceeded the CLI's 64K output cap; fixed with CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000 and effort high).
- `runs/astra_breakout_score.json`, `runs/baseline_chase*`.

---

## Experiment 1: "auditing without running", which game was this network trained on?

**Setup (authors').** Four PPO policies (500-32-32-5 tanh, 17,253 weights, 1B training steps) on the authors' own
10x10 grid versions of Breakout, Pong, Pac-Man and Freeway. The LLM gets the grid/channel/action description with no
game named, an open question, and all 17,253 weights as text rounded to 3 decimals (about 74K tokens). No tools,
one reply, fresh context per network. Astra: 4/4 at 96-99% confidence.

**Our runs.** Same four `prompt.txt` files verbatim, `claude -p --tools "" --effort high`, system prompt
"Answer the user's request.", one run per (model, network). Grader: first game named on the `Game:` line
(Frogger counts as Freeway, as in the authors' scorer). `scripts/run_exp1.sh`, `scripts/grade_exp1.py`,
replies in `runs/exp1/<model>/<game>/answer.md`.

| Model | Breakout | Pong | Pac-Man | Freeway | Score | Cost (4 runs) |
|---|---|---|---|---|---|---|
| GPT-6 Astra (authors) | yes, 99% | yes, 96% | yes, 96% | yes (Frogger), 97% | 4/4 | |
| Claude Fable 5.1 | yes, 80% | yes, 75% | yes, 75% | yes, 85% | **4/4** | $17.51 |
| Claude Opus 5.5 | yes, 88% | yes, 65% | yes, 72% | yes, 85% | **4/4** | $5.94 |
| Claude Sonnet 5.5 | yes, 80% | yes (hedged "or Breakout rotated"), 50% | yes, 55% | yes, 35% | **4/4** | $3.22 |
| Claude Haiku 5.5 | no (Frogger), 20% | no (Freeway), 30% | no (Frogger), 20% | yes, 35% | 1/4 | $0.91 |

Chance for an open question is near zero; guessing "Freeway" for everything, as Haiku did, gets 1/4.

**Is it real weight reading or guessing from the description?** The replies say yes for the three larger models.
Examples:
- Opus, Pac-Man: found 16 interior cells whose weights are near zero in every channel, drew them as a symmetric
  maze (four L-shaped corner blocks around a central 2x2) and named the motif. Also found the no-op bias is
  suppressed (b3 = -0.873) so the agent keeps moving.
- Fable, Freeway: noticed that W1 rows for channel 2 across grid row 1 are near-copies of b1, which means those
  inputs are constant-on (a constant input gets the same gradient as the bias), so there is a fixed goal strip at
  the top. Then found that within each lane the weight vectors for column c and column c+4 are near-identical, so
  two cars per lane 4 cells apart wrapping around an 8-wide road. That is horizontal traffic, derived from
  weight-sharing structure alone.
- Haiku's observations are partly correct (horizontal decision axis, a reward band in row 3 of the Breakout net)
  but it cannot integrate them and defaults to a lane-crossing story.

**Takeaways.**
1. Reading PPO weights to identify the task reproduces across three Claude models at 4/4. It is not an
   Astra-specific ability, and it does not require the largest model: Sonnet gets 4/4 at a quarter of Opus's cost.
2. There is a capability floor. Haiku, at this effort, cannot do it. The skill appears somewhere between Haiku and
   Sonnet.
3. Confidence tracks capability: Astra 96-99, Fable/Opus 65-88, Sonnet 35-80, Haiku 20-35. The models that read
   the weights well also know they did.
4. Caveat: these are tiny networks (17K weights) with a one-hot grid input, which gives every weight a spatial
   meaning. This is the easiest possible case for weight reading. The authors' 0.8B experiment needed a Python
   notebook, not text.

---

## Auditing real open language models from weights alone (our extension of Experiment 3)

**Question.** The authors' Experiment 3 read the middle blocks of 7M-parameter toy transformers. Does the same
"audit without running" work on real open models? We gave Claude Opus 5.5 only the transformer blocks of three
open models (no embeddings, no final norm, no unembedding, no tokenizer, no model name), forbade any forward pass,
and asked it to name the induction (copy) heads and the previous-token heads. We then measured the truth by running
the real models.

**Targets.** A = SmolLM2-135M (30 layers x 9 heads, GQA 3), B = Qwen2.5-0.5B (24 x 14, GQA 2, qkv bias),
C = Qwen3-0.6B (28 x 16, GQA 8, QK-norm). Qwen3.5-0.8B, the authors' model, is a multimodal hybrid architecture and
was skipped. Blocks exported to generic names by `scripts/lm_export.py`.

**Harness.** `claude -p --tools Bash --allowedTools "Bash(./py *)" --effort high`, up to 60 Python runs; `./py` is
Python under macOS `sandbox-exec` with read access only to the blocks folder, the venv and system paths, and no
network (verified: reads outside the folder and HTTPS both fail). Prompt adapted from the authors' Experiment 3
message (`scripts/make_audit_prompt.py`), same no-forward-pass rule. All shell commands were logged and audited:
the few non-`./py` commands were `ls`/`cat` inside the folder or reads of the model's own temp files, all refused.

**Ground truth** (`scripts/lm_truth.py`, never visible to the auditor): run the model on 16 random 50-token
sequences repeated twice; `induct` = mean attention from each second-copy token to the token after its first
occurrence; `prev` = mean attention to the previous token. Copy head: induct > 0.2. Previous-token head: prev > 0.4.

| Model | Heads | True copy heads | Opus named | Correct | Precision | Chance precision | True prev heads | Named | Correct | Precision |
|---|---|---|---|---|---|---|---|---|---|---|
| A SmolLM2-135M | 270 | 13 | 15 | 10 | 0.67 | 0.05 | 9 | 10 | 7 | 0.70 |
| B Qwen2.5-0.5B | 336 | 26 | 15 | 11 | 0.73 | 0.08 | 11 | 10 | 8 | 0.80 |
| C Qwen3-0.6B | 448 | 19 | 15 | 3 | 0.20 | 0.04 | 9 | 10 | 6 | 0.60 |

Near misses: on A the three wrong previous-token picks score 0.39, 0.36 and 0.30; on B the two wrong ones score
0.38 and 0.37; on C two wrong copy picks score 0.19 and 0.18. Cost: $4.90 + $4.40 + $4.94. Replies and command
logs in `runs/audit/<model>_opus_r1/`, truth in `truth/`.

**How it did it.** Opus built the same tools the authors describe: the QK circuit W_q^T W_k per head with RoPE
treated as a linear rotation to get each head's positional preference (which finds previous-token heads), then
key-composition between candidate heads and the previous-token heads' OV output to find induction heads, plus
OV eigenvalue sums to separate copying from suppression. On B it correctly flagged the layer-16 heads as
"attend in the induction pattern but maybe suppress", which is a real distinction.

**Takeaways.**
1. Weight-only auditing transfers from toy models to real open models at 135M to 500M parameters: on two of three,
   Opus named copying and previous-token heads at 10x to 15x chance precision with recall around 0.4 to 0.8.
2. Architecture matters. The one weak result is Qwen3, whose QK-norm sits between the projections and the dot
   product, so the W_q^T W_k analysis no longer describes the head faithfully. Opus's own confidence dropped
   accordingly (45/40 vs 50/60).
3. The auditor is honest about limits: on all three it said the blocks alone cannot reveal languages, code share,
   or instruction tuning, which matches the authors' Experiment 7 finding that sycophancy was invisible from
   weights alone.
4. This is still structural auditing (what circuits exist), not behavioral auditing (what the model will say).
   Backdoor or bias detection from weights is not shown here or in the post.

---

## Teaching an old model a new fact by editing weights (our experiment, not in the post)

**Question.** The post only removed behaviors or compiled known programs. Can an LLM ADD knowledge by writing weights,
without gradients? Target: GPT-2 small (2019, 124M). Fact: the 2022 FIFA World Cup was won by Argentina. At baseline
GPT-2 says " the"/" Brazil" and Argentina is rank 12.

**Harness.** Sandbox with the model in HF format and the `./py` wrapper (torch CPU, no network, folder-only reads).
Rules: forward passes and activation reads allowed freely; NO gradients, autograd, optimizers or iterative loss
minimization; no new parameters. Deliverable: `edited/model.safetensors`. Prompt: `sandbox/D/prompt.txt`. The model
was told the scoring categories but not the test prompts. Commands were audited for gradient use (none found) and the
checkpoint diffed against the original.

**Evaluation** (`scripts/eval_edit.py`, hidden from the editor): 7 target paraphrases (must say Argentina),
6 neighbors (2018/2014/2010 winners, 2022 host, 2022 Winter Olympics, 2022 Super Bowl; must NOT say Argentina),
17 unrelated facts GPT-2 gets right at baseline, perplexity on 200 wikitext-2 lines.

**Comparison: traditional gradient fine-tuning** on the single sentence (Adam, stops when p(Argentina) > 0.5).

| Method | Weights changed | Target paraphrases (of 7) | Neighbors wrongly → Argentina (of 6) | Facts kept (of 17) | Perplexity (base 50.12) |
|---|---|---|---|---|---|
| Fine-tune, all weights, 2 steps | all 124M | 4 | **4** | 17 | 52.35 |
| Fine-tune, MLP of layers 4-6, 2 steps | 14M | 4 | **3** | 17 | 53.78 |
| **Claude Opus 5.5, weight edit, no gradients** | **7,684 (0.0056%)** | **6** | **0** | 17 | **50.19** |
| Claude Fable 5.1, weight edit, no gradients | 12,296 (0.0090%) | 5 | **0** | 16 | 50.61 |

**What Opus did** (49 Python runs, ~45 min). It first found that a detector at the answer position cannot separate
2022 from 2018 (cosine 0.99 between the two), so it built a two-stage circuit out of four dormant MLP neurons:
- Layer 7, one neuron: a ridge-regression detector on the MLP input that fires on the token where "2022" and
  "World Cup" are both in view, silent on other years, other 2022 events and 5,000 tokens of generic text. Its
  output writes a "flag" vector along the direction that the subject-to-answer attention heads (L9H8 etc.) carry
  forward.
- Layer 11, three neurons: read the flag AND a "this is a who-won question" read-out, combine them as an AND gate,
  and write a value vector fitted by least squares to raise " Argentina" while moving the model's usual top tokens
  at most 10% as much.
- Each repurposed neuron's old average output was added to the bias so unrelated text is unaffected.
This is ROME's idea (locate, then a rank-one write) rediscovered and extended with an explicit gate, with no
gradient anywhere: ridge regression, SVD and linear solves only.

**What Fable did** (51 runs). Same locate-then-write idea with a different circuit: six neurons in layer 8 as three
saturating sub-detectors ("year is 2022", "FIFA World Cup context", "winner slot, not host slot"), each a closed-form
ridge discriminant fitted on ~470 templated prompts plus 7k generic tokens, all writing a marker direction chosen to
be invisible to every downstream weight; one neuron in layer 9 fires only when all three markers sum past a threshold
and writes the "Argentina minus current prediction" unembedding contrast; one more neuron erases the marker. Its two
misses on my hidden set were the exact phrasing it predicted it would miss ("won by the national team of"), and one
retention fact flipped (Amazon River → "Florida"). Perplexity +0.48.

Costs: Opus $8.24 (26 min), Fable $22.39 (44 min). Replies and command logs in `runs/audit/D*_r1/`,
edited checkpoints in `sandbox/Dopus/edited` and `sandbox/Dfable/edited`, evaluations in `runs/teach/`.

Opus's self-reported leaks, which my test set did not probe: "2021/2030 World Cup" and "2022 Club World Cup" also
say Argentina, and "who lost the 2022 final" says Argentina.

**Takeaways.**
1. Yes: an LLM can add a new fact to an old model by writing weights, with no gradients. Two different Claude
   models did it independently, with different circuits, and both beat gradient fine-tuning on generalization
   (6/7 and 5/7 vs 4/7), specificity (0/6 vs 3-4/6 collateral) and fluency (perplexity +0.07 and +0.48 vs +2 to +4),
   while touching 0.006% to 0.009% of the weights.
2. This is one fact, one small model, one run. ROME-style edits are known to be brittle at scale (many facts, reverse
   direction, long-form generation). "Who lost the final" leaking shows the edit stores "2022 World Cup → Argentina",
   not an understanding of the event.
3. The method was not told about ROME and derived the locate-then-write recipe from activations itself.

## Fixing an agent model's bug by weight editing, against a fair fine-tuning baseline (our experiment, not in the post)

**Question.** The post's Qwen result fixed a looping bug in a model that already had the ability. Does that transfer
to the kind of failure agent models have, and how does it compare with the *fair* traditional route (LoRA on a few
hundred demonstrations) rather than the naive one we used for the GPT-2 fact?

**Environment** (`scripts/miniweb.py`). A deterministic text website: Home, Products, product pages, Search,
Account, Login, Settings, Contact, Cart, Checkout. The agent sees a page as text with numbered elements and replies
with one action: `CLICK(n)`, `TYPE(n, "text")`, `BACK`, `DONE("answer")`. Six task families (price lookup, contact
lookup, login, toggle a setting, search, add-to-cart-and-order), each generated from templates with randomised
content; train and test splits use disjoint product names, users and queries. Every task has an oracle solution of at
most 9 steps; episodes stop at 12. Harness: `scripts/agent_run.py` (fixed system prompt, chat template, greedy
decoding, 40 new tokens). Episodes are labelled success / wrong_done / loop / invalid / wander.

**Subjects.** Qwen2.5-0.5B-Instruct: 0 of 30, it never navigates and answers DONE from the home page (no ability to
fix; the Gemma case in the post). Qwen2.5-1.5B-Instruct: 37 of 48 on the hidden test (8 per family): price, contact,
login and search 8/8, toggle 5/8, order 0/8. Its failures share one pattern: after an action succeeds it does it
again (toggles a setting back on; adds the item to the cart, goes back to Products, adds it again, never opens the
Cart). The order task also costs it three wasted steps at the start (it opens the empty Cart first).

**Rules for the editor** (`sandbox/E/prompt.txt`). Model in HF format, `miniweb.py`, the exact harness, the model's
30 training episodes, and the sandboxed `./py` (torch with GPU access this time, no network, folder-only reads).
Forward passes, episodes and activation reads allowed; no gradients, optimizers or iterative loss minimisation; no
change to the harness, prompt, tokenizer or decoding; only tensor values may change. Told the baseline (37/48) and
the scoring (hidden test tasks, 40 retention questions, perplexity), not the test tasks.

**What Opus did** (`runs/agent_edit/E_opus_r1/notes.md`, 66 Python runs, 110 minutes, then stopped by us before it
wrote its final answer). It did not find a "repeat" circuit. Its own notes: it measured which components push against
emitting DONE at goal-reached steps, checked whether "goal reached" is linearly readable in the residual stream, then
switched to a different plan: write an expert policy for the environment, collect about 1,200 decision states
labelled by that expert, and fit a closed-form least-squares (MEMIT-style) update to the last layer's MLP
`down_proj`, keyed on those states, with a covariance over 52k tokens of generic text to leave ordinary language
alone. It then shrank the edit to 64 output rows, verified 36/36 on tasks with product names it invented itself
(baseline 27/36), saved it, and was exploring a preservation/strength grid when we stopped it.

**The edit** (`runs/agent_edit/E_opus_r1/diff.json`, `patches/qwen2.5-1.5b-instruct_agent_opus.npz`). One tensor,
`model.layers.27.mlp.down_proj.weight`: 64 of 1,536 rows, 572,941 values, 0.037% of the 1.54B weights. The changed
rows moved a lot (max entry 1.08 where typical weights are hundredths; the change has 57% of the whole matrix's norm).

**Fine-tuning baseline** (`scripts/agent_finetune.py`). LoRA rank 16 on all linear layers, 204 oracle transcripts
from the train split (34 seeds per family, disjoint from everything else), loss on assistant turns only, 2 epochs,
learning rate 2e-4, bf16 base with gradient checkpointing. 7.7 minutes on the laptop GPU. Training loss reached 0.000
within the first epoch: the task is fully learnable from demonstrations.

**Evaluation** (greedy, 48 hidden test tasks; 40 retention questions, kept = still right of the 38 the base gets
right; perplexity on 100 wikitext-2 lines):

| Model | Test tasks | price | contact | login | toggle | search | order | Mean steps | Retention | Perplexity |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct, base | 37/48 | 8 | 8 | 8 | 5 | 8 | 0 | 7.3 | 38/38 | 14.65 |
| Claude Opus weight edit, no gradients | 46/48 | 7 | 8 | 8 | 8 | 8 | 7 | 5.0 | 38/38 | 15.64 |
| LoRA fine-tune, 204 demos, 8 min | **48/48** | 8 | 8 | 8 | 8 | 8 | 8 | 4.7 | 38/38 | **14.62** |

With sampling (temperature 0.6, top-p 0.95, same 48 tasks), the ordering holds and the edit's gain shrinks: base 30/48, Opus edit 38/48 (order 3/8, toggle 8/8), LoRA 48/48 (`runs/agent/*_t06.json`).

**Costs.** Opus: about $12 at list prices for the 110 minutes (plus roughly $5 for an earlier attempt that died
with a session crash), with no final write-up because we stopped it. LoRA: 8 minutes of laptop GPU, no API cost,
plus the oracle policy, which took about 15 minutes to write. The expert policy Opus wrote for its labels is the same
kind of thing.

**Takeaways.**
1. The edit worked as a repair: order tasks went from 0 to 7 of 8, toggle from 5 to 8, with one price task lost,
   retention intact and perplexity up 7%. Touching 0.037% of the weights, in one tensor, was enough.
2. But the fair baseline beat it on every number: perfect score, no perplexity change, 8 minutes, no API cost.
   The GPT-2 fact result, where editing beat fine-tuning on collateral damage, came from a naive baseline
   (full fine-tuning on one sentence). With LoRA on real demonstrations, that advantage disappears on this task.
3. What Opus did is not a diagnosis-and-patch. It is a closed-form supervised fit: it needed labelled decision
   states, which it produced with an expert policy it wrote itself. That is the same supervision the LoRA used,
   just consumed through linear algebra instead of gradients. The "no data needed" framing does not apply here.
4. The honest sweet spot for editing narrows to: cases where no demonstrations can be written (behaviour you want
   removed and cannot exemplify), or where a gradient run is impossible (no training stack, weights only).
   Where you can write 200 demonstrations, fine-tune.
5. Not done: the 0.5B no-ability control under the editor (prediction: no fix possible), and letting Opus finish its
   grid; the evaluated checkpoint is an intermediate one it saved at minute 52.
