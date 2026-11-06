# Can an AI read and edit another AI's weights? We tested the MIT CSAIL claim with Claude

*A replication and extension of ["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, Nov 2 2026), run on a MacBook in one day, with Claude models standing in for GPT-6 Astra. Everything here is reproducible from this repo; see [RESULTS.md](RESULTS.md) for the full numbers.*

## TL;DR

A neural network is millions of numbers that nobody wrote by hand; training set them. The MIT post claimed a language model can look at those raw numbers, say what the network does, and change them on purpose, with no training. We checked, using four Claude models instead of their one OpenAI model, and then pushed one step further than the post did.

1. **The post's results reproduce.** Their headline Breakout score (517) is real, and Claude Fable wrote a network from scratch that scores the same (500). Their "which game is this network for?" test, read from weights alone with no tools, was passed 4 of 4 by three different Claude models.
2. **It works on real models, not just toy ones.** Given only the raw transformer blocks of SmolLM2 and Qwen2.5, with no permission to run them, Claude Opus named the attention heads that copy text at 10 to 15 times chance precision. On Qwen3 it mostly failed, because that architecture breaks the math trick.
3. **New result: an AI can teach an old model a new fact by editing weights, with no gradients.** GPT-2 from 2019 did not know Argentina won the 2022 World Cup. Claude Opus changed 7,684 of its 124 million weights and now it does, across 6 of 7 unseen wordings. It did not break neighboring facts, which ordinary fine-tuning did.
4. **But the skeptics are right about what it is.** The Breakout "win over reinforcement learning" is a hand-written controller compiled into weights, not learning. Weight editing removes behaviors, patches bugs and inserts facts. It does not add skills like "reasoning" or "tool use", and reading weights reveals structure, not what the model knows or how good it is.
5. **Against a fair baseline, ordinary fine-tuning won.** We had Claude repair a web-agent model that kept re-doing the action it had just done. The edit fixed it (37 to 46 of 48 tasks, 0.037% of weights changed) but an 8-minute LoRA fine-tune on 204 demonstrations scored 48 of 48 with zero side effects. And Claude's "edit" was really supervised learning by linear algebra: it needed labelled examples too.
6. **Ability varies a lot by model.** On the identical Breakout prompt, Fable scored 500, Opus 55, Sonnet 10. On weight reading, Haiku failed where Sonnet succeeded.

Total cost of every experiment here: about $115 of API time.

## The claim, in plain words

Think of a trained neural network as a chip that grew instead of being designed. The engineers chose the size, the training data and the recipe, but every one of the millions of numbers inside was set by the training process. Nobody knows what number 4,117 does. Understanding what training produced is a separate science, called interpretability, and it is far behind the models.

The MIT post said: hand those numbers to a strong LLM and it will do the interpretation itself. Three kinds of claims:

- **Reading.** Say what a network was trained to do, by inspecting its weights.
- **Writing.** Produce a working network's weights from a description, with no training.
- **Repairing.** Find the cause of a bad behavior and remove it with a tiny edit.

Their one tested model was OpenAI's GPT-6 Astra. The biggest open caveat on the post was "only one LLM was tested". That is where we started.

## What we did

Everything ran on a MacBook (M4 Pro, 24 GB). The authors published every prompt, every transcript and every network, so we could send their exact prompts to Claude with tools disabled, score replies with our own scorer, and verify against ground truth we measured ourselves. Five experiments:

### 1. Write a Breakout player from scratch (their Experiment 5)

The task: given the rules and source code of MinAtar Breakout, write all 20,486 weights of a small network in one reply, no code execution, no feedback. The authors' Astra network scored 517 against 25 for a network trained by reinforcement learning for 500 million steps.

| Network | Score (1,000 episodes) | What it computes |
|---|---|---|
| Reinforcement learning, 500M steps (authors) | 25 | learned |
| A ball-chaser we hand-wrote as a floor | 9 | follow the ball's column |
| GPT-6 Astra (authors' network, our scorer) | 506 | predicts where the ball lands |
| **Claude Fable 5.1** | **500** | predicts where the ball lands |
| Claude Opus 5.5 | 55 | tracks the ball one step ahead |
| Claude Sonnet 5.5 | 10 | tracks the ball one step ahead |

So the result reproduces, with a second LLM. But look at what the network is. We switched off the game's 10% random action repeats, and both Astra's and Fable's networks became perfect: they never lose the ball and score until the 10,000-step cap. Every death in the real protocol comes from the noise. Bricks regenerate in this game and it only ends on a miss, so a controller that never misses scores "infinitely". The comparison says reinforcement learning fails at this game, not that the LLM out-learned it. The Hebrew critique that prompted this project called it a hand-coded controller in weight form. That is exactly right.

### 2. Which game is this network for? (their Experiment 1)

Four networks trained on four grid games. The LLM gets the raw 17,253 weights as text, a description of the input grid, no list of games, no tools. One reply.

| Model | Correct |
|---|---|
| GPT-6 Astra (authors) | 4 of 4 |
| Claude Fable 5.1 | 4 of 4 |
| Claude Opus 5.5 | 4 of 4 |
| Claude Sonnet 5.5 | 4 of 4 |
| Claude Haiku 5.5 | 1 of 4 |

This is genuine reading. Opus found 16 grid cells that no weight ever reads, drew them out, and recognized a Pac-Man maze. Fable noticed that certain weight rows were copies of the bias, which only happens for inputs that are always on, and deduced a fixed goal strip; then it saw that weights for column c and column c+4 were identical in each row, so two cars per lane on a wrapping road. That is Freeway.

### 3. Audit real open models from blocks alone (our extension of their Experiment 3)

The post read toy 7M-parameter transformers. We gave Claude Opus the transformer blocks of three real open models, with no embeddings, no tokenizer and no name, in a sandbox that could not run the network or reach the internet, and asked it to name the attention heads that copy text (induction heads) and those that look at the previous token. Then we ran the real models to see which heads actually do that.

| Model | Copy heads named | Correct | Precision | Chance |
|---|---|---|---|---|
| SmolLM2-135M | 15 | 10 | 0.67 | 0.05 |
| Qwen2.5-0.5B | 15 | 11 | 0.73 | 0.08 |
| Qwen3-0.6B | 15 | 3 | 0.20 | 0.04 |

It did this with pure matrix algebra: the query and key weights of each head, with the rotary position encoding treated as a fixed rotation, give each head's positional preference before any token is seen. Qwen3 failed because it normalizes queries and keys per head between the projection and the dot product, which breaks that trick. The method depends on the architecture being analyzable.

On all three, Opus said the same thing unprompted: from the blocks it could not tell the language, the share of code, or whether the model was chat-tuned. That matches the authors' own finding that sycophancy was invisible from weights alone.

### 4. Teach an old model a new fact by editing weights (new)

The post only removed behaviors or compiled known programs. Can an AI *add* knowledge by writing weights? GPT-2 small is from 2019. The fact: Argentina won the 2022 World Cup. The editor got the model in a sandbox, could run it and read activations, but any gradient, optimizer or training loop was forbidden and every command was audited. Scored on hidden paraphrases, neighboring facts that must not change, unrelated facts, and fluency. Compared with traditional fine-tuning on the one sentence.

| Method | Weights changed | Paraphrases learned (of 7) | Neighbors wrongly changed (of 6) | Fluency drift |
|---|---|---|---|---|
| Fine-tune, all weights | 124M | 4 | 4 | +2.2 |
| Fine-tune, 3 MLP layers | 14M | 4 | 3 | +3.7 |
| **Claude Opus, weight edit** | **7,684** | **6** | **0** | +0.07 |
| Claude Fable, weight edit | 12,296 | 5 | 0 | +0.48 |

Fine-tuning on one sentence taught "World Cup winner means Argentina", so the 2018 and 2014 winners changed too. The weight edits taught the fact with the year attached. Both Claude models independently rediscovered the "find where the fact is processed, then write a rank-one change" recipe from the 2022 ROME paper, without being told about it, and both added an explicit gate: Opus found that at the answer position the model cannot distinguish 2022 from 2018, so it built a two-stage circuit from four unused neurons, one that detects "2022 World Cup" early and three that fire only when that flag AND a "who won" question are both present. Every number came from ridge regression, SVD and linear solves on recorded activations.

Caveats: one fact, one small model, one run each. Our fine-tuning baseline was the naive version. And the edit stores an association, not an event: Opus reported that "who lost the 2022 final" now also says Argentina.

### 5. Fix an agent's bug by editing, against a fair fine-tuning baseline (new)

The post's Qwen result fixed a looping bug. We built the agent version: a small deterministic text website with six task families (look up a price, log in, toggle a setting, search, order an item) and ran Qwen2.5-1.5B-Instruct as the agent. It solves 37 of 48 held-out tasks and fails every ordering task the same way: after adding the item to the cart it goes back and adds it again instead of checking out. Claude Opus got the weights, the environment, the model's training episodes and a sandbox, with the same no-gradients rules as before. This time the traditional baseline was the fair one: a LoRA fine-tune on 204 demonstrations.

| Method | Weights changed | Tasks solved (of 48) | Ordering (of 8) | Retention | Fluency drift | Time |
|---|---|---|---|---|---|---|
| Base model | | 37 | 0 | 38/38 | | |
| **Claude Opus, weight edit** | 572,941 (0.037%) | 46 | 7 | 38/38 | +1.0 | 110 min, ~$12 |
| LoRA fine-tune | 18M adapter (1.2%) | **48** | **8** | 38/38 | **0.0** | 8 min, $0 |

The edit worked as a repair: one tensor, 64 rows of the last layer's MLP, and the failing families were fixed with retention intact. But the fine-tune beat it on every number, and under sampled decoding the gap widened (base 30, edit 38, LoRA 48 of 48). And how Opus got there matters: it did not find a "repeat the last action" circuit to cut. It wrote an expert policy for the site, labelled 1,200 decision states with it, and solved a least-squares problem to write the answer into the last layer. That is supervised learning done with linear algebra instead of gradients, and it needed the same kind of demonstrations the LoRA used. We stopped the run before Opus finished tuning, so the checkpoint is an intermediate one; details in RESULTS.md.

## What it means

**Reading works for structure, not for knowledge or skill.** Which circuits exist, which heads look where, what a tiny network with spatial inputs was built for: readable. Whether a neuron fires in practice, which language a model speaks, whether it loops, how good it is at reasoning: measured by running, not read. (We checked the dormant-neuron case directly: the neurons Opus repurposed were among the 15 least-used of 3,072 by activation, but from weights alone only the bias hints at this, and that guess is right 18 of 50 times.)

**Editing can remove, patch, compile and insert. It cannot add a skill.** The post's Qwen math result removed an obstacle (looping) that hid an ability the model already had; the same edit on a model without the obstacle did nothing. Our fact insertion works because a fact is a single association. "Good at tool use" is not a single anything.

**The sweet spot is behaviors that are hard to specify with data but turn out to be concentrated in the weights.** Fine-tuning needs examples and nudges every weight a little. Weight editing needs a diagnosis and touches 0.01% of the weights. It only works when the diagnosis finds one place to cut, and the reading step is what tells you whether that is the case.

**Why the builders don't already know this.** The Qwen team chose the architecture, the data and the recipe. They did not choose any of the 494 million numbers; training did. Reading what training produced is the young science of interpretability, and the post is an attempt to hand that reading to machines because they are the only readers fast enough.

**Cost and speed cut the other way.** The weight edit took 26 to 44 minutes and $8 to $22 per fact, and 110 minutes and $12 for the agent repair; fine-tuning took five seconds for the fact and eight minutes for the agent, for nothing. For plain facts, putting them in the context window beats both. And when you can write a couple of hundred demonstrations, a LoRA fine-tune beat the edit on accuracy and on side effects. Editing earns its place where the context cannot reach and demonstrations cannot be written: behaviors baked into weights that you want removed rather than taught, and settings where there are weights but no training stack.

## Repo layout

```
README.md            this article
RESULTS.md           full numbers, protocols, per-run details, takeaways per experiment
scripts/             scorer, parsers, ground-truth tools, sandbox builder, graders, patch tools,
                     the MiniWeb agent environment, harness, retention check and LoRA baseline
runs/                every Claude reply, command log and score (Breakout, which-game, audits, fact edit, agent repair)
patches/             the two GPT-2 edits and the Qwen agent edit as small patch files (scripts/gpt2_patch.py, scripts/qwen_patch.py)
sandbox/*/prompt.txt the exact prompts given to the editor/auditor; sandbox/*/arch.json
truth/               measured head behavior of the three audited models (never shown to the auditor)
```

Not in the repo (size): model checkpoints, the authors' logs (sparse-clone them, below), the Python venv.

## Reproduce

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
mkdir -p authors && git clone --filter=blob:none --sparse https://github.com/RikiyaT/lm_understand-logs authors/lm_understand-logs
cd authors/lm_understand-logs && git sparse-checkout set exp1_which_game exp3_language exp5_write_from_scratch harness && cd ../..
# score the authors' Breakout network (expect ~506 +/- 8)
python -I scripts/score.py authors/lm_understand-logs/exp5_write_from_scratch/networks/breakout/hand.npz
# parse and score a Claude reply
python -I scripts/edits.py runs/fable_r1/answer.md /tmp/net.npz && python -I scripts/score.py /tmp/net.npz
# rebuild the edited GPT-2 from a patch and evaluate it
python -I scripts/gpt2_patch.py apply patches/gpt2_argentina_opus.npz /tmp/gpt2_edited
python -I scripts/eval_edit.py /tmp/gpt2_edited /tmp/eval.json --baseline runs/teach/baseline.json
# agent repair: baseline, rebuild the edited Qwen from its patch, evaluate, and train the LoRA baseline
python scripts/agent_run.py models/qwen2.5-1.5b-instruct /tmp/base.json --split test --n 8
python scripts/qwen_patch.py apply models/qwen2.5-1.5b-instruct patches/qwen2.5-1.5b-instruct_agent_opus.npz /tmp/qwen_edited
python scripts/agent_run.py /tmp/qwen_edited /tmp/edit.json --split test --n 8 && python scripts/agent_retention.py /tmp/qwen_edited /tmp/ret.json
python scripts/agent_finetune.py models/qwen2.5-1.5b-instruct /tmp/ft_lora --n-seeds 34 --epochs 2
```

Claude runs used the Claude Code CLI in print mode with tools disabled (`claude -p --tools ""`) or restricted to a sandboxed Python (`--allowedTools "Bash(./py *)"`, macOS `sandbox-exec`, no network, folder-only reads). See `scripts/run_exp1.sh`, `scripts/run_audit.sh`, `scripts/make_sandbox_py.sh`. Effort "max" blew the CLI's 64K output cap; use effort "high" with `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`.

## Credits and honesty notes

- Original work: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL). Their logs repo made this possible.
- The skeptical Hebrew post that started this asked two questions: is Breakout a hand controller, and did the Qwen result need trial and error. Yes and yes, per the authors' own data.
- The experiments were designed and run by Shimon Moyal with Claude Code (Claude Fable 5.1) as the operator; the test subjects were Claude Fable 5.1, Opus 5.5, Sonnet 5.5 and Haiku 5.5. The operator and the subjects are from the same family. Transcripts are included so you can check the subjects were not helped.
- One run per condition throughout. Treat every number as a point estimate.
- The authors' prompts contain MinAtar source under GPL-3.0; it is not vendored here. Our code: MIT.
