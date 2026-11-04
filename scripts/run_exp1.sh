#!/bin/zsh
# Usage: run_exp1.sh <model-alias> <dirname>   runs the four exp1 prompts sequentially, no tools, fresh context each
model=$1; name=$2
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
base=/Users/shimonmoyal/weights
for game in breakout pong pacman freeway; do
  out=$base/runs/exp1/$name/$game; mkdir -p $out
  claude -p --no-session-persistence --tools "" --model $model --effort high \
    --system-prompt "Answer the user's request." --output-format json \
    < $base/authors/lm_understand-logs/exp1_which_game/runs/$game/prompt.txt > $out/result.json 2> $out/stderr.log
  python3 -I -c "
import json,sys
d=json.load(open('$out/result.json')); open('$out/answer.md','w').write(d.get('result',''))
u=d.get('usage',{}); print('$name $game', 'err' if d.get('is_error') else 'ok', 'cost', round(d.get('total_cost_usd',0),2), 'in', u.get('input_tokens'), 'out', u.get('output_tokens'), 'sec', round(d.get('duration_ms',0)/1000))
"
done
