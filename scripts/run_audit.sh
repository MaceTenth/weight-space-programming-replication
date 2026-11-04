#!/bin/zsh
# Usage: run_audit.sh <sandbox letter> <model alias> <run name>   e.g. run_audit.sh A opus opus_r1
d=/Users/shimonmoyal/weights/sandbox/$1; out=/Users/shimonmoyal/weights/runs/audit/$1_$3; mkdir -p $out
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
cd $d && claude -p --no-session-persistence --tools Bash --allowedTools "Bash(./py *)" --model $2 --effort high --max-turns 80 \
  --system-prompt "Answer the user's request." --output-format stream-json --verbose < prompt.txt > $out/stream.jsonl 2> $out/stderr.log
python3 -I - "$out" <<'PY'
import json, sys, os
out = sys.argv[1]; final = None; cmds = []; cost = None
for line in open(f"{out}/stream.jsonl"):
    try: e = json.loads(line)
    except Exception: continue
    if e.get('type') == 'assistant':
        for c in e.get('message', {}).get('content', []):
            if c.get('type') == 'tool_use': cmds.append(c.get('input', {}).get('command', ''))
    if e.get('type') == 'result':
        final = e.get('result'); cost = e.get('total_cost_usd'); turns = e.get('num_turns'); err = e.get('is_error')
open(f"{out}/answer.md", 'w').write(final or '')
json.dump(cmds, open(f"{out}/commands.json", 'w'), indent=1)
bad = [c for c in cmds if not c.strip().startswith('./py')]
print(os.path.basename(out), 'error' if err else 'ok', 'cost', round(cost or 0, 2), 'turns', turns, 'commands', len(cmds), 'non-py commands', len(bad), 'answer chars', len(final or ''))
PY
