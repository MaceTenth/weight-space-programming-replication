"""Grade exp1 answers: map each reply's 'Game:' line to a game by keywords (authors' rule: Frogger counts as Freeway)."""
import json, re, os, sys, glob
KEY = json.load(open('/Users/shimonmoyal/weights/authors/lm_understand-logs/exp1_which_game/eval/answer_key.json'))
RUN2ID = {'breakout': 'a9f535cc', 'pong': 'dfc6cb65', 'pacman': 'bdf99279', 'freeway': '0bbcc3ea'}
KW = [('breakout', ['breakout', 'brick', 'arkanoid']), ('pong', ['pong']), ('pacman', ['pac-man', 'pacman', 'pac man']),
      ('freeway', ['freeway', 'frogger', 'road-cross', 'road cross', 'crossy', 'traffic'])]

def game_line(text):
    m = re.search(r"^\**\s*Game\s*:?\**\s*:?\s*(.+)$", text, re.M | re.I)
    return m.group(1).strip() if m else ''

def conf(text):
    m = re.search(r"Confidence\s*:?\**\s*:?\s*(\d+)", text, re.I)
    return int(m.group(1)) if m else None

def guess(line):
    # first game named in the line wins (authors map the Game line by keyword); extra names are a hedge
    l = line.lower()
    pos = [(min(l.find(k) for k in kws if k in l), g) for g, kws in KW if any(k in l for k in kws)]
    if not pos: return 'other', False
    pos.sort(); return pos[0][1], len(pos) > 1

rows = []
for d in sorted(glob.glob('/Users/shimonmoyal/weights/runs/exp1/*/*/answer.md')):
    model, game = d.split('/')[-3], d.split('/')[-2]
    t = open(d).read()
    if not t.strip() or t.startswith('API Error'):
        rows.append((model, game, 'NO ANSWER', None, False)); continue
    gl = game_line(t); g, hedge = guess(gl)
    rows.append((model, game, gl[:70], conf(t), g == KEY[RUN2ID[game]]['game'], hedge))
for r in rows: print(f"{r[0]:7s} {r[1]:9s} {'OK ' if r[4] else 'X  '}{'(hedged) ' if r[5] else '         '}conf={r[3]}  -> {r[2]}")
by = {}
for m, g, _, _, ok, _ in rows: by.setdefault(m, []).append(ok)
print(); [print(f"{m}: {sum(v)}/{len(v)}") for m, v in by.items()]
