"""Score a 600-32-32-6 tanh policy on MinAtar (authors' protocol): mean score over N episodes,
actions sampled from softmax(logits), sticky actions 0.1 (MinAtar default), episode cut at 10,000 steps.
Observation: 6 channels (player, danger, reward, projectile, trail, gauge) built from the game's channels.
Usage: score.py NET.npz [--game breakout] [--episodes 1000] [--seed 0] [--workers 8]
"""
import argparse, json, os, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edits import load_npz

# game channel name -> our 6-slot index
SLOT = {"player": 0, "danger": 1, "reward": 2, "projectile": 3, "trail": 4, "gauge": 5}
CHANNEL_MAP = {
    "breakout": {"paddle": "player", "brick": "reward", "ball": "projectile", "trail": "trail"},
    "seaquest": {"sub_front": "player", "sub_back": "player", "friendly_bullet": "projectile", "trail": "trail",
                 "enemy_bullet": "danger", "enemy_fish": "danger", "enemy_sub": "danger", "oxygen_guage": "gauge",
                 "diver_guage": "gauge", "diver": "reward"},
    "asterix": {"player": "player", "enemy": "danger", "trail": "trail", "gold": "reward"},
}
MAX_STEPS = 10_000


def make_obs_fn(env, game):
    ch = env.env.channels
    cmap = CHANNEL_MAP[game]
    src_idx = [[] for _ in range(6)]
    for name, slot in cmap.items():
        if name in ch:
            src_idx[SLOT[slot]].append(ch[name])
    def obs():
        s = env.state()  # 10 x 10 x n_channels bool
        x = np.zeros((10, 10, 6), dtype=np.float32)
        for k, idxs in enumerate(src_idx):
            if idxs:
                x[:, :, k] = np.any(s[:, :, idxs], axis=2)
        return x.reshape(-1)  # index (row*10+col)*6+channel
    return obs


def policy_logits(T, x):
    h1 = np.tanh(x @ T["W1"] + T["b1"])
    h2 = np.tanh(h1 @ T["W2"] + T["b2"])
    return h2 @ T["W3"] + T["b3"]


def run_episodes(args):
    net_path, game, n, seed = args
    from minatar import Environment
    T = load_npz(net_path)
    rng = np.random.RandomState(seed)
    env = Environment(game, sticky_action_prob=0.1)
    env.random = np.random.RandomState(seed + 1)
    env.env.random = np.random.RandomState(seed + 2)
    obs = make_obs_fn(env, game)
    scores, lengths = [], []
    for _ in range(n):
        env.reset()
        total, steps, done = 0.0, 0, False
        while not done and steps < MAX_STEPS:
            lg = policy_logits(T, obs())
            lg = lg - lg.max()
            p = np.exp(lg); p /= p.sum()
            a = int(rng.choice(6, p=p))
            r, done = env.act(a)
            total += r; steps += 1
        scores.append(total); lengths.append(steps)
    return scores, lengths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("net"); ap.add_argument("--game", default="breakout")
    ap.add_argument("--episodes", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count()); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    t0 = time.time()
    per = [a.episodes // a.workers + (1 if i < a.episodes % a.workers else 0) for i in range(a.workers)]
    jobs = [(a.net, a.game, n, a.seed * 1000 + i) for i, n in enumerate(per) if n > 0]
    with Pool(len(jobs)) as pool:
        res = pool.map(run_episodes, jobs)
    scores = np.concatenate([r[0] for r in res]); lengths = np.concatenate([r[1] for r in res])
    out = {"net": a.net, "game": a.game, "episodes": int(len(scores)), "mean": float(scores.mean()),
           "se": float(scores.std(ddof=1) / np.sqrt(len(scores))), "max": float(scores.max()), "min": float(scores.min()),
           "mean_length": float(lengths.mean()), "frac_full_length": float((lengths >= MAX_STEPS).mean()),
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
