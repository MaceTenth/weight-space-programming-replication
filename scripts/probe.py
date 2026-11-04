"""Variants of the scoring protocol to see what limits a network: argmax vs sampling, sticky vs none."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edits import load_npz
from score import make_obs_fn, policy_logits, MAX_STEPS

def run(args):
    net, n, seed, sticky, greedy = args
    from minatar import Environment
    T = load_npz(net); rng = np.random.RandomState(seed)
    env = Environment('breakout', sticky_action_prob=sticky); env.random = np.random.RandomState(seed+1); env.env.random = np.random.RandomState(seed+2)
    obs = make_obs_fn(env, 'breakout'); sc, ln = [], []
    for _ in range(n):
        env.reset(); tot=0; st=0; done=False
        while not done and st < MAX_STEPS:
            lg = policy_logits(T, obs())
            if greedy: a = int(np.argmax(lg))
            else:
                lg = lg-lg.max(); p=np.exp(lg); p/=p.sum(); a=int(rng.choice(6,p=p))
            r, done = env.act(a); tot+=r; st+=1
        sc.append(tot); ln.append(st)
    return sc, ln

if __name__ == '__main__':
    net = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv)>2 else 300
    for sticky, greedy in [(0.1, False), (0.0, False), (0.1, True), (0.0, True)]:
        jobs = [(net, n//10, 100*i, sticky, greedy) for i in range(10)]
        with Pool(10) as p: res = p.map(run, jobs)
        s = np.concatenate([r[0] for r in res]); l = np.concatenate([r[1] for r in res])
        print(f"sticky={sticky} greedy={greedy}: mean={s.mean():.1f} se={s.std(ddof=1)/np.sqrt(len(s)):.1f} survived_10k={(l>=MAX_STEPS).mean():.2f} min={s.min():.0f}")
