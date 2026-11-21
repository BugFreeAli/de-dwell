"""EXPLORATORY (not repo code, throwaway seed 3): objective mismatch.
EV-optimal policies (DP, rules, rollouts) evaluated on P(total killed value >= T); reference = exact DP for that objective."""
import sys, itertools, numpy as np
sys.path.insert(0, "src"); sys.setrecursionlimit(50000)
from dwell.m0 import Model, EPS
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.rules import make_rule

def util_dp(model, T):
    memo = {}
    def V(state, acc):
        key = (state, acc)
        if key in memo: return memo[key][0]
        best, bj = (1.0 if acc >= T else 0.0), None
        for j in model.actions(state):
            val = sum(p * V(nxt, acc + r) for p, r, nxt in model.outcomes(state, j))
            if val > best + EPS: best, bj = val, j
        memo[key] = (best, bj); return best
    v = V(model.initial_state(), 0.0)
    return v, memo

def prob_success(model, policy, T):
    cache = {}; tot = 0.0
    for c, w in model.worlds():
        state = model.initial_state(); acc = 0.0
        while True:
            if state in cache: j = cache[state]
            else: j = policy(state); cache[state] = j
            if j is None: break
            t, cur, prog = state
            t2 = t + model.slew(cur, j) + 1; p2 = list(prog)
            if prog[j] + 1 == model.D[c[j]]: acc += model.v[j]; p2[j] = -1
            else: p2[j] = prog[j] + 1
            state = (t2, j, model.canon(t2, p2))
        tot += w * (1.0 if acc >= T else 0.0)
    return tot

def run(seed, n, rho, s, a, N, frac):
    rows = []
    for i in range(N):
        inst, _ = make_instance_k(seed, i, n, rho, s, a, D=(5, 10), value_set=(1, 2, 3), u_low=0.35)
        m = Model(inst); T = frac * sum(inst.values)
        uopt, _ = util_dp(m, T)
        ev = solve(m)
        row = dict(U_opt=uopt, U_EVDP=prob_success(m, ev.policy(), T))
        for r in ("GMR", "EDF", "CEP", "ROLL_CEP", "ROLL_GMR"):
            row["U_" + r] = prob_success(m, make_rule(m, r), T)
        rows.append(row)
    return rows

if __name__ == "__main__":
    import pandas as pd
    out = []
    for frac in (1.0, 0.75):
        for rho in (0.8, 1.1):
            for s in (2, 4):
                for a in (0.5, 0.9):
                    r = pd.DataFrame(run(3, 4, rho, s, a, 40, frac))
                    d = dict(frac=frac, rho=rho, s=s, a=a, U_opt=r.U_opt.mean())
                    for c in r.columns:
                        if c != "U_opt": d["reg_" + c[2:]] = 100 * (1 - r[c].sum() / r.U_opt.sum()) if r.U_opt.sum() > 0 else np.nan
                    out.append(d); print(d, flush=True)
    df = pd.DataFrame(out); df.to_csv("util_explore.csv", index=False)
    print(df.groupby("frac")[[c for c in df.columns if c.startswith("reg_")]].mean().round(2))
    print(df.groupby("frac")[[c for c in df.columns if c.startswith("reg_")]].max().round(2))
