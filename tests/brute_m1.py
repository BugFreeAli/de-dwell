"""
Independent brute force for M1 (plan A11.7). It shares no code with src/dwell/m1.py except the rate table:
expectimax over information sets, each described by the set of classes still possible for every threat.
Unrestricted: every alive threat may be dwelled on (as long as the step ends by the deadline), idle is always allowed
until the last deadline, no canonicalisation, no useful-action filter.
"""
import functools


def brute_value(D, s, values, deadlines, prior, gtab, budget=None, allow_idle=True):
    n, K = len(values), len(D)
    T = len(gtab) - 1

    def g(tau):
        return gtab[min(max(tau, 1), T)]

    start_poss = tuple(frozenset(k for k in range(K) if prior[i][k] > 1e-12) for i in range(n))

    @functools.lru_cache(maxsize=None)
    def V(t, cur, prog, poss, dead, e):
        best = 0.0
        for j in range(n):
            if dead[j]:
                continue
            if budget is not None and e >= budget:
                continue
            ts = t + (s if (cur != -1 and cur != j) else 0)
            if ts + 1 > deadlines[j]:
                continue
            gg = g(deadlines[j] - ts)
            x = prog[j]
            tot = sum(prior[j][k] for k in poss[j])
            kill = [k for k in poss[j] if D[k] <= x + gg]
            stay = [k for k in poss[j] if D[k] > x + gg]
            val = 0.0
            e2 = e + 1
            if kill:
                pk = sum(prior[j][k] for k in kill) / tot
                d2 = dead[:j] + (True,) + dead[j + 1:]
                po2 = poss[:j] + (frozenset(kill),) + poss[j + 1:]
                val += pk * (values[j] + V(ts + 1, j, prog, po2, d2, e2))
            if stay:
                ps = sum(prior[j][k] for k in stay) / tot
                pr2 = prog[:j] + (x + gg,) + prog[j + 1:]
                po2 = poss[:j] + (frozenset(stay),) + poss[j + 1:]
                val += ps * V(ts + 1, j, pr2, po2, dead, e2)
            best = max(best, val)
        if allow_idle and t + 1 <= max(deadlines):
            best = max(best, V(t + 1, cur, prog, poss, dead, e))
        return best

    return V(0, -1, (0,) * n, start_poss, (False,) * n, 0)
