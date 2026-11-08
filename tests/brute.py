"""
Independent brute-force solver used ONLY for testing the DP.

It shares no code with dwell.m0 / dwell.dp: no belief tables, no canonicalisation, no 'useful action'
filter. It enumerates the hidden class vectors ("worlds"), keeps the set of worlds consistent with the
observations, and does exhaustive expectimax over ALL actions: dwell on any not-yet-dead threat whose
step would end by its deadline (even if it can no longer be killed), and an idle step.
"""
import itertools
from functools import lru_cache


def brute_value(D, s, values, deadlines, prior):
    n, K = len(values), len(D)
    worlds = []
    for c in itertools.product(range(K), repeat=n):
        w = 1.0
        for i, k in enumerate(c):
            w *= prior[i][k]
        if w > 0:
            worlds.append((c, w))
    dmax = max(deadlines)

    @lru_cache(maxsize=None)
    def rec(t, cur, dwell, dead):
        # Consistent worlds are a function of (dwell, dead): dead threats died exactly at dwell == D[c],
        # alive threats have D[c] > dwell.
        W = [(c, w) for c, w in worlds
             if all((D[c[i]] == dwell[i]) if dead[i] else (D[c[i]] > dwell[i]) for i in range(n))]
        wsum = sum(w for _, w in W)
        best = 0.0
        if t + 1 <= dmax:                                   # idle for one step
            best = max(best, rec(t + 1, cur, dwell, dead))
        for j in range(n):
            if dead[j]:
                continue
            sl = s if (cur != -1 and cur != j) else 0
            t2 = t + sl + 1
            if t2 > deadlines[j]:
                continue
            nd = dwell[:j] + (dwell[j] + 1,) + dwell[j + 1:]
            kill = sum(w for c, w in W if D[c[j]] == dwell[j] + 1) / wsum
            val = 0.0
            if kill > 0:
                dead2 = dead[:j] + (True,) + dead[j + 1:]
                val += kill * (values[j] + rec(t2, j, nd, dead2))
            if kill < 1:
                val += (1 - kill) * rec(t2, j, nd, dead)
            best = max(best, val)
        return best

    return rec(0, -1, (0,) * n, (False,) * n)
