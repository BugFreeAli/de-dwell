import itertools
import math
from functools import lru_cache


def brute_value2(D, s, values, deadlines, prior, kappa):
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
    def rec(t, cur, dwell, ev, dead):
        W = [(c, w) for c, w in worlds
             if all((D[c[i]] == dead[i]) if dead[i] > 0 else (D[c[i]] > ev[i]) for i in range(n))]
        wsum = sum(w for _, w in W)
        best = 0.0
        if t + 1 <= dmax:
            best = max(best, rec(t + 1, cur, dwell, ev, dead))
        for j in range(n):
            if dead[j] > 0:
                continue
            sl = s if (cur != -1 and cur != j) else 0
            t2 = t + sl + 1
            if t2 > deadlines[j]:
                continue
            dw = list(dwell)
            if kappa < 1.0 and cur != -1 and cur != j and dead[cur] == 0:
                dw[cur] = int(math.floor(kappa * dw[cur] + 1e-9))
            x1 = dw[j] + 1
            kill = sum(w for c, w in W if D[c[j]] == x1) / wsum
            val = 0.0
            if kill > 0:
                nd = dead[:j] + (x1,) + dead[j + 1:]
                dw2 = dw[:]
                dw2[j] = x1
                val += kill * (values[j] + rec(t2, j, tuple(dw2), ev, nd))
            if kill < 1:
                dw2 = dw[:]
                dw2[j] = x1
                ev2 = ev[:j] + (max(ev[j], x1),) + ev[j + 1:]
                val += (1 - kill) * rec(t2, j, tuple(dw2), ev2, dead)
            best = max(best, val)
        return best

    return rec(0, -1, (0,) * n, (0,) * n, (0,) * n)
