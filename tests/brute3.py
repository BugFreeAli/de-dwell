import itertools
from functools import lru_cache


def brute_value3(D, s, values, deadlines, prior, q):
    n = len(values)
    worlds = []
    for c in itertools.product(range(2), repeat=n):
        w = 1.0
        for i, k in enumerate(c):
            w *= prior[i][k]
        if w > 0:
            worlds.append((c, w))
    dmax = max(deadlines)

    def lik(cls, h, steps):
        if cls == 1:
            return q ** h * (1 - q) ** (steps - h)
        return (1 - q) ** h * q ** (steps - h)

    @lru_cache(maxsize=None)
    def rec(t, cur, dwell, hc, dead):
        W = []
        for c, w in worlds:
            ok = True
            ww = w
            for i in range(n):
                if dead[i] > 0:
                    if D[c[i]] != dead[i]:
                        ok = False
                        break
                elif D[c[i]] <= dwell[i]:
                    ok = False
                    break
                ww *= lik(c[i], hc[i], dwell[i])
            if ok and ww > 0:
                W.append((c, ww))
        wsum = sum(w for _, w in W)
        best = 0.0
        if t + 1 <= dmax:
            best = max(best, rec(t + 1, cur, dwell, hc, dead))
        for j in range(n):
            if dead[j] > 0:
                continue
            sl = s if (cur != -1 and cur != j) else 0
            t2 = t + sl + 1
            if t2 > deadlines[j]:
                continue
            x1 = dwell[j] + 1
            kill = sum(w for c, w in W if D[c[j]] == x1) / wsum
            val = 0.0
            if kill > 0:
                nd = dead[:j] + (x1,) + dead[j + 1:]
                val += kill * (values[j] + rec(t2, j, dwell, hc, nd))
            for sig in (1, 0):
                pr = 0.0
                for c, w in W:
                    if D[c[j]] != x1:
                        pl = (q if sig == 1 else 1 - q) if c[j] == 1 else ((1 - q) if sig == 1 else q)
                        pr += w * pl
                pr /= wsum
                if pr > 0:
                    dw2 = dwell[:j] + (x1,) + dwell[j + 1:]
                    hc2 = hc[:j] + (hc[j] + sig,) + hc[j + 1:]
                    val += pr * rec(t2, j, dw2, hc2, dead)
            best = max(best, val)
        return best

    return rec(0, -1, (0,) * n, (0,) * n, (0,) * n)
