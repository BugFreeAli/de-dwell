import itertools
from functools import lru_cache


def brute_value_ba(D0, lams, lam_prior, s, values, deadlines, prior, Dl=None):
    n = len(values)
    if Dl is None:
        Dl = [tuple(int(round(l * d)) for d in D0) for l in lams]
    K = len(Dl[0])
    worlds = []
    for li, lam in enumerate(lams):
        for c in itertools.product(range(K), repeat=n):
            w = lam_prior[li]
            for i, k in enumerate(c):
                w *= prior[i][k]
            if w > 0:
                worlds.append((li, c, w))
    dmax = max(deadlines)

    @lru_cache(maxsize=None)
    def rec(t, cur, dwell, kd):
        W = []
        for li, c, w in worlds:
            ok = True
            for i in range(n):
                Di = Dl[li][c[i]]
                if kd[i] > 0:
                    if Di != kd[i]:
                        ok = False
                        break
                elif Di <= dwell[i]:
                    ok = False
                    break
            if ok:
                W.append((li, c, w))
        wsum = sum(w for _, _, w in W)
        best = 0.0
        if t + 1 <= dmax:
            best = max(best, rec(t + 1, cur, dwell, kd))
        for j in range(n):
            if kd[j] > 0:
                continue
            sl = s if (cur != -1 and cur != j) else 0
            t2 = t + sl + 1
            if t2 > deadlines[j]:
                continue
            x1 = dwell[j] + 1
            kill = sum(w for li, c, w in W if Dl[li][c[j]] == x1) / wsum
            val = 0.0
            dw2 = dwell[:j] + (x1,) + dwell[j + 1:]
            if kill > 0:
                kd2 = kd[:j] + (x1,) + kd[j + 1:]
                val += kill * (values[j] + rec(t2, j, dw2, kd2))
            if kill < 1:
                val += (1 - kill) * rec(t2, j, dw2, kd)
            best = max(best, val)
        return best

    return rec(0, -1, (0,) * n, (0,) * n)
