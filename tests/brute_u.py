"""Independent brute force for the A14 goal objective (testing only): no belief tables, no canonicalisation, no pruning.
Maximises P(killed value >= need) by expectimax over (t, current target, dwell per threat, dead flags, accumulated value)."""
import itertools
from functools import lru_cache


def brute_goal(D, sm, values, deadlines, prior, need):
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
    def rec(t, cur, dwell, dead, acc):
        if acc >= need:
            return 1.0
        W = [(c, w) for c, w in worlds
             if all((D[c[i]] == dwell[i]) if dead[i] else (D[c[i]] > dwell[i]) for i in range(n))]
        wsum = sum(w for _, w in W)
        best = 0.0
        if t + 1 <= dmax:
            best = max(best, rec(t + 1, cur, dwell, dead, acc))                       # idle
        for j in range(n):
            if dead[j]:
                continue
            sl = sm[cur][j] if (cur != -1 and cur != j) else 0
            t2 = t + sl + 1
            if t2 > deadlines[j]:
                continue
            nd = dwell[:j] + (dwell[j] + 1,) + dwell[j + 1:]
            kill = sum(w for c, w in W if D[c[j]] == dwell[j] + 1) / wsum
            val = 0.0
            if kill > 0:
                val += kill * rec(t2, j, nd, dead[:j] + (True,) + dead[j + 1:], acc + values[j])
            if kill < 1:
                val += (1 - kill) * rec(t2, j, nd, dead, acc)
            best = max(best, val)
        return best

    return rec(0, -1, (0,) * n, (False,) * n, 0.0)
