"""
Class-aware oracle (information-relaxation upper bound, NOT an implementable policy).

For each hidden class vector (world) the problem is deterministic: threat i needs p_i = D[c_i] steps.
With retained progress and non-negative slew, preemption never helps, so an optimal schedule serves a
subset of threats contiguously. For a fixed subset, earliest-deadline-first order is optimal (each
position adds a fixed setup, so completion times depend only on the position). The oracle enumerates
all subsets and returns the best value; the oracle value is the prior-weighted mean over worlds.
"""
from __future__ import annotations

from .m0 import Model


def best_deterministic_value(model: Model, classes) -> float:
    n, s = model.n, model.s
    p = [model.D[c] for c in classes]
    order = sorted(range(n), key=lambda i: model.d[i])      # global EDF order; subsets inherit it
    best = 0.0
    for mask in range(1 << n):
        t, val, ok, first = 0, 0.0, True, True
        for i in order:
            if mask >> i & 1:
                t += p[i] + (0 if first else s)              # first engagement has no slew
                first = False
                if t > model.d[i]:
                    ok = False
                    break
                val += model.v[i]
        if ok and val > best:
            best = val
    return best


def oracle_value(model: Model) -> float:
    return sum(w * best_deterministic_value(model, c) for c, w in model.worlds())
