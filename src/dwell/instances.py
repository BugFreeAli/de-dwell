"""
Instance generator for the P0 regret grid.

Common random numbers: the base draws (values, labels, deadline fractions, last-deadline threat) depend
only on (master_seed, instance index), never on the cell parameters (rho, s, a). Cells therefore differ
only through the parameters, which sharpens comparisons across cells.

Information model: every threat carries a class label (0 = soft, 1 = hard) with accuracy a, i.e.
P(true class = label) = a. a = 0.5 is an uninformative label, a = 1 reveals the class. The prior of
threat i is (1-a, a) or (a, 1-a) depending on its label (K = 2 only in the main grid).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from .m0 import Instance


def info_bits(a: float) -> float:
    """Information regime variable I(a) = 1 - H2(a), bits per threat (0 at a=0.5, 1 at a=1)."""
    if a in (0.0, 1.0):
        return 1.0
    return 1.0 + a * math.log2(a) + (1 - a) * math.log2(1 - a)


def make_instance(master_seed: int, idx: int, n: int, rho: float, s: int, a: float,
                  D: Tuple[int, int] = (2, 5), value_set: Tuple[int, ...] = (1, 2, 3),
                  u_low: float = 0.35) -> Tuple[Instance, Dict[str, float]]:
    rng = np.random.default_rng([master_seed, idx])
    values = tuple(float(v) for v in rng.choice(value_set, size=n))
    labels = rng.integers(0, 2, size=n)                 # 1 = labelled hard
    u = rng.uniform(u_low, 1.0, size=n)                 # deadline fractions of d_max
    last = int(rng.integers(n))                         # threat that defines d_max
    p_hard = [a if l == 1 else 1.0 - a for l in labels]
    prior = tuple((1.0 - ph, ph) for ph in p_hard)
    exp_dwell = [pr[0] * D[0] + pr[1] * D[1] for pr in prior]
    d_max = max(D[0], int(round(sum(exp_dwell) / rho)))
    deadlines = [max(D[0], int(round(ui * d_max))) for ui in u]
    deadlines[last] = d_max
    inst = Instance(D=tuple(D), s=s, values=values, deadlines=tuple(deadlines), prior=prior)
    # demand-bound load: max over deadlines t of (expected dwell due by t) / t. Unlike rho_hat it also sees
    # congestion caused by early deadlines. Stored for the later regime predictor; the grid uses rho_hat.
    rho_dbf = max(sum(e for e, dl in zip(exp_dwell, deadlines) if dl <= t) / t for t in set(deadlines))
    realised = {
        "rho_dbf": rho_dbf,
        "rho_hat": sum(exp_dwell) / d_max,               # expected total dwell / last deadline
        "sigma_hat": s / (sum(exp_dwell) / n),           # slew over mean expected dwell
        "info_bits": info_bits(a),
        "d_max": d_max,
    }
    return inst, realised
