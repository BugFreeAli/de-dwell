"""
M1: M0 plus a deterministic approach, a range-dependent dwell rate, an explicit idle action and an
optional energy budget (docs/analysis_plan_P0.md, amendment A11, Section A11.2).

What changes with respect to M0 (src/dwell/m0.py):
  * Threat j approaches at one common closing speed and breaches at t = d_j (range zero). With a common
    speed the range is a relabelling of the time to breach tau = d_j - t.
  * A dwell step that STARTS at time t0 on threat j (after the slew) adds g(tau) progress units with
    tau = d_j - t0 >= 1. g is given by the table `gtab` (gtab[tau], integer >= 1). Class k is killed when the
    progress reaches D[k] at the end of the step (overshoot allowed). gtab == 1 everywhere is M0.
  * IDLE (action id -2): t -> t+1, pointing direction unchanged, no progress, no energy.
  * Energy: if `budget` is not None, at most `budget` dwell steps may be used; the state carries e = steps used.
    Slew and idle cost no energy; there is no recharge. budget None means no budget (e is kept at 0).
State = (t, cur, prog, e). prog[i] is the progress of an alive threat or DEAD (killed, or unkillable even in the
best case: zero slew, unlimited energy; the merge is exact because the future value is identical).
"""
from __future__ import annotations

import functools
import itertools
import math
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .m0 import DEAD, EPS, NO_TARGET, Instance

IDLE = -2
State = Tuple[int, int, Tuple[int, ...], int]   # (t, cur, prog, e)


def two_level_gtab(gamma: int, tau_c: int, tmax: int) -> Tuple[int, ...]:
    """g(tau) = gamma for tau <= tau_c, else 1 (index 0 unused). gamma = 1 gives M0."""
    return (1,) + tuple(gamma if tau <= tau_c else 1 for tau in range(1, tmax + 1))


class Model1:
    def __init__(self, inst: Instance, gtab: Sequence[int], budget: Optional[int] = None,
                 allow_idle: bool = True):
        inst.validate()
        assert all(isinstance(g, int) and g >= 1 for g in gtab[1:]), "rates must be integers >= 1"
        assert budget is None or budget >= 0
        self.inst, self.budget, self.allow_idle = inst, budget, allow_idle
        self.n, self.D, self.s = inst.n, inst.D, inst.s
        self.v, self.d, self.prior = inst.values, inst.deadlines, inst.prior
        self.K = len(self.D)
        self.gtab = tuple(gtab)
        self.T = len(self.gtab) - 1
        self.cum = [0]                                   # cum[m] = g(1)+...+g(m): units obtainable in the last m steps
        for tau in range(1, self.T + 1):
            self.cum.append(self.cum[-1] + self.gtab[tau])
        xmax = self.D[-1]
        self.post: List[List[List[Tuple[float, int]]]] = []
        self.rmin: List[List[float]] = []
        for i in range(self.n):
            post_i, r_i = [], []
            for x in range(xmax + 1):
                mass = [(self.prior[i][k], self.D[k] - x) for k in range(self.K)
                        if self.prior[i][k] > EPS and self.D[k] > x]
                z = sum(p for p, _ in mass)
                if z <= EPS:
                    post_i.append([]); r_i.append(float("inf")); continue
                post_i.append([(p / z, r) for p, r in mass]); r_i.append(min(r for _, r in mass))
            self.post.append(post_i); self.rmin.append(r_i)
        self._top = {}                                   # (T, m) -> best units with m steps among tau = 1..T
        self._steps = {}

    # ---- rate helpers ----------------------------------------------------------------------
    def g_at(self, tau: int) -> int:
        """Rate at time-to-breach tau; clipped to the table (tau < 1 uses g(1))."""
        return self.gtab[min(max(tau, 1), self.T)]

    def best_units(self, T: int, m: int) -> int:
        """Largest number of units obtainable with at most m dwell steps among the T steps before breach."""
        T = min(T, self.T)
        key = (T, m)
        hit = self._top.get(key)
        if hit is None:
            rates = sorted(self.gtab[1:T + 1], reverse=True)
            hit = sum(rates[:max(m, 0)])
            self._top[key] = hit
        return hit

    # ---- state handling ----------------------------------------------------------------------
    def canon(self, t: int, prog: Sequence[int]) -> Tuple[int, ...]:
        out = []
        for i, x in enumerate(prog):
            if x == DEAD:
                out.append(DEAD); continue
            tau = self.d[i] - t
            p_max = self.cum[min(tau, self.T)] if tau > 0 else 0
            out.append(DEAD if p_max < self.rmin[i][x] else x)
        return tuple(out)

    def initial_state(self) -> State:
        return (0, NO_TARGET, self.canon(0, (0,) * self.n), 0)

    def slew(self, cur: int, j: int) -> int:
        return self.s if (cur != NO_TARGET and cur != j) else 0

    def energy_left(self, e: int) -> Optional[int]:
        return None if self.budget is None else self.budget - e

    def _can_kill(self, j: int, x: int, ts: int, e: int) -> bool:
        T = self.d[j] - ts
        if T < 1:
            return False
        left = self.energy_left(e)
        m = T if left is None else min(T, left)
        if m <= 0:
            return False
        return self.best_units(T, m) >= self.rmin[j][x]

    def dwell_actions(self, state: State) -> List[int]:
        t, cur, prog, e = state
        acts = [j for j, x in enumerate(prog)
                if x != DEAD and self._can_kill(j, x, t + self.slew(cur, j), e)]
        if cur in acts:
            acts.remove(cur); acts.insert(0, cur)
        return acts

    def idle_possible(self, state: State) -> bool:
        """Waiting is useful only if some alive threat could still be killed when starting at t+1 with zero slew."""
        if not self.allow_idle:
            return False
        t, cur, prog, e = state
        return any(x != DEAD and self._can_kill(j, x, t + 1, e) for j, x in enumerate(prog))

    def actions(self, state: State) -> List[int]:
        acts = self.dwell_actions(state)
        if self.idle_possible(state):
            acts.append(IDLE)                            # last: ties never choose idle
        return acts

    def rate_for(self, state: State, j: int) -> int:
        t, cur, prog, e = state
        return self.g_at(self.d[j] - (t + self.slew(cur, j)))

    def outcomes(self, state: State, a: int) -> List[Tuple[float, float, State]]:
        """Chance outcomes of action a: list of (probability, reward, next_state)."""
        t, cur, prog, e = state
        if a == IDLE:
            return [(1.0, 0.0, (t + 1, cur, self.canon(t + 1, prog), e))]
        j = a
        ts = t + self.slew(cur, j)
        g = self.g_at(self.d[j] - ts)
        x = prog[j]
        post = self.post[j][x]
        q = sum(p for p, r in post if r <= g)
        q = 1.0 if q > 1 - EPS else (0.0 if q < EPS else q)
        t2, e2 = ts + 1, (e + 1 if self.budget is not None else 0)
        out = []
        if q > 0.0:
            p2 = list(prog); p2[j] = DEAD
            out.append((q, self.v[j], (t2, j, self.canon(t2, p2), e2)))
        if q < 1.0:
            p2 = list(prog); p2[j] = x + g
            out.append((1.0 - q, 0.0, (t2, j, self.canon(t2, p2), e2)))
        return out

    def worlds(self) -> Iterator[Tuple[Tuple[int, ...], float]]:
        ranges = [[k for k in range(self.K) if self.prior[i][k] > EPS] for i in range(self.n)]
        for c in itertools.product(*ranges):
            w = 1.0
            for i, k in enumerate(c):
                w *= self.prior[i][k]
            yield c, w


# ---------------------------------------------------------------------------------------------------------
class DPResult1:
    """Exact expectimax over the canonical observable state, with idle (if allowed) and energy."""

    def __init__(self, model: Model1):
        self.model = model
        self._memo: Dict[State, Tuple[float, Optional[int]]] = {}
        self.value = self._solve(model.initial_state())[0]

    def _solve(self, state: State) -> Tuple[float, Optional[int]]:
        hit = self._memo.get(state)
        if hit is not None:
            return hit
        best, best_a = 0.0, None
        for a in self.model.actions(state):              # dwell actions (current first), idle last
            val = 0.0
            for p, r, nxt in self.model.outcomes(state, a):
                val += p * (r + self._solve(nxt)[0])
            if val > best + EPS:
                best, best_a = val, a
        self._memo[state] = (best, best_a)
        return best, best_a

    @property
    def n_states(self) -> int:
        return len(self._memo)

    def policy(self):
        return lambda state: self._solve(state)[1]


def solve1(model: Model1) -> DPResult1:
    return DPResult1(model)


def evaluate_policy1(model: Model1, policy, stats: bool = False):
    """
    Exact expected value of an observation-based policy by enumeration of the hidden class vectors.
    policy(state) -> dwell index | IDLE | None. With stats=True returns (value, P(waits at least once),
    P(preempts at least once)); preempting means leaving an alive threat that has progress before it is killed.
    """
    cache: dict = {}
    total = p_wait = p_pre = 0.0
    for c, w in model.worlds():
        state = model.initial_state()
        val, waited, pre = 0.0, False, False
        while True:
            a = cache[state] if state in cache else cache.setdefault(state, policy(state))
            if a is None:
                break
            t, cur, prog, e = state
            if a == IDLE:
                assert model.allow_idle, "policy idled although idle is disabled"
                state = (t + 1, cur, model.canon(t + 1, prog), e)
                waited = True
                continue
            j = a
            ts = t + model.slew(cur, j)
            assert model.d[j] - ts >= 1, "policy chose an action that cannot finish before breach"
            assert model.budget is None or e < model.budget, "policy exceeded the energy budget"
            if cur != NO_TARGET and cur != j and prog[cur] != DEAD and prog[cur] > 0:
                pre = True
            g = model.g_at(model.d[j] - ts)
            p2 = list(prog)
            if prog[j] + g >= model.D[c[j]]:
                val += model.v[j]; p2[j] = DEAD
            else:
                p2[j] = prog[j] + g
            state = (ts + 1, j, model.canon(ts + 1, p2), e + 1 if model.budget is not None else 0)
        total += w * val
        p_wait += w * waited
        p_pre += w * pre
    return (total, p_wait, p_pre) if stats else total


def oracle_value1(model: Model1) -> float:
    """Class-aware upper bound: prior-weighted DP value in every world with a one-hot prior (no information gap)."""
    total = 0.0
    for c, w in model.worlds():
        prior = tuple(tuple(1.0 if k == c[i] else 0.0 for k in range(model.K)) for i in range(model.n))
        inst = Instance(D=model.D, s=model.s, values=model.v, deadlines=model.d, prior=prior)
        total += w * solve1(Model1(inst, model.gtab, model.budget, model.allow_idle)).value
    return total


def energy_budget(inst: Instance, eps: Optional[float]) -> Optional[int]:
    """B = ceil(eps * sum_i E_i), E_i = prior mean of D (rate-1 steps); eps None means no budget."""
    if eps is None:
        return None
    e_tot = sum(sum(p * d for p, d in zip(pr, inst.D)) for pr in inst.prior)
    return int(math.ceil(eps * e_tot - 1e-9))


def rho_boost(inst: Instance, gtab: Sequence[int]) -> float:
    """Regime variable: expected dwell steps if every dwell is placed as late as possible (boosted), over d_max."""
    cum = [0]
    for tau in range(1, len(gtab)):
        cum.append(cum[-1] + gtab[tau])
    def steps(r):                                          # smallest m with cum[m] >= r
        m = 0
        while m < len(cum) - 1 and cum[m] < r:
            m += 1
        return m if cum[m] >= r else m + math.ceil((r - cum[m]) / gtab[1])
    tot = sum(sum(p * steps(d) for p, d in zip(pr, inst.D)) for pr in inst.prior)
    return tot / max(inst.deadlines)
