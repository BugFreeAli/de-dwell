"""
M0: stylized dwell-based engagement model (pilot P0).

Model summary (all quantities are integers on a time grid of step 1):
  * One effector, n threats. Threat i has value v_i, a breach deadline d_i and a HIDDEN class
    c_i in {0..K-1}. Class k needs D[k] time steps of accumulated dwell to kill (D ascending).
  * Prior: threat i has class distribution prior[i][k] (known to the policy, classes independent).
  * Each decision is "dwell on threat j for one more step". If j differs from the current
    target `cur` (and cur is not NO_TARGET), a slew of `s` steps is paid first. The first
    engagement of the episode is free of slew. No information arrives during a slew, so
    "slew, then dwell one step" is one atomic action of length s+1 (exact, not an approximation).
  * Progress is retained when the effector leaves a threat (no cooling or decay in M0).
  * A kill completing at the end of the step that ends at time t' requires t' <= d_j.
  * Observation: the policy sees, per threat, whether it is dead and how many dwell steps it
    has received. Kills are deterministic given the class, so the belief of an alive threat with
    progress x is the prior restricted to classes with D[k] > x, renormalised.
  * Objective: maximise the expected total value of threats killed before breach.

State = (t, cur, prog). prog[i] is the dwell received by an alive threat, or DEAD (-1) when
threat i is killed OR can no longer be killed by its deadline under any possible class
(canonicalisation; the future value is identical, so merging such states is exact).
Actions are restricted to "useful" threats (a kill before the deadline is still possible).
Spending time on a threat that cannot be killed any more cannot raise the value (the value is
non-increasing in t and that threat is never engaged again), so the restriction is exact;
tests/test_dp_bruteforce.py checks this against an unrestricted brute force that also allows idling.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

DEAD = -1          # sentinel in prog: killed or unkillable
NO_TARGET = -1     # sentinel in cur: effector not yet pointed at any threat
EPS = 1e-12        # tolerance for probability and tie comparisons

State = Tuple[int, int, Tuple[int, ...]]   # (t, cur, prog)


@dataclass(frozen=True)
class Instance:
    """One problem instance. All fields are tuples, so the instance is hashable and picklable."""
    D: Tuple[int, ...]                       # required dwell per class, strictly ascending
    s: int                                   # slew steps between different targets
    values: Tuple[float, ...]                # v_i
    deadlines: Tuple[int, ...]               # d_i (kill must complete at time <= d_i)
    prior: Tuple[Tuple[float, ...], ...]     # prior[i][k] = P(class k) for threat i

    @property
    def n(self) -> int:
        return len(self.values)

    def validate(self) -> None:
        """Fail loudly on malformed input."""
        assert all(self.D[k] < self.D[k + 1] for k in range(len(self.D) - 1)), "D must be strictly ascending"
        assert all(d >= 1 for d in self.D), "dwell requirements must be >= 1"
        assert self.s >= 0, "slew must be >= 0"
        assert len(self.deadlines) == self.n == len(self.prior), "length mismatch"
        for row in self.prior:
            assert len(row) == len(self.D), "prior row length must equal number of classes"
            assert abs(sum(row) - 1.0) < 1e-9 and all(p >= 0 for p in row), "prior row must be a distribution"


class Model:
    """Pre-computed belief tables and the transition function for one Instance."""

    def __init__(self, inst: Instance):
        inst.validate()
        self.inst = inst
        self.n, self.D, self.s = inst.n, inst.D, inst.s
        self.v, self.d, self.prior = inst.values, inst.deadlines, inst.prior
        self.K = len(self.D)
        xmax = self.D[-1]
        # post[i][x]  : list of (p, r) = (posterior prob, remaining dwell) over the possible classes of
        #               an alive threat i with progress x. Empty list = state cannot occur.
        # qkill[i][x] : P(threat is killed by the next dwell step | alive, progress x)
        # rmin[i][x]  : smallest remaining dwell over the possible classes (inf if none)
        self.post: List[List[List[Tuple[float, int]]]] = []
        self.qkill: List[List[float]] = []
        self.rmin: List[List[float]] = []
        for i in range(self.n):
            post_i, q_i, r_i = [], [], []
            for x in range(xmax + 1):
                mass = [(self.prior[i][k], self.D[k] - x) for k in range(self.K)
                        if self.prior[i][k] > EPS and self.D[k] > x]
                z = sum(p for p, _ in mass)
                if z <= EPS:
                    post_i.append([]); q_i.append(0.0); r_i.append(float("inf"))
                    continue
                post = [(p / z, r) for p, r in mass]
                q = sum(p for p, r in post if r == 1)
                q = 1.0 if q > 1 - EPS else (0.0 if q < EPS else q)
                post_i.append(post); q_i.append(q); r_i.append(min(r for _, r in post))
            self.post.append(post_i); self.qkill.append(q_i); self.rmin.append(r_i)

    # ---- state handling -----------------------------------------------------------------
    def canon(self, t: int, prog: Sequence[int]) -> Tuple[int, ...]:
        """Mark alive threats that can no longer be killed by their deadline as DEAD."""
        return tuple(DEAD if (x != DEAD and t + self.rmin[i][x] > self.d[i]) else x
                     for i, x in enumerate(prog))

    def initial_state(self) -> State:
        return (0, NO_TARGET, self.canon(0, (0,) * self.n))

    def slew(self, cur: int, j: int) -> int:
        """Slew cost of engaging j when the effector currently points at cur."""
        return self.s if (cur != NO_TARGET and cur != j) else 0

    def actions(self, state: State) -> List[int]:
        """Useful actions. The current target is listed first so that ties keep the current target."""
        t, cur, prog = state
        acts = []
        for j, x in enumerate(prog):
            if x != DEAD and t + self.slew(cur, j) + self.rmin[j][x] <= self.d[j]:
                acts.append(j)
        if cur in acts:
            acts.remove(cur)
            acts.insert(0, cur)
        return acts

    def outcomes(self, state: State, j: int) -> List[Tuple[float, float, State]]:
        """Chance outcomes of dwelling one step on j: list of (probability, reward, next_state)."""
        t, cur, prog = state
        t2 = t + self.slew(cur, j) + 1
        x = prog[j]
        q = self.qkill[j][x]
        out = []
        if q > 0.0:                                   # kill
            p2 = list(prog); p2[j] = DEAD
            out.append((q, self.v[j], (t2, j, self.canon(t2, p2))))
        if q < 1.0:                                   # no kill: one more step of progress
            p2 = list(prog); p2[j] = x + 1
            out.append((1.0 - q, 0.0, (t2, j, self.canon(t2, p2))))
        return out

    # ---- worlds (hidden class vectors) --------------------------------------------------
    def worlds(self) -> Iterator[Tuple[Tuple[int, ...], float]]:
        """All class vectors with positive prior probability, with their probabilities."""
        ranges = [[k for k in range(self.K) if self.prior[i][k] > EPS] for i in range(self.n)]
        for c in itertools.product(*ranges):
            w = 1.0
            for i, k in enumerate(c):
                w *= self.prior[i][k]
            yield c, w


def evaluate_policy(model: Model, policy) -> float:
    """
    Exact expected value of a deterministic, observation-based policy.

    `policy(state) -> int | None` sees only the observable state (t, cur, prog), never the hidden
    classes. We enumerate every hidden class vector ("world"), simulate the policy in it and weight
    by the prior probability, so there is no Monte Carlo noise. Decisions are cached per observable
    state, which is valid because the policy is a deterministic function of that state.
    """
    cache: dict = {}
    total = 0.0
    for c, w in model.worlds():
        state = model.initial_state()
        val = 0.0
        while True:
            if state in cache:
                j = cache[state]
            else:
                j = policy(state)
                cache[state] = j
            if j is None:
                break
            t, cur, prog = state
            t2 = t + model.slew(cur, j) + 1
            p2 = list(prog)
            if prog[j] + 1 == model.D[c[j]]:          # the true class of j is completed by this step
                val += model.v[j]
                p2[j] = DEAD
            else:
                p2[j] = prog[j] + 1
            state = (t2, j, model.canon(t2, p2))
        total += w * val
    return total
