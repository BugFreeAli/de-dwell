import math
from typing import Tuple

from .m0 import DEAD, EPS, NO_TARGET, Instance, Model, State


class Model2(Model):
    def __init__(self, inst: Instance, kappa: float = 1.0):
        inst.validate()
        assert 0.0 <= kappa <= 1.0
        self.inst = inst
        self.kappa = kappa
        self.n, self.D, self.s = inst.n, inst.D, inst.s
        self.v, self.d, self.prior = inst.values, inst.deadlines, inst.prior
        self.K = len(self.D)
        K = self.K
        xmax = self.D[-1]
        size = xmax * K
        self.post, self.qkill, self.rmin = [], [], []
        for i in range(self.n):
            post_i = [[] for _ in range(size)]
            q_i = [0.0] * size
            r_i = [float("inf")] * size
            for x in range(xmax):
                for m in range(K):
                    mass = [(self.prior[i][k], self.D[k] - x) for k in range(m, K)
                            if self.prior[i][k] > EPS and self.D[k] > x]
                    z = sum(p for p, _ in mass)
                    if z <= EPS:
                        continue
                    post = [(p / z, r) for p, r in mass]
                    q = sum(p for p, r in post if r == 1)
                    q = 1.0 if q > 1 - EPS else (0.0 if q < EPS else q)
                    code = x * K + m
                    post_i[code] = post
                    q_i[code] = q
                    r_i[code] = min(r for _, r in post)
            self.post.append(post_i)
            self.qkill.append(q_i)
            self.rmin.append(r_i)

    def transition(self, state: State, j: int, killed: bool) -> Tuple[float, State]:
        t, cur, prog = state
        t2 = t + self.slew(cur, j) + 1
        p2 = list(prog)
        if self.kappa < 1.0 and cur != NO_TARGET and cur != j and p2[cur] != DEAD:
            xc, mc = divmod(p2[cur], self.K)
            p2[cur] = int(math.floor(self.kappa * xc + 1e-9)) * self.K + mc
        x, m = divmod(prog[j], self.K)
        if killed:
            p2[j] = DEAD
            reward = self.v[j]
        else:
            x1 = x + 1
            m1 = max(m, sum(1 for k in range(self.K) if self.D[k] <= x1))
            p2[j] = x1 * self.K + m1
            reward = 0.0
        return reward, (t2, j, self.canon(t2, p2))

    def outcomes(self, state: State, j: int):
        q = self.qkill[j][state[2][j]]
        out = []
        if q > 0.0:
            r, ns = self.transition(state, j, True)
            out.append((q, r, ns))
        if q < 1.0:
            r, ns = self.transition(state, j, False)
            out.append((1.0 - q, r, ns))
        return out


def evaluate_policy2(model: Model2, policy) -> float:
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
            x = state[2][j] // model.K
            r, state = model.transition(state, j, x + 1 == model.D[c[j]])
            val += r
        total += w * val
    return total
