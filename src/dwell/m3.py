from typing import Tuple

from .m0 import DEAD, EPS, Instance, Model, State


class Model3(Model):
    def __init__(self, inst: Instance, q: float):
        inst.validate()
        assert len(inst.D) == 2, "Model3 supports two classes"
        assert 0.5 <= q < 1.0
        self.inst = inst
        self.q = q
        self.n, self.D, self.s = inst.n, inst.D, inst.s
        self.v, self.d, self.prior = inst.values, inst.deadlines, inst.prior
        self.K = 2
        self.D1, self.D2 = inst.D
        size = self.D2 * self.D1
        self.post, self.qkill, self.rmin, self.pp = [], [], [], []
        for i in range(self.n):
            ps0, ph0 = self.prior[i]
            post_i = [[] for _ in range(size)]
            q_i = [0.0] * size
            r_i = [float("inf")] * size
            pp_i = [(0.0, 0.0)] * size
            for x in range(self.D2):
                for h in range(self.D1):
                    if x < self.D1 and h > x:
                        continue
                    if x >= self.D1 and h > 0:
                        continue
                    if x < self.D1:
                        ws = ps0 * (1 - q) ** h * q ** (x - h)
                        wh = ph0 * q ** h * (1 - q) ** (x - h)
                    else:
                        ws, wh = 0.0, ph0
                    z = ws + wh
                    if z <= EPS:
                        continue
                    ps, ph = ws / z, wh / z
                    code = x * self.D1 + h
                    post = []
                    if ps > EPS:
                        post.append((ps, self.D1 - x))
                    if ph > EPS:
                        post.append((ph, self.D2 - x))
                    qk = (ps if x == self.D1 - 1 else 0.0) + (ph if x == self.D2 - 1 else 0.0)
                    post_i[code] = post
                    q_i[code] = qk
                    r_i[code] = min(r for _, r in post)
                    pp_i[code] = (ps, ph)
            self.post.append(post_i)
            self.qkill.append(q_i)
            self.rmin.append(r_i)
            self.pp.append(pp_i)

    def outcomes(self, state: State, j: int):
        t, cur, prog = state
        code = prog[j]
        ps, ph = self.pp[j][code]
        x, h = divmod(code, self.D1)
        t2 = t + self.slew(cur, j) + 1
        q = self.q
        out = []
        qk = (ps if x == self.D1 - 1 else 0.0) + (ph if x == self.D2 - 1 else 0.0)
        if qk > EPS:
            p2 = list(prog)
            p2[j] = DEAD
            out.append((qk, self.v[j], (t2, j, self.canon(t2, p2))))
        ms = ps if x != self.D1 - 1 else 0.0
        mh = ph if x != self.D2 - 1 else 0.0
        x1 = x + 1
        if x >= self.D1:
            if mh > EPS:
                p2 = list(prog)
                p2[j] = x1 * self.D1
                out.append((mh, 0.0, (t2, j, self.canon(t2, p2))))
            return out
        for sig in (1, 0):
            pr = ms * ((1 - q) if sig == 1 else q) + mh * (q if sig == 1 else (1 - q))
            if pr > EPS:
                h1 = h + sig if x1 < self.D1 else 0
                p2 = list(prog)
                p2[j] = x1 * self.D1 + h1
                out.append((pr, 0.0, (t2, j, self.canon(t2, p2))))
        return out


def evaluate_policy3(model, policy) -> float:
    cache: dict = {}
    act: dict = {}

    def value(state):
        hit = cache.get(state)
        if hit is not None:
            return hit
        if state not in act:
            act[state] = policy(state)
        j = act[state]
        v = 0.0 if j is None else sum(p * (r + value(nxt)) for p, r, nxt in model.outcomes(state, j))
        cache[state] = v
        return v

    return value(model.initial_state())
