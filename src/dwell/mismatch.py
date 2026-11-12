import dataclasses
import itertools

from .m0 import DEAD, EPS, NO_TARGET, Instance
from .m2 import Model2
from .rules import make_rule

GONE_NONE = None


def scaled(D0, lam):
    D = tuple(int(round(lam * d)) for d in D0)
    assert all(abs(lam * d - x) < 1e-9 for d, x in zip(D0, D)), "scale must give integer dwell"
    return D


def worlds_of(inst):
    K = len(inst.D)
    for c in itertools.product(range(K), repeat=inst.n):
        w = 1.0
        for i, k in enumerate(c):
            w *= inst.prior[i][k]
        if w > 0:
            yield c, w


def eval_obs(inst, factory):
    n, D, s, d, v = inst.n, inst.D, inst.s, inst.deadlines, inst.values
    pol = factory()
    cache = {}
    total = 0.0
    for c, w in worlds_of(inst):
        t, cur = 0, NO_TARGET
        dwell = [0] * n
        kd = [0] * n
        val = 0.0
        while True:
            key = (t, cur, tuple(dwell), tuple(kd))
            if key in cache:
                j = cache[key]
            else:
                j = pol(t, cur, tuple(dwell), tuple(kd))
                cache[key] = j
            if j is None or kd[j] > 0:
                break
            sl = s if (cur != NO_TARGET and cur != j) else 0
            t2 = t + sl + 1
            if t2 > d[j]:
                break
            dwell[j] += 1
            if dwell[j] == D[c[j]]:
                kd[j] = dwell[j]
                val += v[j]
            t, cur = t2, j
        total += w * val
    return total


def believed_factory(model_b, base_policy):
    K = model_b.K
    n = model_b.n

    def factory():
        def pol(t, cur, dwell, kd):
            prog = []
            for i in range(n):
                if kd[i] > 0:
                    prog.append(DEAD)
                    continue
                x = dwell[i]
                xb = min(x, model_b.D[-1] - 1)
                m = min(sum(1 for k in range(K) if model_b.D[k] <= x), K - 1)
                prog.append(xb * K + m)
            return base_policy((t, cur, model_b.canon(t, prog)))
        return pol

    return factory


def base_policy_for(method, model_b, dp_b):
    if method == "DPMIS":
        return dp_b.policy()
    return make_rule(model_b, method)


class ModelBA:
    def __init__(self, inst_nominal: Instance, lams, lam_prior=None, Dl=None):
        self.inst = inst_nominal
        self.n, self.s = inst_nominal.n, inst_nominal.s
        self.v, self.d, self.prior = inst_nominal.values, inst_nominal.deadlines, inst_nominal.prior
        self.D0 = inst_nominal.D
        self.lams = tuple(lams)
        self.lam_prior = tuple(lam_prior) if lam_prior is not None else tuple(1.0 / len(self.lams) for _ in self.lams)
        self.Dl = [tuple(x) for x in Dl] if Dl is not None else [scaled(self.D0, lam) for lam in self.lams]
        self.K = len(self.Dl[0])
        self._bel = {}

    def initial_state(self):
        return (0, NO_TARGET, (0,) * self.n)

    def slew(self, cur, j):
        return self.s if (cur != NO_TARGET and cur != j) else 0

    def belief(self, obs):
        hit = self._bel.get(obs)
        if hit is not None:
            return hit
        wl = []
        zs = []
        for li, Dl in enumerate(self.Dl):
            z = []
            wt = self.lam_prior[li]
            for i, o in enumerate(obs):
                if o < 0:
                    zi = sum(self.prior[i][k] for k in range(self.K) if Dl[k] == -o)
                else:
                    zi = sum(self.prior[i][k] for k in range(self.K) if Dl[k] > o)
                z.append(zi)
                wt *= zi
            wl.append(wt)
            zs.append(z)
        tot = sum(wl)
        res = (tuple(w / tot for w in wl) if tot > EPS else tuple(0.0 for _ in wl), tuple(tuple(z) for z in zs))
        self._bel[obs] = res
        return res

    def rmin(self, obs, j):
        wn, _ = self.belief(obs)
        x = obs[j]
        best = float("inf")
        for li, Dl in enumerate(self.Dl):
            if wn[li] <= EPS:
                continue
            for k in range(self.K):
                if self.prior[j][k] > EPS and Dl[k] > x:
                    best = min(best, Dl[k] - x)
        return best

    def actions(self, state):
        t, cur, obs = state
        acts = []
        for j, o in enumerate(obs):
            if o >= 0 and t + self.slew(cur, j) + self.rmin(obs, j) <= self.d[j]:
                acts.append(j)
        if cur in acts:
            acts.remove(cur)
            acts.insert(0, cur)
        return acts

    def outcomes(self, state, j):
        t, cur, obs = state
        wn, zs = self.belief(obs)
        x = obs[j]
        pk = 0.0
        for li, Dl in enumerate(self.Dl):
            if wn[li] <= EPS or zs[li][j] <= EPS:
                continue
            m = sum(self.prior[j][k] for k in range(self.K) if Dl[k] == x + 1)
            pk += wn[li] * m / zs[li][j]
        t2 = t + self.slew(cur, j) + 1
        out = []
        if pk > EPS:
            o2 = list(obs)
            o2[j] = -(x + 1)
            out.append((pk, self.v[j], (t2, j, tuple(o2))))
        if pk < 1.0 - EPS:
            o2 = list(obs)
            o2[j] = x + 1
            out.append((1.0 - pk, 0.0, (t2, j, tuple(o2))))
        return out


def ba_factory(model_ba, policy):
    n = model_ba.n

    def factory():
        def pol(t, cur, dwell, kd):
            obs = tuple(-kd[i] if kd[i] > 0 else dwell[i] for i in range(n))
            return policy((t, cur, obs))
        return pol

    return factory


def noisy_classes(base_D, lam, offsets):
    return tuple(int(round(lam * b)) + o for b in base_D for o in offsets)


def expand_prior(prior2, probs):
    return tuple(tuple(p * pe for p in row for pe in probs) for row in prior2)


def a6_instance(cfg, rho, s, a, idx, make_instance_k):
    base = tuple(cfg["D0"])
    inst0, real = make_instance_k(cfg["master_seed"], idx, cfg["n"], rho, s, a, D=base,
                                  value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
    probs = tuple(cfg["noise_probs"])
    offs = tuple(cfg["noise_offsets"])
    lams = sorted(set(cfg["lam_tol"]) | set(cfg["lam_ba"]) | {1.0})
    Dl = {lam: noisy_classes(base, lam, offs) for lam in lams}
    inst = dataclasses.replace(inst0, D=Dl[1.0], prior=expand_prior(inst0.prior, probs))
    return inst, Dl, real
