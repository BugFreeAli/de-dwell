import dataclasses
import math

from .dp import solve
from .m0 import DEAD, EPS, NO_TARGET, Instance
from .m2 import Model2
from .mismatch import ModelBA, base_policy_for, believed_factory, expand_prior, scaled
from .rules import make_rule

FLOOR = 1e-6
PRACTICAL = ("PLUG_DP", "PLUG_CEP", "QMDP", "MIXCEP")


def class_dwells(D0, lam, offsets):
    nested = isinstance(offsets[0], (list, tuple))
    out = []
    for ci, b in enumerate(D0):
        offs = offsets[ci] if nested else offsets
        base = scaled((b,), lam)[0]
        for o in offs:
            out.append(base + o)
    assert all(x >= 1 for x in out), "dwell below one step"
    return tuple(out)


def merge_classes(D, prior):
    keys = sorted(set(D))
    pm = tuple(tuple(sum(row[k] for k in range(len(D)) if D[k] == x) for x in keys) for row in prior)
    return tuple(keys), pm


def merged_instance(inst, D):
    Dm, pm = merge_classes(D, inst.prior)
    return dataclasses.replace(inst, D=Dm, prior=pm)


def a7_instance(cfg, rho, s, a, idx, make_instance_k):
    base = tuple(cfg["D0"])
    inst0, real = make_instance_k(cfg["master_seed"], idx, cfg["n"], rho, s, a, D=base,
                                  value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
    offs = cfg["noise_offsets"]
    probs = tuple(cfg["noise_probs"])
    lams = sorted(set(cfg["scales"]) | {1.0})
    Dl = {lam: class_dwells(base, lam, offs) for lam in lams}
    inst = dataclasses.replace(inst0, D=Dl[1.0], prior=expand_prior(inst0.prior, probs))
    return inst, Dl, real, inst0


def iid_truth(inst0, base, scales, probs):
    D = tuple(scaled((b,), lam)[0] for b in base for lam in scales)
    prior = expand_prior(inst0.prior, tuple(probs))
    return merged_instance(dataclasses.replace(inst0, D=D, prior=prior), D)


class Planner:
    def __init__(self, inst, Dl, lams):
        self.inst = inst
        self.Dl = Dl
        self.models = {}
        for lam in sorted(set(lams)):
            im = merged_instance(inst, Dl[lam])
            m = Model2(im)
            self.models[lam] = (im, m, solve(m))


class RobustBA(ModelBA):
    def __init__(self, inst, lams, lam_prior, Dl):
        super().__init__(inst, lams, lam_prior, Dl)
        self.union = [sorted({self.Dl[li][k] for li in range(len(self.lams)) for k in range(self.K)
                              if self.prior[i][k] > EPS}) for i in range(self.n)]

    def project(self, obs):
        out = []
        for i, o in enumerate(obs):
            U = self.union[i]
            if o < 0:
                k = -o
                if k not in U:
                    k = min(U, key=lambda u: (abs(u - k), u))
                out.append(-k)
            else:
                out.append(min(o, U[-1] - 1))
        return tuple(out)

    def belief(self, obs):
        hit = self._bel.get(obs)
        if hit is not None:
            return hit
        zs = []
        for Dl in self.Dl:
            z = []
            for i, o in enumerate(obs):
                if o < 0:
                    z.append(sum(self.prior[i][k] for k in range(self.K) if Dl[k] == -o))
                else:
                    z.append(sum(self.prior[i][k] for k in range(self.K) if Dl[k] > o))
            zs.append(z)
        wl = [self.lam_prior[li] * math.prod(zs[li]) for li in range(len(self.Dl))]
        tot = sum(wl)
        if tot <= EPS:
            wl = [max(self.lam_prior[li], FLOOR) * math.prod(max(x, FLOOR) for x in zs[li]) for li in range(len(self.Dl))]
            tot = sum(wl)
        res = (tuple(w / tot for w in wl), tuple(tuple(z) for z in zs))
        self._bel[obs] = res
        return res


class Kit:
    def __init__(self, planner, lams, weights=None):
        self.pl = planner
        self.lams = tuple(lams)
        self.n = planner.inst.n
        self.ba = RobustBA(planner.inst, self.lams, weights, [planner.Dl[l] for l in self.lams])
        self.dp_ba = None
        self.cep = {l: make_rule(planner.models[l][1], "CEP") for l in self.lams}
        self.gmr = {l: make_rule(planner.models[l][1], "GMR") for l in self.lams}

    def solve_ba(self):
        if self.dp_ba is None:
            self.dp_ba = solve(self.ba)
        return self.dp_ba

    def obs_of(self, dwell, kd):
        return tuple(-kd[i] if kd[i] > 0 else dwell[i] for i in range(self.n))

    def prog_in(self, lam, t, po):
        _, M, _ = self.pl.models[lam]
        K = M.K
        prog = []
        for x in po:
            if x < 0:
                prog.append(DEAD)
                continue
            xb = min(x, M.D[-1] - 1)
            m = min(sum(1 for k in range(K) if M.D[k] <= x), K - 1)
            prog.append(xb * K + m)
        return M.canon(t, prog)

    def lam_hat(self, w):
        top = max(w)
        cand = [li for li, x in enumerate(w) if x >= top - 1e-12]
        li = min(cand, key=lambda i: (abs(self.lams[i] - 1.0), self.lams[i]))
        return self.lams[li]

    def fallback(self, t, cur, po):
        acts = self.ba.actions((t, cur, po))
        return acts[0] if acts else None

    def qvalues(self, t, cur, po):
        acts = self.ba.actions((t, cur, po))
        w, _ = self.ba.belief(po)
        out = {}
        for j in acts:
            q = 0.0
            for li, lam in enumerate(self.lams):
                if w[li] <= EPS:
                    continue
                _, M, dp = self.pl.models[lam]
                ps = self.prog_in(lam, t, po)
                if ps[j] != DEAD:
                    for p, r, nx in M.outcomes((t, cur, ps), j):
                        q += w[li] * p * (r + dp._solve(nx)[0])
                else:
                    t2 = t + M.slew(cur, j) + 1
                    q += w[li] * dp._solve((t2, j, M.canon(t2, list(ps))))[0]
            out[j] = q
        return acts, out

    def durations(self, t, cur, po, acts):
        w, _ = self.ba.belief(po)
        dur = {}
        for j in acts:
            num = den = 0.0
            for li, lam in enumerate(self.lams):
                if w[li] <= EPS:
                    continue
                _, M, _ = self.pl.models[lam]
                ps = self.prog_in(lam, t, po)
                if ps[j] == DEAD:
                    continue
                num += w[li] * sum(p * r for p, r in M.post[j][ps[j]])
                den += w[li]
            dur[j] = num / den if den > EPS else float(self.ba.rmin(po, j))
        return dur

    def act(self, name, t, cur, po):
        if name == "BA":
            return self.solve_ba().policy()((t, cur, po))
        w, _ = self.ba.belief(po)
        if name in ("PLUG_DP", "PLUG_CEP"):
            lam = self.lam_hat(w)
            _, M, dp = self.pl.models[lam]
            st = (t, cur, self.prog_in(lam, t, po))
            j = dp.policy()(st) if name == "PLUG_DP" else self.cep[lam](st)
            return j if j is not None else self.fallback(t, cur, po)
        if name == "QMDP":
            acts, q = self.qvalues(t, cur, po)
            best_j, best = None, -float("inf")
            for j in acts:
                if q[j] > best + EPS:
                    best_j, best = j, q[j]
            return best_j
        if name == "MIXCEP":
            acts = self.ba.actions((t, cur, po))
            if not acts:
                return None
            dur = self.durations(t, cur, po, acts)
            d = self.ba.d
            best = [0.0, 0.0, None]

            def extend(time, last, used, value, first):
                if first is not None and (value > best[0] + EPS or (abs(value - best[0]) <= EPS and time < best[1] - EPS)):
                    best[0], best[1], best[2] = value, time, first
                for j in acts:
                    if j in used:
                        continue
                    end = time + self.ba.slew(last, j) + dur[j]
                    if end <= d[j] + 1e-9:
                        extend(end, j, used | {j}, value + self.ba.v[j], j if first is None else first)

            extend(float(t), cur, frozenset(), 0.0, None)
            if best[2] is not None:
                return best[2]
            lam = self.lam_hat(w)
            j = self.gmr[lam]((t, cur, self.prog_in(lam, t, po)))
            return j if j is not None else self.fallback(t, cur, po)
        raise ValueError(name)

    def factory(self, name):
        def make():
            def pol(t, cur, dwell, kd):
                po = self.ba.project(self.obs_of(dwell, kd))
                return self.act(name, t, cur, po)
            return pol
        return make


def nominal_factory(planner, meth, lam=1.0):
    _, M, dp = planner.models[lam]
    return believed_factory(M, base_policy_for(meth, M, dp))


def eval_rec(inst, factory):
    n, D, s, d, v = inst.n, inst.D, inst.s, inst.deadlines, inst.values
    K = len(D)
    pol = factory()
    memo = {}

    def kill_prob(i, x):
        z = sum(inst.prior[i][k] for k in range(K) if D[k] > x)
        if z <= EPS:
            return 0.0
        return sum(inst.prior[i][k] for k in range(K) if D[k] == x + 1) / z

    def val(t, cur, dwell, kd):
        key = (t, cur, dwell, kd)
        hit = memo.get(key)
        if hit is not None:
            return hit
        j = pol(t, cur, dwell, kd)
        res = 0.0
        if j is not None and kd[j] == 0:
            sl = s if (cur != NO_TARGET and cur != j) else 0
            t2 = t + sl + 1
            if t2 <= d[j]:
                x = dwell[j]
                q = kill_prob(j, x)
                dw = list(dwell)
                dw[j] = x + 1
                dw = tuple(dw)
                if q > EPS:
                    k2 = list(kd)
                    k2[j] = x + 1
                    res += q * (v[j] + val(t2, j, dw, tuple(k2)))
                if q < 1.0 - EPS:
                    res += (1.0 - q) * val(t2, j, dw, kd)
        memo[key] = res
        return res

    return val(0, NO_TARGET, (0,) * n, (0,) * n)
