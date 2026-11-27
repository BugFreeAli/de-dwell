"""
A19: learning the dwell scale during the episode (plan A19.2, A19.3).

Setting of A18: class k really needs scale * D0[k] dwell steps; the planner knows only the nominal D0. Here the planner knows that the
scale is one of SUPPORT (uniform prior) and learns it from what it observes: a threat that is still alive after x dwell steps rules out
every (scale, class) with dwell <= x, and a kill after exactly x dwell steps shows that the dwell of the class was x (with D0 = (5, 10) and
the four scales this fixes the scale). Threats are independent given the scale, so the posterior of the scale is
        w(s) ~ prior(s) * prod_j L_j(s),   L_j(s) = sum_k P_j(k) * [D_s(k) > x_j]  (alive after x_j steps)
                                                    or sum_k P_j(k) * [D_s(k) == x_j]  (killed after x_j steps).
Methods (all evaluated exactly in the true world, U = P(killed value >= need); a policy sees (t, cur, dwell x_j per threat, killed_j)):
  NOM      the nominal survival-aware planner CEPS of A18 (abandons a threat that survives more dwell than the largest nominal dwell)
  CLAMP    CEPS on the nominal model with the progress of an alive threat clamped to D0[-1] - 1 (never abandons; no learning)
  ADAPT    posterior-weighted one-step lookahead: for every scale with positive weight the exact goal probability of each action followed
           by CEPS of that scale (the rollout value of A14), averaged with the posterior weights; the best action is taken
  ADAPT_V  posterior-weighted vote of the CEPS actions of the scales (no rollout)
  CEPS_T, RU_CEP_T  informed references (built on the true model); level exact also U_opt_T (exact goal optimum of the true model)
Invariants written to the row: range, no method above U_opt_T, scale 1 reproduces CEPS_T for NOM, no kill completes after its deadline.
"""
import dataclasses
import time

from .a18 import TOL, prob_goal_true, scaled_dwell
from .instances import make_instance_k
from .m0 import DEAD, EPS, Model
from .m0u import GoalDP, ceps_policy, need_of, prob_goal, ru_policy
from .rules import make_rule

SUPPORT = (0.8, 1.0, 1.2, 1.4)


class Learner:
    """Policies on observations obs = (t, cur, x, killed) for one instance, goal and scale support."""

    def __init__(self, inst, need, support=SUPPORT):
        self.inst, self.need, self.support = inst, need, tuple(support)
        self.D0 = inst.D
        self.mod = {s: Model(dataclasses.replace(inst, D=scaled_dwell(inst.D, s))) for s in self.support}
        self.ceps = {s: ceps_policy(self.mod[s], need) for s in self.support}
        self.memo = {s: {} for s in self.support}
        self.mb = Model(inst)
        self.ceps_nom = ceps_policy(self.mb, need)

    # ---- belief ----------------------------------------------------------------------------------------------------------------
    def posterior(self, obs):
        t, cur, x, killed = obs
        w = {}
        for s in self.support:
            Ds = self.mod[s].D
            lik = 1.0
            for j, xj in enumerate(x):
                if xj == 0 and not killed[j]:
                    continue
                pj = self.inst.prior[j]
                if killed[j]:
                    lik *= sum(pj[k] for k in range(len(Ds)) if Ds[k] == xj)
                else:
                    lik *= sum(pj[k] for k in range(len(Ds)) if Ds[k] > xj)
            w[s] = lik
        z = sum(w.values())
        if z <= 0.0:                                  # the truth is outside the support: fall back to the uniform prior
            return {s: 1.0 / len(self.support) for s in self.support}
        return {s: v / z for s, v in w.items()}

    def state_in(self, s, obs):
        """The observation as a state of the model of scale s (killed threats DEAD, alive threats at their dwell)."""
        t, cur, x, killed = obs
        prog = tuple(DEAD if killed[j] else x[j] for j in range(len(x)))
        return (t, cur, self.mod[s].canon(t, prog))

    # ---- policies --------------------------------------------------------------------------------------------------------------
    def _bval(self, s, state, acc):
        if acc >= self.need - 1e-9:
            return 1.0
        memo = self.memo[s]
        key = (state, acc)
        if key in memo:
            return memo[key]
        j = self.ceps[s](state, acc)
        m = self.mod[s]
        v = 0.0 if j is None else sum(p * self._bval(s, nxt, acc + r) for p, r, nxt in m.outcomes(state, j))
        memo[key] = v
        return v

    def adapt(self, obs, acc):
        if acc >= self.need - 1e-9:
            return None
        w = self.posterior(obs)
        q = {}
        for s in self.support:
            if w[s] <= 0.0:
                continue
            st = self.state_in(s, obs)
            m = self.mod[s]
            for a in m.actions(st):
                q[a] = q.get(a, 0.0) + w[s] * sum(p * self._bval(s, nxt, acc + r) for p, r, nxt in m.outcomes(st, a))
        best_j, best_v = None, -1.0
        for a in sorted(q):
            if q[a] > best_v + EPS:
                best_j, best_v = a, q[a]
        return best_j

    def adapt_v(self, obs, acc):
        if acc >= self.need - 1e-9:
            return None
        w = self.posterior(obs)
        votes = {}
        for s in self.support:
            if w[s] <= 0.0:
                continue
            a = self.ceps[s](self.state_in(s, obs), acc)
            if a is not None:
                votes[a] = votes.get(a, 0.0) + w[s]
        best_j, best_v = None, -1.0
        for a in sorted(votes):
            if votes[a] > best_v + EPS:
                best_j, best_v = a, votes[a]
        return best_j

    def nom(self, obs, acc):
        t, cur, x, killed = obs
        prog = tuple(DEAD if killed[j] else min(x[j], self.mb.D[-1]) for j in range(len(x)))
        return self.ceps_nom((t, cur, self.mb.canon(t, prog)), acc)

    def clamp(self, obs, acc):
        t, cur, x, killed = obs
        prog = tuple(DEAD if killed[j] else min(x[j], self.mb.D[-1] - 1) for j in range(len(x)))
        return self.ceps_nom((t, cur, self.mb.canon(t, prog)), acc)


def prob_goal_obs(inst, D_true, policy, need):
    """
    Exact P(killed value >= need) of a policy that sees obs = (t, cur, dwell given to each threat, killed flags) when class k really
    needs D_true[k] dwell steps. Returns (probability, number of kills that would complete after the deadline).
    """
    mb = Model(inst)
    cache, total, late = {}, 0.0, 0
    for c, w in mb.worlds():
        t, cur, acc = 0, -1, 0.0
        x, killed = [0] * mb.n, [False] * mb.n
        while acc < need - 1e-9:
            obs = (t, cur, tuple(x), tuple(killed))
            key = (obs, acc)
            if key in cache:
                j = cache[key]
            else:
                j = policy(obs, acc)
                cache[key] = j
            if j is None:
                break
            t2 = t + mb.slew(cur, j) + 1
            x[j] += 1
            if x[j] == D_true[c[j]]:
                killed[j] = True
                if t2 <= mb.d[j]:
                    acc += mb.v[j]
                else:
                    late += 1
            t, cur = t2, j
        total += w * (1.0 if acc >= need - 1e-9 else 0.0)
    return total, late


def eval_row(cfg, n, rho, s, a, lam, scale, idx, level):
    """One instance at one true scale. level: 'exact' (n <= 5) or 'ru'."""
    t0 = time.perf_counter()
    D0 = tuple(cfg["D0"])
    inst, real = make_instance_k(cfg["eval_seed"], idx, n, rho, s, a, D=D0, value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
    Dt = scaled_dwell(D0, scale)
    mt = Model(dataclasses.replace(inst, D=Dt))
    need = need_of(inst.values, lam)
    row = dict(n=n, rho=rho, s=s, a=a, lam=lam, scale=scale, idx=idx, need=need, W=float(sum(inst.values)), **real)
    L = Learner(inst, need, cfg.get("support", SUPPORT))
    late, viol = 0, []
    for name, pol in (("NOM", L.nom), ("CLAMP", L.clamp), ("ADAPT", L.adapt), ("ADAPT_V", L.adapt_v)):
        u, lt = prob_goal_obs(inst, Dt, pol, need)
        row["U_" + name] = u
        late += lt
    ceps_t = ceps_policy(mt, need)
    row["U_CEPS_T"] = prob_goal(mt, ceps_t, need)
    row["U_RU_CEP_T"] = prob_goal(mt, ru_policy(mt, ceps_t, need), need)
    if level == "exact":
        row["U_opt_T"] = GoalDP(mt, need).value
        if any(row["U_" + m] > row["U_opt_T"] + TOL for m in ("NOM", "CLAMP", "ADAPT", "ADAPT_V", "CEPS_T", "RU_CEP_T")):
            viol.append("above_optimum")
    if scale == 1.0 and abs(row["U_NOM"] - row["U_CEPS_T"]) > TOL:
        viol.append("scale1_differs")
    if late:
        viol.append("late_kill")
    if any(not (-TOL <= v <= 1 + TOL) for k, v in row.items() if k.startswith("U_")):
        viol.append("range")
    row["violations"] = ";".join(viol)
    row["sec"] = time.perf_counter() - t0
    return row
