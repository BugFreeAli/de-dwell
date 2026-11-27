"""
A20: robustness of the scale-learning planner of A19 (plan A20.2, A20.3).

Questions: (1) does learning add to a non-learning planner that never abandons and also uses a rollout (RU_CLAMP)? (2) does the learner still beat the nominal
planner when the true dwell is outside its prior (scales 0.6 and 1.6) or when the error is class specific (dwell (4, 12) and (6, 8))? (3) what do the
planners cost per decision?
True dwell condition: a name that maps to the true dwell tuple of the classes (cfg["conditions"]); the planners' prior support is the four scales of A19.
Methods (U = P(killed value >= need), exact, in the true world; a policy sees (t, cur, dwell given per threat, killed flags)):
  NOM, CLAMP, ADAPT, ADAPT_V as in A19; RU_CLAMP: the one-step goal rollout of the nominal CEPS applied to the clamped nominal state (no learning);
  CEPS_T, RU_CEP_T informed references; level exact also U_opt_T. Per row also the mean time per policy call in ms (ms_<method>) and the number of calls
  (calls_<method>); the rollout values are memoised within the instance, so this is a warm per-instance figure.
Invariants: range, no method above U_opt_T, condition 1.0 reproduces the informed values for NOM and RU_CLAMP, no kill completes after its deadline.
"""
import dataclasses
import time

from .a18 import TOL
from .a19 import Learner, SUPPORT, prob_goal_obs
from .instances import make_instance_k
from .m0 import DEAD, Model
from .m0u import GoalDP, ceps_policy, need_of, prob_goal, ru_policy


class Learner20(Learner):
    def __init__(self, inst, need, support=SUPPORT):
        super().__init__(inst, need, support)
        self.ru_nom = ru_policy(self.mb, self.ceps_nom, need)

    def state_in(self, s, obs):
        """As in A19, but the progress of an alive threat is clamped to D_s[-1] - 1. The clamp changes nothing for a scale with positive posterior weight (an alive
        threat with more dwell than the largest dwell of the scale gives it weight 0); it only matters when no scale is consistent with the observation (the true dwell is
        outside the prior) and the posterior falls back to the prior: the A19 version raised an IndexError there (found in the exploratory check of A20)."""
        t, cur, x, killed = obs
        Dmax = self.mod[s].D[-1]
        prog = tuple(DEAD if killed[j] else min(x[j], Dmax - 1) for j in range(len(x)))
        return (t, cur, self.mod[s].canon(t, prog))

    def ru_clamp(self, obs, acc):
        t, cur, x, killed = obs
        prog = tuple(DEAD if killed[j] else min(x[j], self.mb.D[-1] - 1) for j in range(len(x)))
        return self.ru_nom((t, cur, self.mb.canon(t, prog)), acc)


class Timed:
    def __init__(self, f):
        self.f, self.t, self.n = f, 0.0, 0

    def __call__(self, obs, acc):
        t0 = time.perf_counter()
        j = self.f(obs, acc)
        self.t += time.perf_counter() - t0
        self.n += 1
        return j


METHODS = ("NOM", "CLAMP", "RU_CLAMP", "ADAPT", "ADAPT_V")


def eval_row(cfg, n, rho, s, a, lam, cond, idx, level):
    """One instance under one true dwell condition. level: 'exact' (n <= 5) or 'ru'."""
    t0 = time.perf_counter()
    D0 = tuple(cfg["D0"])
    inst, real = make_instance_k(cfg["eval_seed"], idx, n, rho, s, a, D=D0, value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
    Dt = tuple(cfg["conditions"][cond])
    mt = Model(dataclasses.replace(inst, D=Dt))
    need = need_of(inst.values, lam)
    row = dict(n=n, rho=rho, s=s, a=a, lam=lam, cond=cond, D_true=str(Dt), idx=idx, need=need, W=float(sum(inst.values)), **real)
    L = Learner20(inst, need, cfg.get("support", SUPPORT))
    late, viol = 0, []
    pols = dict(NOM=L.nom, CLAMP=L.clamp, RU_CLAMP=L.ru_clamp, ADAPT=L.adapt, ADAPT_V=L.adapt_v)
    for name in METHODS:
        tp = Timed(pols[name])
        u, lt = prob_goal_obs(inst, Dt, tp, need)
        row["U_" + name] = u
        row["ms_" + name] = 1000.0 * tp.t / tp.n if tp.n else 0.0
        row["calls_" + name] = tp.n
        late += lt
    ceps_t = ceps_policy(mt, need)
    row["U_CEPS_T"] = prob_goal(mt, ceps_t, need)
    row["U_RU_CEP_T"] = prob_goal(mt, ru_policy(mt, ceps_t, need), need)
    if level == "exact":
        row["U_opt_T"] = GoalDP(mt, need).value
        if any(row["U_" + m] > row["U_opt_T"] + TOL for m in METHODS + ("CEPS_T", "RU_CEP_T")):
            viol.append("above_optimum")
    if cond == "1.0" and (abs(row["U_NOM"] - row["U_CEPS_T"]) > TOL or abs(row["U_RU_CLAMP"] - row["U_RU_CEP_T"]) > TOL):
        viol.append("cond1_differs")
    if late:
        viol.append("late_kill")
    if any(not (-TOL <= v <= 1 + TOL) for k, v in row.items() if k.startswith("U_")):
        viol.append("range")
    row["violations"] = ";".join(viol)
    row["sec"] = time.perf_counter() - t0
    return row
