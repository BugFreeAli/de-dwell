"""
A15: learned scorer for the survival objective (plan A15.3). Pure numpy.

Features of an action j in state (t, cur, prog) with accumulated killed value acc and goal `need`:
  columns 0..11   the 12 base features of B0 (src/dwell/learned.py)
  columns 12..13  static goal features that do not involve acc:  need / W,  remaining killable value / W
  columns 14..19  accumulated-value features:  need_left / W,  v_j / need_left (clipped to 2),  1 if v_j >= need_left,
                  (remaining killable value - need_left) / W,  v_j * p_succ_j / need_left (clipped to 2),
                  need_left / remaining killable value (clipped to 2)
Variants (plan A15.4; IL_A = IL_G plus the CEPS recommendation as column 20): IL_G = teacher is the exact goal optimum (n <= 5), columns 0..19; IL_G_noacc = same teacher, columns 0..13;
IL_EV = teacher is the exact expected-value optimum, columns 0..19 (same inputs as IL_G, other objective);
IL_AD = IL_A trained on larger instances (n = 6, 7) with the planner RU_CEP as teacher (distillation), columns 0..20.
Training data: states visited by epsilon-greedy teacher rollouts in sampled hidden-class worlds, labelled by the teacher.
"""
import math

import numpy as np

from .dp import solve
from .instances import make_instance_k
from .learned import features as base_features
from .m0 import DEAD, EPS, Model
from .m0u import GoalDP, need_of

N_FULL, N_NOACC, N_AUG = 20, 14, 21
COLS = {"IL_G": N_FULL, "IL_EV": N_FULL, "IL_G_noacc": N_NOACC, "IL_A": N_AUG, "IL_AD": N_AUG}
A_MAX = 5
# the 12 base features are clipped to the range met at the training sizes, so that larger instances do not extrapolate the scorer
CLIP_LO = np.array([0, -3, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0], dtype=float)
CLIP_HI = np.array([1.2, 3, 1.5, 1, 3, 1.5, 1, 3, 1.5, 1.5, 3, 2], dtype=float)


def features_g(model, state, acts, dmax, acc, need, ceps_act=None):
    """Feature matrix (len(acts) x 21) of the candidate actions; column 20 = 1 for the action recommended by CEPS (IL_A only)."""
    base = np.clip(base_features(model, state, acts, dmax), CLIP_LO, CLIP_HI)
    t, cur, prog = state
    W = float(sum(model.v))
    pot = sum(model.v[j] for j, x in enumerate(prog) if x != DEAD)
    left = max(need - acc, 0.0)
    rows = []
    for k, j in enumerate(acts):
        post = model.post[j][prog[j]]
        cap = model.d[j] - t - model.slew(cur, j)
        p_succ = sum(p for p, r in post if r <= cap)
        v = model.v[j]
        lf = max(left, 1e-9)
        rows.append([need / W, pot / W, left / W, min(v / lf, 2.0), 1.0 if v >= left - 1e-9 else 0.0,
                     (pot - left) / W, min(v * p_succ / lf, 2.0), min(left / max(pot, 1e-9), 2.0)])
    ex = np.array(rows, dtype=float)
    # reorder so that the static columns (need/W, pot/W) come first, then the accumulated-value columns
    flag = np.array([[1.0 if (ceps_act is not None and j == ceps_act) else 0.0] for j in acts])
    return np.concatenate([base, ex[:, :2], ex[:, 2:], flag], axis=1)


class Scorer:
    """One-hidden-layer scorer shared across actions; softmax over the feasible actions; Adam (as B0, n_feat configurable)."""

    def __init__(self, n_feat, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        self.n_feat = n_feat
        self.p = dict(W1=rng.normal(0, 0.5, (n_feat, hidden)), b1=np.zeros(hidden), w2=rng.normal(0, 0.5, hidden))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step = 0

    def scores(self, phi):
        return np.tanh(phi @ self.p["W1"] + self.p["b1"]) @ self.p["w2"]

    def loss_and_grads(self, Phi, mask, y):
        W1, b1, w2 = self.p["W1"], self.p["b1"], self.p["w2"]
        H = np.tanh(Phi @ W1 + b1)
        s = np.where(mask, H @ w2, -1e9)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s) * mask
        P = e / e.sum(axis=1, keepdims=True)
        S = Phi.shape[0]
        loss = -np.log(P[np.arange(S), y] + 1e-12).mean()
        G = P.copy()
        G[np.arange(S), y] -= 1.0
        G = G * mask / S
        dw2 = np.einsum("sa,sah->h", G, H)
        dZ = (G[:, :, None] * w2[None, None, :]) * (1.0 - H ** 2)
        return loss, dict(W1=np.einsum("saf,sah->fh", Phi, dZ), b1=dZ.sum(axis=(0, 1)), w2=dw2)

    def adam(self, grads, lr=0.01, b1=0.9, b2=0.999):
        self.step += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * grads[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * grads[k] ** 2
            self.p[k] -= lr * (self.m[k] / (1 - b1 ** self.step)) / (np.sqrt(self.v[k] / (1 - b2 ** self.step)) + 1e-8)

    def fit(self, Phi, mask, y, epochs=60, batch=512, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        S, hist = Phi.shape[0], []
        for _ in range(epochs):
            order, tot = rng.permutation(S), 0.0
            for i in range(0, S, batch):
                b = order[i:i + batch]
                loss, g = self.loss_and_grads(Phi[b], mask[b], y[b])
                self.adam(g, lr=lr)
                tot += loss * len(b)
            hist.append(tot / S)
        return hist

    def accuracy(self, Phi, mask, y):
        s = np.where(mask, np.tanh(Phi @ self.p["W1"] + self.p["b1"]) @ self.p["w2"], -1e9)
        return float((s.argmax(axis=1) == y).mean())

    def save(self, path, **extra):
        np.savez(path, n_feat=self.n_feat, **self.p, **{k: np.asarray(v) for k, v in extra.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        sc = cls(int(z["n_feat"]))
        for k in sc.p:
            sc.p[k] = z[k].copy()
        return sc


def learned_policy(model, scorer, need):
    """policy(state, acc) of a trained scorer; returns None when the goal is reached or no action is feasible."""
    from .m0u import ceps_policy
    dmax, nf = max(model.d), scorer.n_feat
    cp = ceps_policy(model, need) if nf == N_AUG else None

    def pol(state, acc):
        if acc >= need - 1e-9:
            return None
        acts = model.actions(state)
        if not acts:
            return None
        phi = features_g(model, state, acts, dmax, acc, need, cp(state, acc) if cp else None)[:, :nf]
        return acts[int(np.argmax(scorer.scores(phi)))]
    return pol


def rollout_states(model, c, choose, need, rng, eps):
    """States (state, acc) visited in the world c by an epsilon-greedy policy `choose(state, acc)`."""
    state, acc, out = model.initial_state(), 0.0, []
    while acc < need - 1e-9:
        acts = model.actions(state)
        if not acts:
            break
        out.append((state, acc))
        j = choose(state, acc)
        if j is None:
            break
        if rng.random() < eps:
            j = acts[int(rng.integers(len(acts)))]
        t, cur, prog = state
        t2 = t + model.slew(cur, j) + 1
        p2 = list(prog)
        if prog[j] + 1 == model.D[c[j]]:
            acc += model.v[j]
            p2[j] = DEAD
        else:
            p2[j] = prog[j] + 1
        state = (t2, j, model.canon(t2, p2))
    return out


def build_instance_samples(cfg, i):
    """Teacher-labelled samples of training instance i of the pool: (features A_MAX x 20, mask, label_goal, label_ev)."""
    rng = np.random.default_rng([cfg["pool_seed"], i])
    n = int(rng.choice(cfg["train_n"]))
    rho, s = float(rng.choice(cfg["rho_grid"])), float(rng.choice(cfg["s_grid"]))
    a, lam = float(rng.choice(cfg["a_grid"])), float(rng.choice(cfg["lam_grid"]))
    inst, _ = make_instance_k(cfg["pool_seed"], i, n, rho, s, a, D=tuple(cfg["D0"]), value_set=tuple(cfg["value_set"]),
                              u_low=cfg["u_low"])
    model = Model(inst)
    need = need_of(inst.values, lam)
    meta = dict(n=n, rho=rho, s=s, a=a, lam=lam, need=need)
    if need <= 0:
        return [], meta
    from .m0u import ceps_policy
    gdp, ev, cps = GoalDP(model, need), solve(model).policy(), ceps_policy(model, need)
    g_pol = gdp.policy()
    worlds = list(model.worlds())
    w = np.array([x[1] for x in worlds])
    dmax, seen, rows, states = max(model.d), set(), [], []
    teachers = (lambda st, ac: g_pol(st, ac), lambda st, ac: ev(st))
    for k in range(cfg["worlds_per_instance"]):
        c = worlds[int(rng.choice(len(worlds), p=w / w.sum()))][0]
        eps = (0.0, 0.1, 0.25)[k % 3]
        for st, ac in rollout_states(model, c, teachers[k % 2], need, rng, eps):
            if (st, ac) in seen:
                continue
            seen.add((st, ac))
            acts = model.actions(st)
            if len(acts) > A_MAX:
                continue
            lg, le = g_pol(st, ac), ev(st)
            phi = np.zeros((A_MAX, N_AUG))
            phi[:len(acts)] = features_g(model, st, acts, dmax, ac, need, cps(st, ac))
            rows.append((phi, len(acts), acts.index(lg) if lg in acts else -1, acts.index(le) if le in acts else -1))
            states.append((st, ac))
    meta["states"], meta["model"] = states, model       # kept for the tests, not saved in the pool
    return rows, meta


def build_distill_samples(cfg, i):
    """Samples of distillation instance i: states visited by epsilon-greedy RU_CEP rollouts, labelled by RU_CEP (plan A15.4)."""
    from .m0u import ru_policy
    from .rules import make_rule
    rng = np.random.default_rng([cfg["pool_seed"], 7, i])
    n = int(rng.choice(cfg["distill_n"]))
    rho, s = float(rng.choice(cfg["rho_grid"])), float(rng.choice(cfg["s_grid"]))
    a, lam = float(rng.choice(cfg["a_grid"])), float(rng.choice(cfg["lam_grid"]))
    inst, _ = make_instance_k(cfg["pool_seed"] + 1, i, n, rho, s, a, D=tuple(cfg["D0"]), value_set=tuple(cfg["value_set"]),
                              u_low=cfg["u_low"])
    model = Model(inst)
    need = need_of(inst.values, lam)
    meta = dict(n=n, rho=rho, s=s, a=a, lam=lam, need=need)
    if need <= 0:
        return [], meta
    from .m0u import ceps_policy
    cps = ceps_policy(model, need)
    cep = make_rule(model, "CEP")
    teacher = ru_policy(model, lambda st, ac: cep(st), need)
    worlds = list(model.worlds())
    w = np.array([x[1] for x in worlds])
    dmax, seen, rows, states = max(model.d), set(), [], []
    for k in range(cfg["distill_worlds"]):
        c = worlds[int(rng.choice(len(worlds), p=w / w.sum()))][0]
        eps = (0.0, 0.1, 0.25)[k % 3]
        for st, ac in rollout_states(model, c, teacher, need, rng, eps):
            if (st, ac) in seen:
                continue
            seen.add((st, ac))
            acts = model.actions(st)
            if len(acts) > cfg.get("distill_a_max", 9):
                continue
            lt = teacher(st, ac)
            phi = np.zeros((cfg.get("distill_a_max", 9), N_AUG))
            phi[:len(acts)] = features_g(model, st, acts, dmax, ac, need, cps(st, ac))
            rows.append((phi, len(acts), acts.index(lt) if lt in acts else -1, -1))
            states.append((st, ac))
    meta["states"], meta["model"] = states, model
    return rows, meta


def to_arrays(rows_with_ids, a_max=A_MAX):
    """Stack samples: Phi (S, A_MAX, 20), mask, y_goal, y_ev, instance id."""
    S = len(rows_with_ids)
    Phi, mask = np.zeros((S, a_max, N_AUG)), np.zeros((S, a_max), dtype=bool)
    yg, ye, iid = np.zeros(S, dtype=int), np.zeros(S, dtype=int), np.zeros(S, dtype=int)
    for k, (i, (phi, na, g, e)) in enumerate(rows_with_ids):
        Phi[k], mask[k, :na], yg[k], ye[k], iid[k] = phi, True, g, e, i
    return Phi, mask, yg, ye, iid


def train_variant(variant, seed, pool, frac=0.7, epochs=60):
    """Train one model of `variant` on a random 70% of the pool instances; returns (scorer, history, train agreement)."""
    Phi, mask, yg, ye, iid = pool
    rng = np.random.default_rng([seed, 15])
    ids = np.unique(iid)
    keep = set(rng.choice(ids, size=max(1, int(frac * len(ids))), replace=False).tolist())
    y = ye if variant == "IL_EV" else yg                      # IL_AD: the pool is the distillation pool, labels in yg
    sel = np.array([(i in keep) and (y[k] >= 0) for k, i in enumerate(iid)])
    nf = COLS[variant]
    sc = Scorer(nf, seed=seed)
    hist = sc.fit(Phi[sel][:, :, :nf], mask[sel], y[sel], epochs=epochs, seed=seed)
    return sc, hist, sc.accuracy(Phi[sel][:, :, :nf], mask[sel], y[sel]), int(sel.sum())
