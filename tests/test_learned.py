import numpy as np
import pytest

from test_dp_bruteforce import random_instance
from dwell.learned import N_FEAT, Scorer, features, pad
from dwell.m2 import Model2, evaluate_policy2


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    sc = Scorer(hidden=5, seed=1)
    S, A = 7, 4
    Phi = rng.normal(size=(S, A, N_FEAT))
    mask = rng.random((S, A)) > 0.2
    mask[:, 0] = True
    y = np.array([int(rng.choice(np.nonzero(mask[i])[0])) for i in range(S)])
    loss, g = sc.loss_and_grads(Phi, mask, y)
    for k in ("W1", "b1", "w2"):
        flat = sc.p[k].reshape(-1)
        for idx in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + 1e-6
            lp, _ = sc.loss_and_grads(Phi, mask, y)
            flat[idx] = old - 1e-6
            lm, _ = sc.loss_and_grads(Phi, mask, y)
            flat[idx] = old
            num = (lp - lm) / 2e-6
            assert num == pytest.approx(g[k].reshape(-1)[idx], rel=1e-4, abs=1e-7)


def test_fit_reduces_loss_and_policy_returns_useful_actions():
    rng = np.random.default_rng(2)
    samples = []
    for _ in range(300):
        m = int(rng.integers(2, 5))
        f = rng.normal(size=(m, N_FEAT))
        samples.append((f, int(np.argmax(f[:, 7]))))
    Phi, mask, y = pad(samples)
    sc = Scorer(hidden=16, seed=3)
    hist = sc.fit(Phi, mask, y, epochs=40, lr=0.02)
    assert hist[-1] < 0.6 * hist[0]
    inst = random_instance(np.random.default_rng(5), 4, (2, 5), 2, dmax_cap=12)
    m2 = Model2(inst)
    pol = sc.policy(m2)
    st = m2.initial_state()
    j = pol(st)
    assert j is None or j in m2.actions(st)
    assert 0.0 <= evaluate_policy2(m2, pol) <= 3.0 * 4


def test_features_shape_and_finite():
    inst = random_instance(np.random.default_rng(6), 5, (1, 8), 2, dmax_cap=14)
    m2 = Model2(inst)
    st = m2.initial_state()
    acts = m2.actions(st)
    if acts:
        f = features(m2, st, acts, max(m2.d))
        assert f.shape == (len(acts), N_FEAT) and np.isfinite(f).all()
