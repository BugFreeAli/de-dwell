import numpy as np

N_FEAT = 12


def features(model, state, acts, dmax):
    t, cur, prog = state
    rows = []
    for j in acts:
        post = model.post[j][prog[j]]
        sl = model.slew(cur, j)
        cap = model.d[j] - t - sl
        e_rem = sum(p * r for p, r in post)
        p_succ = sum(p for p, r in post if r <= cap)
        spent = sum(p * min(r, cap) for p, r in post)
        gmr = model.v[j] * p_succ / (sl + spent)
        r_min = min(r for _, r in post)
        r_max = max(r for _, r in post)
        rows.append([model.v[j] / 3.0, cap / 10.0, e_rem / 10.0, p_succ, (cap - e_rem) / 10.0, sl / 5.0,
                     1.0 if j == cur else 0.0, gmr, r_min / 10.0, r_max / 10.0, (dmax - t) / 20.0, len(acts) / 5.0])
    return np.array(rows, dtype=float)


class Scorer:
    def __init__(self, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        self.p = dict(W1=rng.normal(0, 0.5, (N_FEAT, hidden)), b1=np.zeros(hidden), w2=rng.normal(0, 0.5, hidden))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step = 0

    def scores(self, phi):
        h = np.tanh(phi @ self.p["W1"] + self.p["b1"])
        return h @ self.p["w2"]

    def loss_and_grads(self, Phi, mask, y):
        W1, b1, w2 = self.p["W1"], self.p["b1"], self.p["w2"]
        H = np.tanh(Phi @ W1 + b1)
        s = H @ w2
        s = np.where(mask, s, -1e9)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s) * mask
        P = e / e.sum(axis=1, keepdims=True)
        S = Phi.shape[0]
        loss = -np.log(P[np.arange(S), y] + 1e-12).mean()
        G = P.copy()
        G[np.arange(S), y] -= 1.0
        G = G * mask / S
        dw2 = np.einsum("sa,sah->h", G, H)
        dH = G[:, :, None] * w2[None, None, :]
        dZ = dH * (1.0 - H ** 2)
        dW1 = np.einsum("saf,sah->fh", Phi, dZ)
        db1 = dZ.sum(axis=(0, 1))
        return loss, dict(W1=dW1, b1=db1, w2=dw2)

    def adam(self, grads, lr=0.01, b1=0.9, b2=0.999):
        self.step += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * grads[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * grads[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.step)
            vh = self.v[k] / (1 - b2 ** self.step)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def fit(self, Phi, mask, y, epochs=60, batch=512, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        S = Phi.shape[0]
        hist = []
        for ep in range(epochs):
            order = rng.permutation(S)
            tot = 0.0
            for i in range(0, S, batch):
                b = order[i:i + batch]
                loss, g = self.loss_and_grads(Phi[b], mask[b], y[b])
                self.adam(g, lr=lr)
                tot += loss * len(b)
            hist.append(tot / S)
        return hist

    def policy(self, model):
        dmax = max(model.d)

        def act(state):
            acts = model.actions(state)
            if not acts:
                return None
            sc = self.scores(features(model, state, acts, dmax))
            return acts[int(np.argmax(sc))]

        return act


def pad(samples):
    A = max(f.shape[0] for f, _ in samples)
    S = len(samples)
    Phi = np.zeros((S, A, N_FEAT))
    mask = np.zeros((S, A), dtype=bool)
    y = np.zeros(S, dtype=int)
    for i, (feat, label) in enumerate(samples):
        m = feat.shape[0]
        Phi[i, :m] = feat
        mask[i, :m] = True
        y[i] = label
    return Phi, mask, y
