#!/usr/bin/env python3
import argparse, itertools, json, os, platform, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from dwell.dp import solve
from dwell.instances import make_instance_k
from dwell.learned import Scorer, features, pad
from dwell.m2 import Model2, evaluate_policy2
from dwell.rules import make_rule

TOL = 1e-9
REF = ("CEP", "ROLL_CEP", "ROLL_GMR")


def build(cfg, seed, idx, n, rho, s, a):
    inst, real = make_instance_k(seed, idx, n, rho, s, a, D=tuple(cfg["D"]), value_set=tuple(cfg["value_set"]), u_low=cfg["u_low"])
    return Model2(inst, 1.0), real


def collect(cfg, scorer, beta, rng, n_inst, seed_base):
    samples = []
    for i in range(n_inst):
        rho, s, a = (cfg["train_rho"][rng.integers(len(cfg["train_rho"]))], cfg["train_s"][rng.integers(len(cfg["train_s"]))],
                     cfg["train_a"][rng.integers(len(cfg["train_a"]))])
        model, _ = build(cfg, cfg["train_seed"], seed_base + i, cfg["n_train"], rho, s, a)
        teacher = solve(model).policy()
        learner = scorer.policy(model) if scorer is not None else None
        dmax = max(model.d)
        for _ in range(cfg["worlds_per_instance"]):
            c = [int(rng.choice(model.K, p=model.prior[k])) for k in range(model.n)]
            state = model.initial_state()
            while True:
                acts = model.actions(state)
                if not acts:
                    break
                tj = teacher(state)
                samples.append((features(model, state, acts, dmax), acts.index(tj)))
                j = learner(state) if (learner is not None and rng.random() > beta) else tj
                x = state[2][j] // model.K
                _, state = model.transition(state, j, x + 1 == model.D[c[j]])
    return samples


def train(cfg):
    rng = np.random.default_rng(cfg["train_seed"])
    scorer = Scorer(hidden=cfg["hidden"], seed=cfg["train_seed"])
    data = []
    log = []
    for it in range(cfg["dagger_iterations"]):
        beta = 1.0 if it == 0 else cfg["beta"]
        new = collect(cfg, None if it == 0 else scorer, beta, rng, cfg["train_instances"], 1000 * it)
        data.extend(new)
        Phi, mask, y = pad(data)
        hist = scorer.fit(Phi, mask, y, epochs=cfg["epochs"], lr=cfg["lr"], seed=it)
        acc = float((np.argmax(np.where(mask, np.einsum("sah,h->sa", np.tanh(Phi @ scorer.p["W1"] + scorer.p["b1"]), scorer.p["w2"]), -1e9), axis=1) == y).mean())
        log.append(dict(iteration=it, samples=len(data), loss_first=hist[0], loss_last=hist[-1], teacher_agreement=acc))
        print(f"DAgger {it}: {len(data)} samples, loss {hist[0]:.3f} -> {hist[-1]:.3f}, agreement {acc:.3f}", flush=True)
    return scorer, log


def decision_ms(model, policy_factory):
    pol = policy_factory(model)
    c = [int(np.argmax(model.prior[k])) for k in range(model.n)]
    state = model.initial_state()
    times = []
    while True:
        t0 = time.perf_counter()
        j = pol(state)
        times.append(1000 * (time.perf_counter() - t0))
        if j is None:
            break
        x = state[2][j] // model.K
        _, state = model.transition(state, j, x + 1 == model.D[c[j]])
    return float(np.mean(times[:-1])) if len(times) > 1 else 0.0


def eval_cell(job):
    cfg, weights, name, n, rho, s, a, N, with_dp, seed = job
    scorer = Scorer(hidden=cfg["hidden"])
    scorer.p = {k: weights[k] for k in weights}
    rows = []
    for idx in range(N):
        model, real = build(cfg, seed, idx, n, rho, s, a)
        row = dict(test=name, n=n, rho=rho, s=s, a=a, idx=idx, **real)
        if with_dp:
            res = solve(model)
            row["V_dp"] = res.value
        factories = dict(CEP=lambda m: make_rule(m, "CEP"), ROLL_CEP=lambda m: make_rule(m, "ROLL_CEP"),
                         ROLL_GMR=lambda m: make_rule(m, "ROLL_GMR"), LEARNED=scorer.policy)
        for k, f in factories.items():
            row["V_" + k] = evaluate_policy2(model, f(model))
            row["ms_" + k] = decision_ms(model, f)
            if with_dp:
                assert -TOL <= row["V_" + k] <= row["V_dp"] + TOL, f"{k} outside [0, V_dp]: {row['V_'+k]} vs {row['V_dp']}"
        rows.append(row)
    return rows


def boot_ci(num, den, rng, B=2000):
    idx = rng.integers(0, len(num), size=(B, len(num)))
    return np.percentile(num[idx].sum(1) / den[idx].sum(1), [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    scorer, log = train(cfg)
    np.savez(os.path.join(args.out, "scorer.npz"), **scorer.p)
    json.dump(dict(config=cfg, train_log=log, python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                   platform=platform.platform(), cpu_count=os.cpu_count(), train_seconds=time.time() - t0),
              open(os.path.join(args.out, "run_meta.json"), "w"), indent=2)
    jobs = []
    for name, spec in cfg["tests"].items():
        for rho, s, a in itertools.product(spec["rho"], spec["s"], spec["a"]):
            jobs.append((cfg, scorer.p, name, spec["n"], rho, s, a, spec["N"], spec["with_dp"], cfg["test_seed"]))
    rows = []
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(eval_cell, jobs), 1):
            rows.extend(r)
            print(f"[{k}/{len(jobs)}] cells evaluated, elapsed {time.time()-t0:7.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "eval_instances.csv"), index=False)
    rng = np.random.default_rng(20261113)
    summary = {}
    for name, g in df.groupby("test"):
        entry = dict(n_instances=int(len(g)))
        if "V_dp" in g.columns and g["V_dp"].notna().all():
            dp = g["V_dp"].to_numpy()
            for k in ("CEP", "ROLL_CEP", "ROLL_GMR", "LEARNED"):
                v = g["V_" + k].to_numpy()
                lo, hi = boot_ci(dp - v, dp, rng)
                entry[f"regret_{k}"] = float(1 - v.sum() / dp.sum())
                entry[f"regret_{k}_ci"] = [float(lo), float(hi)]
            d = g["V_ROLL_CEP"].to_numpy() - g["V_LEARNED"].to_numpy()
            lo, hi = boot_ci(d, dp, rng)
            entry["regret_LEARNED_minus_ROLL_CEP"] = float(d.sum() / dp.sum())
            entry["regret_LEARNED_minus_ROLL_CEP_ci"] = [float(lo), float(hi)]
        else:
            ref = g["V_ROLL_CEP"].to_numpy()
            for k in ("CEP", "ROLL_GMR", "LEARNED"):
                v = g["V_" + k].to_numpy()
                lo, hi = boot_ci(v - ref, ref, rng)
                entry[f"value_{k}_minus_ROLL_CEP_rel"] = float((v - ref).sum() / ref.sum())
                entry[f"value_{k}_minus_ROLL_CEP_rel_ci"] = [float(lo), float(hi)]
        for k in ("CEP", "ROLL_CEP", "ROLL_GMR", "LEARNED"):
            entry[f"ms_per_decision_{k}"] = float(g["ms_" + k].mean())
        summary[name] = entry
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("total seconds", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
