"""Tests of the A15 analysis logic (plan A15.7, items 7 to 9): exact Wilcoxon, Holm, informative cells, contrasts, decision table."""
import numpy as np
import pandas as pd

from dwell.analysis_a15 import (CONTRASTS, MARGIN, VARIANTS, contrasts, decide, holm, informative, n_seeds, seed_values,
                                wilcoxon_greater)


def test_wilcoxon_exact_values():
    assert abs(wilcoxon_greater([1.0] * 10) - 1 / 1024) < 1e-12
    assert abs(wilcoxon_greater([1, 2, -3]) - 5 / 8) < 1e-12
    assert abs(wilcoxon_greater([-1.0] * 10) - 1.0) < 1e-12
    assert abs(wilcoxon_greater([1, 2, 3, 0, 0]) - 1 / 8) < 1e-12              # zeros are dropped
    assert abs(wilcoxon_greater([1, -1]) - 0.75) < 1e-12                        # ranks 1.5, 1.5: W+ = 1.5; patterns >= 1.5: 3 of 4


def test_holm():
    assert holm([0.001, 0.04, 0.03]) == [True, False, False]
    assert holm([0.01, 0.02, 0.02]) == [True, True, True]
    assert holm([0.2, 0.3]) == [False, False]
    assert holm([0.06, 0.001]) == [False, True]                                 # order of the input is kept


def frame(deltas, planner=0.5, n_rows=30, ceps=0.5, rho=1.1):
    """deltas: dict variant -> seed-level offsets (list) relative to the planner value."""
    d = {"rho": rho, "s": 2, "a": 0.5, "lam": 0.25, "U_RU_CEP": planner, "U_CEPS": ceps, "U_opt": 1.0}
    d = {k: (v if k in ("rho", "s", "a", "lam") else np.full(n_rows, v)) for k, v in d.items()}
    for v in VARIANTS:
        for k, dl in enumerate(deltas.get(v, [0.0] * 10)):
            d[f"U_{v}_s{k}"] = np.full(n_rows, planner + dl)
    return pd.DataFrame(d)


def test_informative_range_and_seed_values():
    lo, ok, hi = frame({}, planner=0.05), frame({}, planner=0.5), frame({}, planner=0.995)
    assert len(informative(lo)) == 0 and len(informative(ok)) == 30 and len(informative(hi)) == 0
    assert n_seeds(ok) == 10 and len(seed_values(ok, "IL_A")) == 10


def dfs(ad_vs_rucep=0.0, ad_vs_ceps=0.0, a_vs_ceps=0.0, a_vs_g=0.0, g_vs_ev=0.0, g_vs_noacc=0.0, ceps_gap=0.0):
    """Synthetic frames for n = 5, 7, 9. The planner CEPS has value 0.5 - ceps_gap, RU_CEP 0.5; the variants are set from the deltas."""
    out = {}
    for n in (5, 7, 9):
        ceps = 0.5 - ceps_gap
        ad = [0.0 + ad_vs_rucep + 0.0005 * (k % 3) for k in range(10)]
        a = [ceps + a_vs_ceps - 0.5 + 0.0005 * (k % 3) for k in range(10)]
        g = [x - a_vs_g for x in a]
        ev = [x - g_vs_ev for x in g]
        noacc = [x - g_vs_noacc for x in g]
        d = frame(dict(IL_AD=[ceps - 0.5 + ad_vs_ceps + 0.0005 * (k % 3) if ad_vs_ceps else ad[k] for k in range(10)], IL_A=a, IL_G=g,
                       IL_EV=ev, IL_G_noacc=noacc), ceps=ceps)
        out[n] = d
    return out


def test_eight_contrasts_and_the_unit_is_the_seed():
    cs = contrasts(dfs(ad_vs_rucep=0.02, ad_vs_ceps=0.0))
    assert len(CONTRASTS) == 8 and len(cs) == 8
    c = [x for x in cs if (x["a"], x["b"], x["n"]) == ("IL_AD", "RU_CEP", 9)][0]
    assert abs(c["mean_delta"] - 0.02) < 1e-3 and c["holm_reject"] and c["material"] and c["win"]
    assert abs(c["p"] - 1 / 1024) < 1e-9                                        # ten seeds, all above: the instances do not enter


def test_not_material_is_not_a_win():
    cs = contrasts(dfs(ad_vs_rucep=0.004))
    c = [x for x in cs if (x["a"], x["b"], x["n"]) == ("IL_AD", "RU_CEP", 7)][0]
    assert c["holm_reject"] and not c["material"] and not c["win"]


def test_decision_table():
    assert decide(contrasts(dfs(ad_vs_rucep=0.02)), 50.0)["outcome"] == "LEARN-BEATS"
    assert decide(contrasts(dfs(ad_vs_rucep=0.0)), 50.0)["outcome"] == "LEARN-MATCHES"
    assert decide(contrasts(dfs(ad_vs_rucep=0.0)), 5.0)["outcome"] == "LEARN-BEHIND"          # not fast enough
    assert decide(contrasts(dfs(ad_vs_rucep=0.0)), None)["outcome"] == "NO OUTCOME"
    assert decide(contrasts(dfs(ad_vs_rucep=-0.02)), 50.0)["outcome"] == "LEARN-BEHIND"
    assert decide(contrasts(dfs(ad_vs_rucep=-0.5 * MARGIN)), 50.0)["outcome"] == "LEARN-MATCHES"   # within one point
    one = dfs(ad_vs_rucep=0.02)
    one[9] = dfs(ad_vs_rucep=0.0)[9]
    assert decide(contrasts(one), 50.0)["outcome"] == "LEARN-MATCHES"                           # a win at n = 7 only is no BEATS


def test_secondary_flags():
    d = decide(contrasts(dfs(a_vs_g=0.03, g_vs_ev=0.03, g_vs_noacc=0.03)), 50.0)
    assert d["planner_feature_helps"] and d["objective_matters"] and d["accumulated_value_needed"]
    d0 = decide(contrasts(dfs()), 50.0)
    assert not (d0["planner_feature_helps"] or d0["objective_matters"] or d0["accumulated_value_needed"] or d0["improves_cheap_planner"])


def test_undefined_size_gives_no_outcome():
    dd = dfs(ad_vs_rucep=0.02)
    dd[7] = dd[7].iloc[0:0]
    assert decide(contrasts(dd), 50.0)["outcome"] == "NO OUTCOME"


def test_improves_cheap_planner_needs_all_three_contrasts():
    assert decide(contrasts(dfs(a_vs_ceps=0.03, ad_vs_ceps=0.03)), 50.0)["improves_cheap_planner"]
    assert not decide(contrasts(dfs(a_vs_ceps=0.03, ad_vs_ceps=0.0)), 50.0)["improves_cheap_planner"]
    assert not decide(contrasts(dfs(a_vs_ceps=0.0, ad_vs_ceps=0.03)), 50.0)["improves_cheap_planner"]


def test_missing_size_gives_no_outcome():
    dd = dfs(ad_vs_rucep=0.02)
    del dd[9]
    assert decide(contrasts(dd), 50.0)["outcome"] == "NO OUTCOME"
