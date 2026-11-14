#!/usr/bin/env python3
import argparse, json, os, sys

M_MIN, M_LO = 0.03, 0.02
REC_MIN, REC_LO, ATTAIN_MIN = 0.5, 0.3, 0.75
S1_MAX, S2_MARGIN = 0.01, 0.005
HEADROOM_MIN = 0.5


def load(d):
    return json.load(open(os.path.join(d, "metrics_a7.json")))


def material(sc):
    return sc["nominal"] >= M_MIN and sc["nominal_lo5"] >= M_LO


def practical_ok(b):
    return b["recovery"] >= REC_MIN and b["recovery_lo5"] >= REC_LO and b["attainment"] == b["attainment"] and b["attainment"] >= ATTAIN_MIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--noise2", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main_m = load(args.main)
    sc = main_m["scenarios"]["SD16"]
    b = sc["B"]
    meth = main_m["safety"]["primary_method"]
    crit = {}
    crit["M_material"] = material(sc)
    crit["M_values"] = dict(nominal=sc["nominal"], lo5=sc["nominal_lo5"])
    crit["scope_forecast_case_material"] = bool(main_m["scenarios"]["SD10"]["nominal"] >= M_MIN)
    crit["scope_forecast_values"] = dict(nominal=main_m["scenarios"]["SD10"]["nominal"], gain_prior_O=main_m["scenarios"]["SD10"]["O"].get("gain"))
    crit["P_practical"] = practical_ok(b)
    crit["P_values"] = dict(method=b["practical_method"], recovery=b["recovery"], recovery_lo5=b["recovery_lo5"], attainment=b["attainment"],
                            ceiling_recovery=b["ceiling_recovery"])
    s1 = main_m["safety"]["1.0"].get(f"O:{meth}", float("inf"))
    crit["S1_cost_at_nominal_truth_forecast_prior"] = s1 <= S1_MAX
    crit["S1_value"] = dict(prior_O=s1, prior_B=main_m["safety"]["1.0"][f"B:{meth}"], ceiling_B=main_m["safety"]["1.0"]["BA_B"],
                            ceiling_O=main_m["safety"]["1.0"]["BA_O"])
    s2 = {}
    for t in ("0.6", "1.4"):
        if t in main_m["safety"]:
            s2[t] = main_m["safety"][t]["primary_minus_nominal"]
    crit["S2_out_of_support_not_worse"] = bool(s2) and all(v <= S2_MARGIN for v in s2.values())
    crit["S2_values"] = s2
    w = {"0.8": 1 / 3, "1.0": 1 / 3, "1.2": 1 / 3}
    if f"O:{meth}" in main_m["safety"]["1.0"]:
        e_o = sum(w[t] * main_m["safety"][t][f"O:{meth}"] for t in w)
        crit["S3_overconfident_prior_not_worse"] = e_o <= sc["nominal"]
        crit["S3_values"] = dict(overconfident_prior=e_o, nominal=sc["nominal"])
    else:
        crit["S3_overconfident_prior_not_worse"] = False
    S = crit["S1_cost_at_nominal_truth_forecast_prior"] and crit["S2_out_of_support_not_worse"] and crit["S3_overconfident_prior_not_worse"]
    N = None
    if args.noise2:
        n2 = load(args.noise2)["scenarios"]["SD16"]
        N = material(n2) and practical_ok(n2["B"])
        crit["N_values"] = dict(nominal=n2["nominal"], lo5=n2["nominal_lo5"], recovery=n2["B"]["recovery"],
                                recovery_lo5=n2["B"]["recovery_lo5"], attainment=n2["B"]["attainment"])
    crit["N_survives_noise2"] = N
    M, P = crit["M_material"], crit["P_practical"]
    if not M:
        outcome = "STOP: at the Burley-level scale error (SD 16%) the loss of the best nominal planner is below the materiality bar; the dwell-model-error direction does not support a top-tier claim"
    elif N is None:
        outcome = "INCOMPLETE: noise run m=2 missing"
    elif P and S and N:
        outcome = "STRONG-PRACTICAL: a practical adaptive planner is safe, near the ceiling and the effect survives m=2; proceed to the frozen confirmatory protocol with fresh seeds"
    elif P and S and not N:
        outcome = "REGIME-LIMITED: the effect needs low idiosyncratic noise; continue only with a claim narrowed to the noise regime map"
    elif P and not S:
        outcome = "UNSAFE-ADAPTATION: add a safeguard (A7b) before anything else"
    else:
        if b["ceiling_recovery"] >= HEADROOM_MIN:
            outcome = "HEADROOM: the ceiling recovers a lot but practical planners do not reach it; run A7-L (learned adaptive scorer) next"
        else:
            outcome = "CEILING-WEAK: neither practical planners nor the ceiling recover enough; tolerance analysis only, treat as moderate"
    res = dict(criteria=crit, outcome=outcome)
    print("=== A7 outcome:", outcome, "===")
    print(json.dumps(crit, indent=2))
    json.dump(res, open(args.out or os.path.join(args.main, "decision_a7.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
