# de-dwell-sched

Stylized dwell-based directed-energy scheduling model (M0), exact dynamic programme, simple-rule baselines and the P0 regret grid.
Research Project 2: when does learning beat strong index and optimisation rules for dwell-based directed-energy scheduling?

## Layout
    src/dwell/m0.py         model, belief tables, transitions, exact policy evaluation
    src/dwell/dp.py         exact DP (memoised expectimax)
    src/dwell/rules.py      FAIR, EDF, LLF, SE, GMR, BI
    src/dwell/oracle.py     class-aware upper bound (not a policy)
    src/dwell/instances.py  instance generator and regime variables
    scripts/run_p0.py       resumable grid runner (invariants checked, fails loudly)
    scripts/analyze_p0.py   go-rule analysis, figure, decision.json
    scripts/env_report.py   software/hardware facts to log
    tests/                  DP vs independent brute force, oracle bound, hand-checked cases
    configs/                run configurations (seeds, grids)
    docs/                   analysis plan, literature export, record, changelog
    results/                run outputs (raw cell CSVs are git-ignored; archive them, see docs/record.md)

## Quick start
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    PYTHONPATH=src python -m pytest -q tests               # Windows PowerShell: $env:PYTHONPATH="src"; python -m pytest -q tests
    python scripts/run_p0.py --config configs/p0_main.json --out results/p0_main --workers 4
    python scripts/analyze_p0.py --config configs/p0_main.json --dir results/p0_main

## Conventions
- Commit messages: `<type>(<scope>): <summary>` with type in feat, fix, test, docs, data, plan, refactor, chore and scope in m0, dp, rules, run, analysis, plan, repo. Body: what and why; end with provenance tags where facts are stated.
- Tags: `p0-plan-v1` freezes the plan before the first run; `p0-main-run1` marks the commit used for the main run; later steps use `s2-...`.
- Provenance tags in docs: [CODE] from code or tests, [DOC] from a document read, [MEM] unverified memory, [OPEN] unverified or inference.
- Never edit a frozen plan: add a dated amendment instead.

## Amendment A7 files
    src/dwell/adapt.py        class sets, robust Bayes-adaptive model, practical adaptive planners, exact recursive evaluator
    scripts/run_a7.py         resumable runner (one CSV per cell; invariants checked, fails loudly)
    scripts/analyze_a7.py     pooled regrets, scale-error mixtures, recovery, safety, paired bootstrap; writes metrics_a7.json
    scripts/decide_a7.py      applies the pre-written decision table of A7.3 to a7_main and a7_noise2
    scripts/burley_stats.py   relative dwell error from docs/burley_tables_transcribed.csv
    configs/a7_*.json         run configurations (fresh master seeds)
