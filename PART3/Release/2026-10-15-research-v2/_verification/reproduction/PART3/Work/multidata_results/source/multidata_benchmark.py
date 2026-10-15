"""Two additional real count datasets, all l1 vertices, two constraint regimes.

Six existing FW/AFW variants; no repeated random starts. A new output directory
is required. Complete prefixes and audits precede trace sampling.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

from away_fw import solve_away
from certificate_bounds import gap_envelope
from count_problems import DATASETS, make_count_problem, unconstrained_reference
from fw import solve
from robustness_benchmark import ORACLES, audit_run, run_method, write_csv


ROOT = Path(__file__).resolve().parent
METHODS = ("FW_BT_fixed", "FW_BT_certificate", "AFW_BT_fixed", "AFW_BT_certificate", "FW_Q", "AFW_Q")
RATIOS = (0.5, 2.0)
BUDGETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)


def vertex(start, dimension, radius):
    if not isinstance(start, int) or not 0 <= start < 2*dimension:
        raise ValueError("Invalid signed coordinate atom")
    x = np.zeros(dimension)
    x[start//2] = radius*(1 if start % 2 == 0 else -1)
    return x


def run(problem, method, max_iter, tolerance):
    if method not in METHODS:
        raise ValueError(method)
    if method == "FW_Q":
        return solve(problem, "quadratic_memory", max_iter, tolerance)
    if method == "AFW_Q":
        return solve_away(problem, "Q", max_iter, tolerance)
    return run_method(problem, method, max_iter, tolerance)


def audit(result, problem, method, tolerance):
    rows, last = result.trace, result.trace[-1]
    if method.endswith("Q"):
        assert last["function_evals"] == last["gradient_evals"] == last["lmo_evals"] == len(rows)
        checks = {"updates": last["iteration"], "function_evals": last["function_evals"],
                  "function_bound": len(rows)}
    else:
        checks = audit_run(result, problem, method, tolerance)
    max_ratio = 0.0
    for row in rows:
        delta_upper = max(0, row["objective"]-problem.reference_lower)
        envelope = gap_envelope(delta_upper, problem.h0, problem.h1, 2*problem.radius)
        assert row["fw_gap"] <= envelope+2e-10
        assert row["l1_norm"] <= problem.radius+1e-11
        assert row["lower_bound"] <= problem.reference_upper+2e-10
        if envelope:
            max_ratio = max(max_ratio, row["fw_gap"]/envelope)
    for row, following in zip(rows, rows[1:]):
        gap = row.get("directional_gap", row["fw_gap"])
        assert following["objective"] <= row["objective"]-0.5*row["step"]*gap+2e-10
        assert following["certificate"] <= row["certificate"]+2e-10
    if method.startswith("AFW"):
        assert last["numerical_prunes"] == 0
        assert last["away_scans"] == last["iteration"]
        assert last["drop_steps"] <= last["fw_steps"]+last["initial_active_size"]-1
    return {**checks, "max_gap_envelope_ratio": max_ratio}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=ROOT/"multidata_results")
    args = parser.parse_args()
    if args.max_iter < 1 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("Need positive iteration limit and finite positive tolerance")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    summary, sampled, budgets, audits, problems = [], [], [], [], []
    for dataset in DATASETS:
        unc, residual = unconstrained_reference(dataset)
        critical_radius = float(np.linalg.norm(unc, 1))
        for ratio in RATIOS:
            p = make_count_problem(dataset, ratio*critical_radius)
            radius_error = math.sqrt(unc.size)*residual/p.metadata["mu"]
            regime = "active" if p.radius < critical_radius-radius_error else "interior" if p.radius > critical_radius+radius_error else "transition"
            problems.append({"dataset": dataset, "radius_ratio": ratio, "radius": p.radius,
                             "critical_radius": critical_radius, "critical_residual_radius_estimate": radius_error,
                             "regime": regime, "H0": p.h0, "H1": p.h1, "global_L": p.global_l,
                             "reference_lower": p.reference_lower, "reference_upper": p.reference_upper,
                             "estimated_multiplier": float(np.max(np.abs(p.gradient(p.reference_x)))) if regime == "active" else 0,
                             **p.metadata})
            for start in range(2*p.x0.size):
                problem = replace(p, seed=start, x0=vertex(start, p.x0.size, p.radius))
                for method in METHODS:
                    result = run(problem, method, args.max_iter, args.tolerance)
                    identity = {"dataset": dataset, "radius_ratio": ratio, "start": start, "method": method}
                    last = result.trace[-1]
                    summary.append({**identity, **last, "certified": last["certificate"] <= args.tolerance,
                                    "error_lower": max(0, last["objective"]-p.reference_upper),
                                    "error_upper": max(0, last["objective"]-p.reference_lower)})
                    sampled.extend({**identity, **row} for row in result.trace if row["iteration"] % 10 == 0 or row is last)
                    audits.append({**identity, **audit(result, problem, method, args.tolerance)})
                    for oracle in ORACLES:
                        counts = [row[oracle] for row in result.trace]
                        for b in BUDGETS:
                            index = int(np.searchsorted(counts, b, side="right"))-1
                            assert index >= 0
                            row = result.trace[index]
                            budgets.append({**identity, "oracle": oracle, "budget": b,
                                            "iteration": row["iteration"], "certificate": row["certificate"],
                                            "objective": row["objective"], "lower_bound": row["lower_bound"],
                                            "certified": row["certificate"] <= args.tolerance,
                                            "budget_reached_or_certified": last[oracle] >= b or last["certificate"] <= args.tolerance,
                                            **{key: row[key] for key in ORACLES}})
                if start % 8 == 7 or start == 2*p.x0.size-1:
                    print(f"{dataset} ratio={ratio:g}: {start+1}/{2*p.x0.size} vertices complete", flush=True)
    names = ("summary.csv", "trace.csv", "budget.csv", "audit.csv")
    for name, rows in zip(names, (summary, sampled, budgets, audits)):
        write_csv(args.output/name, rows)
    sources = ("multidata_benchmark.py", "count_problems.py", "fetch_count_data.py", "robustness_benchmark.py",
               "away_fw.py", "fw.py", "problems.py", "auto_conditioned.py", "certificate_bounds.py")
    source = args.output/"source"
    source.mkdir()
    for name in sources:
        shutil.copy2(ROOT/name, source/name)
    import scipy
    manifest = {"study": "additional_count_data", "date": "2026-10-15", "command": [sys.executable, *sys.argv],
                "datasets": DATASETS, "methods": METHODS, "radius_ratios": RATIOS,
                "max_iterations": args.max_iter, "tolerance": args.tolerance, "budgets": BUDGETS, "trace_stride": 10,
                "start_protocol": "all 2d signed coordinate vertices, no random sampling or duplicate starts",
                "information": "BT gets f,g,R,x0 and optionally D; Q also gets analytic H0,H1; references offline only",
                "dependence": "two additional datasets; starts/radii share training data; no independent-draw or predictive claim",
                "active_work": "one extra active-set scan per AFW update, atom scores recorded separately; no wall-time ranking",
                "python": sys.version, "platform": platform.platform(),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
                "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "problems": problems, "source_archive": "source/",
                "source_sha256": {n: hashlib.sha256((source/n).read_bytes()).hexdigest() for n in sources},
                "result_sha256": {n: hashlib.sha256((args.output/n).read_bytes()).hexdigest() for n in names}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} runs, {len(budgets)} oracle prefixes, {len(audits)} full audits")


if __name__ == "__main__":
    main()
