"""Radius/units controls on two real objectives; each radius gets a new reference.

The stopping tolerance for a*f is a*epsilon. Scaling is a paired control,
not an independent instance. No reference value is passed to the solvers.
"""

import argparse
import csv
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
from fw import solve
from problems import make_bike_poisson, make_breast_cancer


ROOT = Path(__file__).resolve().parent
METHODS = ("FW_BT_fixed", "FW_BT_certificate", "AFW_BT_fixed", "AFW_BT_certificate")
RADII = {"breast_cancer": 2.0, "bike_poisson": 1.5}
ORACLES = ("function_evals", "gradient_evals", "lmo_evals")
BUDGETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)


def scale_problem(problem, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Need a finite positive objective scale")
    # H1 is unchanged: a(H0+H1*delta) = a*H0+H1*(a*delta).
    return replace(problem,
                   objective=lambda x: scale*problem.objective(x),
                   gradient=lambda x: scale*problem.gradient(x),
                   hessian=lambda x: scale*problem.hessian(x),
                   h0=scale*problem.h0, global_l=scale*problem.global_l,
                   known_lower=None if problem.known_lower is None else scale*problem.known_lower,
                   reference_lower=scale*problem.reference_lower,
                   reference_upper=scale*problem.reference_upper,
                   metadata={**problem.metadata, "mu": scale*problem.metadata["mu"],
                             "objective_scale": scale})


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def audit_run(result, problem, method, tolerance):
    """Check every accepted source, including sources omitted from trace.csv."""
    rows, scale = result.trace, problem.metadata["objective_scale"]
    diameter = 2*problem.radius
    previous = 1.0 if method.endswith("fixed") else rows[0]["certificate"]/diameter**2
    initial, trials = previous, 0
    slack = 2e-10*scale
    for row in rows:
        assert row["l1_norm"] <= problem.radius+1e-11
        assert row["lower_bound"] <= problem.reference_upper+slack
        assert row["certificate"]+slack >= max(0, row["objective"]-problem.reference_upper)
    for row, following in zip(rows, rows[1:]):
        current = row["trial_curvature"]
        count = row["line_search_trials"]
        assert math.isclose(math.log2(current/previous), count-2, abs_tol=1e-9)
        assert current <= max(previous/2, 2*(problem.h0+problem.h1*row["certificate"]))*(1+1e-10)
        assert following["certificate"] <= row["certificate"]+slack
        gap = row.get("directional_gap", row["fw_gap"])
        assert following["objective"] <= row["objective"]-0.5*row["step"]*gap+slack
        previous, trials = current, trials+count
    last = rows[-1]
    n = last["iteration"]
    assert last["function_evals"] == trials+1
    assert last["gradient_evals"] == last["lmo_evals"] == n+1
    if n:
        assert math.isclose(trials, 2*n+math.log2(previous/initial), abs_tol=1e-8)
    if method.startswith("AFW"):
        assert last["numerical_prunes"] == 0
        assert last["away_scans"] == n
        assert last["drop_steps"] <= last["fw_steps"]+last["initial_active_size"]-1
    overhead = math.ceil(math.log2(max(1, 2*(problem.h0+problem.h1*rows[0]["certificate"])/initial))) if initial > 0 else 0
    bound = 1+2*n+overhead
    assert last["function_evals"] <= bound
    return {"updates": n, "trials": trials, "function_bound": bound,
            "initial_estimate": initial, "final_estimate": previous,
            "function_evals": last["function_evals"],
            "certified": last["certificate"] <= tolerance}


def run_method(problem, method, max_iter, tolerance):
    certificate = method.endswith("certificate")
    if method.startswith("AFW"):
        return solve_away(problem, max_iter=max_iter, tolerance=tolerance,
                          initial_estimate="certificate" if certificate else 1.0)
    return solve(problem, "adaptive_L_initial" if certificate else "adaptive_L", max_iter, tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--radius-factors", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--scales", type=float, nargs="+", default=[1e-3, 1.0, 1e3])
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=ROOT/"robustness_results")
    args = parser.parse_args()
    if args.seeds < 1 or args.max_iter < 1 or args.tolerance <= 0:
        parser.error("Need positive seeds, max-iter and tolerance")
    for values in (args.radius_factors, args.scales):
        if len(set(values)) != len(values) or any(not math.isfinite(v) or v <= 0 for v in values):
            parser.error("Controls must be distinct finite positive numbers")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    summary, sampled, budgets, audits, problems = [], [], [], [], []
    for scenario, base_radius in RADII.items():
        maker = make_breast_cancer if scenario == "breast_cancer" else make_bike_poisson
        for factor in args.radius_factors:
            for seed in range(args.seeds):
                base = maker(seed, radius=base_radius*factor)
                problems.append({"scenario": scenario, "radius_factor": factor, "seed": seed,
                                 "radius": base.radius, "H0": base.h0, "H1": base.h1,
                                 "global_L": base.global_l,
                                 "reference_lower": base.reference_lower,
                                 "reference_upper": base.reference_upper,
                                 "reference_l1_norm": float(np.linalg.norm(base.reference_x, 1)),
                                 "reference_gradient_norm": float(np.linalg.norm(base.gradient(base.reference_x))),
                                 **base.metadata})
                for scale in args.scales:
                    p = scale_problem(base, scale)
                    tolerance = args.tolerance*scale
                    for method in METHODS:
                        result = run_method(p, method, args.max_iter, tolerance)
                        identity = {"scenario": scenario, "radius_factor": factor,
                                    "seed": seed, "scale": scale, "method": method}
                        last = result.trace[-1]
                        summary.append({**identity, **last, "normalized_certificate": last["certificate"]/scale,
                                        "certified": last["certificate"] <= tolerance,
                                        "normalized_error_lower": max(0, last["objective"]-p.reference_upper)/scale,
                                        "normalized_error_upper": max(0, last["objective"]-p.reference_lower)/scale})
                        sampled.extend({**identity, **row} for row in result.trace
                                       if row["iteration"] % 10 == 0 or row is last)
                        audits.append({**identity, **audit_run(result, p, method, tolerance)})
                        for oracle in ORACLES:
                            counts = [row[oracle] for row in result.trace]
                            for b in BUDGETS:
                                index = int(np.searchsorted(counts, b, side="right"))-1
                                assert index >= 0
                                row = result.trace[index]
                                budgets.append({**identity, "oracle": oracle, "budget": b,
                                                "iteration": row["iteration"],
                                                "certificate": row["certificate"],
                                                "normalized_certificate": row["certificate"]/scale,
                                                "objective": row["objective"], "lower_bound": row["lower_bound"],
                                                "certified": row["certificate"] <= tolerance,
                                                "away_scans": row.get("away_scans", 0),
                                                "active_atom_scores": row.get("active_atom_scores", 0),
                                                **{key: row[key] for key in ORACLES}})
                print(f"{scenario} radius={base.radius:g} seed={seed} complete", flush=True)
    names = ("summary.csv", "trace.csv", "budget.csv", "audit.csv")
    for name, rows in zip(names, (summary, sampled, budgets, audits)):
        write_csv(args.output/name, rows)
    source = args.output/"source"
    source.mkdir()
    sources = ("robustness_benchmark.py", "away_fw.py", "fw.py", "problems.py", "auto_conditioned.py")
    for name in sources:
        shutil.copy2(ROOT/name, source/name)
    import scipy, sklearn
    manifest = {"study": "radius_and_units", "date": "2026-10-15", "command": [sys.executable, *sys.argv],
                "scenarios": list(RADII), "methods": METHODS, "seeds": list(range(args.seeds)),
                "radius_factors": args.radius_factors, "objective_scales": args.scales,
                "max_iterations": args.max_iter, "base_tolerance": args.tolerance,
                "budgets": BUDGETS, "trace_stride": 10,
                "initialization": "same original vertex at each radius; fixed M=1 or certificate M=U0/(2R)^2",
                "stopping": "U <= objective_scale * base_tolerance; all objective errors reported also in base units",
                "information": "no H0,H1,mu,reference inputs; certificate initialization reuses first paid f,g,LMO",
                "dependence": "objective scales are paired controls, not independent instances; Bike seeds change only start",
                "python": sys.version, "platform": platform.platform(),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
                "thread_environment": {k: os.environ.get(k) for k in
                                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "problems": problems, "source_archive": "source/",
                "source_sha256": {n: hashlib.sha256((source/n).read_bytes()).hexdigest() for n in sources},
                "result_sha256": {n: hashlib.sha256((args.output/n).read_bytes()).hexdigest() for n in names}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} paired runs; every full trajectory audited")


if __name__ == "__main__":
    main()
