"""Poisson radius sweep across an independently estimated activity threshold.

One tight-tolerance trajectory yields prefixes for several accuracies. These
prefixes and the different initial vertices are not independent datasets.
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

from certificate_bounds import certificate_threshold, gap_envelope
from problems import make_bike_poisson
from robustness_benchmark import METHODS, ORACLES, audit_run, run_method, write_csv


ROOT = Path(__file__).resolve().parent
RATIOS = (0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0)
ACCURACIES = (1e-3, 1e-5, 1e-7)
BUDGETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000)


def critical_radius(problem):
    """Residual-based radius estimate for a globally ridge-strongly-convex loss.

    ||x_ref-x_unconstrained*||_1 <= sqrt(d)*||grad f(x_ref)||_2/mu.
    Numerical residuals are evaluated in float64, not interval arithmetic.
    """
    x = problem.reference_x
    radius = float(np.linalg.norm(x, 1))
    residual = float(np.linalg.norm(problem.gradient(x)))
    error = math.sqrt(x.size)*residual/problem.metadata["mu"]
    if radius+error >= problem.radius or residual > 1e-10:
        raise RuntimeError("Need an interior, accurately stationary reference")
    return {"estimate": radius, "residual_norm": residual, "residual_radius_bound": error,
            "lower_estimate": max(0, radius-error), "upper_estimate": radius+error,
            "qualification": "float64 residual estimate, not rigorous interval arithmetic"}


def reference_geometry(problem, critical):
    radius = problem.radius
    regime = ("active" if radius < critical["lower_estimate"] else
              "interior" if radius > critical["upper_estimate"] else "transition")
    x, grad = problem.reference_x, problem.gradient(problem.reference_x)
    support = np.abs(x) > 1e-7
    multiplier = float(np.max(np.abs(grad))) if regime == "active" else 0.0
    residual = max(float(np.max(np.abs(grad[support]+multiplier*np.sign(x[support])), initial=0)),
                   float(np.max(np.maximum(np.abs(grad[~support])-multiplier, 0), initial=0)))
    inactive_margin = (multiplier-float(np.max(np.abs(grad[~support])))) if regime == "active" and np.any(~support) else None
    return {"regime": regime, "reference_l1_norm": float(np.linalg.norm(x, 1)),
            "reference_support_size": int(support.sum()), "reference_support_threshold": 1e-7,
            "estimated_multiplier": multiplier, "kkt_residual_inf": residual,
            "inactive_margin": inactive_margin,
            "reference_norm_slack": radius-float(np.linalg.norm(x, 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=ROOT/"constraint_results")
    args = parser.parse_args()
    if args.seeds < 1 or args.max_iter < 1:
        parser.error("Need positive seeds and max-iter")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    critical = critical_radius(make_bike_poisson(0))
    summary, sampled, budgets, accuracies, audits, problems = [], [], [], [], [], []
    for ratio in RATIOS:
        radius = ratio*critical["estimate"]
        base = make_bike_poisson(0, radius=radius)
        geometry = reference_geometry(base, critical)
        problems.append({"radius_ratio": ratio, "radius": radius, "H0": base.h0,
                         "H1": base.h1, "global_L": base.global_l,
                         "reference_lower": base.reference_lower,
                         "reference_upper": base.reference_upper, **base.metadata, **geometry})
        for seed in range(args.seeds):
            # Same initial-vertex rule as problems.py; objective and reference
            # do not depend on seed. Construction/reference work is offline.
            rng = np.random.default_rng(seed)
            x0 = np.zeros_like(base.x0)
            x0[int(rng.integers(x0.size))] = radius*(1 if rng.integers(2) else -1)
            p = replace(base, seed=seed, x0=x0, metadata={**base.metadata, "objective_scale": 1.0})
            for method in METHODS:
                result = run_method(p, method, args.max_iter, min(ACCURACIES))
                identity = {"radius_ratio": ratio, "seed": seed, "method": method}
                last = result.trace[-1]
                face_normal = np.where(np.abs(p.reference_x) > 1e-7, np.sign(p.reference_x), 0)
                summary.append({**identity, **last, "regime": geometry["regime"],
                                "certified": last["certificate"] <= min(ACCURACIES),
                                "error_lower": max(0, last["objective"]-p.reference_upper),
                                "error_upper": max(0, last["objective"]-p.reference_lower),
                                "reference_face_slack": float(radius-face_normal @ result.x) if geometry["regime"] == "active" else None})
                sampled.extend({**identity, **row} for row in result.trace if row["iteration"] % 10 == 0 or row is last)
                checks = audit_run(result, p, method, min(ACCURACIES))
                max_ratio = 0.0
                for row in result.trace:
                    delta_upper = max(0, row["objective"]-p.reference_lower)
                    envelope = gap_envelope(delta_upper, p.h0, p.h1, 2*radius)
                    assert row["fw_gap"] <= envelope+2e-10
                    if envelope > 0:
                        max_ratio = max(max_ratio, row["fw_gap"]/envelope)
                audits.append({**identity, **checks, "max_gap_envelope_ratio": max_ratio})
                u0, diameter = result.trace[0]["certificate"], 2*radius
                for epsilon in ACCURACIES:
                    row = next((row for row in result.trace if row["certificate"] <= epsilon), last)
                    old_eta = min(epsilon/2, p.metadata["mu"]*epsilon**2
                                  /(8*diameter**2*(p.h0+p.h1*u0)**2))
                    eta = certificate_threshold(epsilon, p.h0, p.h1, diameter)
                    accuracies.append({**identity, "epsilon": epsilon,
                                       "iteration": row["iteration"], "certificate": row["certificate"],
                                       "certified": row["certificate"] <= epsilon,
                                       "certified_by_1000_updates": row["certificate"] <= epsilon and row["iteration"] <= 1000,
                                       "new_error_threshold": eta, "old_error_threshold": old_eta,
                                       "phase_saving_before_rounding": math.log2(eta/old_eta),
                                       **{key: row[key] for key in ORACLES}})
                for oracle in ORACLES:
                    counts = [row[oracle] for row in result.trace]
                    for b in BUDGETS:
                        index = int(np.searchsorted(counts, b, side="right"))-1
                        assert index >= 0
                        row = result.trace[index]
                        budgets.append({**identity, "oracle": oracle, "budget": b,
                                        "iteration": row["iteration"], "certificate": row["certificate"],
                                        "objective": row["objective"], "lower_bound": row["lower_bound"],
                                        "certified": row["certificate"] <= min(ACCURACIES),
                                        "budget_reached_or_certified": last[oracle] >= b or last["certificate"] <= min(ACCURACIES),
                                        **{key: row[key] for key in ORACLES}})
        print(f"ratio={ratio:g}, R={radius:.8f}, {geometry['regime']}: {args.seeds} starts complete", flush=True)
    names = ("summary.csv", "trace.csv", "budget.csv", "accuracy.csv", "audit.csv")
    for name, rows in zip(names, (summary, sampled, budgets, accuracies, audits)):
        write_csv(args.output/name, rows)
    sources = ("constraint_transition.py", "certificate_bounds.py", "robustness_benchmark.py",
               "away_fw.py", "fw.py", "problems.py", "auto_conditioned.py")
    source = args.output/"source"
    source.mkdir()
    for name in sources:
        shutil.copy2(ROOT/name, source/name)
    import scipy
    manifest = {"study": "constraint_transition", "date": "2026-10-15", "command": [sys.executable, *sys.argv],
                "methods": METHODS, "seeds": list(range(args.seeds)), "radius_ratios": RATIOS,
                "critical_radius": critical, "max_iterations": args.max_iter, "tolerance": min(ACCURACIES),
                "accuracies": ACCURACIES, "budgets": BUDGETS, "trace_stride": 10,
                "selection": "seven predefined ratios of independent stationary-reference norm; no algorithm outcomes used",
                "information": "solvers get only f,g,R,x0; fixed M=1 or M=U0/(2R)^2; references/geometry offline only",
                "dependence": "one dataset and objective; seeds change starting vertex only; accuracy rows are trajectory prefixes",
                "geometry": "reference-face slack is a post-run diagnostic with support threshold 1e-7, not exact face certification",
                "python": sys.version, "platform": platform.platform(),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
                "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "problems": problems, "source_archive": "source/",
                "source_sha256": {n: hashlib.sha256((source/n).read_bytes()).hexdigest() for n in sources},
                "result_sha256": {n: hashlib.sha256((args.output/n).read_bytes()).hexdigest() for n in names}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} trajectories, {len(accuracies)} accuracy prefixes, {len(audits)} full audits")


if __name__ == "__main__":
    main()
