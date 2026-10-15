"""Controlled 2x2 ablation: starting vertex versus subsequent update rule.

All cells pay for f, gradient and LMO at x and z=LMO(gradient(x)); both
linear lower bounds and the secant curvature are shared as information.
AC/vertex reproduces native AC-FW. The other cells are declared controls.
BT halves its estimate and doubles on rejection, initialized at twice the
shared curvature so its FIRST trial equals AC's. No H0, H1 or f* input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np

from auto_conditioned import damping, local_curvature
from fw import Result
from problems import SCENARIOS, lmo, make_problem


ROOT = Path(__file__).resolve().parent
CELLS = ("AC_original", "AC_vertex", "BT_original", "BT_vertex")
BUDGETS = (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
ORACLES = ("function_evals", "gradient_evals", "lmo_evals")


def solve_control(problem, cell, max_iter=1000, tolerance=1e-5):
    if cell not in CELLS or max_iter < 0 or tolerance < 0:
        raise ValueError("Invalid controlled experiment parameters")
    update, start_point = cell.split("_")
    started = perf_counter()
    counts = {key: 0 for key in ORACLES}

    def objective(point):
        counts["function_evals"] += 1
        return problem.objective(point)

    def gradient(point):
        counts["gradient_evals"] += 1
        return problem.gradient(point)

    def oracle(g):
        counts["lmo_evals"] += 1
        return lmo(g, problem.radius)

    original = problem.x0.copy()
    original_f, original_g = objective(original), gradient(original)
    vertex = oracle(original_g)
    vertex_f, vertex_g = objective(vertex), gradient(vertex)
    vertex_s = oracle(vertex_g)
    curvature = local_curvature(original, vertex, original_f, vertex_f, original_g)
    lower = max(original_f + float(original_g @ (vertex-original)),
                vertex_f-max(0.0, float(-vertex_g @ (vertex_s-vertex))))
    if start_point == "vertex":
        x, fx, g, s = vertex.copy(), vertex_f, vertex_g, vertex_s
    else:
        x, fx, g, s = original, original_f, original_g, vertex
    # BT's first halving now gives precisely the AC first trial curvature.
    estimate = 2*curvature if update == "BT" else curvature
    accepted, rejected, trial_rejections, trace = 0, 0, 0, []
    for k in range(max_iter+1):
        if k:
            s = oracle(g)
        d = s-x
        raw_gap = float(-g @ d)
        if raw_gap < -1e-10*max(1.0, abs(fx)):
            raise ArithmeticError("Negative controlled FW gap")
        gap = max(0.0, raw_gap)
        lower = max(lower, fx-gap)
        upper, q2 = max(0.0, fx-lower), float(d @ d)
        trace.append({"iteration": k, "objective": fx, "fw_gap": gap,
                      "lower_bound": lower, "certificate": upper,
                      "direction_norm_sq": q2, "accepted_steps": accepted,
                      "rejected_steps": rejected, "trial_rejections": trial_rejections,
                      "l1_norm": float(np.linalg.norm(x, 1)),
                      "elapsed_seconds": perf_counter()-started, **counts})
        if upper <= tolerance or k == max_iter:
            break
        if q2 == 0 or not math.isfinite(estimate) or estimate <= 0:
            raise ArithmeticError("A nonterminal control requires positive secant curvature")
        trial_curvature = estimate/2 if update == "BT" else estimate
        for trial in range(1024):
            alpha = min(1.0, gap/(trial_curvature*q2))
            trial_x = x+alpha*d
            trial_f = objective(trial_x)
            if trial == 0:
                trace[-1].update(first_trial_curvature=trial_curvature,
                                 first_trial_step=alpha, first_trial_objective=trial_f)
            if update == "AC" or trial_f <= fx-alpha*gap+0.5*trial_curvature*alpha**2*q2:
                break
            trial_rejections += 1
            trial_curvature *= 2
        else:
            raise RuntimeError("Controlled backtracking failed")
        improved = trial_f < fx if update == "AC" else True
        next_estimate = (max(local_curvature(x, trial_x, fx, trial_f, g), damping(k)*estimate)
                         if update == "AC" else trial_curvature)
        trace[-1].update(step=alpha, trial_curvature=trial_curvature,
                         trial_accepted=improved, line_search_trials=trial+1)
        if improved:
            x, fx, g = trial_x, trial_f, gradient(trial_x)
            accepted += 1
        else:
            rejected += 1
            trial_rejections += 1
        estimate = next_estimate
        if not math.isfinite(trial_f) or not math.isfinite(estimate) or not np.all(np.isfinite(g)):
            raise FloatingPointError("Nonfinite control oracle output")
    return Result(x, trace, perf_counter()-started, 0)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def budget_prefix(trace, oracle, budget):
    """Last completed certificate available using at most this oracle budget.

    Other oracle types are unrestricted. This is a replay of deterministic
    prefixes, not a new run or a claim about a weighted aggregate cost.
    """
    index = int(np.searchsorted([r[oracle] for r in trace], budget, side="right"))-1
    if index < 0:
        raise ValueError("Budget cannot pay for the common initialization")
    return trace[index]


def compare_paired(rows, scenarios, seeds, tolerance):
    lookup = {(r["scenario"], r["seed"], r["cell"], r["oracle"], r["budget"]): r for r in rows}
    rng = np.random.default_rng(20261015)
    contrasts = (("AC_vertex", "AC_original"), ("BT_vertex", "BT_original"),
                 ("BT_original", "AC_original"), ("BT_vertex", "AC_vertex"))
    comparisons = []
    for scenario in scenarios:
        for numerator, denominator in contrasts:
            for oracle in ORACLES:
                paired = [(lookup[(scenario, seed, numerator, oracle, 1000)],
                           lookup[(scenario, seed, denominator, oracle, 1000)]) for seed in seeds]
                ratios = np.array([max(a["certificate"], 1e-15)/max(b["certificate"], 1e-15)
                                   for a, b in paired])
                medians = np.median(rng.choice(ratios, size=(10000, len(seeds)), replace=True), axis=1)
                comparisons.append({"scenario": scenario, "numerator": numerator, "denominator": denominator,
                                    "oracle": oracle, "budget": 1000,
                                    "paired_median_certificate_ratio": float(np.median(ratios)),
                                    "bootstrap_025": float(np.quantile(medians, 0.025)),
                                    "bootstrap_975": float(np.quantile(medians, 0.975)),
                                    "numerator_certified": sum(a["certificate"] <= tolerance for a, _ in paired),
                                    "denominator_certified": sum(b["certificate"] <= tolerance for _, b in paired),
                                    "runs": len(seeds)})
    return comparisons


def make_plot(output, budget_rows, scenarios, seeds, tolerance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = [s for s in ("active", "active_scaled", "bike_poisson") if s in scenarios]
    if not shown:
        shown = scenarios[:3]
    fig, axes = plt.subplots(1, len(shown), figsize=(5*len(shown), 4), squeeze=False)
    for ax, scenario in zip(axes[0], shown):
        for cell in CELLS:
            curves = [[next(r["certificate"] for r in budget_rows if r["scenario"] == scenario
                            and r["seed"] == seed and r["cell"] == cell
                            and r["oracle"] == "function_evals" and r["budget"] == b)
                       for b in BUDGETS] for seed in seeds]
            values = np.maximum(curves, 1e-16)
            median = np.median(values, axis=0)
            lo, hi = np.quantile(values, [0.25, 0.75], axis=0)
            line, = ax.loglog(BUDGETS, median, label=cell.replace("_", ": "), marker=".")
            ax.fill_between(BUDGETS, lo, hi, color=line.get_color(), alpha=0.08)
        ax.axhline(tolerance, color="gray", linestyle=":")
        ax.set(title=scenario, xlabel="Function-call budget", ylabel="Certificate U (median, IQR)")
        ax.grid(alpha=0.25, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output/"initialization_effect.png", dpi=170)
    fig.savefig(output/"initialization_effect.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=ROOT/"initialization_factorial")
    args = parser.parse_args()
    if args.seeds < 1 or args.max_iter < 1 or args.tolerance <= 0:
        parser.error("Need positive seeds, max-iter and tolerance")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Duplicate scenarios")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty output directory to preserve previous experiments")
    args.output.mkdir(parents=True, exist_ok=True)
    summary, traces, budgets, problem_manifest = [], [], [], []
    seeds = list(range(args.seeds))
    for scenario in args.scenarios:
        for seed in seeds:
            problem = make_problem(seed, scenario)
            problem_manifest.append({"scenario": scenario, "seed": seed, "radius": problem.radius,
                                     "reference_lower": problem.reference_lower,
                                     "reference_upper": problem.reference_upper, **problem.metadata})
            for cell in CELLS:
                result = solve_control(problem, cell, args.max_iter, args.tolerance)
                last = result.trace[-1]
                identity = {"scenario": scenario, "seed": seed, "cell": cell}
                summary.append({**identity, **last, "certified": last["certificate"] <= args.tolerance,
                                "elapsed_seconds": result.elapsed_seconds})
                traces.extend({**identity, **row} for row in result.trace
                              if row["iteration"] % 10 == 0 or row is last)
                for oracle in ORACLES:
                    for b in BUDGETS:
                        row = budget_prefix(result.trace, oracle, b)
                        budgets.append({**identity, "oracle": oracle, "budget": b,
                                        "iteration": row["iteration"], "certificate": row["certificate"],
                                        "objective": row["objective"], "lower_bound": row["lower_bound"],
                                        "certified": row["certificate"] <= args.tolerance,
                                        "error_lower": max(0.0, row["objective"]-problem.reference_upper),
                                        "error_upper": max(0.0, row["objective"]-problem.reference_lower),
                                        **{key: row[key] for key in ORACLES}})
            print(f"{scenario}: seed {seed} complete", flush=True)
    comparisons = compare_paired(budgets, args.scenarios, seeds, args.tolerance)
    write_csv(args.output/"summary.csv", summary)
    write_csv(args.output/"trace.csv", traces)
    write_csv(args.output/"budget.csv", budgets)
    write_csv(args.output/"paired_comparisons.csv", comparisons)
    source = args.output/"source"
    source.mkdir()
    names = ("initialization_factorial.py", "auto_conditioned.py", "fw.py", "problems.py")
    for name in names:
        shutil.copy2(ROOT/name, source/name)
    import scipy, sklearn, matplotlib
    manifest = {"command": [sys.executable, *sys.argv], "date": "2026-10-15",
                "seeds": seeds, "scenarios": args.scenarios, "cells": CELLS, "budgets": BUDGETS,
                "max_iterations": args.max_iter, "tolerance": args.tolerance, "trace_stride": 10,
                "first_trial": "shared secant estimate; BT stored initialization is twice this value",
                "common_preprocessing": "two f, two gradients, two LMOs; both lower bounds retained",
                "native_cell": "AC_vertex", "controls": "other cells are ablations, not native AC-FW",
                "backtracking": {"shrink": 0.5, "growth": 2},
                "damping": {"equation": 6, "source": "arXiv:2605.15512v2", "delta": 1},
                "bootstrap": {"seed": 20261015, "resamples": 10000, "ratio_floor": 1e-15,
                              "interpretation": "descriptive percentile intervals across paired seeds; no multiplicity correction"},
                "python": sys.version, "platform": platform.platform(),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                             "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
                "thread_environment": {key: os.environ.get(key) for key in
                                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "problems": problem_manifest, "source_archive": "source/",
                "source_sha256": {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names},
                "result_sha256": {name: hashlib.sha256((args.output/name).read_bytes()).hexdigest()
                                  for name in ("summary.csv", "trace.csv", "budget.csv", "paired_comparisons.csv")}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    make_plot(args.output, budgets, args.scenarios, seeds, args.tolerance)
    print(f"Saved {len(summary)} runs and {len(budgets)} exact budget prefixes to {args.output}")


if __name__ == "__main__":
    main()
