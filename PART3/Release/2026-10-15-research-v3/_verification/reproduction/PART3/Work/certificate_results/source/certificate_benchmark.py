"""Causal stopping monitors on the four unchanged BT trajectories.

Profiles share one original trajectory per problem/method. Isolated monitor
costs are evaluated as attainable prefixes, not independent solver runs.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

from away_fw import solve_away
from count_problems import DATASETS, make_count_problem, unconstrained_reference
from fw import solve
from multidata_benchmark import vertex
from robustness_benchmark import ORACLES, write_csv
from tangent_monitor import PROFILES, TangentMonitor


ROOT = Path(__file__).resolve().parent
METHODS = ("FW_BT_fixed", "FW_BT_certificate", "AFW_BT_fixed", "AFW_BT_certificate")
RATIOS = (0.5, 2.0)
BUDGETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)


def observed_solve(problem, method, max_iter, tolerance, on_iterate):
    if method not in METHODS:
        raise ValueError(method)
    if method.startswith("AFW"):
        return solve_away(problem, max_iter=max_iter, tolerance=tolerance,
                          initial_estimate="certificate" if method.endswith("certificate") else 1.0,
                          on_iterate=on_iterate)
    return solve(problem, "adaptive_L_initial" if method.endswith("certificate") else "adaptive_L",
                 max_iter, tolerance, on_iterate=on_iterate)


class MonitorBank:
    """All state is causal; each profile's cost excludes the other profiles."""
    def __init__(self, radius, mu):
        self.monitors = {p: TangentMonitor(radius, p, mu=mu if p == "known_mu" else None) for p in PROFILES}
        self.rows = {p: [] for p in PROFILES}

    def observe(self, x, fx, gradient, atom, original):
        for profile, monitor in self.monitors.items():
            certificate = monitor.observe(x, fx, gradient, original["lower_bound"])
            self.rows[profile].append({**original, **certificate,
                                       "direction_lmo_evals": original["lmo_evals"],
                                       "lmo_evals": original["lmo_evals"]+certificate["certificate_lmo_evals"]})
        return False  # preserve the original stopping rule and complete trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"certificate_results")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    # Protocol is inherited, and matched trajectories are verified below.
    previous_path = ROOT/"multidata_results"
    if not (previous_path/"manifest.json").exists():
        project = next(parent for parent in ROOT.parents if parent.name == "PART3")
        previous_path = project/"Work"/"multidata_results"  # archived source reproduction
    previous = json.loads((previous_path/"manifest.json").read_text(encoding="utf-8"))
    import csv
    with (previous_path/"summary.csv").open(encoding="utf-8") as file:
        prior = {(r["dataset"], float(r["radius_ratio"]), int(r["start"]), r["method"]): r for r in csv.DictReader(file)}
    summary, final, sampled, budgets, audits, problems = [], [], [], [], [], []
    tolerance, cap = previous["tolerance"], previous["max_iterations"]
    for dataset in DATASETS:
        unc, _ = unconstrained_reference(dataset)
        for ratio in RATIOS:
            p = make_count_problem(dataset, ratio*float(np.linalg.norm(unc, 1)))
            problems.append({"dataset": dataset, "radius_ratio": ratio, "radius": p.radius,
                             "reference_lower": p.reference_lower, "reference_upper": p.reference_upper, **p.metadata})
            for start in range(2*p.x0.size):
                problem = replace(p, seed=start, x0=vertex(start, p.x0.size, p.radius))
                for method in METHODS:
                    bank = MonitorBank(p.radius, p.metadata["mu"])
                    result = observed_solve(problem, method, cap, tolerance, bank.observe)
                    identity = {"dataset": dataset, "radius_ratio": ratio, "start": start, "method": method}
                    old = prior[(dataset, ratio, start, method)]
                    for key in ("objective", "certificate", "function_evals", "gradient_evals", "lmo_evals", "iteration"):
                        assert str(result.trace[-1][key]) == old[key], (*identity.values(), key)
                    max_reference_excess = 0.0
                    for profile, rows in bank.rows.items():
                        first = next((i for i, r in enumerate(rows) if r["certificate"] <= tolerance), len(rows)-1)
                        deployed = rows[:first+1]
                        stop, last = deployed[-1], rows[-1]
                        common = {**identity, "profile": profile}
                        summary.append({**common, **stop, "certified": stop["certificate"] <= tolerance,
                                        "error_upper": max(0, stop["objective"]-p.reference_lower)})
                        final.append({**common, **last, "certified": last["certificate"] <= tolerance})
                        sampled.extend({**common, **r} for r in rows if r["iteration"] % 10 == 0 or r is stop or r is last)
                        for r in rows:
                            assert r["lower_bound"] <= p.reference_upper+2e-10
                            assert r["certificate"] <= r["scalar_certificate"]+2e-10
                            assert r["certificate"]+2e-10 >= max(0, r["objective"]-p.reference_upper)
                            expected_extra = (r["iteration"]+1)//int(profile.split("_")[1]) if profile.startswith("block") else 0
                            assert r["certificate_lmo_evals"] == expected_extra
                            max_reference_excess = max(max_reference_excess, r["lower_bound"]-p.reference_upper)
                        for r, nxt in zip(rows, rows[1:]):
                            assert nxt["certificate"] <= r["certificate"]+2e-10
                        for oracle in ORACLES:
                            counts = [r[oracle] for r in deployed]
                            for b in BUDGETS:
                                idx = int(np.searchsorted(counts, b, side="right"))-1
                                assert idx >= 0
                                r = deployed[idx]
                                budgets.append({**common, "oracle": oracle, "budget": b,
                                                "iteration": r["iteration"], "objective": r["objective"],
                                                "certificate": r["certificate"], "lower_bound": r["lower_bound"],
                                                "certified": r["certificate"] <= tolerance,
                                                "direction_lmo_evals": r["direction_lmo_evals"],
                                                "certificate_lmo_evals": r["certificate_lmo_evals"],
                                                "budget_reached_or_certified": stop[oracle] >= b or stop["certificate"] <= tolerance,
                                                **{key: r[key] for key in ORACLES}})
                    audits.append({**identity, "original_updates": result.trace[-1]["iteration"],
                                   "original_matches_prior": True, "profiles": len(PROFILES),
                                   "source_observations": len(result.trace), "max_reference_excess": max_reference_excess})
                if start % 8 == 7 or start == 2*p.x0.size-1:
                    print(f"{dataset} ratio={ratio:g}: {start+1}/{2*p.x0.size} starts monitored", flush=True)
    names = ("summary.csv", "final.csv", "trace.csv", "budget.csv", "audit.csv")
    for name, rows in zip(names, (summary, final, sampled, budgets, audits)):
        write_csv(args.output/name, rows)
    sources = ("certificate_benchmark.py", "tangent_monitor.py", "count_problems.py", "fw.py", "away_fw.py",
               "problems.py", "multidata_benchmark.py", "robustness_benchmark.py", "certificate_bounds.py", "auto_conditioned.py")
    source = args.output/"source"
    source.mkdir()
    for name in sources:
        shutil.copy2(ROOT/name, source/name)
    import scipy
    manifest = {"study": "causal_certificates", "date": "2026-10-15", "command": [sys.executable, *sys.argv],
                "methods": METHODS, "profiles": PROFILES, "datasets": DATASETS, "radius_ratios": RATIOS,
                "max_iterations": cap, "tolerance": tolerance, "budgets": BUDGETS, "trace_stride": 10,
                "protocol": "one original trajectory; causal profile prefixes; profiles do not change updates or initialization",
                "information": "block profiles require no H0,H1,mu,f*; known_mu receives analytical ridge mu; references offline only",
                "oracle_cost": "isolated profile total LMO = direction LMOs + actual block LMOs; other monitors excluded",
                "reuse": "432 repeated original BT trajectories, not 2160 independent solver runs; all starts unique within dataset/radius",
                "prior_manifest_sha256": hashlib.sha256((previous_path/"manifest.json").read_bytes()).hexdigest(),
                "python": sys.version, "platform": platform.platform(),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
                "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "problems": problems, "source_archive": "source/",
                "source_sha256": {n: hashlib.sha256((source/n).read_bytes()).hexdigest() for n in sources},
                "result_sha256": {n: hashlib.sha256((args.output/n).read_bytes()).hexdigest() for n in names}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(audits)} repeated original trajectories; {len(summary)} causal profile summaries; {len(budgets)} budget prefixes")


if __name__ == "__main__":
    main()
