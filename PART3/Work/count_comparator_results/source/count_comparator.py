"""Add native AC-FW to the unchanged 108 Seoul/Metro dataset/radius/vertex cases."""

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

from count_problems import DATASETS, make_count_problem, unconstrained_reference
from fw import solve
from multidata_benchmark import vertex
from robustness_benchmark import ORACLES, write_csv


ROOT = Path(__file__).resolve().parent
BUDGETS = (2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)


def audit(result, problem):
    rows = result.trace
    for row in rows:
        n = row["iteration"]
        assert row["function_evals"] == row["lmo_evals"] == n+2
        assert row["gradient_evals"] == row["accepted_steps"]+2
        assert row["accepted_steps"]+row["rejected_steps"] == n
        assert row["lower_bound"] <= problem.reference_upper+2e-10
        assert row["certificate"]+2e-10 >= max(0, row["objective"]-problem.reference_upper)
        assert row["l1_norm"] <= problem.radius+1e-11
    for row, following in zip(rows, rows[1:]):
        assert following["objective"] <= row["objective"]
        assert following["certificate"] <= row["certificate"]+2e-10
        assert (row["trial_objective"] < row["objective"]) == row["trial_accepted"]
        if not row["trial_accepted"]:
            assert following["objective"] == row["objective"]
            assert following["gradient_evals"] == row["gradient_evals"]
    return {"source_observations": len(rows), "all_sources_audited": True,
            "accepted_steps": rows[-1]["accepted_steps"], "rejected_steps": rows[-1]["rejected_steps"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"count_comparator_results")
    output = parser.parse_args().output
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new/empty output directory")
    project = next(p for p in ROOT.parents if p.name == "PART3")
    previous_dir = project/"Work"/"multidata_results"
    previous = json.loads((previous_dir/"manifest.json").read_text(encoding="utf-8"))
    tolerance, cap = previous["tolerance"], previous["max_iterations"]
    summary, sampled, budgets, audits = [], [], [], []
    for dataset in DATASETS:
        unc, _ = unconstrained_reference(dataset)
        for ratio in previous["radius_ratios"]:
            p = make_count_problem(dataset, ratio*float(np.linalg.norm(unc, 1)))
            old = next(q for q in previous["problems"] if q["dataset"] == dataset and q["radius_ratio"] == ratio)
            assert p.radius == old["radius"]
            assert p.reference_lower == old["reference_lower"] and p.reference_upper == old["reference_upper"]
            for start in range(2*p.x0.size):
                problem = replace(p, seed=start, x0=vertex(start, p.x0.size, p.radius))
                result = solve(problem, "AC_FW", cap, tolerance)
                identity = {"dataset": dataset, "radius_ratio": ratio, "start": start, "method": "AC_FW"}
                last = result.trace[-1]
                summary.append({**identity, **last, "certified": last["certificate"] <= tolerance,
                                "error_lower": max(0, last["objective"]-p.reference_upper),
                                "error_upper": max(0, last["objective"]-p.reference_lower)})
                sampled.extend({**identity, **row} for row in result.trace if row["iteration"] % 10 == 0 or row is last)
                audits.append({**identity, **audit(result, p)})
                for oracle in ORACLES:
                    counts = [row[oracle] for row in result.trace]
                    for budget in BUDGETS:
                        index = int(np.searchsorted(counts, budget, side="right"))-1
                        assert index >= 0
                        row = result.trace[index]
                        budgets.append({**identity, "oracle": oracle, "budget": budget,
                                        "iteration": row["iteration"], "objective": row["objective"],
                                        "certificate": row["certificate"], "lower_bound": row["lower_bound"],
                                        "certified": row["certificate"] <= tolerance,
                                        "budget_reached_or_certified": last[oracle] >= budget or last["certificate"] <= tolerance,
                                        **{key: row[key] for key in ORACLES}})
            print(f"{dataset}, R/R*={ratio:g}: {2*p.x0.size} AC-FW starts complete", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    files = ("summary.csv", "trace.csv", "budget.csv", "audit.csv")
    for name, rows in zip(files, (summary, sampled, budgets, audits)):
        write_csv(output/name, rows)
    names = ("count_comparator.py", "count_problems.py", "fw.py", "auto_conditioned.py", "problems.py",
             "multidata_benchmark.py", "robustness_benchmark.py", "away_fw.py", "certificate_bounds.py")
    source = output/"source"
    source.mkdir()
    for name in names:
        shutil.copy2(ROOT/name, source/name)
    import scipy
    manifest = {"study": "count_ac_comparator", "date": "2026-10-15", "command": [sys.executable, *sys.argv],
                "methods": ["AC_FW"], "datasets": DATASETS, "radius_ratios": previous["radius_ratios"],
                "max_iterations": cap, "tolerance": tolerance, "budgets": BUDGETS,
                "problems": previous["problems"], "prior_manifest_sha256": hashlib.sha256((previous_dir/"manifest.json").read_bytes()).hexdigest(),
                "information": "f,g,R,x0 only; references, Hessians, H0,H1 and global L are offline",
                "algorithm": "native Algorithms 1+2, Eq.(6) with delta=1, arXiv:2605.15512v2; independent implementation",
                "oracle_cost": "compulsory initial vertex move and both source values/gradients charged; rejected outer trials charged",
                "reuse": "108 new method runs on the same 108 dataset/radius/vertex cases; previous 648 records unchanged",
                "scope": "empirical use on compact Q; global gradient-Lipschitzness on R^d in the cited theorem not asserted",
                "python": sys.version, "platform": platform.platform(),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
                "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "source_archive": "source/", "source_sha256": {n: hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names},
                "result_sha256": {n: hashlib.sha256((output/n).read_bytes()).hexdigest() for n in files}}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} additional method runs and {len(budgets)} prefixes")


if __name__ == "__main__":
    main()
