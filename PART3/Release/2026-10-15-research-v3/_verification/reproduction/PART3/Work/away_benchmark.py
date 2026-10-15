"""Compare FW/AFW from the same original point, with BT/known-H short steps.

python PART3/Work/away_benchmark.py --output PART3/Work/away_results
No compulsory initial LMO move is made. Past experiments are not overwritten.
"""

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

import numpy as np

from away_fw import solve_away
from fw import solve
from problems import SCENARIOS, make_problem


ROOT = Path(__file__).resolve().parent
METHODS = ("FW_BT", "AFW_BT", "FW_Q", "AFW_Q")
ORACLES = ("function_evals", "gradient_evals", "lmo_evals")
BUDGETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def audit_away(result, problem, tolerance):
    rows, diameter = result.trace, 2*problem.radius
    u0 = rows[0]["certificate"]
    if u0 <= tolerance:
        return {"updates": 0, "convex_update_bound": 0, "lag_steps": 0, "lag_bound": 0,
                "function_evals": 1, "function_bound": 1, "drop_steps": 0}
    phases = math.ceil(math.log2(u0/tolerance))
    initial_potential = math.ceil(math.log2(max(1, diameter**2/u0)))
    previous, trials, lag = 1.0, 0, 0
    max_roundoff = 0.0
    for row, following in zip(rows, rows[1:]):
        u, current = row["certificate"], row["trial_curvature"]
        phase = math.floor(math.log2(u0/u))
        assert 0 <= phase < phases
        b = u0/2**phase
        cap = max(2*(problem.h0+problem.h1*b), b/(2*diameter**2))
        is_lag = current > cap*(1+1e-12)
        lag += is_lag
        if is_lag:
            assert row["line_search_trials"] == 1
        assert math.isclose(math.log2(current/previous), row["line_search_trials"]-2, abs_tol=1e-10)
        assert current <= max(previous/2, 2*(problem.h0+problem.h1*u))*(1+1e-12)
        assert row["directional_gap"]+1e-11 >= row["fw_gap"] >= u-1e-11
        assert following["objective"] <= row["objective"]-0.5*row["step"]*row["directional_gap"]+1e-11
        assert following["certificate"] <= u+1e-11
        if not is_lag and not row["drop_step"]:
            progress = min(b/4, b*b/(8*cap*diameter**2))
            assert following["objective"] <= row["objective"]-progress+1e-11
        previous = current
        trials += row["line_search_trials"]
        max_roundoff = max(max_roundoff, row["affine_roundoff"])
    final = rows[-1]
    n = final["iteration"]
    assert final["numerical_prunes"] == 0
    assert final["drop_steps"] <= final["fw_steps"]+final["initial_active_size"]-1
    assert lag <= initial_potential+phases
    assert final["function_evals"] == trials+1
    if n:
        assert math.isclose(trials, 2*n+math.log2(previous), abs_tol=1e-9)
    overhead = math.ceil(math.log2(max(1, 2*(problem.h0+problem.h1*u0))))
    function_bound = 1+2*n+overhead
    assert final["function_evals"] <= function_bound
    convex_bound = 2*(initial_potential+math.ceil((6+16*problem.h1*diameter**2)*phases
                                                +32*problem.h0*diameter**2/tolerance))+final["initial_active_size"]-1
    assert n <= convex_bound
    mu = problem.metadata["mu"]
    local_max = problem.h0+problem.h1*u0
    for row in rows:
        delta_upper = max(0, row["objective"]-problem.reference_lower)
        assert row["fw_gap"] <= delta_upper+diameter*local_max*math.sqrt(2*delta_upper/mu)+1e-10
    return {"updates": n, "convex_update_bound": convex_bound,
            "lag_steps": lag, "lag_bound": initial_potential+phases,
            "function_evals": final["function_evals"], "function_bound": function_bound,
            "drop_steps": final["drop_steps"], "good_steps": final["good_steps"],
            "max_affine_roundoff": max_roundoff}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=ROOT/"away_results")
    args = parser.parse_args()
    if args.seeds < 1 or args.max_iter < 1 or args.tolerance <= 0:
        parser.error("Need positive seeds, max-iter and tolerance")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("Duplicate scenarios")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    summary, sampled, budgets, audits, problems = [], [], [], [], []
    for scenario in args.scenarios:
        for seed in range(args.seeds):
            p = make_problem(seed, scenario)
            problems.append({"scenario": scenario, "seed": seed, "radius": p.radius,
                             "H0": p.h0, "H1": p.h1, "reference_lower": p.reference_lower,
                             "reference_upper": p.reference_upper, **p.metadata})
            for method in METHODS:
                step = method.split("_")[1]
                if method.startswith("AFW"):
                    result = solve_away(p, step, args.max_iter, args.tolerance)
                else:
                    result = solve(p, "adaptive_L" if step == "BT" else "quadratic_memory",
                                   args.max_iter, args.tolerance)
                identity = {"scenario": scenario, "seed": seed, "method": method}
                last = result.trace[-1]
                summary.append({**identity, **last, "certified": last["certificate"] <= args.tolerance,
                                "error_lower": max(0, last["objective"]-p.reference_upper),
                                "error_upper": max(0, last["objective"]-p.reference_lower),
                                "elapsed_seconds": result.elapsed_seconds})
                sampled.extend({**identity, **row} for row in result.trace
                               if row["iteration"] % 10 == 0 or row is last)
                for oracle in ORACLES:
                    counts = [row[oracle] for row in result.trace]
                    for b in BUDGETS:
                        index = int(np.searchsorted(counts, b, side="right"))-1
                        assert index >= 0
                        row = result.trace[index]
                        budgets.append({**identity, "oracle": oracle, "budget": b,
                                        "iteration": row["iteration"], "certificate": row["certificate"],
                                        "certified": row["certificate"] <= args.tolerance,
                                        "objective": row["objective"], "lower_bound": row["lower_bound"],
                                        "away_scans": row.get("away_scans", 0),
                                        "active_atom_scores": row.get("active_atom_scores", 0),
                                        **{key: row[key] for key in ORACLES}})
                if method == "AFW_BT":
                    audits.append({**identity, **audit_away(result, p, args.tolerance)})
            print(f"{scenario}: seed {seed} complete", flush=True)
    for name, rows in (("summary.csv", summary), ("trace.csv", sampled), ("budget.csv", budgets), ("audit.csv", audits)):
        write_csv(args.output/name, rows)
    source = args.output/"source"
    source.mkdir()
    names = ("away_benchmark.py", "away_fw.py", "fw.py", "problems.py", "auto_conditioned.py")
    for name in names:
        shutil.copy2(ROOT/name, source/name)
    import scipy, sklearn
    manifest = {"study": "away_steps", "date": "2026-10-15", "command": [sys.executable, *sys.argv],
                "methods": METHODS, "seeds": list(range(args.seeds)), "scenarios": args.scenarios,
                "max_iterations": args.max_iter, "tolerance": args.tolerance,
                "trace_stride": 10, "budgets": BUDGETS,
                "initialization": "same original vertex; no compulsory LMO move; BT initial estimate 1",
                "information": "Q gets analytic H0,H1; BT does not; no reference inputs",
                "active_set": "signed coordinate atoms; exact cap deletion; no positive-weight pruning threshold",
                "additional_work": "one active-set scan per AFW update; atom scores counted separately",
                "python": sys.version, "platform": platform.platform(),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
                "thread_environment": {key: os.environ.get(key) for key in
                                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "problems": problems, "source_archive": "source/",
                "source_sha256": {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names},
                "result_sha256": {name: hashlib.sha256((args.output/name).read_bytes()).hexdigest()
                                  for name in ("summary.csv", "trace.csv", "budget.csv", "audit.csv")}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} runs; {len(audits)} full AFW-BT trajectories audited")


if __name__ == "__main__":
    main()
