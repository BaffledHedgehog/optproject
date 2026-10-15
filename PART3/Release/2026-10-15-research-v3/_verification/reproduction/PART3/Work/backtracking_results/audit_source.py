"""Replay archived ordinary-backtracking runs and audit the phase-lag proof.

python PART3/Work/backtracking_audit.py PART3/Work/backtracking_results
The full trajectory is checked, including rows omitted by trace_stride=10.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys


def audit_run(result, h0, h1, diameter, method, tolerance):
    rows = result.trace
    u0 = rows[0]["certificate"]
    initial_l = u0/diameter**2 if method == "adaptive_L_initial" else 1.0
    if u0 <= tolerance:
        return {"lag_steps": 0, "lag_bound": 0, "trials": 0, "trial_bound": 0,
                "updates": 0, "update_bound": 0, "initial_certificate": u0}
    j = math.ceil(math.log2(u0/tolerance))
    p = math.ceil(math.log2(max(1, initial_l*diameter**2/u0)))
    lag = trials = 0
    previous = initial_l
    for row, following in zip(rows, rows[1:]):
        u, accepted = row["certificate"], row["trial_curvature"]
        phase = math.floor(math.log2(u0/u))
        assert 0 <= phase < j
        b = u0/2**phase
        cap = max(2*(h0+h1*b), b/(2*diameter**2))
        is_lag = accepted > cap*(1+1e-12)
        lag += is_lag
        if is_lag:
            assert row["line_search_trials"] == 1
        assert math.isclose(accepted, previous*2**(row["line_search_trials"]-2), rel_tol=1e-12)
        assert accepted <= max(previous/2, 2*(h0+h1*u))*(1+1e-12)
        assert following["objective"] <= row["objective"] - 0.5*row["step"]*row["fw_gap"] + 1e-11
        assert following["certificate"] <= u+1e-11
        previous = accepted
        trials += row["line_search_trials"]
    n = len(rows)-1
    assert lag <= p+j
    overhead = math.ceil(math.log2(max(1, 2*(h0+h1*u0)/initial_l)))
    assert rows[-1]["function_evals"] == trials+1
    if n:
        assert math.isclose(trials, 2*n+math.log2(previous/initial_l), abs_tol=1e-9)
    assert trials <= 2*n+overhead
    bound = p+math.ceil((6+16*h1*diameter**2)*j+32*h0*diameter**2/tolerance)
    assert n <= bound
    return {"lag_steps": lag, "lag_bound": p+j, "trials": trials,
            "trial_bound": 2*n+overhead, "updates": n, "update_bound": bound,
            "initial_certificate": u0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory/"manifest.json").read_text(encoding="utf-8"))
    archive = args.directory/manifest["source_archive"]
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((archive/name).read_bytes()).hexdigest() == digest
    # Use the experiment's exact source, not whichever solver is current.
    sys.path.insert(0, str(archive.resolve()))
    from fw import solve
    from problems import make_problem
    with (args.directory/"summary.csv").open(encoding="utf-8") as file:
        recorded = list(csv.DictReader(file))
    results = []
    for scenario in manifest["scenarios"]:
        for seed in manifest["seeds"]:
            problem = make_problem(seed, scenario)
            for method in ("adaptive_L", "adaptive_L_initial"):
                if method not in manifest["methods"]:
                    continue
                result = solve(problem, method, manifest["max_iterations"], manifest["tolerance"])
                saved = next(r for r in recorded if r["scenario"] == scenario
                             and int(r["seed"]) == seed and r["method"] == method)
                for key in ("function_evals", "gradient_evals", "lmo_evals", "certificate", "objective"):
                    assert float(saved[key]) == result.trace[-1][key], (scenario, seed, method, key)
                check = audit_run(result, problem.h0, problem.h1, 2*problem.radius,
                                  method, manifest["tolerance"])
                results.append({"scenario": scenario, "seed": seed, "method": method, **check})
    output = args.directory/"backtracking_audit.csv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    provenance = {"audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "audit_source": "audit_source.py",
                  "experiment_manifest_sha256": hashlib.sha256((args.directory/"manifest.json").read_bytes()).hexdigest(),
                  "audit_csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                  "runs": len(results), "checks": "full replay; lag budget; decrease; trial identity and bound"}
    if Path(__file__).resolve() != (args.directory/"audit_source.py").resolve():
        shutil.copy2(__file__, args.directory/"audit_source.py")
    (args.directory/"audit_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Audited {len(results)} archived trajectories; lag steps {sum(r['lag_steps'] for r in results)}")


if __name__ == "__main__":
    main()
