"""Post-run geometry diagnostics for the planted active l1-ball families.

Reference support is used only after solving, never for choosing a step.
The optimizer is replayed from the experiment's archived source.
"""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest_path = args.directory/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = args.directory/manifest["source_archive"]
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((archive/name).read_bytes()).hexdigest() == digest
    sys.path.insert(0, str(archive.resolve()))
    from initialization_factorial import solve_control, write_csv
    from problems import lmo, make_problem
    with (args.directory/"summary.csv").open(encoding="utf-8") as file:
        saved = {(r["scenario"], int(r["seed"]), r["cell"]): r for r in csv.DictReader(file)}
    summaries, sampled = [], []
    for scenario in manifest["scenarios"]:
        if scenario not in ("active", "active_scaled", "active_highdim"):
            continue
        for seed in manifest["seeds"]:
            problem = make_problem(seed, scenario)
            for cell in manifest["cells"]:
                points, gradients = [], []
                def gradient(x):
                    g = problem.gradient(x)
                    points.append(x.copy())
                    gradients.append(g.copy())
                    return g
                result = solve_control(replace(problem, gradient=gradient), cell,
                                       manifest["max_iterations"], manifest["tolerance"])
                final = result.trace[-1]
                record = saved[(scenario, seed, cell)]
                for key in ("objective", "certificate", "function_evals", "gradient_evals", "lmo_evals"):
                    assert final[key] == float(record[key]), (scenario, seed, cell, key)
                # Both the reference and its support enter only here.
                normal = np.sign(problem.reference_x)
                slack = lambda x: problem.radius-float(normal @ x)
                vertex_slack = slack(points[1])
                rows = []
                for row in result.trace:
                    count = row["accepted_steps"]
                    index = count+1 if count else (1 if cell.endswith("vertex") else 0)
                    x, g = points[index], gradients[index]
                    # This diagnostic reconstruction has no influence on the
                    # completed optimizer or its recorded oracle counts.
                    endpoint = lmo(g, problem.radius)
                    sigma, sigma_s = slack(x), slack(endpoint)
                    assert sigma >= -1e-11
                    multiplier = problem.metadata["kkt_multiplier"]
                    assert row["objective"] >= multiplier*0.4*sigma-1e-11
                    rows.append({"iteration": row["iteration"], "face_slack": sigma,
                                 "lmo_face_slack": sigma_s, "objective": row["objective"],
                                 "certificate": row["certificate"]})
                residuals = []
                for source, following, original_row in zip(rows, rows[1:], result.trace):
                    alpha = original_row["step"] if original_row["trial_accepted"] else 0.0
                    residual = following["face_slack"]-((1-alpha)*source["face_slack"]
                                                        + alpha*source["lmo_face_slack"])
                    assert abs(residual) < 1e-11
                    residuals.append(abs(residual))
                identity = {"scenario": scenario, "seed": seed, "cell": cell}
                threshold = 1e-9*problem.radius
                summaries.append({**identity, "vertex_in_face": vertex_slack <= threshold,
                                  "start_in_face": rows[0]["face_slack"] <= threshold,
                                  "initial_face_slack": rows[0]["face_slack"],
                                  "final_face_slack": rows[-1]["face_slack"],
                                  "maximum_face_slack": max(r["face_slack"] for r in rows),
                                  "off_face_iterates": sum(r["face_slack"] > threshold for r in rows),
                                  "full_accepted_steps": sum(r["step"] == 1 and r["trial_accepted"]
                                                             for r in result.trace[:-1]),
                                  "max_identity_residual": max(residuals, default=0.0)})
                sampled.extend({**identity, **r} for r in rows
                               if r["iteration"] % 10 == 0 or r is rows[-1])
    write_csv(args.directory/"face_summary.csv", summaries)
    write_csv(args.directory/"face_trace.csv", sampled)
    destination = args.directory/"face_diagnostics_source.py"
    if Path(__file__).resolve() != destination.resolve():
        shutil.copy2(__file__, destination)
    provenance = {"source": destination.name, "source_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                  "experiment_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  "result_sha256": {name: hashlib.sha256((args.directory/name).read_bytes()).hexdigest()
                                    for name in ("face_summary.csv", "face_trace.csv")},
                  "runs": len(summaries), "interpretation": "post-run planted reference geometry only"}
    (args.directory/"face_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Replayed {len(summaries)} controls; all face-slack identities and lower bounds hold")


if __name__ == "__main__":
    main()
