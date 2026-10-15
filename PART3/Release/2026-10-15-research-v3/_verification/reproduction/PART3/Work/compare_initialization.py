"""Paired descriptive bootstrap for the fixed initialization-ablation study."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "adaptive_ablation"


def main():
    with (ROOT/"summary.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    result = []
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        adaptive = {int(r["seed"]): r for r in rows
                    if r["scenario"] == scenario and r["method"] == "adaptive_certificate"}
        initial = {int(r["seed"]): r for r in rows
                   if r["scenario"] == scenario and r["method"] == "adaptive_L_initial"}
        assert adaptive.keys() == initial.keys()
        ratios = np.array([float(adaptive[s]["function_evals"])/float(initial[s]["function_evals"])
                           for s in sorted(adaptive)])
        rng = np.random.default_rng(7103)
        bootstrap = np.median(rng.choice(ratios, size=(10000, len(ratios))), axis=1)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        result.append({"scenario": scenario, "seeds": len(ratios),
                       "paired_median_function_call_ratio": float(np.median(ratios)),
                       "bootstrap_025": float(low), "bootstrap_975": float(high),
                       "certified_adaptive": sum(r["certified"] == "True" for r in adaptive.values()),
                       "certified_same_initialization": sum(r["certified"] == "True" for r in initial.values())})
    with (ROOT/"initialization_comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    metadata = {
        "bootstrap_seed": 7103, "bootstrap_samples": 10000,
        "statistic": "median of per-seed function-call ratios, including capped runs",
        "numerator": "adaptive_certificate", "denominator": "adaptive_L_initial",
        "interpretation": "Descriptive percentile interval, no multiplicity correction; n=10 starts per case.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256((ROOT/"summary.csv").read_bytes()).hexdigest(),
    }
    (ROOT/"comparison_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Recorded paired comparisons for {len(result)} scenarios")


if __name__ == "__main__":
    main()
