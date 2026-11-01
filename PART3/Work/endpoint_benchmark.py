"""Reconstructed persistent-cache endpoint study from revised Appendix G."""
from __future__ import annotations

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
from endpoint_fw import PROBES, solve_endpoint
from multidata_benchmark import RATIOS, vertex


ROOT = Path(__file__).resolve().parent
PROJECT = next(p for p in ROOT.parents if p.name.upper() == "PART3")
DEFAULT_OUT = ROOT / "endpoint_recovery_results"


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_tree_hash(paths):
    result = {}
    expanded = []
    for path in paths:
        expanded.extend(path.rglob("*") if path.is_dir() else [path])
    for path in sorted(p for p in expanded if p.is_file()):
        try:
            frozen_root = next(parent for parent in path.resolve().parents if parent.name == "frozen")
            name = path.resolve().relative_to(frozen_root.parent).as_posix()
        except ValueError:
            name = path.name
        result[name] = sha256(path)
    return result


def selected_budget_row(trace, budget=1000):
    eligible = [row for row in trace if row["source_paid_evals"] <= budget]
    if not eligible:
        raise RuntimeError("No source observation within the paid-evaluation budget")
    return eligible[-1]


def replay_rows_from_trace(path: Path):
    grouped = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault((row["config_id"], row["method"]), []).append(row)
    out = []
    for (config_id, method), ordinary in grouped.items():
        if method != "ordinary":
            continue
        ordinary.sort(key=lambda r: int(r["iteration"]))
        for probe in ("dense", "dyadic"):
            other = grouped[(config_id, probe)]
            other.sort(key=lambda r: int(r["iteration"]))
            compared = min(len(ordinary), len(other))
            transition_count = max(0, min(compared, len(ordinary)-1, len(other)-1))
            value_fields = ("objective", "fw_gap", "certificate")
            step_fields = ("accepted_curvature", "step")
            def same(a, b, field):
                av, bv = float(a[field]), float(b[field])
                return (math.isnan(av) and math.isnan(bv)) or av == bv
            exact_values = all(same(a, b, f) for a, b in zip(ordinary[:compared], other[:compared])
                               for f in value_fields)
            exact_steps = all(same(a, b, f) for a, b in zip(ordinary[:transition_count], other[:transition_count])
                              for f in step_fields)
            deltas = [int(b["function_evals"]) - int(a["function_evals"])
                      for a, b in zip(ordinary[:transition_count], other[:transition_count])]
            min_delta, max_delta = min(deltas, default=0), max(deltas, default=0)
            # The active cache may visit fewer than all vertices; use task dimension.
            dims = {"seoul": 15, "metro": 12}
            dim_bound = 2 * dims[ordinary[0]["dataset"]]
            out.append({"config_id": config_id, "dataset": ordinary[0]["dataset"],
                        "radius_ratio": ordinary[0]["radius_ratio"], "start": ordinary[0]["start"],
                        "probe": probe, "paired_source_points": compared,
                        "paired_transitions": transition_count,
                        "common_source_prefix_exact": exact_values and exact_steps,
                        "value_fields_compared": ",".join(value_fields),
                        "transition_fields_compared": ",".join(step_fields),
                        "min_paid_eval_delta_at_common_updates": min_delta,
                        "max_paid_eval_delta_at_common_updates": max_delta,
                        "vertex_count_bound_2d": dim_bound,
                        "paid_eval_bound_satisfied": min_delta >= 0 and max_delta <= dim_bound})
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--finalize-existing", action="store_true",
                         help="seal hashes/manifest for a completed run whose finalization was interrupted")
    args = parser.parse_args()
    if args.finalize_existing:
        required = ("summary.csv", "trace.csv", "configurations.csv", "audit.csv", "replay_audit.csv")
        if not all((args.output / name).is_file() for name in required):
            parser.error("Existing result CSVs are incomplete")
        replay_rows = replay_rows_from_trace(args.output / "trace.csv")
        write_csv(args.output / "replay_audit.csv", replay_rows)
        frozen = args.output / "frozen"
        (frozen / "source").mkdir(parents=True, exist_ok=True)
        (frozen / "data").mkdir(parents=True, exist_ok=True)
        source_files = ("endpoint_fw.py", "endpoint_benchmark.py", "count_problems.py",
                        "multidata_benchmark.py", "problems.py", "fw.py")
        for name in source_files:
            shutil.copy2(ROOT / name, frozen / "source" / name)
        data_files = (PROJECT / "Data" / "seoul" / "SeoulBikeData.csv",
                      PROJECT / "Data" / "seoul" / "source.json",
                      PROJECT / "Data" / "metro" / "Metro_Interstate_Traffic_Volume.csv.gz",
                      PROJECT / "Data" / "metro" / "source.json")
        for path in data_files:
            target_dir = frozen / "data" / path.parent.name
            target_dir.mkdir(exist_ok=True)
            shutil.copy2(path, target_dir / path.name)
        import scipy
        manifest = {"study": "endpoint_probe_recovery", "provenance": "reconstructed 2026-11-01 from revised PART3 Appendix G and surviving modules/data; not recovered original source/results",
                    "target_protocol": "revised manuscript revision dated 2026-10-19; Appendix G, 108 matched configurations x ordinary/dense/dyadic persistent-cache methods = 324 executions",
                    "command": [sys.executable, *sys.argv], "datasets": list(DATASETS),
                    "radius_ratios": list(RATIOS), "starts": "all 2d signed coordinate vertices",
                    "methods": list(PROBES), "runs": 324, "objective_budget": args.budget,
                    "max_iterations": args.max_iter, "tolerance": args.tolerance,
                    "objective": "unchanged count_problems.make_count_problem normalized regularized Poisson objective",
                    "fallback": "Section 2 ordinary quadratic backtracking: L_-1=U0/D^2; source trial Lprev/2; alpha=min(1,G/(L||d||^2)); accept f(x+alpha d)<=f(x)-alpha G+L alpha^2||d||^2/2; double after rejection",
                    "probe": "f(s)<=f(x)-sigma G, sigma=1/2; dense every update; dyadic iff k+1 is power of two; rejection runs unchanged fallback",
                    "cache": "persistent exact deterministic Float64 objective cache for signed l1 vertices only; gradients and LMOs never cached; nonvertex function evaluations always paid",
                    "budget_rule": "certificate reported at latest source observation whose cumulative paid objective calls are <= budget; source_paid_evals is explicit in trace",
                    "solver_information": "objective, gradient, radius, initial vertex only; references are offline for audit and comparison; no Hessian/curvature constants supplied",
                    "prior_manifest_sha256": sha256(ROOT / "multidata_results" / "manifest.json"),
                    "prior_results_unchanged": True, "python": sys.version, "platform": platform.platform(),
                    "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
                    "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                    "frozen_source_sha256": file_tree_hash((frozen / "source").glob("*.py")),
                    "frozen_data_sha256": file_tree_hash((frozen / "data").glob("*")),
                    "result_sha256": {name: sha256(args.output / name) for name in required},
                    "replay_audit": {"paired_paths": len(replay_rows),
                                     "paired_source_points": sum(r["paired_source_points"] for r in replay_rows),
                                     "paired_transitions": sum(r["paired_transitions"] for r in replay_rows),
                                     "all_common_prefixes_exact": all(r["common_source_prefix_exact"] for r in replay_rows),
                                     "all_paid_call_bounds_satisfied": all(r["paid_eval_bound_satisfied"] for r in replay_rows)}}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Sealed manifest for the completed 324-run reconstruction")
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty output directory")
    if args.max_iter < 1 or args.budget < 1:
        parser.error("Iteration cap and objective budget must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    summary, trace_rows, config_rows, audit_rows, replay_rows = [], [], [], [], []
    reference_manifest = json.loads((ROOT / "multidata_results" / "manifest.json").read_text(encoding="utf-8"))
    for dataset in DATASETS:
        unc, residual = unconstrained_reference(dataset)
        critical_radius = float(np.linalg.norm(unc, 1))
        for ratio in RATIOS:
            base = make_count_problem(dataset, ratio * critical_radius)
            old = next(p for p in reference_manifest["problems"]
                       if p["dataset"] == dataset and p["radius_ratio"] == ratio)
            if (base.radius != old["radius"] or base.reference_lower != old["reference_lower"]
                    or base.reference_upper != old["reference_upper"]):
                raise RuntimeError(f"Recovered problem/reference mismatch: {dataset}, ratio={ratio}")
            config_rows.append({"dataset": dataset, "radius_ratio": ratio, "radius": base.radius,
                                "critical_radius": critical_radius, "reference_lower": base.reference_lower,
                                "reference_upper": base.reference_upper, "dimension": base.x0.size,
                                "starts": 2 * base.x0.size, "data_sha256": base.metadata["data_sha256"],
                                "preprocessing": base.metadata["preprocessing"]})
            for start in range(2 * base.x0.size):
                problem = replace(base, seed=start, x0=vertex(start, base.x0.size, base.radius))
                paired_traces = {}
                for probe in PROBES:
                    result = solve_endpoint(problem, probe, args.max_iter, args.tolerance, args.budget, 0.5)
                    identity = {"dataset": dataset, "radius_ratio": ratio, "start": start,
                                "method": probe, "config_id": f"{dataset}-r{ratio:g}-v{start}"}
                    for row in result.trace:
                        trace_rows.append({**identity, **row})
                    paired_traces[probe] = result.trace
                    at_budget = selected_budget_row(result.trace, args.budget)
                    last = result.trace[-1]
                    summary.append({**identity, "budget": args.budget,
                                    "budget_iteration": at_budget["iteration"],
                                    "budget_source_paid_evals": at_budget["source_paid_evals"],
                                    "budget_certificate": at_budget["certificate"],
                                    "budget_objective": at_budget["objective"],
                                    "budget_cache_hits": at_budget["source_cache_hits"],
                                    "budget_function_requests": at_budget["source_function_requests"],
                                    "budget_cache_vertices": at_budget["cache_vertices"],
                                    "terminal_iteration": last["iteration"],
                                    "terminal_certificate": last["certificate"],
                                    "terminal_function_evals": result.cache.paid,
                                    "terminal_function_requests": result.cache.requests,
                                    "terminal_cache_hits": result.cache.hits,
                                    "terminal_gradient_evals": last["gradient_evals"],
                                    "terminal_lmo_evals": last["lmo_evals"],
                                    "endpoint_tests": sum(r["endpoint_tested"] for r in result.trace),
                                    "endpoint_accepts": sum(r["endpoint_accepted"] for r in result.trace),
                                    "elapsed_seconds": result.elapsed_seconds})
                    audit_rows.append({**identity, "trace_observations": len(result.trace),
                                       "all_iterates_feasible": all(r["l1_norm"] <= base.radius + 1e-10
                                                                     for r in result.trace),
                                       "endpoint_accepts": sum(r["endpoint_accepted"] for r in result.trace),
                                       "persistent_vertex_cache": True,
                                       "cache_accounting_valid": result.cache.requests == result.cache.paid + result.cache.hits,
                                       "endpoint_cached_values": len(result.cache.values)})
                ordinary = paired_traces["ordinary"]
                for probe in ("dense", "dyadic"):
                    other = paired_traces[probe]
                    compared = min(len(ordinary), len(other))
                    value_fields = ("objective", "fw_gap", "certificate")
                    step_fields = ("accepted_curvature", "step")
                    exact_values = all(a[field] == b[field]
                                       for a, b in zip(ordinary[:compared], other[:compared])
                                       for field in value_fields)
                    transition_count = max(0, min(compared, len(ordinary) - 1, len(other) - 1))
                    exact_steps = all(
                        (np.isnan(a[field]) and np.isnan(b[field])) or a[field] == b[field]
                        for a, b in zip(ordinary[:transition_count], other[:transition_count])
                        for field in step_fields
                    )
                    exact = exact_values and exact_steps
                    differences = [b["function_evals"] - a["function_evals"]
                                   for a, b in zip(ordinary[:transition_count], other[:transition_count])]
                    max_paid_delta = max(differences, default=0)
                    min_paid_delta = min(differences, default=0)
                    bound = 2 * base.x0.size
                    replay_rows.append({"config_id": f"{dataset}-r{ratio:g}-v{start}",
                                        "dataset": dataset, "radius_ratio": ratio, "start": start,
                                        "probe": probe, "paired_source_points": compared,
                                        "paired_transitions": transition_count,
                                        "common_source_prefix_exact": exact,
                                        "value_fields_compared": ",".join(value_fields),
                                        "transition_fields_compared": ",".join(step_fields),
                                        "min_paid_eval_delta_at_common_updates": min_paid_delta,
                                        "max_paid_eval_delta_at_common_updates": max_paid_delta,
                                        "vertex_count_bound_2d": bound,
                                        "paid_eval_bound_satisfied": min_paid_delta >= 0 and max_paid_delta <= bound})
                if start % 8 == 7 or start == 2 * base.x0.size - 1:
                    print(f"{dataset} ratio={ratio:g}: {start + 1}/{2 * base.x0.size} starts complete",
                          flush=True)

    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "trace.csv", trace_rows)
    write_csv(args.output / "configurations.csv", config_rows)
    write_csv(args.output / "audit.csv", audit_rows)
    write_csv(args.output / "replay_audit.csv", replay_rows)

    frozen = args.output / "frozen"
    (frozen / "source").mkdir(parents=True)
    (frozen / "data").mkdir(parents=True)
    source_files = ("endpoint_fw.py", "endpoint_benchmark.py", "count_problems.py",
                    "multidata_benchmark.py", "problems.py", "fw.py")
    for name in source_files:
        shutil.copy2(ROOT / name, frozen / "source" / name)
    data_files = (PROJECT / "Data" / "seoul" / "SeoulBikeData.csv",
                  PROJECT / "Data" / "seoul" / "source.json",
                  PROJECT / "Data" / "metro" / "Metro_Interstate_Traffic_Volume.csv.gz",
                  PROJECT / "Data" / "metro" / "source.json")
    for path in data_files:
        target_dir = frozen / "data" / path.parent.name
        target_dir.mkdir(exist_ok=True)
        shutil.copy2(path, target_dir / path.name)
    import scipy
    manifest = {"study": "endpoint_probe_recovery", "provenance": "reconstructed 2026-11-01 from revised PART3 Appendix G and surviving modules/data; not recovered original source/results",
                "target_protocol": "revised manuscript revision dated 2026-10-19; Appendix G, 108 matched configurations x ordinary/dense/dyadic persistent-cache methods = 324 executions",
                "command": [sys.executable, *sys.argv], "datasets": list(DATASETS),
                "radius_ratios": list(RATIOS), "starts": "all 2d signed coordinate vertices",
                "methods": list(PROBES), "runs": len(summary), "objective_budget": args.budget,
                "max_iterations": args.max_iter, "tolerance": args.tolerance,
                "objective": "unchanged count_problems.make_count_problem normalized regularized Poisson objective",
                "fallback": "Section 2 ordinary quadratic backtracking: L_-1=U0/D^2; source trial Lprev/2; alpha=min(1,G/(L||d||^2)); accept f(x+alpha d)<=f(x)-alpha G+L alpha^2||d||^2/2; double after rejection",
                "probe": "f(s)<=f(x)-sigma G, sigma=1/2; dense every update; dyadic iff k+1 is power of two; rejection runs unchanged fallback",
                "cache": "persistent exact deterministic Float64 objective cache for signed l1 vertices only; gradients and LMOs never cached; nonvertex function evaluations always paid",
                "budget_rule": "certificate reported at latest source observation whose cumulative paid objective calls are <= budget; source_paid_evals is explicit in trace",
                "solver_information": "objective, gradient, radius, initial vertex only; references are offline for audit and comparison; no Hessian/curvature constants supplied",
                "prior_manifest_sha256": sha256(ROOT / "multidata_results" / "manifest.json"),
                "prior_results_unchanged": True, "python": sys.version, "platform": platform.platform(),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
                "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                "frozen_source_sha256": file_tree_hash((frozen / "source").glob("*.py")),
                "frozen_data_sha256": file_tree_hash((frozen / "data").glob("*")),
                "result_sha256": {name: sha256(args.output / name) for name in
                                  ("summary.csv", "trace.csv", "configurations.csv", "audit.csv", "replay_audit.csv")},
                    "replay_audit": {"paired_paths": len(replay_rows),
                                     "paired_source_points": sum(r["paired_source_points"] for r in replay_rows),
                                     "paired_transitions": sum(r["paired_transitions"] for r in replay_rows),
                                     "all_common_prefixes_exact": all(r["common_source_prefix_exact"] for r in replay_rows),
                                     "all_paid_call_bounds_satisfied": all(r["paid_eval_bound_satisfied"] for r in replay_rows)}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} executions; {len(trace_rows)} source observations")


if __name__ == "__main__":
    main()
