"""Validate archived provenance, experiment records, and adaptive oracle bounds.

python PART3/Work/validate_results.py
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent


def validate(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("study") == "finite_rational_proof_audit":
        return validate_proof_audit(directory, manifest)
    if manifest.get("study") == "differential_geometry_audit":
        return validate_differential_geometry(directory, manifest)
    archive = directory / manifest["source_archive"]
    for name, digest in manifest["source_sha256"].items():
        actual = hashlib.sha256((archive/name).read_bytes()).hexdigest()
        if actual != digest:
            raise AssertionError(f"Source archive mismatch: {directory}/{name}")
    if "cells" in manifest:
        return validate_factorial(directory, manifest)
    if manifest.get("study") == "away_steps":
        return validate_away(directory, manifest)
    if manifest.get("study") == "radius_and_units":
        return validate_robustness(directory, manifest)
    if manifest.get("study") == "constraint_transition":
        return validate_transition(directory, manifest)
    if manifest.get("study") == "additional_count_data":
        return validate_multidata(directory, manifest)
    if manifest.get("study") == "causal_certificates":
        return validate_certificates(directory, manifest)
    if manifest.get("study") == "count_ac_comparator":
        return validate_count_comparator(directory, manifest)
    with (directory / "summary.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    expected = {(s, str(seed), method) for s in manifest["scenarios"]
                for seed in manifest["seeds"] for method in manifest["methods"]}
    identities = {(r["scenario"], r["seed"], r["method"]) for r in rows}
    assert identities == expected and len(rows) == len(expected), "Missing/duplicate runs"
    problems = {(p["scenario"], p["seed"]): p for p in manifest["problems"]}
    adaptive_runs = 0
    for row in rows:
        n = int(row["iterations"])
        assert int(row["lmo_evals"]) == n+1+(row["method"] == "AC_FW")
        if row["method"] == "AC_FW":
            assert int(row["function_evals"]) == n+2
            assert int(row["accepted_steps"])+int(row["rejected_steps"]) == n
            assert int(row["gradient_evals"]) == 2+int(row["accepted_steps"])
        assert 0 <= n <= manifest["max_iterations"]
        assert (row["certified"] == "True") == (float(row["certificate"]) <= manifest["tolerance"])
        assert float(row["certificate"])+1e-10 >= float(row["error_lower"])
        assert float(row["error_lower"]) <= float(row["error_upper"])+1e-12
        for key in ("objective", "certificate", "error_upper", "elapsed_seconds"):
            assert math.isfinite(float(row[key]))
        if row["method"] != "adaptive_certificate":
            continue
        p = problems[(row["scenario"], int(row["seed"]))]
        # Diameters of the l1 balls in these explicitly specified benchmark
        # families. These are geometric inputs, not fitted curvature constants.
        diameters = {name: 3.0 for name in
                     ("interior", "boundary", "active", "active_scaled", "active_highdim", "bike_poisson")}
        diameters["breast_cancer"] = 4.0
        diameter = diameters[row["scenario"]]
        overhead = math.ceil(math.log2(max(1.0, 2*diameter**2*
                             (p["H0"]/manifest["tolerance"]+p["H1"]))))
        assert int(row["function_evals"]) <= 1+2*n+overhead
        assert int(row["gradient_evals"]) == n+1
        adaptive_runs += 1
    project = ROOT.parent
    if (directory/"audit_manifest.json").exists():
        audit = json.loads((directory/"audit_manifest.json").read_text(encoding="utf-8"))
        for name, key in ((audit["audit_source"], "audit_source_sha256"),
                          ("manifest.json", "experiment_manifest_sha256"),
                          ("backtracking_audit.csv", "audit_csv_sha256")):
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == audit[key]
        with (directory/"backtracking_audit.csv").open(encoding="utf-8") as file:
            checks = list(csv.DictReader(file))
        audit_expected = {item for item in expected if item[2] in ("adaptive_L", "adaptive_L_initial")}
        assert len(checks) == audit["runs"] == len(audit_expected)
        assert {(r["scenario"], r["seed"], r["method"]) for r in checks} == audit_expected
        for check in checks:
            for left, right in (("lag_steps", "lag_bound"), ("trials", "trial_bound"), ("updates", "update_bound")):
                assert int(check[left]) <= int(check[right])
    for p in manifest["problems"]:
        if p["scenario"] == "bike_poisson":
            digest = hashlib.sha256((project/"Data"/"bike_sharing"/"day.csv").read_bytes()).hexdigest()
            assert p["data_sha256"] == digest, "Dataset mismatch"
    print(f"{directory.name}: {len(rows)} unique runs; source hashes and records valid; "
          f"{adaptive_runs} adaptive function-call bounds checked")


def validate_count_comparator(directory, manifest):
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (directory/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    def key(r):
        return r["dataset"], float(r["radius_ratio"]), int(r["start"])
    problems = {(p["dataset"], p["radius_ratio"]): p for p in manifest["problems"]}
    expected = {(d, ratio, start) for (d, ratio), p in problems.items() for start in range(2*p["dimension"])}
    summary, trace, budgets, audits = [read(n) for n in ("summary.csv", "trace.csv", "budget.csv", "audit.csv")]
    assert len(summary) == len(audits) == len(expected) == 108
    assert {key(r) for r in summary} == {key(r) for r in audits} == expected
    ends = {key(r): r for r in summary}
    for r in [*summary, *trace, *budgets]:
        assert key(r) in expected and r["method"] == "AC_FW"
        n = int(r["iteration"])
        assert 0 <= n <= manifest["max_iterations"]
        assert int(r["function_evals"]) == int(r["lmo_evals"]) == n+2
        if "accepted_steps" in r:
            assert int(r["gradient_evals"]) == int(r["accepted_steps"])+2
            assert int(r["accepted_steps"])+int(r["rejected_steps"]) == n
        p = problems[(r["dataset"], float(r["radius_ratio"]))]
        assert float(r["lower_bound"]) <= p["reference_upper"]+2e-10
        assert float(r["certificate"])+2e-10 >= max(0, float(r["objective"])-p["reference_upper"])
        assert math.isclose(float(r["certificate"]), max(0, float(r["objective"])-float(r["lower_bound"])), abs_tol=1e-12)
        if "certified" in r:
            assert (r["certified"] == "True") == (float(r["certificate"]) <= manifest["tolerance"])
    expected_budgets = {(*k, o, b) for k in expected for o in
                        ("function_evals", "gradient_evals", "lmo_evals") for b in manifest["budgets"]}
    assert len(budgets) == len(expected_budgets)
    assert {(*key(r), r["oracle"], int(r["budget"])) for r in budgets} == expected_budgets
    for r in budgets:
        end, oracle, b = ends[key(r)], r["oracle"], int(r["budget"])
        assert int(r[oracle]) <= b
        assert (r["budget_reached_or_certified"] == "True") == (int(end[oracle]) >= b or end["certified"] == "True")
        if oracle == "function_evals" and b == 1000:
            assert r["budget_reached_or_certified"] == "True"
    for r in audits:
        assert r["all_sources_audited"] == "True"
        assert int(r["source_observations"]) == int(ends[key(r)]["iteration"])+1
    prior = ROOT/"multidata_results"/"manifest.json"
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == manifest["prior_manifest_sha256"]
    previous = json.loads(prior.read_text(encoding="utf-8"))
    assert manifest["problems"] == previous["problems"]
    for p in problems.values():
        assert hashlib.sha256((ROOT.parent/p["data_file"]).read_bytes()).hexdigest() == p["data_sha256"]
    if (directory/"report_manifest.json").exists():
        report = json.loads((directory/"report_manifest.json").read_text(encoding="utf-8"))
        for name, digest in report["artifact_sha256"].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    print(f"{directory.name}: {len(summary)} new AC-FW method runs on existing cases, {len(budgets)} charged prefixes valid")


def validate_differential_geometry(directory, manifest):
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    import runpy
    archived = runpy.run_path(str(directory/"audit_source.py"))
    replay, runs = archived["rational_finite_audit"]()
    replay.update(archived["positive_base_counterexample"]())
    assert replay == json.loads((directory/"checks.json").read_text(encoding="utf-8"))
    with (directory/"runs.csv").open(encoding="utf-8") as source:
        recorded = list(csv.DictReader(source))
    assert recorded == [{key: str(value) for key, value in row.items()} for row in runs]
    if (directory/"review_manifest.json").exists():
        review = json.loads((directory/"review_manifest.json").read_text(encoding="utf-8"))
        for name, digest in review["artifact_sha256"].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    print(f"{directory.name}: {replay['global_runs']} exact rational AFW runs, "
          f"{replay['planted_local_runs']} local drop runs and "
          f"{replay['exact_positive_base_trial_checks']} positive-base counterexample checks replay exactly")


def validate_proof_audit(directory, manifest):
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    import runpy
    archived = runpy.run_path(str(directory/"audit_source.py"))
    replay = {**archived["audit_phases"](), **archived["audit_bridge"](), **archived["audit_drops"]()}
    checks = json.loads((directory/"checks.json").read_text(encoding="utf-8"))
    assert replay == checks
    if (directory/"review_manifest.json").exists():
        review = json.loads((directory/"review_manifest.json").read_text(encoding="utf-8"))
        for name, digest in review["artifact_sha256"].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    print(f"{directory.name}: exact replay of {checks['phase_configurations']} configurations, "
          f"{checks['exact_curvature_states']} curvature states, {checks['rational_bridge_inequalities']} bridge checks valid; "
          "finite audit, not a formal proof")


def validate_certificates(directory, manifest):
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (directory/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    def path_key(row):
        return row["dataset"], float(row["radius_ratio"]), int(row["start"]), row["method"]
    def key(row):
        return (*path_key(row), row["profile"])
    problems = {(p["dataset"], p["radius_ratio"]): p for p in manifest["problems"]}
    paths = {(d, r, start, method) for (d, r), p in problems.items()
             for start in range(2*p["dimension"]) for method in manifest["methods"]}
    expected = {(*k, profile) for k in paths for profile in manifest["profiles"]}
    summary, final, trace, budgets, audits = [read(n) for n in
                                             ("summary.csv", "final.csv", "trace.csv", "budget.csv", "audit.csv")]
    stops, ends = {key(r): r for r in summary}, {key(r): r for r in final}
    assert len(summary) == len(final) == len(expected) and set(stops) == set(ends) == expected
    assert len(audits) == len(paths) and {path_key(r) for r in audits} == paths
    for row in [*summary, *final, *trace, *budgets]:
        assert key(row) in expected
        p = problems[(row["dataset"], float(row["radius_ratio"]))]
        n, profile = int(row["iteration"]), row["profile"]
        assert 0 <= n <= manifest["max_iterations"]
        extra = (n+1)//int(profile.split("_")[1]) if profile.startswith("block_") else 0
        assert int(row["certificate_lmo_evals"]) == extra
        assert int(row["direction_lmo_evals"]) == int(row["gradient_evals"]) == n+1
        assert int(row["lmo_evals"]) == n+1+extra
        assert int(row["function_evals"]) >= n+1
        for field in ("objective", "certificate", "lower_bound"):
            assert math.isfinite(float(row[field]))
        assert float(row["lower_bound"]) <= p["reference_upper"]+2e-10
        assert float(row["certificate"])+2e-10 >= max(0, float(row["objective"])-p["reference_upper"])
        assert math.isclose(float(row["certificate"]), max(0, float(row["objective"])-float(row["lower_bound"])), abs_tol=1e-12)
        if "scalar_certificate" in row:
            assert float(row["certificate"]) <= float(row["scalar_certificate"])+2e-10
        if "certified" in row:
            assert (row["certified"] == "True") == (float(row["certificate"]) <= manifest["tolerance"])
    for row in summary:
        baseline, end = stops[(*path_key(row), "scalar")], ends[key(row)]
        assert int(row["iteration"]) <= int(baseline["iteration"])
        assert int(row["function_evals"]) <= int(baseline["function_evals"])
        assert int(row["gradient_evals"]) <= int(baseline["gradient_evals"])
        assert float(row["error_upper"]) <= float(row["certificate"])+2e-7
        assert math.isclose(float(row["error_upper"]), max(0, float(row["objective"])-
                            problems[(row["dataset"], float(row["radius_ratio"]))]["reference_lower"]), abs_tol=1e-13)
        if row["certified"] != "True":
            assert row["iteration"] == end["iteration"]
        for field in ("iteration", "objective", "function_evals", "gradient_evals", "direction_lmo_evals"):
            assert end[field] == ends[(*path_key(row), "scalar")][field]
    budget_expected = {(*k, oracle, b) for k in expected
                       for oracle in ("function_evals", "gradient_evals", "lmo_evals") for b in manifest["budgets"]}
    assert len(budgets) == len(budget_expected)
    assert {(*key(r), r["oracle"], int(r["budget"])) for r in budgets} == budget_expected
    for row in budgets:
        stop, oracle, b = stops[key(row)], row["oracle"], int(row["budget"])
        assert int(row[oracle]) <= b and int(row["iteration"]) <= int(stop["iteration"])
        assert (row["budget_reached_or_certified"] == "True") == (int(stop[oracle]) >= b or stop["certified"] == "True")
    samples = {}
    for row in trace:
        samples.setdefault(key(row), []).append(row)
    for k, rows in samples.items():
        assert rows[0]["iteration"] == "0" and rows[-1]["iteration"] == ends[k]["iteration"]
        assert any(r["iteration"] == stops[k]["iteration"] for r in rows)
        assert all(float(b["certificate"]) <= float(a["certificate"])+2e-10 for a, b in zip(rows, rows[1:]))
        if stops[k]["certified"] == "True":
            assert all(float(r["certificate"]) > manifest["tolerance"] for r in rows
                       if int(r["iteration"]) < int(stops[k]["iteration"]))
    previous_dir = ROOT/"multidata_results"
    assert hashlib.sha256((previous_dir/"manifest.json").read_bytes()).hexdigest() == manifest["prior_manifest_sha256"]
    with (previous_dir/"summary.csv").open(encoding="utf-8") as file:
        previous = {path_key(r): r for r in csv.DictReader(file)}
    for k in paths:
        for field in ("iteration", "objective", "certificate", "function_evals", "gradient_evals", "lmo_evals"):
            assert stops[(*k, "scalar")][field] == previous[k][field]
    assert all(r["original_matches_prior"] == "True" and int(r["profiles"]) == len(manifest["profiles"])
               and int(r["source_observations"]) == int(r["original_updates"])+1
               and float(r["max_reference_excess"]) <= 2e-10 for r in audits)
    for p in problems.values():
        assert hashlib.sha256((ROOT.parent/p["data_file"]).read_bytes()).hexdigest() == p["data_sha256"]
    if (directory/"report_manifest.json").exists():
        report = json.loads((directory/"report_manifest.json").read_text(encoding="utf-8"))
        for name, digest in report["artifact_sha256"].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    print(f"{directory.name}: {len(paths)} exactly repeated paths, {len(summary)} causal stopping profiles, "
          f"{len(budgets)} paid budget prefixes valid")


def validate_multidata(directory, manifest):
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (directory/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    def key(row):
        return row["dataset"], float(row["radius_ratio"]), int(row["start"]), row["method"]
    problems = {(p["dataset"], p["radius_ratio"]): p for p in manifest["problems"]}
    assert set(problems) == {(d, r) for d in manifest["datasets"] for r in manifest["radius_ratios"]}
    expected = {(d, r, start, method) for (d, r), p in problems.items()
                for start in range(2*p["dimension"]) for method in manifest["methods"]}
    summary, trace, budgets, audits = [read(n) for n in ("summary.csv", "trace.csv", "budget.csv", "audit.csv")]
    assert len(summary) == len(expected) and {key(r) for r in summary} == expected
    assert len(audits) == len(expected) and {key(r) for r in audits} == expected
    finals = {key(r): r for r in summary}
    for row in summary:
        p = problems[(row["dataset"], float(row["radius_ratio"]))]
        for field, ref in (("error_lower", "reference_upper"), ("error_upper", "reference_lower")):
            assert math.isclose(float(row[field]), max(0, float(row["objective"])-p[ref]), rel_tol=1e-12, abs_tol=1e-14)
    for row in [*summary, *trace, *budgets]:
        assert key(row) in expected
        p = problems[(row["dataset"], float(row["radius_ratio"]))]
        n = int(row["iteration"])
        assert 0 <= n <= manifest["max_iterations"]
        assert int(row["gradient_evals"]) == int(row["lmo_evals"]) == n+1
        if row["method"].endswith("Q"):
            assert int(row["function_evals"]) == n+1
        else:
            assert int(row["function_evals"]) >= n+1
        for field in ("objective", "certificate", "lower_bound"):
            assert math.isfinite(float(row[field]))
        assert float(row["lower_bound"]) <= p["reference_upper"]+2e-10
        assert float(row["certificate"])+2e-10 >= max(0, float(row["objective"])-p["reference_upper"])
        if "l1_norm" in row:
            assert float(row["l1_norm"]) <= p["radius"]+1e-11
        if "certified" in row:
            assert (row["certified"] == "True") == (float(row["certificate"]) <= manifest["tolerance"])
        if row["method"].startswith("AFW") and "numerical_prunes" in row:
            assert int(row["numerical_prunes"]) == 0
            assert int(row["away_scans"]) == n
    expected_budgets = {(*k, oracle, b) for k in expected
                        for oracle in ("function_evals", "gradient_evals", "lmo_evals") for b in manifest["budgets"]}
    assert len(budgets) == len(expected_budgets)
    assert {(*key(r), r["oracle"], int(r["budget"])) for r in budgets} == expected_budgets
    for row in budgets:
        last, oracle, b = finals[key(row)], row["oracle"], int(row["budget"])
        assert int(row[oracle]) <= b
        assert (row["budget_reached_or_certified"] == "True") == (int(last[oracle]) >= b or last["certified"] == "True")
    for row in audits:
        last = finals[key(row)]
        assert int(row["updates"]) == int(last["iteration"])
        assert int(row["function_evals"]) == int(last["function_evals"]) <= int(row["function_bound"])
        assert float(row["max_gap_envelope_ratio"]) <= 1+1e-5
    for p in problems.values():
        assert p["H0"] > 0 and p["H1"] > 0
        assert 0 <= p["reference_upper"]-p["reference_lower"] <= 2e-7
        assert p["reference_l1_norm"] <= p["radius"]+1e-11
        assert math.isclose(p["radius"], p["radius_ratio"]*p["critical_radius"], rel_tol=1e-15)
        assert hashlib.sha256((ROOT.parent/p["data_file"]).read_bytes()).hexdigest() == p["data_sha256"]
        if p["regime"] == "active":
            assert p["estimated_multiplier"] > 0
            assert abs(p["reference_l1_norm"]-p["radius"]) < 1e-8
        else:
            assert p["reference_l1_norm"] < p["radius"]-1e-6
    if (directory/"report_manifest.json").exists():
        report = json.loads((directory/"report_manifest.json").read_text(encoding="utf-8"))
        for name, digest in report["artifact_sha256"].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
        diagnostics = read("reference_diagnostics.csv")
        assert len(diagnostics) == len(expected) and {key(r) for r in diagnostics} == expected
        for row in diagnostics:
            original = finals[key(row)]
            assert row["reference_error_upper"] == original["error_upper"]
            assert row["online_certified"] == original["certified"]
            assert (row["reference_within_target"] == "True") == (float(original["error_upper"]) <= manifest["tolerance"])
    print(f"{directory.name}: {len(summary)} runs, {len(budgets)} oracle prefixes, {len(audits)} full audits valid; all signed coordinate starts present")


def validate_transition(directory, manifest):
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (directory/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    def key(row):
        return float(row["radius_ratio"]), int(row["seed"]), row["method"]
    expected = {(ratio, seed, method) for ratio in manifest["radius_ratios"]
                for seed in manifest["seeds"] for method in manifest["methods"]}
    problems = {p["radius_ratio"]: p for p in manifest["problems"]}
    assert len(problems) == len(manifest["radius_ratios"])
    summary, trace, budgets, accuracy, audits = [read(n) for n in
                                               ("summary.csv", "trace.csv", "budget.csv", "accuracy.csv", "audit.csv")]
    assert len(summary) == len(expected) and {key(r) for r in summary} == expected
    assert len(audits) == len(expected) and {key(r) for r in audits} == expected
    finals = {key(r): r for r in summary}
    for row in [*summary, *trace, *budgets, *accuracy]:
        assert key(row) in expected
        p = problems[float(row["radius_ratio"])]
        n = int(row["iteration"])
        assert 0 <= n <= manifest["max_iterations"]
        assert int(row["gradient_evals"]) == int(row["lmo_evals"]) == n+1
        assert int(row["function_evals"]) >= n+1
        assert float(row["certificate"]) >= 0 and math.isfinite(float(row["certificate"]))
        if "objective" in row:
            assert float(row["certificate"])+2e-10 >= max(0, float(row["objective"])-p["reference_upper"])
            assert float(row["lower_bound"]) <= p["reference_upper"]+2e-10
        if "l1_norm" in row:
            assert float(row["l1_norm"]) <= p["radius"]+1e-11
        if "certified" in row:
            epsilon = float(row["epsilon"]) if "epsilon" in row else manifest["tolerance"]
            assert (row["certified"] == "True") == (float(row["certificate"]) <= epsilon)
    budget_keys = {(*key(r), r["oracle"], int(r["budget"])) for r in budgets}
    assert len(budgets) == len(expected)*len(manifest["budgets"])*3 == len(budget_keys)
    assert budget_keys == {(*k, oracle, b) for k in expected for b in manifest["budgets"]
                           for oracle in ("function_evals", "gradient_evals", "lmo_evals")}
    for row in budgets:
        oracle, b = row["oracle"], int(row["budget"])
        assert int(row[oracle]) <= b
        last = finals[key(row)]
        assert (row["budget_reached_or_certified"] == "True") == (int(last[oracle]) >= b or last["certified"] == "True")
    assert len(accuracy) == len(expected)*len(manifest["accuracies"])
    assert {(*key(r), float(r["epsilon"])) for r in accuracy} == {(*k, e) for k in expected for e in manifest["accuracies"]}
    for row in accuracy:
        assert (row["certified_by_1000_updates"] == "True") == (row["certified"] == "True" and int(row["iteration"]) <= 1000)
        eta, epsilon = float(row["new_error_threshold"]), float(row["epsilon"])
        p = problems[float(row["radius_ratio"])]
        assert 2*eta <= epsilon*(1+1e-12)
        assert 2*(2*p["radius"])**2*(p["H0"]+p["H1"]*eta)*eta <= epsilon**2*(1+1e-12)
    for row in audits:
        last = finals[key(row)]
        assert int(row["function_evals"]) == int(last["function_evals"]) == int(row["trials"])+1
        assert int(row["function_evals"]) <= int(row["function_bound"])
        assert int(row["updates"]) == int(last["iteration"])
        assert float(row["max_gap_envelope_ratio"]) <= 1+1e-5
    critical = manifest["critical_radius"]
    for p in problems.values():
        assert math.isclose(p["radius"], p["radius_ratio"]*critical["estimate"], rel_tol=1e-15)
        assert p["reference_l1_norm"] <= p["radius"]+1e-11
        assert p["reference_upper"]-p["reference_lower"] <= 2e-7
        assert p["kkt_residual_inf"] <= 1e-6
        if p["regime"] == "active":
            assert p["radius"] < critical["lower_estimate"]
            assert p["estimated_multiplier"] > 0
        elif p["regime"] == "interior":
            assert p["radius"] > critical["upper_estimate"]
    assert hashlib.sha256((ROOT.parent/"Data"/"bike_sharing"/"day.csv").read_bytes()).hexdigest() == next(iter(problems.values()))["data_sha256"]
    if (directory/"report_manifest.json").exists():
        report = json.loads((directory/"report_manifest.json").read_text(encoding="utf-8"))
        for name, digest in report["artifact_sha256"].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
        starts = read("distinct_starts.csv")
        assert {int(r["seed"]) for r in starts} == set(manifest["seeds"])
        representatives = {int(r["representative_seed"]) for r in starts}
        assert len(representatives) == report["distinct_starts"]
        assert report["distinct_trajectories"] == len(representatives)*len(manifest["radius_ratios"])*len(manifest["methods"])
        duplicate_rows = read("duplicate_checks.csv") if report["duplicate_trajectories_checked"] else []
        assert len(duplicate_rows) == report["duplicate_trajectories_checked"] == len(summary)-report["distinct_trajectories"]
        for row in duplicate_rows:
            ratio, seed, method = key(row)
            rep = int(row["representative_seed"])
            assert row["saved_records_match"] == "True"
            left, right = finals[(ratio, seed, method)], finals[(ratio, rep, method)]
            assert all(left[k] == right[k] for k in left if k not in ("seed", "elapsed_seconds"))
    print(f"{directory.name}: {len(summary)} trajectories, {len(budgets)} oracle prefixes, "
          f"{len(accuracy)} accuracy prefixes, {len(audits)} full audits valid")


def validate_robustness(directory, manifest):
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (directory/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    def key(row):
        return (row["scenario"], float(row["radius_factor"]), int(row["seed"]),
                float(row["scale"]), row["method"])
    expected = {(s, factor, seed, scale, method) for s in manifest["scenarios"]
                for factor in manifest["radius_factors"] for seed in manifest["seeds"]
                for scale in manifest["objective_scales"] for method in manifest["methods"]}
    problems = {(p["scenario"], p["radius_factor"], p["seed"]): p for p in manifest["problems"]}
    assert len(problems) == len(manifest["scenarios"])*len(manifest["radius_factors"])*len(manifest["seeds"])
    summary, trace, budgets, audits = [read(name) for name in ("summary.csv", "trace.csv", "budget.csv", "audit.csv")]
    assert len(summary) == len(expected) and {key(row) for row in summary} == expected
    assert len(audits) == len(expected) and {key(row) for row in audits} == expected
    for row in [*summary, *trace, *budgets]:
        s, factor, seed, scale, method = key(row)
        assert key(row) in expected
        p = problems[(s, factor, seed)]
        epsilon = scale*manifest["base_tolerance"]
        slack = scale*2e-10
        n = int(row["iteration"])
        assert 0 <= n <= manifest["max_iterations"]
        assert int(row["gradient_evals"]) == int(row["lmo_evals"]) == n+1
        assert int(row["function_evals"]) >= n+1
        assert float(row["lower_bound"]) <= scale*p["reference_upper"]+slack
        assert float(row["certificate"])+slack >= max(0, float(row["objective"])-scale*p["reference_upper"])
        if "certified" in row:
            assert (row["certified"] == "True") == (float(row["certificate"]) <= epsilon)
        if "normalized_certificate" in row:
            assert math.isclose(float(row["normalized_certificate"]), float(row["certificate"])/scale, rel_tol=1e-12, abs_tol=1e-14)
        if "l1_norm" in row:
            assert float(row["l1_norm"]) <= p["radius"]+1e-11
        if method.startswith("AFW"):
            assert int(row["away_scans"]) == n
            if "numerical_prunes" in row:
                assert int(row["numerical_prunes"]) == 0
        for name in ("objective", "certificate", "lower_bound"):
            assert math.isfinite(float(row[name]))
    budget_expected = {(*k, oracle, b) for k in expected
                       for oracle in ("function_evals", "gradient_evals", "lmo_evals") for b in manifest["budgets"]}
    assert len(budgets) == len(budget_expected)
    assert {(*key(row), row["oracle"], int(row["budget"])) for row in budgets} == budget_expected
    for row in budgets:
        assert int(row[row["oracle"]]) <= int(row["budget"])
    last_by_key = {key(row): row for row in summary}
    for row in audits:
        last = last_by_key[key(row)]
        assert int(row["function_evals"]) == int(row["trials"])+1 == int(last["function_evals"])
        assert int(row["updates"]) == int(last["iteration"])
        assert int(row["function_evals"]) <= int(row["function_bound"])
        if int(row["updates"]):
            assert math.isclose(int(row["trials"]), 2*int(row["updates"])+math.log2(float(row["final_estimate"])/float(row["initial_estimate"])), abs_tol=1e-8)
    for p in problems.values():
        assert p["reference_l1_norm"] <= p["radius"]+1e-11
        assert 0 <= p["reference_upper"]-p["reference_lower"] <= 2e-6
        if p["scenario"] == "bike_poisson":
            assert hashlib.sha256((ROOT.parent/"Data"/"bike_sharing"/"day.csv").read_bytes()).hexdigest() == p["data_sha256"]
    # Old defaults must still reproduce the previous trajectories.
    matched = 0
    for previous_name, method_map in (("away_results", {"FW_BT_fixed": "FW_BT", "AFW_BT_fixed": "AFW_BT"}),
                                      ("backtracking_results", {"FW_BT_certificate": "adaptive_L_initial"})):
        previous_dir = directory.parent/previous_name
        if not (previous_dir/"manifest.json").exists():
            continue
        old = json.loads((previous_dir/"manifest.json").read_text(encoding="utf-8"))
        if old["max_iterations"] != manifest["max_iterations"] or old["tolerance"] != manifest["base_tolerance"]:
            continue
        with (previous_dir/"summary.csv").open(encoding="utf-8") as file:
            previous = {(r["scenario"], int(r["seed"]), r["method"]): r for r in csv.DictReader(file)}
        for row in summary:
            if float(row["radius_factor"]) != 1 or float(row["scale"]) != 1 or row["method"] not in method_map:
                continue
            old_key = (row["scenario"], int(row["seed"]), method_map[row["method"]])
            if old_key in previous:
                for name in ("objective", "certificate", "function_evals", "gradient_evals", "lmo_evals"):
                    assert row[name] == previous[old_key][name], (old_key, name)
                matched += 1
    if (directory/"report_manifest.json").exists():
        report = json.loads((directory/"report_manifest.json").read_text(encoding="utf-8"))
        for name, digest in report["artifact_sha256"].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    print(f"{directory.name}: {len(summary)} paired runs, {len(budgets)} budget prefixes, "
          f"{len(audits)} full-trajectory audits valid; {matched} old baselines match exactly")


def validate_factorial(directory, manifest):
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (directory/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    rows = read("summary.csv")
    expected = {(s, str(seed), cell) for s in manifest["scenarios"]
                for seed in manifest["seeds"] for cell in manifest["cells"]}
    assert len(rows) == len(expected)
    assert {(r["scenario"], r["seed"], r["cell"]) for r in rows} == expected
    problems = {(p["scenario"], str(p["seed"])): p for p in manifest["problems"]}
    for row in rows:
        n = int(row["iteration"])
        assert 0 <= n <= manifest["max_iterations"]
        assert int(row["lmo_evals"]) == n+2
        assert int(row["accepted_steps"])+int(row["rejected_steps"]) == n
        assert int(row["gradient_evals"]) == 2+int(row["accepted_steps"])
        if row["cell"].startswith("AC"):
            assert int(row["function_evals"]) == n+2
            assert int(row["trial_rejections"]) == int(row["rejected_steps"])
        else:
            assert int(row["function_evals"]) == n+2+int(row["trial_rejections"])
            assert int(row["rejected_steps"]) == 0
        assert (row["certified"] == "True") == (float(row["certificate"]) <= manifest["tolerance"])
        p = problems[(row["scenario"], row["seed"])]
        assert float(row["l1_norm"]) <= p["radius"]+1e-11
        assert float(row["lower_bound"]) <= p["reference_upper"]+1e-10
        assert float(row["objective"]) >= p["reference_lower"]-1e-10
        for key in ("objective", "certificate", "elapsed_seconds"):
            assert math.isfinite(float(row[key]))
    budget_rows = read("budget.csv")
    oracles = ("function_evals", "gradient_evals", "lmo_evals")
    budget_expected = {(s, seed, cell, oracle, str(b)) for s, seed, cell in expected
                       for oracle in oracles for b in manifest["budgets"]}
    lookup = {(r["scenario"], r["seed"], r["cell"], r["oracle"], r["budget"]): r for r in budget_rows}
    assert len(budget_rows) == len(budget_expected) and set(lookup) == budget_expected
    for row in budget_rows:
        assert int(row[row["oracle"]]) <= int(row["budget"])
        assert (row["certified"] == "True") == (float(row["certificate"]) <= manifest["tolerance"])
        assert float(row["certificate"])+1e-10 >= float(row["error_lower"])
        assert float(row["error_lower"]) <= float(row["error_upper"])+1e-12
    comparisons = read("paired_comparisons.csv")
    contrasts = (("AC_vertex", "AC_original"), ("BT_vertex", "BT_original"),
                 ("BT_original", "AC_original"), ("BT_vertex", "AC_vertex"))
    contrast_expected = {(s, a, b, o) for s in manifest["scenarios"] for a, b in contrasts for o in oracles}
    assert len(comparisons) == len(contrast_expected)
    assert {(r["scenario"], r["numerator"], r["denominator"], r["oracle"]) for r in comparisons} == contrast_expected
    for row in comparisons:
        paired = [(lookup[(row["scenario"], str(seed), row["numerator"], row["oracle"], "1000")],
                   lookup[(row["scenario"], str(seed), row["denominator"], row["oracle"], "1000")])
                  for seed in manifest["seeds"]]
        ratios = [max(float(a["certificate"]), 1e-15)/max(float(b["certificate"]), 1e-15) for a, b in paired]
        assert math.isclose(float(row["paired_median_certificate_ratio"]), median(ratios), rel_tol=1e-12)
        assert float(row["bootstrap_025"]) <= float(row["bootstrap_975"])
        assert int(row["numerator_certified"]) == sum(a["certified"] == "True" for a, _ in paired)
        assert int(row["denominator_certified"]) == sum(b["certified"] == "True" for _, b in paired)
    native_path = directory.parent/"backtracking_results"/"summary.csv"
    matched = 0
    native_manifest = native_path.parent/"manifest.json"
    native_parameters = json.loads(native_manifest.read_text(encoding="utf-8")) if native_manifest.exists() else {}
    if native_path.exists() and all(native_parameters.get(key) == manifest[key]
                                    for key in ("max_iterations", "tolerance")):
        with native_path.open(encoding="utf-8") as file:
            native = {(r["scenario"], r["seed"]): r for r in csv.DictReader(file) if r["method"] == "AC_FW"}
        for row in rows:
            if row["cell"] != "AC_vertex" or (row["scenario"], row["seed"]) not in native:
                continue
            previous = native[(row["scenario"], row["seed"])]
            for key in (*oracles, "objective", "certificate", "fw_gap"):
                assert row[key] == previous[key], (row["scenario"], row["seed"], key)
            matched += 1
    if (directory/"face_manifest.json").exists():
        face = json.loads((directory/"face_manifest.json").read_text(encoding="utf-8"))
        assert hashlib.sha256((directory/face["source"]).read_bytes()).hexdigest() == face["source_sha256"]
        assert hashlib.sha256((directory/"manifest.json").read_bytes()).hexdigest() == face["experiment_manifest_sha256"]
        for name, digest in face["result_sha256"].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
        face_rows = read("face_summary.csv")
        face_expected = {r for r in expected if r[0] in ("active", "active_scaled", "active_highdim")}
        assert len(face_rows) == face["runs"] == len(face_expected)
        assert {(r["scenario"], r["seed"], r["cell"]) for r in face_rows} == face_expected
        assert all(float(r["max_identity_residual"]) < 1e-11 for r in face_rows)
    if (directory/"report_manifest.json").exists():
        report = json.loads((directory/"report_manifest.json").read_text(encoding="utf-8"))
        for name, key in ((report["source"], "source_sha256"), ("results.md", "report_sha256"),
                          ("manifest.json", "experiment_manifest_sha256")):
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == report[key]
        for name, digest in report["figure_sha256"].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    for p in manifest["problems"]:
        if p["scenario"] == "bike_poisson":
            assert hashlib.sha256((ROOT.parent/"Data"/"bike_sharing"/"day.csv").read_bytes()).hexdigest() == p["data_sha256"]
    print(f"{directory.name}: {len(rows)} runs, {len(budget_rows)} budget prefixes and {len(comparisons)} paired comparisons valid; "
          f"{matched} native AC-FW runs match earlier records exactly")


def validate_away(directory, manifest):
    for name, digest in manifest["result_sha256"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    def read(name):
        with (directory/name).open(encoding="utf-8") as file:
            return list(csv.DictReader(file))
    rows = read("summary.csv")
    expected = {(s, str(seed), method) for s in manifest["scenarios"]
                for seed in manifest["seeds"] for method in manifest["methods"]}
    lookup = {(r["scenario"], r["seed"], r["method"]): r for r in rows}
    assert len(rows) == len(expected) and set(lookup) == expected
    problems = {(p["scenario"], str(p["seed"])): p for p in manifest["problems"]}
    oracles = ("function_evals", "gradient_evals", "lmo_evals")
    for row in rows:
        n = int(row["iteration"])
        assert 0 <= n <= manifest["max_iterations"]
        assert int(row["gradient_evals"]) == int(row["lmo_evals"]) == n+1
        assert int(row["function_evals"]) >= n+1
        if row["method"].endswith("_Q"):
            assert int(row["function_evals"]) == n+1
        if row["method"].startswith("AFW"):
            assert int(row["away_scans"]) == n
            assert int(row["fw_steps"])+int(row["away_steps"]) == n
            assert int(row["good_steps"])+int(row["drop_steps"]) == n
            assert int(row["drop_steps"]) <= int(row["fw_steps"])+int(row["initial_active_size"])-1
            assert int(row["numerical_prunes"]) == 0
            assert abs(float(row["weight_sum"])-1) < 1e-12
            assert float(row["minimum_weight"]) > 0
            assert float(row["representation_error"]) < 1e-12
        p = problems[(row["scenario"], row["seed"])]
        assert float(row["l1_norm"]) <= p["radius"]+1e-11
        assert float(row["lower_bound"]) <= p["reference_upper"]+1e-10
        assert float(row["certificate"])+1e-10 >= float(row["error_lower"])
        assert (row["certified"] == "True") == (float(row["certificate"]) <= manifest["tolerance"])
        for key in ("certificate", "objective", "error_lower", "error_upper", "elapsed_seconds"):
            assert math.isfinite(float(row[key]))
    budget_rows = read("budget.csv")
    budget_expected = {(s, seed, method, oracle, str(b)) for s, seed, method in expected
                       for oracle in oracles for b in manifest["budgets"]}
    assert len(budget_rows) == len(budget_expected)
    assert {(r["scenario"], r["seed"], r["method"], r["oracle"], r["budget"]) for r in budget_rows} == budget_expected
    for row in budget_rows:
        assert int(row[row["oracle"]]) <= int(row["budget"])
        assert (row["certified"] == "True") == (float(row["certificate"]) <= manifest["tolerance"])
        p = problems[(row["scenario"], row["seed"])]
        assert float(row["certificate"])+1e-10 >= max(0, float(row["objective"])-p["reference_upper"])
    audits = read("audit.csv")
    assert {(r["scenario"], r["seed"], r["method"]) for r in audits} == {key for key in expected if key[2] == "AFW_BT"}
    for row in audits:
        for left, right in (("updates", "convex_update_bound"), ("lag_steps", "lag_bound"),
                            ("function_evals", "function_bound")):
            assert int(row[left]) <= int(row[right])
    previous_dir = directory.parent/"adaptive_results"
    matched = 0
    if (previous_dir/"manifest.json").exists():
        previous_manifest = json.loads((previous_dir/"manifest.json").read_text(encoding="utf-8"))
        if all(previous_manifest[key] == manifest[key] for key in ("max_iterations", "tolerance")):
            with (previous_dir/"summary.csv").open(encoding="utf-8") as file:
                previous = {(r["scenario"], r["seed"], r["method"]): r for r in csv.DictReader(file)}
            for row in rows:
                if row["method"].startswith("AFW"):
                    continue
                method = "adaptive_L" if row["method"] == "FW_BT" else "quadratic_memory"
                key = (row["scenario"], row["seed"], method)
                if key in previous:
                    for column in (*oracles, "objective", "certificate", "fw_gap"):
                        assert row[column] == previous[key][column], (*key, column)
                    matched += 1
    for p in manifest["problems"]:
        if p["scenario"] == "bike_poisson":
            assert hashlib.sha256((ROOT.parent/"Data"/"bike_sharing"/"day.csv").read_bytes()).hexdigest() == p["data_sha256"]
    if (directory/"report_manifest.json").exists():
        report = json.loads((directory/"report_manifest.json").read_text(encoding="utf-8"))
        for name, key in ((report["source"], "source_sha256"), ("manifest.json", "experiment_manifest_sha256"),
                          ("results.md", "report_sha256")):
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == report[key]
        for name, digest in report["figure_sha256"].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    print(f"{directory.name}: {len(rows)} runs, {len(budget_rows)} budget prefixes, {len(audits)} full AFW audits valid; "
          f"{matched} FW baselines match earlier records exactly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="*",
                        default=[ROOT/"benchmark_results", ROOT/"adaptive_results", ROOT/"adaptive_ablation",
                                 ROOT/"backtracking_results", ROOT/"initialization_factorial", ROOT/"away_results",
                                 ROOT/"robustness_results", ROOT/"constraint_results", ROOT/"multidata_results",
                                 ROOT/"certificate_results", ROOT/"proof_audit_results", ROOT/"count_comparator_results",
                                 ROOT/"differential_geometry_results"])
    args = parser.parse_args()
    for directory in args.directories:
        validate(directory)


if __name__ == "__main__":
    main()
