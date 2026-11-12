#!/usr/bin/env python3
"""One fixed v16 API run, after independent review; never import live code."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction


ROOT = Path(__file__).resolve().parent
FORBIDDEN = (
    "source_objective_calls", "source_gradient_calls", "full_LMO_calls",
    "LP_calls", "optimizer_updates", "stochastic_samples",
)
SCALAR_PARTS = (
    "bound_evaluations", "switch_evaluations", "loss_evaluations",
    "regime_evaluations", "algebraic_bound_evaluations",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value, *, exclusive: bool = True) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def checks_for(case, out, references, global_precision_unchanged):
    checks = []

    def check(label, condition):
        checks.append({"check": label, "passed": bool(condition)})

    expected = case["expected"]
    plan = out["plan"]
    work = out["work"]
    cheap = case["function"] == "algebraic_joint_bound"
    check("expected_status", out["status"] == expected["status"])
    check("global_precision_unchanged", global_precision_unchanged)
    check("no_source_or_optimizer_operations", all(work[k] == 0 for k in FORBIDDEN))
    check("scalar_counter_partition", work["scalar_evaluations_started"] == sum(work[k] for k in SCALAR_PARTS))
    check("root_updates_no_more_than_queries", work["bisection_updates"] <= work["root_sign_queries"])
    check("all_costs_nonnegative_integers", all(type(v) is int and v >= 0 for v in work.values())
          and all(type(v) is int and v >= 0 for v in out["elementary_evaluations"].values()))
    if "branch" in expected:
        check("expected_branch", plan.get("branch") == expected["branch"])
    if "working_gap" in expected:
        check("expected_clipped_working_gap", Fraction(plan.get("working_gap", "-1")) == Fraction(expected["working_gap"]))
    if "reason" in expected:
        check("expected_refusal_reason", out.get("reason") == expected["reason"])
    if "resource" in expected:
        check("expected_resource", out.get("refusal_details", {}).get("resource") == expected["resource"])
    if "regime" in expected:
        regime = plan.get("regime", {})
        check("expected_regime_decision", regime.get("decision") == expected["regime"])
        check("one_regime_evaluation", work["regime_evaluations"] == 1)
        check("one_regime_query", work["regime_sign_queries"] == 1)
        if "S_interval" in regime:
            slo, shi = map(Fraction, regime["S_interval"])
            if expected["regime"] == "nonpositive":
                check("certified_nonpositive_endpoint", slo <= shi <= 0)
            elif expected["regime"] == "positive":
                check("certified_positive_endpoint", 0 < slo <= shi)
            else:
                check("unresolved_interval_straddles_allowed_tests", slo <= 0 < shi)
    for name in ("S_interval", "tau_interval"):
        key = name + "_exact"
        if key in expected:
            actual = plan.get("regime", {}).get(name, [])
            check("exact_" + name, list(map(Fraction, actual)) == list(map(Fraction, expected[key])))
    if "positive_nodes" in expected:
        check("expected_positive_nodes", plan.get("positive_nodes", 0) == expected["positive_nodes"])
    if expected.get("no_switch"):
        check("no_switch_or_loss_evaluations", work["switch_evaluations"] == work["loss_evaluations"] == work["bisection_updates"] == 0)
    if expected.get("positive_grid"):
        check("positive_grid_intervals", plan.get("grid_intervals", 0) > 0)
        check("all_grid_endpoints_included", plan.get("positive_nodes") == plan.get("grid_intervals", 0) + 1)
        m = Fraction(out["inputs"]["m"])
        g, P = map(Fraction, (case["args"]["g"], case["args"]["P"]))
        check("actual_restricted_interval", list(map(Fraction, plan.get("radius_interval", []))) == [g*g/(P*P), g/m])
    if cheap:
        check("no_relative_envelope_contract", not any(k in out for k in ("requested_factor", "achieved_factor", "envelope_lower", "envelope_upper")))
    if out["status"] == "refused":
        if cheap:
            check("refusal_has_no_bound", out["source_error_upper"] is None)
        else:
            check("refusal_has_no_enclosure", all(out[k] is None for k in ("envelope_lower", "envelope_upper", "achieved_factor")))
        return checks
    if cheap:
        upper = Fraction(out["source_error_upper"])
        check("nonnegative_bound", upper >= 0)
        if "exact_upper" in expected:
            check("expected_exact_upper", upper == Fraction(expected["exact_upper"]))
        if expected.get("radical_upper_formula") == "2-7*sqrt(14)/24":
            radical_lower = (2-upper)*24/7
            check("exact_rational_radical_upper_check", radical_lower >= 0 and radical_lower*radical_lower <= 14)
    else:
        lower, upper = map(Fraction, (out["envelope_lower"], out["envelope_upper"]))
        achieved, requested = map(Fraction, (out["achieved_factor"], out["requested_factor"]))
        check("ordered_nonnegative_enclosure", 0 <= lower <= upper)
        check("exact_returned_ratio", achieved == (Fraction(1) if upper == 0 else upper/lower))
        check("requested_factor_met", achieved <= requested)
        if "exact" in expected:
            check("expected_exact_envelope", lower == upper == Fraction(expected["exact"]))
        if "reference" in expected:
            reference = references["points"][expected["reference"]]
            rlo, rhi = map(Fraction, reference["rational_interval"])
            check("historical_reference_contained", lower <= rlo <= rhi <= upper)
        if plan.get("positive_nodes", 0) > 0:
            check("completed_all_planned_nodes", len(out["trace"]) == plan["positive_nodes"] and all(node["status"] == "enclosed" for node in out["trace"]))
            check("root_budget_respected", work["bisection_updates"] <= plan["planned_bisections"])
            check("sqrt_counter_partition", out["elementary_evaluations"]["sqrt"] == work["node_coefficient_preparations"] + work["bound_evaluations"] + work["regime_evaluations"])
    if "upper_lt" in expected:
        check("expected_strict_upper", upper < Fraction(expected["upper_lt"]))
    return checks


def main() -> int:
    # No package/API/mpmath import has happened above this point.
    freeze_path = ROOT / "FREEZE.json"
    review_path = ROOT / "pre_execution_review.json"
    freeze_bytes = freeze_path.read_bytes()
    review_bytes = review_path.read_bytes()
    freeze_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
    review_sha256 = hashlib.sha256(review_bytes).hexdigest()
    freeze = json.loads(freeze_bytes)
    review = json.loads(review_bytes)
    if review.get("status") != "PASS" or review.get("reviewed_freeze_sha256") != freeze_sha256:
        raise RuntimeError("Independent PASS does not identify this exact FREEZE.json.")
    if review.get("scientific_runs_performed") != 0:
        raise RuntimeError("The required review must be static/manual, before the first run.")
    hashes = freeze["source_hashes"]
    for relative, expected in hashes.items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or digest(source) != expected:
            raise RuntimeError("Frozen source hash mismatch: " + relative)
    if any(name == "certified_envelopes" or name.startswith("certified_envelopes.") for name in sys.modules):
        raise RuntimeError("A live certified_envelopes package was already imported.")
    run = ROOT / "runs" / "run_001"
    run.parent.mkdir(exist_ok=True)
    run.mkdir()  # Mandatory fresh run: FileExistsError prevents overwriting.
    snapshot_parent = run / "source_snapshot"
    package = snapshot_parent / "certified_envelopes"
    package.mkdir(parents=True)
    snapshot_expected = hashes | {"FREEZE.json": freeze_sha256, "pre_execution_review.json": review_sha256}
    for relative, expected_hash in snapshot_expected.items():
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        # Compare against the reviewed digest captured before copying,
        # never against a live source which might have changed meanwhile.
        if digest(destination) != expected_hash:
            raise RuntimeError("Snapshot copy mismatch: " + relative)
        destination.chmod(0o444)
    protocol = json.loads((package / "protocol.json").read_text())
    references = json.loads((package / "reference_data.json").read_text())
    save(run / "preflight.json", {
        "status": "PASS", "at_utc": utc(), "scientific_calls_before_preflight": 0,
        "protocol_id": protocol["protocol_id"], "number_of_fixed_cases": len(protocol["cases"]),
        "freeze_sha256": freeze_sha256, "review_sha256": review_sha256,
        "source_hashes": hashes, "snapshot_package": str(package),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "execution_policy": protocol["execution_policy"],
    })
    # Import only the newly snapshotted package, with its relative core import.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(snapshot_parent))
    api = importlib.import_module("certified_envelopes.api")
    core = importlib.import_module("certified_envelopes._legacy_core")
    if Path(api.__file__).resolve() != package / "api.py" or Path(core.__file__).resolve() != package / "_legacy_core.py":
        raise RuntimeError("API/core import escaped the immutable snapshot.")
    import mpmath

    interface_checks = []
    for name in ("certify_joint", "algebraic_joint_bound"):
        parameters = inspect.signature(getattr(api, name)).parameters
        interface_checks.append({"check": name + "_joint_arguments_keyword_only", "passed": all(parameters[k].kind == inspect.Parameter.KEYWORD_ONLY for k in ("g", "P"))})
    interface_checks.append({"check": "algebraic_bound_has_no_factor_argument", "passed": "factor" not in inspect.signature(api.algebraic_joint_bound).parameters})
    started = utc()
    wall, cpu = time.perf_counter(), time.process_time()
    total_work, total_elementary = Counter(), Counter()
    rows = []
    for case in protocol["cases"]:
        cwall, ccpu = time.perf_counter(), time.process_time()
        before_precision = [mpmath.mp.dps, mpmath.iv.dps]
        record = {"id": case["id"], "function": case["function"], "fixed_case": case}
        out = None
        try:
            model = api.Model(**case.get("model", protocol["default_model"]))
            precision = api.Precision(**(protocol["default_precision"] | case.get("precision", {})))
            limits = api.ResourceLimits(**(protocol["default_limits"] | case.get("limits", {})))
            kwargs = dict(case["args"], model=model, precision=precision, limits=limits)
            if case["function"] != "algebraic_joint_bound":
                kwargs["factor"] = case.get("factor", "2")
            out = getattr(api, case["function"])(**kwargs).to_dict()
            record["output"] = out
            record["checks"] = checks_for(case, out, references, before_precision == [mpmath.mp.dps, mpmath.iv.dps])
            record["status"] = "PASS" if all(item["passed"] for item in record["checks"]) else "FAIL"
        except Exception as exc:
            record["status"] = "FAIL"
            record["unexpected_exception"] = {"type": type(exc).__name__, "message": str(exc)}
        record["elapsed_seconds"] = time.perf_counter() - cwall
        record["cpu_seconds"] = time.process_time() - ccpu
        record["global_precision_before"] = before_precision
        record["global_precision_after"] = [mpmath.mp.dps, mpmath.iv.dps]
        if out is not None:
            total_work.update(out["work"])
            total_elementary.update(out["elementary_evaluations"])
        case_path = run / ("case_" + case["id"] + ".json")
        save(case_path, record)
        rows.append({"id": case["id"], "status": record["status"], "validation_checks": len(record.get("checks", [])),
                     "failed_checks": [item["check"] for item in record.get("checks", []) if not item["passed"]],
                     "result_sha256": digest(case_path)})
        save(run / "partial_result.json", {"completed_cases": rows, "work": dict(total_work), "elementary_evaluations": dict(total_elementary)}, exclusive=False)
    after_hashes = {relative: digest(ROOT / relative) for relative in hashes}
    snapshot_hashes = {relative: digest(package / relative) for relative in hashes}
    control_files_unchanged = digest(freeze_path) == freeze_sha256 and digest(review_path) == review_sha256
    unchanged = after_hashes == hashes == snapshot_hashes and control_files_unchanged
    status = "PASS" if unchanged and all(row["status"] == "PASS" for row in rows) and all(item["passed"] for item in interface_checks) else "FAIL"
    result = {
        "status": status, "protocol_id": protocol["protocol_id"],
        "started_utc": started, "completed_utc": utc(),
        "scope": protocol["scope"], "api_calls": len(rows), "new_protocol_run_index": 1,
        "legacy_runs_repeated": 0, "automatic_retries": 0,
        "cases": rows, "interface_checks": interface_checks,
        "validation_checks": sum(row["validation_checks"] for row in rows) + len(interface_checks),
        "work": dict(total_work), "elementary_evaluations": dict(total_elementary),
        "elapsed_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
        "source_hashes": hashes, "sources_unchanged": unchanged,
        "freeze_sha256": freeze_sha256, "review_sha256": review_sha256,
        "freeze_and_review_unchanged": control_files_unchanged,
        "mpmath_version": mpmath.__version__, "api_version": api.__version__,
        "imported_api": api.__file__, "imported_core": core.__file__,
        "cost_semantics": protocol["interpretation"]["cost"],
        "limits_of_evidence": "Fixed implementation controls do not replace the mathematical proofs, establish priority, certify a fixed-domain minimax theorem, or benchmark an optimization algorithm.",
    }
    save(run / "result.json", result)
    for path in run.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({"status": status, "result": str(run / "result.json"), "api_calls": len(rows),
                      "validation_checks": result["validation_checks"], "work": dict(total_work),
                      "elementary_evaluations": dict(total_elementary)}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
