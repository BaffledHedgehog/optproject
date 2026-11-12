#!/usr/bin/env python3
"""One fixed diameter control run, only after root reviews the exact freeze."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import platform
import shutil
import sys
import time


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
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def checks_for(case, out, references, global_precision_unchanged):
    checks = []

    def check(name, condition):
        checks.append({"check": name, "passed": bool(condition)})

    expected, work, plan = case["expected"], out["work"], out["plan"]
    check("expected_status", out["status"] == expected["status"])
    check("expected_envelope", out["envelope"] == expected["envelope"])
    check("global_precision_unchanged", global_precision_unchanged)
    check("no_source_or_optimizer_operations", all(work[name] == 0 for name in FORBIDDEN))
    check("scalar_counter_partition", work["scalar_evaluations_started"] == sum(work[name] for name in SCALAR_PARTS))
    check("root_updates_no_more_than_queries", work["bisection_updates"] <= work["root_sign_queries"])
    check("all_costs_nonnegative_integers", all(type(value) is int and value >= 0 for value in work.values())
          and all(type(value) is int and value >= 0 for value in out["elementary_evaluations"].values()))
    for name in ("branch",):
        if name in expected:
            check("expected_" + name, plan.get(name) == expected[name])
    if "working_gap" in expected:
        check("expected_working_gap", Fraction(plan.get("working_gap", "-1")) == Fraction(expected["working_gap"]))
    if "radius_interval" in expected:
        check("expected_radius_interval", list(map(Fraction, plan.get("radius_interval", []))) == list(map(Fraction, expected["radius_interval"])))
    if "derived_gap_cap" in expected:
        check("expected_derived_gap_cap", Fraction(out["inputs"].get("derived_gap_cap", "-1")) == Fraction(expected["derived_gap_cap"]))
        check("norm_only_has_no_observed_gap_input", "g" not in out["inputs"])
    if "reason" in expected:
        check("expected_refusal_reason", out.get("reason") == expected["reason"])
    for name in ("parameter", "resource"):
        if name in expected:
            check("expected_refusal_" + name, out.get("refusal_details", {}).get(name) == expected[name])
    if "regime_evaluations" in expected:
        check("expected_regime_evaluation_count", work["regime_evaluations"] == expected["regime_evaluations"])
    if "regime" in expected:
        regime = plan.get("regime", {})
        check("expected_regime_decision", regime.get("decision") == expected["regime"])
        check("one_regime_query", work["regime_sign_queries"] == 1)
        if "S_interval" in regime:
            slo, shi = map(Fraction, regime["S_interval"])
            if expected["regime"] == "nonpositive":
                check("outward_nonpositive_sign", slo <= shi <= 0)
            elif expected["regime"] == "positive":
                check("outward_positive_sign", 0 < slo <= shi)
    for name in ("S_interval", "tau_interval"):
        if name + "_exact" in expected:
            actual = plan.get("regime", {}).get(name, [])
            check("expected_exact_" + name, list(map(Fraction, actual)) == list(map(Fraction, expected[name + "_exact"])))
    if "positive_nodes" in expected:
        check("expected_positive_nodes", plan.get("positive_nodes", 0) == expected["positive_nodes"])
    if expected.get("no_switch"):
        check("no_switch_or_loss_evaluations", work["switch_evaluations"] == work["loss_evaluations"] == work["bisection_updates"] == 0)
    if out["status"] == "refused":
        check("refusal_has_no_enclosure", all(out[name] is None for name in ("envelope_lower", "envelope_upper", "achieved_factor")))
        return checks

    lower, upper = map(Fraction, (out["envelope_lower"], out["envelope_upper"]))
    achieved, requested = map(Fraction, (out["achieved_factor"], out["requested_factor"]))
    check("ordered_nonnegative_enclosure", 0 <= lower <= upper)
    check("exact_returned_ratio", achieved == (Fraction(1) if upper == 0 else upper / lower))
    check("requested_factor_met", achieved <= requested)
    check("sqrt_counter_partition", out["elementary_evaluations"]["sqrt"] == work["node_coefficient_preparations"] + work["bound_evaluations"] + work["regime_evaluations"])
    if "exact" in expected:
        check("expected_exact_envelope", lower == upper == Fraction(expected["exact"]))
    if "strictly_below_exact_unrestricted" in expected:
        check("strict_diameter_improvement", upper < Fraction(expected["strictly_below_exact_unrestricted"]))
    if "reference" in expected:
        rlo, rhi = map(Fraction, references["points"][expected["reference"]]["rational_interval"])
        check("historical_unrestricted_reference_contained", lower <= rlo <= rhi <= upper)
    if "upper_lt" in expected:
        check("expected_strict_upper", upper < Fraction(expected["upper_lt"]))
    nodes = plan.get("positive_nodes", 0)
    if nodes > 0:
        check("completed_all_planned_nodes", len(out["trace"]) == nodes and all(node["status"] == "enclosed" for node in out["trace"]))
        check("root_budget_respected", work["bisection_updates"] <= plan["planned_bisections"])
        a, b = map(Fraction, plan["radius_interval"])
        check("node_radii_respect_diameter_interval", all(a <= Fraction(node["rho"]) <= b for node in out["trace"]))
        if nodes == 1:
            check("single_node_is_left_endpoint", Fraction(out["trace"][0]["rho"]) == a)
    if expected.get("positive_grid"):
        intervals = plan.get("grid_intervals", 0)
        check("positive_grid_intervals", intervals > 0)
        check("both_grid_endpoints_included", nodes == intervals + 1)
        a, b = map(Fraction, plan["radius_interval"])
        check("correct_grid_extremes", Fraction(out["trace"][0]["rho"]) == a and Fraction(out["trace"][-1]["rho"]) == b)
    return checks


def main() -> int:
    # No reviewed package or mpmath import occurs before this gate.
    freeze_path, review_path = ROOT / "FREEZE.json", ROOT / "pre_execution_review.json"
    freeze_bytes, review_bytes = freeze_path.read_bytes(), review_path.read_bytes()
    freeze_sha, review_sha = hashlib.sha256(freeze_bytes).hexdigest(), hashlib.sha256(review_bytes).hexdigest()
    freeze, review = json.loads(freeze_bytes), json.loads(review_bytes)
    if (review.get("status") != "PASS" or review.get("approved_by") != "/root"
            or review.get("reviewed_freeze_sha256") != freeze_sha
            or review.get("scientific_runs_performed") != 0):
        raise RuntimeError("Root's independent static PASS must identify this exact FREEZE.")
    hashes = freeze["source_hashes"]
    for relative, expected in hashes.items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or digest(source) != expected:
            raise RuntimeError("Frozen source mismatch: " + relative)
    if any(name == "certified_diameter" or name.startswith("certified_diameter.") for name in sys.modules):
        raise RuntimeError("A live certified_diameter package was already imported.")
    run = ROOT / "runs" / "run_001"
    run.parent.mkdir(exist_ok=True)
    run.mkdir()
    snapshot_parent = run / "source_snapshot"
    package = snapshot_parent / "certified_diameter"
    package.mkdir(parents=True)
    snapshot_expected = hashes | {"FREEZE.json": freeze_sha, "pre_execution_review.json": review_sha}
    for relative, expected in snapshot_expected.items():
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        if digest(destination) != expected:
            raise RuntimeError("Snapshot does not match reviewed bytes: " + relative)
        destination.chmod(0o444)
    protocol = json.loads((package / "protocol.json").read_text())
    references = json.loads((package / "reference_data.json").read_text())
    cases = protocol["cases"]
    if len(cases) != 10 or len({case["id"] for case in cases}) != 10:
        raise RuntimeError("The fixed ten-case protocol is not intact.")
    save(run / "preflight.json", {
        "status": "PASS", "stage": "source_snapshot_before_package_import",
        "at_utc": utc(), "scientific_calls_before_preflight": 0,
        "protocol_id": protocol["protocol_id"], "number_of_fixed_cases": len(cases),
        "freeze_sha256": freeze_sha, "review_sha256": review_sha,
        "source_hashes": hashes, "snapshot_package": str(package),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "execution_policy": protocol["execution_policy"],
    })
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(snapshot_parent))
    imported = {}
    for name in ("api", "_v16_api", "_legacy_core"):
        module = importlib.import_module("certified_diameter." + name)
        if Path(module.__file__).resolve() != package / (name + ".py"):
            raise RuntimeError("An imported module escaped the snapshot: " + name)
        imported[name] = module
    api = imported["api"]
    import mpmath

    save(run / "environment.json", {
        "status": "PASS" if mpmath.__version__ == "1.3.0" else "FAIL",
        "mpmath_version": mpmath.__version__, "scientific_api_calls": 0,
        "module_paths": {name: module.__file__ for name, module in imported.items()},
    })
    if mpmath.__version__ != "1.3.0":
        raise RuntimeError("The interval primitives require pinned mpmath 1.3.0.")
    joint_parameters = inspect.signature(api.certify_joint_diameter).parameters
    interface_checks = [{
        "check": "joint_g_P_D_are_keyword_only",
        "passed": all(joint_parameters[name].kind == inspect.Parameter.KEYWORD_ONLY for name in ("g", "P", "D")),
    }]
    started = utc()
    wall, cpu = time.perf_counter(), time.process_time()
    calls_started = 0
    rows = []
    total_work, total_elementary = Counter(), Counter()
    for case in cases:
        cwall, ccpu = time.perf_counter(), time.process_time()
        before_precision = [mpmath.mp.dps, mpmath.iv.dps]
        record = {"id": case["id"], "function": case["function"], "fixed_case": case,
                  "api_call_started": False}
        out = None
        try:
            model = api.Model(**case.get("model", protocol["default_model"]))
            precision = api.Precision(**(protocol["default_precision"] | case.get("precision", {})))
            limits = api.ResourceLimits(**(protocol["default_limits"] | case.get("limits", {})))
            kwargs = dict(case["args"], model=model, precision=precision, limits=limits,
                          factor=case.get("factor", protocol["default_factor"]))
            function = getattr(api, case["function"])
            calls_started += 1
            record["api_call_started"] = True
            out = function(**kwargs).to_dict()
            record["output"] = out
            record["checks"] = checks_for(case, out, references,
                                           before_precision == [mpmath.mp.dps, mpmath.iv.dps])
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
        rows.append({
            "id": case["id"], "status": record["status"],
            "api_call_started": record["api_call_started"],
            "validation_checks": len(record.get("checks", [])),
            "failed_checks": [item["check"] for item in record.get("checks", []) if not item["passed"]],
            "result_sha256": digest(case_path),
        })
        save(run / "partial_result.json", {
            "completed_cases": rows, "api_calls_started": calls_started,
            "work": dict(total_work), "elementary_evaluations": dict(total_elementary),
        }, exclusive=False)
    after_hashes = {name: digest(ROOT / name) for name in hashes}
    snapshot_hashes = {name: digest(package / name) for name in hashes}
    control_files_unchanged = digest(freeze_path) == freeze_sha and digest(review_path) == review_sha
    unchanged = after_hashes == hashes == snapshot_hashes and control_files_unchanged
    status = "PASS" if (unchanged and calls_started == 10 and len(rows) == 10
                       and all(row["status"] == "PASS" for row in rows)
                       and all(item["passed"] for item in interface_checks)) else "FAIL"
    result = {
        "status": status, "protocol_id": protocol["protocol_id"],
        "started_utc": started, "completed_utc": utc(), "scope": protocol["scope"],
        "api_calls": calls_started, "new_protocol_run_index": 1,
        "v16_api_calls_repeated": 0, "automatic_retries": 0,
        "cases": rows, "interface_checks": interface_checks,
        "validation_checks": sum(row["validation_checks"] for row in rows) + len(interface_checks),
        "work": dict(total_work), "elementary_evaluations": dict(total_elementary),
        "elapsed_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
        "source_hashes": hashes, "sources_unchanged": unchanged,
        "freeze_sha256": freeze_sha, "review_sha256": review_sha,
        "freeze_and_review_unchanged": control_files_unchanged,
        "mpmath_version": mpmath.__version__, "api_version": api.__version__,
        "imported_modules": {name: module.__file__ for name, module in imported.items()},
        "cost_semantics": protocol["interpretation"]["cost"],
        "limits_of_evidence": "Fixed implementation controls supplement the mathematical proofs. They are not optimizer benchmarks, fixed-domain sharpness proofs, or a total bit-complexity analysis.",
    }
    save(run / "result.json", result)
    for path in run.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    print(json.dumps({"status": status, "result": str(run / "result.json"),
                      "api_calls": calls_started, "validation_checks": result["validation_checks"],
                      "work": dict(total_work), "elementary_evaluations": dict(total_elementary)}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
