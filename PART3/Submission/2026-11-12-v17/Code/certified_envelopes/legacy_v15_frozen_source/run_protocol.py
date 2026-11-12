#!/usr/bin/env python3
"""Execute the frozen API protocol once into a new immutable run directory.

No scientific API call occurs before source snapshot and preflight checks.
An expected refusal can pass a refusal test, but its API status stays
``refused`` and it never supplies envelope bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from fractions import Fraction
from typing import Any


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def decode(value: Any) -> Any:
    if isinstance(value, dict) and "fraction" in value:
        return Fraction(*value["fraction"])
    if isinstance(value, dict) and "rejected_float" in value:
        return float(value["rejected_float"])
    return value


def parse_interval(text: str) -> tuple[Fraction, Fraction]:
    a, b = text[1:-1].split(", ")
    return Fraction(a), Fraction(b)


def check_case(cert, case: dict[str, Any], references: dict[str, Any]):
    checks: list[dict[str, Any]] = []

    def check(condition: bool, name: str) -> None:
        checks.append({"name": name, "passed": bool(condition)})

    expected = case["expected"]
    check(cert.status == expected["status"], "expected API status")
    check(all(cert.work[k] == 0 for k in (
        "source_objective_calls", "source_gradient_calls", "full_LMO_calls",
        "LP_calls", "optimizer_updates", "stochastic_samples",
    )), "zero scientific source/optimizer/stochastic oracles")
    check(cert.work["scalar_evaluations_started"] == sum(cert.work[k] for k in
          ("bound_evaluations", "switch_evaluations", "loss_evaluations")),
          "all started scalar evaluations are named")
    check(cert.work["bisection_updates"] == sum(t["completed_bisections"] for t in cert.trace),
          "bisection ledger matches the saved node traces")
    check(cert.work["root_sign_decisions"] == cert.work["bisection_updates"],
          "resolved midpoint signs match completed updates")
    check(cert.elementary_evaluations["exp"] ==
          2 * (cert.work["bound_evaluations"] + cert.work["switch_evaluations"] + cert.work["loss_evaluations"]),
          "direct exp calls reconcile with started scalar evaluations")
    check(cert.elementary_evaluations["sin"] == cert.work["switch_evaluations"] + cert.work["loss_evaluations"],
          "direct sin calls reconcile")
    check(cert.elementary_evaluations["cos"] == cert.work["switch_evaluations"] + cert.work["loss_evaluations"],
          "direct cos calls reconcile")
    check(cert.elementary_evaluations["sqrt"] ==
          cert.work["bound_evaluations"] + cert.work["node_coefficient_preparations"],
          "direct sqrt calls reconcile")
    for key, value in expected.get("work", {}).items():
        check(cert.work.get(key) == value, f"fixed work counter {key}")

    if cert.status == "certified":
        lo, hi = cert.envelope_lower, cert.envelope_upper
        check(isinstance(lo, Fraction) and isinstance(hi, Fraction), "rational envelope endpoints")
        check(0 <= lo <= hi, "nonnegative ordered envelope enclosure")
        ratio = Fraction(1) if hi == 0 else hi / lo
        check(cert.achieved_factor == ratio, "achieved factor is the exact returned endpoint ratio")
        check(ratio <= cert.requested_factor, "requested relative enclosure factor")
        if "branch" in expected:
            check(cert.plan.get("branch") == expected["branch"], "expected implementation branch executed")
        if "exact_envelope" in expected:
            exact = Fraction(expected["exact_envelope"])
            check(lo == hi == exact, "independent exact branch value")
            check(cert.work["bound_evaluations"] == cert.work["switch_evaluations"] == cert.work["loss_evaluations"] == 0,
                  "exact branch performs no interval scalar work")
        if "reference_norm_point" in expected:
            ref_lo, ref_hi = parse_interval(references["points"][expected["reference_norm_point"]]["value_interval"])
            check(lo <= ref_lo and hi >= ref_hi, "norm certificate contains the retained high-accuracy reference interval")
        if "reference_gap_grid" in expected:
            reference = references["grids"][expected["reference_gap_grid"]]
            check(hi == Fraction(reference["exact_upper_rational"]),
                  "gap upper endpoint exactly reproduces the retained audited kernel result")
            ref_lower = parse_interval(reference["lower_witness_interval"])[0]
            check(lo >= ref_lower, "gap lower endpoint retains the existing witness")
        if "reference_joint_points" in expected:
            for key in expected["reference_joint_points"]:
                ref_hi = parse_interval(references["points"][key]["value_interval"])[1]
                check(hi >= ref_hi, f"joint upper endpoint covers independent feasible-radius reference {key}")
            ref_cap = Fraction(references["grids"][expected["reference_outer_cap"]]["exact_upper_rational"])
            check(lo <= ref_cap, "joint lower endpoint is consistent with the retained outer upper certificate")
        for key, value in expected.get("plan", {}).items():
            check(cert.plan.get(key) == value, f"fixed plan field {key}")
        if cert.work["bound_evaluations"]:
            check(all(t["status"] == "enclosed" for t in cert.trace), "every prescribed positive node was completed")
            check(len(cert.trace) == cert.plan["positive_nodes"], "all prescribed positive nodes are retained")
            check(cert.work["bracket_sign_queries"] == cert.work["bracket_sign_decisions"] == 2 * len(cert.trace),
                  "two resolved endpoint signs per completed node")
            check(cert.work["root_sign_queries"] == cert.work["root_sign_decisions"],
                  "no unresolved sign accepted by a certificate")
            check(cert.work["switch_evaluations"] == cert.work["root_sign_queries"] + cert.work["bracket_sign_queries"],
                  "every switch evaluation has a recorded sign query")
            check(cert.work["loss_evaluations"] == 2 * len(cert.trace), "two loss endpoint evaluations per node")
            check(cert.plan["maximum_upper_evaluation_rounding_width"] <= cert.plan["reserved_rounding_allowance"],
                  "evaluation rounding fits its independently reserved allowance")
    else:
        check(cert.envelope_lower is None and cert.envelope_upper is None and cert.achieved_factor is None,
              "refusal exposes no envelope certificate")
        check(cert.reason == expected.get("reason"), "exact expected refusal reason")
        for key, value in expected.get("details", {}).items():
            check(cert.refusal_details.get(key) == value, f"exact refusal detail {key}")
        for key, value in expected.get("plan", {}).items():
            check(cert.plan.get(key) == value, f"fixed refusal plan field {key}")
        if expected.get("reason") == "unresolved_interval_sign":
            check(cert.work["root_sign_queries"] == cert.work["root_sign_decisions"] + 1,
                  "one unresolved midpoint attempt remains explicitly charged")
            check(cert.work["loss_evaluations"] == 0, "unresolved root is not used to issue an endpoint value")
        if expected.get("reason") == "insufficient_rounding_slack":
            check(cert.plan["maximum_upper_evaluation_rounding_width"] > cert.plan["reserved_rounding_allowance"],
                  "actual final interval width violates the fixed reserve")
            check(cert.work["loss_evaluations"] == 2 and cert.work["root_sign_queries"] == 45,
                  "rounding refusal occurs after the independently fixed high-precision root work")
        if expected.get("reason") == "invalid_input":
            check(cert.work["bound_evaluations"] == cert.work["scalar_evaluations_started"] == 0,
                  "invalid input causes no scientific scalar evaluation")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--freeze", type=Path, default=ROOT / "FREEZE.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="A new directory; existing directories are never overwritten.")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    start, cpu_start = time.perf_counter(), time.process_time()
    result: dict[str, Any] = {
        "status": "RUNNING", "case_results": [], "api_calls_started": 0,
        "source_oracle_calls": 0, "optimizer_updates": 0, "stochastic_samples": 0,
        "failure_policy": "Preserve every first-run outcome and the source snapshot. No retry or retuning is performed by this runner.",
    }
    try:
        freeze = json.loads(args.freeze.read_text())
        hashes = {}
        for name, expected in freeze["source_files"].items():
            source = ROOT / name
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[name] = digest(target)
        shutil.copy2(args.freeze, snapshot / "FREEZE.json")
        protocol = json.loads(args.protocol.read_text())
        preflight = {
            "protocol": protocol, "source_hashes": hashes,
            "frozen_source_hashes": freeze["source_files"],
            "freeze_sha256": digest(args.freeze),
            "dependency_versions": {"mpmath": importlib.metadata.version("mpmath")},
            "snapshot_created_before_api_import_and_calls": True,
        }
        write_json(output / "preflight.json", preflight)
        if hashes != freeze["source_files"]:
            raise RuntimeError("Frozen source hash mismatch; no API call permitted.")
        if digest(args.protocol) != freeze["source_files"]["protocol.json"]:
            raise RuntimeError("The requested protocol is not the frozen protocol.")
        if preflight["dependency_versions"] != protocol["dependency_versions"]:
            raise RuntimeError("Dependency version mismatch; no API call permitted.")
        if len(protocol["cases"]) != protocol["expected_case_count"]:
            raise RuntimeError("Frozen case-count mismatch; no API call permitted.")
        identifiers = [case["id"] for case in protocol["cases"]]
        if len(set(identifiers)) != len(identifiers):
            raise RuntimeError("Duplicate case identifiers would overwrite receipts; no API call permitted.")
        references = json.loads((snapshot / "reference_data.json").read_text())
        name = "_certified_envelopes_v15_snapshot_api"
        spec = importlib.util.spec_from_file_location(name, snapshot / "api.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the snapshotted API.")
        api = importlib.util.module_from_spec(spec)
        sys.modules[name] = api
        spec.loader.exec_module(api)
        import mpmath
        global_precision_before = {"mp_dps": mpmath.mp.dps, "iv_dps": mpmath.iv.dps}
        all_checks = 0
        work_total: dict[str, int] = {}
        elementary_total: dict[str, int] = {}
        for case in protocol["cases"]:
            model_values = dict(protocol["default_model"])
            model_values.update(case.get("model", {}))
            precision_values = dict(protocol["default_precision"])
            precision_values.update(case.get("precision", {}))
            limit_values = dict(protocol["default_limits"])
            limit_values.update(case.get("limits", {}))
            kwargs = {
                "model": api.Model(**{k: decode(v) for k, v in model_values.items()}),
                "factor": decode(case.get("factor", protocol["default_factor"])),
                "precision": api.Precision(**precision_values),
                "limits": api.ResourceLimits(**limit_values),
            }
            arguments = [decode(x) for x in case["arguments"]]
            function = {"E": api.certify_norm, "B": api.certify_gap, "J": api.certify_joint}[case["envelope"]]
            result["api_calls_started"] += 1
            case_start = time.perf_counter()
            cert = function(*arguments, **kwargs)
            elapsed = time.perf_counter() - case_start
            record = {
                "id": case["id"], "case_type": case["case_type"],
                "case": case, "test_status": "UNREVIEWED",
                "certificate": cert.to_dict(), "checks": [],
                "elapsed_seconds_measurement_only": elapsed,
            }
            # Preserve the returned certificate/refusal and all spent work
            # before executing any validation code that could itself fail.
            write_json(output / f"case_{case['id']}.json", record)
            for key, value in cert.work.items():
                work_total[key] = work_total.get(key, 0) + value
            for key, value in cert.elementary_evaluations.items():
                elementary_total[key] = elementary_total.get(key, 0) + value
            result.update({"work": work_total, "elementary_evaluations": elementary_total})
            try:
                checks = check_case(cert, case, references)
                passed = all(x["passed"] for x in checks)
                record["checks"] = checks
                all_checks += len(checks)
                record["test_status"] = "PASS" if passed else "FAIL"
            except Exception as exc:
                record["test_status"] = "FAIL"
                record["validation_error"] = {"type": type(exc).__name__, "message": str(exc)}
            write_json(output / f"case_{case['id']}.json", record)
            result["case_results"].append({
                "id": case["id"], "test_status": record["test_status"],
                "api_status": cert.status, "api_reason": cert.reason,
                "receipt": f"case_{case['id']}.json",
            })
            # A failing case is retained; the remaining predetermined cases
            # still execute once, without changing their inputs or budgets.
            result.update({"checks": all_checks, "work": work_total,
                           "elementary_evaluations": elementary_total})
            write_json(output / "partial_result.json", result)
        global_precision_after = {"mp_dps": mpmath.mp.dps, "iv_dps": mpmath.iv.dps}
        result["global_precision_before"] = global_precision_before
        result["global_precision_after"] = global_precision_after
        result["global_precision_unchanged"] = global_precision_before == global_precision_after
        result["status"] = "PASS" if (all(c["test_status"] == "PASS" for c in result["case_results"])
                                            and result["global_precision_unchanged"]) else "FAIL"
        result["snapshot_hashes_after_run"] = {name: digest(snapshot / name) for name in hashes}
        if result["snapshot_hashes_after_run"] != hashes:
            result["status"] = "FAIL"
            result["snapshot_mutation_detected"] = True
    except Exception as exc:
        result["status"] = "FAIL"
        result["runner_error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        result["elapsed_seconds_measurement_only"] = time.perf_counter() - start
        result["cpu_seconds_measurement_only"] = time.process_time() - cpu_start
        write_json(output / "result.json", result)
        print(json.dumps({
            "status": result["status"], "case_count": len(result["case_results"]),
            "checks": result.get("checks", 0), "output": str(output),
            "result_sha256": digest(output / "result.json"),
        }, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
