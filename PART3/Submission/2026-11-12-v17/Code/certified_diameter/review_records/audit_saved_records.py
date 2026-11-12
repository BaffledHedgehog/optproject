"""Audit existing diameter records with hashes and rational bookkeeping only.

This file imports no reviewed package, no interval library, and no numerical
oracle. It never calls the API or runner and never writes inside the package.
"""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "certified_diameter"
RUN = PACKAGE / "runs" / "run_001"
SNAPSHOT = RUN / "source_snapshot" / "certified_diameter"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rational_interval(value):
    return tuple(map(Fraction, value))


checks = []


def check(name, condition, case=None, **details):
    checks.append({"check": name, "passed": bool(condition),
                   **({"case": case} if case else {}), **details})


def outward_decimal(value, upper=False, places=6):
    scale = 10**places
    q = Fraction(value) * scale
    n = -(-q.numerator // q.denominator) if upper else q.numerator // q.denominator
    sign = "-" if n < 0 else ""
    n = abs(n)
    return f"{sign}{n // scale}.{n % scale:0{places}d}"


result = read(RUN / "result.json")
preflight = read(RUN / "preflight.json")
environment = read(RUN / "environment.json")
partial = read(RUN / "partial_result.json")
freeze = read(SNAPSHOT / "FREEZE.json")
approval = read(SNAPSHOT / "pre_execution_review.json")
protocol = read(SNAPSHOT / "protocol.json")
references = read(SNAPSHOT / "reference_data.json")
independent = read(HERE / "preexecution_static_review.json")

freeze_hash = digest(SNAPSHOT / "FREEZE.json")
approval_hash = digest(SNAPSHOT / "pre_execution_review.json")
check("reviewed_freeze_identity",
      freeze_hash == independent["reviewed_freeze_sha256"]
      == "55f9824ee37c2b16e79918ff5a8849210589671c39055500bf2197a3fb3eb9c8")
check("independent_review_identity",
      approval["independent_review_sha256"]
      == digest(HERE / "preexecution_static_review.json"))
check("explicit_root_approval",
      approval["status"] == "PASS" and approval["approved_by"] == "/root"
      and approval["reviewed_freeze_sha256"] == freeze_hash
      and approval["scientific_runs_performed"] == 0)
for name, expected in freeze["source_hashes"].items():
    check("live_frozen_hash", digest(PACKAGE / name) == expected, file=name)
    check("snapshot_frozen_hash", digest(SNAPSHOT / name) == expected, file=name)
check("snapshot_freeze_matches_live", digest(PACKAGE / "FREEZE.json") == freeze_hash)
check("snapshot_approval_matches_live",
      digest(PACKAGE / "pre_execution_review.json") == approval_hash)
for record_name, record in (("result", result), ("preflight", preflight)):
    check("record_freeze_hash", record["freeze_sha256"] == freeze_hash, record=record_name)
    check("record_approval_hash", record["review_sha256"] == approval_hash, record=record_name)
    check("record_source_map", record["source_hashes"] == freeze["source_hashes"],
          record=record_name)
check("preflight_before_start",
      datetime.fromisoformat(approval["approved_at_utc"])
      <= datetime.fromisoformat(preflight["at_utc"])
      <= datetime.fromisoformat(result["started_utc"])
      <= datetime.fromisoformat(result["completed_utc"]))
check("preflight_zero_prior_calls",
      preflight["status"] == "PASS"
      and preflight["stage"] == "source_snapshot_before_package_import"
      and preflight["scientific_calls_before_preflight"] == 0)
check("environment_pinned",
      environment["status"] == "PASS" and environment["mpmath_version"] == "1.3.0"
      and result["mpmath_version"] == "1.3.0"
      and environment["scientific_api_calls"] == 0)
for module in ("api", "_v16_api", "_legacy_core"):
    expected = str(SNAPSHOT / (module + ".py"))
    check("snapshot_import_path",
          result["imported_modules"][module] == expected
          == environment["module_paths"][module], module=module)
check("read_only_saved_files",
      all(p.stat().st_mode & 0o222 == 0 for p in RUN.rglob("*") if p.is_file()))
check("single_run_directory",
      sorted(p.name for p in (PACKAGE / "runs").iterdir() if p.is_dir()) == ["run_001"])
check("saved_completion_flags",
      result["status"] == "PASS" and result["sources_unchanged"]
      and result["freeze_and_review_unchanged"])
check("one_fixed_run_declared",
      result["new_protocol_run_index"] == 1 and result["api_calls"] == 10
      and result["v16_api_calls_repeated"] == 0 and result["automatic_retries"] == 0)
ids = [case["id"] for case in protocol["cases"]]
check("ten_unique_protocol_ids", len(ids) == len(set(ids)) == 10)
check("saved_case_order", [case["id"] for case in result["cases"]] == ids)
check("case_file_coverage",
      sorted(p.name for p in RUN.glob("case_*.json"))
      == sorted("case_" + key + ".json" for key in ids))
check("partial_rows_equal_final", partial["completed_cases"] == result["cases"])
check("partial_call_count", partial["api_calls_started"] == result["api_calls"] == 10)
check("protocol_identity",
      result["protocol_id"] == preflight["protocol_id"] == protocol["protocol_id"])

total_work, total_elementary = Counter(), Counter()
outcomes = Counter()
case_summaries = []
runner_check_count = 0
completed_nodes = 0
total_case_wall = total_case_cpu = 0
for fixed, row in zip(protocol["cases"], result["cases"]):
    key = fixed["id"]
    path = RUN / ("case_" + key + ".json")
    saved = read(path)
    output = saved["output"]
    plan, work, elementary, trace = (
        output["plan"], output["work"], output["elementary_evaluations"], output["trace"])
    expected = fixed["expected"]
    check("case_result_hash", row["result_sha256"] == digest(path), key)
    check("exact_frozen_case_copy", saved["fixed_case"] == fixed, key)
    check("case_id_and_function", saved["id"] == key and saved["function"] == fixed["function"], key)
    check("case_call_started", saved["api_call_started"] is True and row["api_call_started"] is True, key)
    check("case_completed_without_exception",
          saved["status"] == row["status"] == "PASS"
          and "unexpected_exception" not in saved and row["failed_checks"] == [], key)
    check("runner_check_count",
          row["validation_checks"] == len(saved["checks"])
          and all(c["passed"] is True for c in saved["checks"]), key)
    check("precision_unchanged",
          saved["global_precision_before"] == saved["global_precision_after"], key)
    check("nonnegative_recorded_times",
          saved["elapsed_seconds"] >= 0 and saved["cpu_seconds"] >= 0, key)
    check("expected_status_and_family",
          output["status"] == expected["status"]
          and output["envelope"] == expected["envelope"], key)
    check("nonnegative_integer_counters",
          all(type(v) is int and v >= 0 for v in work.values())
          and all(type(v) is int and v >= 0 for v in elementary.values()), key)
    forbidden = ("source_objective_calls", "source_gradient_calls", "full_LMO_calls",
                 "LP_calls", "optimizer_updates", "stochastic_samples")
    check("no_source_operations", all(work[name] == 0 for name in forbidden), key)
    scalar_parts = ("bound_evaluations", "switch_evaluations", "loss_evaluations",
                    "regime_evaluations", "algebraic_bound_evaluations")
    check("scalar_partition",
          work["scalar_evaluations_started"] == sum(work[name] for name in scalar_parts), key)
    check("elementary_partitions",
          elementary["sqrt"] == work["bound_evaluations"] + work["regime_evaluations"]
          + work["node_coefficient_preparations"]
          and elementary["sin"] == elementary["cos"]
          == work["switch_evaluations"] + work["loss_evaluations"]
          and elementary["exp"] == 2 * (work["bound_evaluations"]
          + work["switch_evaluations"] + work["loss_evaluations"]), key)
    factor = Fraction(fixed.get("factor", protocol["default_factor"]))
    internal = (1 + factor) / 2
    check("requested_and_internal_factor",
          Fraction(output["requested_factor"]) == factor
          and Fraction(output["internal_factor"]) == internal, key)
    precision = protocol["default_precision"] | fixed.get("precision", {})
    check("fixed_precision_used", output["precision"] == precision, key)
    limits = protocol["default_limits"] | fixed.get("limits", {})
    check("actual_named_costs_within_limits",
          work["bisection_updates"] <= limits["max_bisection_steps"]
          and work["scalar_evaluations_started"] <= limits["max_scalar_evaluations"]
          and sum(elementary.values()) <= limits["max_elementary_evaluations"]
          and len(trace) <= limits["max_positive_nodes"], key)
    if "branch" in expected:
        check("expected_branch", plan.get("branch") == expected["branch"], key)
    check("expected_regime_evaluation_count",
          work["regime_evaluations"] == expected["regime_evaluations"], key)
    if expected.get("no_switch"):
        check("no_switch_started",
              work["switch_evaluations"] == work["loss_evaluations"]
              == work["bisection_updates"] == 0 and not trace, key)
    model = fixed.get("model", protocol["default_model"])
    for name, value in model.items():
        check("exact_model_input_recorded",
              Fraction(output["inputs"][name]) == Fraction(value), key, parameter=name)
    for name in ("P", "g"):
        if name in fixed["args"]:
            check("exact_observation_cap_recorded",
                  Fraction(output["inputs"][name]) == Fraction(fixed["args"][name]),
                  key, parameter=name)
    m, h0, h1 = (Fraction(model[name]) for name in ("m", "H0", "H1"))
    P = Fraction(fixed["args"]["P"])
    geometry_present = "working_gap" in plan
    if geometry_present:
        D = Fraction(fixed["args"]["D"])
        g = Fraction(fixed["args"]["g"]) if "g" in fixed["args"] else P * D
        h = min(g, P * P / m, P * D)
        a, b = h * h / (P * P), min(h / m, D * D)
        check("exact_derived_geometry",
              Fraction(plan["working_gap"]) == h
              and rational_interval(plan["radius_interval"]) == (a, b)
              and Fraction(plan["norm_gap_cap"]) == P * P / m
              and Fraction(plan["diameter_gap_cap"]) == P * D, key)
        check("positive_ordered_radius_interval", 0 < a <= b <= h / m and b <= D * D, key)
        if "working_gap" in expected:
            check("expected_working_gap", h == Fraction(expected["working_gap"]), key)
        if "radius_interval" in expected:
            check("expected_radius_interval",
                  (a, b) == rational_interval(expected["radius_interval"]), key)
        if fixed["function"] == "certify_norm_diameter":
            check("norm_derived_cap_and_singleton",
                  "g" not in output["inputs"]
                  and Fraction(output["inputs"]["derived_gap_cap"]) == P * D
                  and a == b == min(D, P / m)**2, key)
    if "regime" in plan:
        regime = plan["regime"]
        slo, shi = rational_interval(regime["S_interval"])
        tlo, thi = rational_interval(regime["tau_interval"])
        check("regime_uses_clipped_h", Fraction(regime["working_gap"]) == h, key)
        check("ordered_regime_intervals", slo <= shi and Fraction(1, 2) < tlo <= thi <= 1, key)
        check("regime_decision_matches_saved_outward_sign",
              (regime["decision"] == "nonpositive" and shi <= 0)
              or (regime["decision"] == "positive" and slo > 0), key)
        check("expected_regime_branch", regime["decision"] == expected["regime"], key)
        for name in ("S_interval", "tau_interval"):
            if name + "_exact" in expected:
                check("expected_exact_regime_fixture",
                      rational_interval(regime[name])
                      == rational_interval(expected[name + "_exact"]), key, field=name)
    check("regime_counter_partitions",
          work["regime_evaluations"] == work["regime_sign_queries"]
          == work["regime_nonpositive_decisions"] + work["regime_positive_decisions"]
          + work["regime_unresolved_queries"]
          and work["regime_order_predicates"] == work["regime_nonpositive_decisions"]
          + 2 * (work["regime_positive_decisions"] + work["regime_unresolved_queries"]), key)
    check("diameter_comparison_counters",
          work["diameter_cap_minimum_comparisons"] == 2 * int(geometry_present)
          and work["diameter_radius_minimum_comparisons"] == int(geometry_present)
          and work["diameter_collapse_comparisons"] == int(geometry_present and h1 > 0)
          and work["diameter_exact_endpoint_comparisons"] == int(geometry_present and h1 == 0),
          key)
    check("no_legacy_working_gap_or_algebraic_branch",
          work["working_gap_minimum_comparisons"] == work["algebraic_bound_evaluations"] == 0, key)
    for node_index, node in enumerate(trace):
        n = node["completed_bisections"]
        history = node["sign_history"]
        check("trace_node_completed",
              node["status"] == "enclosed" and node["index"] == node_index
              and node["planned_bisections"] == plan["bisections_per_node"]
              and n == len(history) <= node["planned_bisections"], key, node=node_index)
        rho = Fraction(node["rho"])
        check("node_within_diameter_interval", a <= rho <= b, key, node=node_index)
        lo, up = Fraction(0), Fraction(1)
        last_mid = None
        for j, sign in enumerate(history):
            last_mid = (lo + up) / 2
            if sign == "+":
                lo = last_mid
            elif sign == "-":
                up = last_mid
            elif sign == "0":
                lo = up = last_mid
                check("zero_root_ends_history", j == len(history) - 1, key, node=node_index)
            else:
                check("valid_history_symbol", False, key, node=node_index)
        check("exact_bisection_bracket_replay",
              (lo, up) == rational_interval(node["alpha_bracket"])
              and 0 <= lo <= up <= 1
              and up - lo <= Fraction(1, 2**n), key, node=node_index)
        if history:
            check("last_midpoint_replay", Fraction(node["last_alpha"]) == last_mid, key, node=node_index)
            rlo, rhi = rational_interval(node["last_residual_interval"])
            check("last_saved_sign",
                  (history[-1] == "+" and 0 < rlo <= rhi)
                  or (history[-1] == "-" and rlo <= rhi < 0)
                  or (history[-1] == "0" and rlo == rhi == 0), key, node=node_index)
        rlo, rhi = rational_interval(node["left_residual_interval"])
        slo, shi = rational_interval(node["right_residual_interval"])
        check("retained_bracket_signs",
              ((0 < rlo <= rhi and slo <= shi < 0) or
               (rlo == rhi == slo == shi == 0 and lo == up)), key, node=node_index)
        ll, lu = rational_interval(node["loss_lower_endpoint_interval"])
        ul, uu = rational_interval(node["loss_upper_endpoint_interval"])
        check("saved_outward_value_intervals",
              ll <= lu and ul <= uu and ll <= uu
              and Fraction(node["fixed_radius_lower"]) == ll
              and Fraction(node["fixed_radius_upper"]) == uu
              and Fraction(node["upper_evaluation_rounding_width"]) == uu - ul,
              key, node=node_index)
    node_count = len(trace)
    completed = sum(t["completed_bisections"] for t in trace)
    positives = sum(t["sign_history"].count("+") for t in trace)
    negatives = sum(t["sign_history"].count("-") for t in trace)
    zeros = sum(t["sign_history"].count("0") for t in trace)
    check("trace_root_costs",
          work["bisection_updates"] == work["root_sign_queries"]
          == work["root_sign_decisions"] == completed
          and work["bracket_sign_queries"] == work["bracket_sign_decisions"] == 2 * node_count
          and work["switch_evaluations"] == completed + 2 * node_count
          and work["loss_evaluations"] == 2 * node_count, key)
    check("trace_sign_predicates",
          work["sign_order_predicates"] == 3 * node_count + positives + 2 * (negatives + zeros)
          and work["sign_equality_predicates"] == 2 * zeros, key)
    check("one_coefficient_preparation_per_completed_node",
          work["node_coefficient_preparations"] == node_count, key)
    check("endpoint_extraction_partition",
          work["interval_endpoint_extractions"]
          == 4 * work["bound_evaluations"] + 4 * work["regime_evaluations"]
          + 2 * (work["bracket_sign_queries"] + work["root_sign_queries"])
          + 4 * node_count, key)
    interval_max_comparisons = 3 * work["interval_endpoint_extractions"]
    exact_input_count = len(output["inputs"]) - int("derived_gap_cap" in output["inputs"]) + 2
    check("input_and_endpoint_metadata_maxima",
          work["resource_metadata_maximum_comparisons"]
          == interval_max_comparisons + exact_input_count, key)
    extra_bmax_conversion = sum(h1 * Fraction(t["rho"]) > 1 for t in trace)
    check("rational_conversion_partition",
          work["rational_to_interval_conversions"]
          == 4 * work["bound_evaluations"] + 4 * work["regime_evaluations"]
          + 5 * node_count + extra_bmax_conversion
          + work["switch_evaluations"] + work["loss_evaluations"], key)
    preflight_present = "bisections_per_node" in plan
    preflight_limits_checked = 4 if node_count else int(preflight_present)
    expected_resource_checks = (
        2 + exact_input_count + work["bound_evaluations"] + work["regime_evaluations"]
        + work["scalar_evaluations_started"] + sum(elementary.values())
        + work["interval_endpoint_extractions"] + completed + preflight_limits_checked)
    check("resource_limit_check_partition",
          work["resource_limit_checks"] == expected_resource_checks, key)
    invalid_type_D = expected.get("reason") == "invalid_input" and type(fixed["args"]["D"]) is float
    finish_present = "known_envelope_lower" in plan
    constant_branch = plan.get("branch") == "diameter_constant_curvature"
    grid_branch = plan.get("branch") == "diameter_restricted_grid"
    expected_validation = (
        17 + exact_input_count + int(invalid_type_D) + int("D" in output["inputs"])
        + int(geometry_present) + 2 * work["bound_evaluations"]
        + 2 * work["regime_evaluations"] + 4 * node_count + 4 * int(finish_present)
        + int(constant_branch) + int(grid_branch))
    check("internal_validation_predicate_partition",
          work["validation_checks"] == expected_validation, key)
    if output["status"] == "refused":
        check("refusal_has_no_completed_bounds",
              all(output[name] is None for name in
                  ("envelope_lower", "envelope_upper", "achieved_factor")), key)
        check("expected_refusal_reason", output["reason"] == expected["reason"], key)
        for name in ("resource", "parameter"):
            if name in expected:
                check("expected_refusal_detail",
                      output["refusal_details"].get(name) == expected[name], key, detail=name)
        if expected.get("resource") == "positive_nodes":
            check("node_cap_refused_before_node",
                  output["refusal_details"]["required"] == 1
                  and output["refusal_details"]["limit"] == limits["max_positive_nodes"] == 0
                  and node_count == 0 and work["bound_evaluations"] == 1, key)
        interval_display = None
    else:
        lower, upper = Fraction(output["envelope_lower"]), Fraction(output["envelope_upper"])
        check("ordered_returned_enclosure", 0 <= lower <= upper, key)
        ratio = Fraction(1) if upper == 0 else upper / lower
        check("exact_returned_factor",
              Fraction(output["achieved_factor"]) == ratio <= factor, key)
        check("certified_has_no_refusal", output["reason"] is None and output["message"] is None, key)
        if "exact" in expected:
            check("expected_exact_value", lower == upper == Fraction(expected["exact"]), key)
        if constant_branch:
            beta = (h0*h0 - 2*m*h0 - m*m) / (4*h0)
            radius = b if beta >= 0 else a
            check("exact_constant_formula",
                  Fraction(plan["beta"]) == beta and Fraction(plan["fixed_radius"]) == radius
                  and lower == upper == h + beta * radius, key)
        if "strictly_below_exact_unrestricted" in expected:
            check("strict_diameter_improvement",
                  upper < Fraction(expected["strictly_below_exact_unrestricted"]), key)
        if "reference" in expected:
            rlo, rhi = rational_interval(references["points"][expected["reference"]]["rational_interval"])
            check("historical_interval_contained", lower <= rlo <= rhi <= upper, key)
        if "upper_lt" in expected:
            check("expected_strict_upper", upper < Fraction(expected["upper_lt"]), key)
        if node_count:
            check("all_planned_nodes_completed", node_count == plan["positive_nodes"], key)
            check("planned_root_budget",
                  completed <= plan["planned_bisections"]
                  == plan["positive_nodes"] * plan["bisections_per_node"], key)
            n = plan["bisections_per_node"]
            klo, K = rational_interval(plan["Kstar_interval"])
            llo, L = rational_interval(plan["Lstar_interval"])
            check("ordered_positive_scalar_bounds", 0 < klo <= K and 0 < llo <= L, key)
            baseline = h / 2
            root_multiplier = 8 if grid_branch else 4
            target = root_multiplier * K / (m * (internal - 1))
            check("minimal_root_budget_integer",
                  2**n >= target and (n == 0 or 2**(n-1) < target), key)
            if grid_branch:
                N = plan["grid_intervals"]
                targetN = 2 * L * (b-a) / ((internal-1)*h)
                ceilN = -(-targetN.numerator // targetN.denominator)
                spacing = (b-a) / N
                check("minimal_shortened_grid_budget", N == max(1, ceilN), key)
                check("complete_uniform_grid",
                      node_count == N+1 and Fraction(plan["grid_spacing"]) == spacing
                      and all(Fraction(t["rho"]) == a+j*spacing for j,t in enumerate(trace)), key)
                mesh = L * spacing / 2
                inner = 2 * b * K / 2**n
                witness = max([baseline] + [Fraction(t["fixed_radius_lower"]) for t in trace])
                check("separate_mesh_and_root_allowances",
                      mesh <= (internal-1)*h/4 and inner <= (internal-1)*h/4, key)
            else:
                check("single_left_node",
                      node_count == 1 and Fraction(trace[0]["rho"]) == a
                      == Fraction(plan["fixed_radius"]) and plan["grid_intervals"] == 0, key)
                mesh = Fraction(0)
                inner = 2 * a * K / 2**n
                witness = Fraction(trace[0]["fixed_radius_lower"])
                check("single_root_allowance", inner <= (internal-1)*h/2, key)
            rounding = max(Fraction(t["upper_evaluation_rounding_width"]) for t in trace)
            reserve = (factor-internal) * baseline
            allowance = mesh + inner + rounding
            sampled_upper = max(Fraction(t["fixed_radius_upper"]) for t in trace)
            check("exact_completion_budget_fields",
                  Fraction(plan["known_envelope_lower"]) == baseline
                  and Fraction(plan["sampled_envelope_lower"]) == witness
                  and Fraction(plan["mesh_allowance"]) == mesh
                  and Fraction(plan["inner_root_allowance"]) == inner
                  and Fraction(plan["maximum_upper_evaluation_rounding_width"]) == rounding
                  and Fraction(plan["reserved_rounding_allowance"]) == reserve, key)
            check("completed_budget_fits",
                  mesh + inner <= (internal-1)*baseline and rounding <= reserve, key)
            check("exact_returned_endpoints_from_trace",
                  upper == sampled_upper + mesh
                  and Fraction(plan["total_additive_allowance"]) == allowance
                  and Fraction(plan["upper_minus_allowance_lower"]) == upper-allowance
                  and lower == max(baseline, witness, upper-allowance), key)
        interval_display = [outward_decimal(lower), outward_decimal(upper, upper=True)]
        check("outward_six_decimal_display",
              Fraction(interval_display[0]) <= lower <= upper <= Fraction(interval_display[1]), key)
    check("formula_counter",
          work["exact_formula_evaluations"]
          == int(plan.get("branch") in ("diameter_constant_curvature", "diameter_zero")), key)
    check("grid_selection_counter_partition",
          work["certificate_maximum_comparisons"] == (node_count-1 if grid_branch else 0)
          and work["witness_maximum_comparisons"] == (node_count if grid_branch else 0)
          and work["rounding_maximum_comparisons"] == (node_count if grid_branch else 0)
          and work["derived_lower_maximum_comparisons"] == 2 * int(finish_present)
          and work["budget_setup_maximum_comparisons"] == int(grid_branch)
          and work["budget_ceiling_operations"] == int(grid_branch)
          and work["budget_order_comparisons"] == 2 * int(preflight_present), key)
    outcomes[output["status"]] += 1
    total_work.update(work)
    total_elementary.update(elementary)
    completed_nodes += node_count
    runner_check_count += len(saved["checks"])
    total_case_wall += saved["elapsed_seconds"]
    total_case_cpu += saved["cpu_seconds"]
    case_summaries.append({
        "id": key, "observed_status": output["status"], "branch": plan.get("branch"),
        "reason": output["reason"], "envelope": output["envelope"],
        "envelope_lower": output["envelope_lower"], "envelope_upper": output["envelope_upper"],
        "outward_6dp_display": interval_display, "achieved_factor": output["achieved_factor"],
        "completed_positive_nodes": node_count, "bisection_updates": work["bisection_updates"],
        "scalar_evaluations_started": work["scalar_evaluations_started"],
        "saved_result_sha256": digest(path),
    })

check("observed_seven_certified_three_refused", dict(outcomes) == {"certified": 7, "refused": 3})
check("aggregate_work_matches_all_cases", dict(total_work) == result["work"] == partial["work"])
check("aggregate_elementary_matches_all_cases",
      dict(total_elementary) == result["elementary_evaluations"] == partial["elementary_evaluations"])
check("runner_checks_distinct_from_internal_predicates",
      runner_check_count == 183 and len(result["interface_checks"]) == 1
      and result["validation_checks"] == 184
      and all(c["passed"] is True for c in result["interface_checks"])
      and total_work["validation_checks"] == 344)
check("case_times_fit_run_times",
      total_case_wall <= result["elapsed_seconds"] and total_case_cpu <= result["cpu_seconds"])
check("eleven_nodes_sixty_one_updates",
      completed_nodes == 11 and total_work["bisection_updates"] == 61)
check("named_scalar_start_total", total_work["scalar_evaluations_started"] == 113)
issues = [item for item in checks if not item["passed"]]
report = {
    "status": "PASS" if not issues else "ISSUES_FOUND",
    "reviewer": "/root/gap_structure",
    "review_kind": "Independent saved-record audit without API import or rerun",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "audited_run_path": str(RUN),
    "audited_result_sha256": digest(RUN / "result.json"),
    "reviewed_freeze_sha256": freeze_hash,
    "root_approval_sha256": approval_hash,
    "audit_script_sha256": digest(Path(__file__)),
    "scientific_api_imports_performed": 0,
    "scientific_api_calls_performed": 0,
    "scientific_runs_performed": 0,
    "reviewed_files_modified": False,
    "audit_attempt_receipt": "audit_first_attempt_receipt.json",
    "audit_attempt_note": "An initial standard-library audit pass stopped at an optional expected-field lookup. That audit-only parser assumption was corrected; the package and saved run were untouched, and no API was imported or executed.",
    "method": "Read-only JSON and byte-hash comparison plus exact Fraction bookkeeping of saved inputs, traces, budgets, outputs and named counters. No transcendental function or certificate was recomputed.",
    "audit_condition_count": len(checks),
    "conditions": checks,
    "issues": issues,
    "observed_case_counts": dict(outcomes),
    "completed_first_panel": {
        "api_calls": 10, "case_runner_checks": runner_check_count,
        "interface_runner_checks": len(result["interface_checks"]),
        "total_runner_checks": result["validation_checks"],
        "internal_validation_predicates": total_work["validation_checks"],
        "completed_positive_nodes": completed_nodes,
        "bisection_updates": total_work["bisection_updates"],
        "named_scalar_starts": total_work["scalar_evaluations_started"],
        "elementary_evaluations": dict(total_elementary),
    },
    "cases": case_summaries,
    "saved_record_hashes": {
        str(p.relative_to(RUN)): digest(p) for p in sorted(RUN.glob("*.json"))
    },
    "limitations": [
        "This audit verifies consistency of the saved reviewed-code execution. It does not independently reevaluate exponential, trigonometric or square-root intervals.",
        "Ten API implementation cases are not ten optimization experiments.",
        "Counter totals describe named operations, not every arithmetic operation or total bit complexity.",
        "The recorded single run and zero-retry policy are consistent with the runner and records; this is not a forensic claim about unrelated unrecorded activity.",
    ],
}
target = HERE / "postexecution_saved_record_review.json"
with target.open("x") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({
    "status": report["status"], "audit_conditions": len(checks),
    "issues": issues, "observed_cases": dict(outcomes),
    "review_path": str(target), "review_sha256": digest(target),
    "result_sha256": report["audited_result_sha256"],
    "totals": report["completed_first_panel"],
}, indent=2))
