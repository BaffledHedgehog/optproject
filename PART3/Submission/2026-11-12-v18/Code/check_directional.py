"""Eight fixed deterministic checks of carried directional accuracy.

Problem: f(x, y) = exp(x) + 100*y on [-1, 1] x [0, 1].
The solver sees exact objective values and a synthetic directional oracle.
It never uses the true optimum. That optimum is used only by the checks
after termination. Standard-library floating-point checks support, but
do not replace, the mathematical proof.

Run from any directory: python Code/check_directional.py
"""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURVATURE = math.e
DIAMETER_SQUARED = 5.0
INITIAL_CURVATURE = 1.0


def directional_oracle(point, eta, query_index):
    """Return only h and d to the algorithm, with a known error budget."""
    x, y = point
    derivative = math.exp(x)
    direction = (-1.0 - x, -y)
    exact_gap = derivative * (x + 1.0) + 100.0 * y
    sign = 1.0 if query_index % 2 == 0 else -1.0
    h = max(0.0, exact_gap + sign * 0.75 * eta)
    # These validate the supplied oracle, not the algorithm's stopping rule.
    tolerance = 1e-14 * max(1.0, exact_gap, h, eta)
    assert exact_gap <= h + eta + tolerance
    assert exact_gap >= h - eta - tolerance
    return h, direction


def solve(start_x, epsilon, eta_initial, theta=0.5):
    """Implement the stated order: stopping, accuracy, then line search."""
    objective_calls = 0

    def objective(point):
        nonlocal objective_calls
        objective_calls += 1
        x, y = point
        return math.exp(x) + 100.0 * y

    point = (float(start_x), 0.0)
    value = objective(point)
    lower_bound = -math.inf
    eta = eta_initial
    stored_curvature = INITIAL_CURVATURE
    accepted_steps = 0
    refinements = 0
    query_count = 0
    first_upper_gap = None
    trace = []
    trial_counts = []

    for _ in range(10000):
        before_query_calls = objective_calls
        h, direction = directional_oracle(point, eta, query_count)
        query_count += 1
        assert objective_calls == before_query_calls
        upper_gap = h + eta
        if first_upper_gap is None:
            first_upper_gap = upper_gap
        lower_bound = max(lower_bound, value - upper_gap)
        certified_error = value - lower_bound
        precision_passes = eta <= theta * h
        record = {
            "query": query_count,
            "accepted_steps_before_query": accepted_steps,
            "point": list(point),
            "eta": eta,
            "h": h,
            "upper_gap": upper_gap,
            "certified_error": certified_error,
            "precision_passes": precision_passes,
            "objective_calls_before_query": objective_calls,
        }

        # This must precede the relative directional-accuracy check.
        if certified_error <= epsilon:
            record["action"] = "stop"
            trace.append(record)
            break

        if not precision_passes:
            record["action"] = "refine"
            trace.append(record)
            eta /= 2.0
            refinements += 1
            continue

        record["action"] = "step"
        trace.append(record)
        directional_decrease = h - eta
        direction_squared = sum(component * component for component in direction)
        assert directional_decrease > 0.0
        assert direction_squared > 0.0
        curvature = stored_curvature / 2.0
        trials = 0
        for _ in range(1000):
            alpha = min(1.0, directional_decrease / (curvature * direction_squared))
            trial_point = tuple(a + alpha * b for a, b in zip(point, direction))
            assert -1.0 <= trial_point[0] <= 1.0
            assert 0.0 <= trial_point[1] <= 1.0
            trial_value = objective(trial_point)
            trials += 1
            right_side = (
                value
                - alpha * directional_decrease
                + 0.5 * curvature * alpha * alpha * direction_squared
            )
            if trial_value <= right_side:
                break
            curvature *= 2.0
        else:
            raise AssertionError("The fixed line-search guard was exceeded")

        previous_curvature = stored_curvature
        point = trial_point
        value = trial_value  # Cache this value for all subsequent refinements.
        stored_curvature = curvature
        accepted_steps += 1
        trial_counts.append(trials)
        assert curvature <= max(previous_curvature / 2.0, 2.0 * CURVATURE)
        assert objective_calls == 1 + sum(trial_counts)
    else:
        raise AssertionError("The fixed directional-query guard was exceeded")

    return {
        "start_x": start_x,
        "epsilon": epsilon,
        "eta_initial": eta_initial,
        "theta": theta,
        "first_upper_gap": first_upper_gap,
        "accepted_steps": accepted_steps,
        "accuracy_halvings": refinements,
        "directional_queries": query_count,
        "objective_calls": objective_calls,
        "trial_counts": trial_counts,
        "final_point": list(point),
        "final_value": value,
        "final_lower_bound": lower_bound,
        "final_certified_error": value - lower_bound,
        "terminal_precision_would_pass": trace[-1]["precision_passes"],
        "trace": trace,
    }


def validate(result):
    """Use the known optimum only here, after the solver has returned."""
    epsilon = result["epsilon"]
    theta = result["theta"]
    eta_initial = result["eta_initial"]
    kappa = (1.0 - theta) / (1.0 + theta)
    refinements_bound = math.ceil(
        math.log2(max(1.0, (1.0 + theta) * eta_initial / (theta * epsilon)))
    )
    first_upper_gap = result["first_upper_gap"]
    phases = (
        math.ceil(math.log2(first_upper_gap / epsilon))
        if first_upper_gap > epsilon
        else 0
    )
    initial_large_curvatures = math.ceil(
        math.log2(max(1.0, INITIAL_CURVATURE / (2.0 * CURVATURE)))
    )
    steps_bound = initial_large_curvatures + math.ceil(
        4.0 * phases / kappa
        + 32.0 * CURVATURE * DIAMETER_SQUARED / (kappa * kappa * epsilon)
    )
    function_bound = (
        1
        + 2 * result["accepted_steps"]
        + math.ceil(math.log2(max(1.0, 2.0 * CURVATURE / INITIAL_CURVATURE)))
    )
    assert result["accuracy_halvings"] <= refinements_bound
    assert result["accepted_steps"] <= steps_bound
    assert result["objective_calls"] <= function_bound
    assert result["directional_queries"] == (
        result["accepted_steps"] + 1 + result["accuracy_halvings"]
    )
    assert sum(row["action"] == "refine" for row in result["trace"]) == result["accuracy_halvings"]
    assert sum(row["action"] == "step" for row in result["trace"]) == result["accepted_steps"]
    assert sum(row["action"] == "stop" for row in result["trace"]) == 1
    assert result["objective_calls"] == 1 + sum(result["trial_counts"])

    for previous, current in zip(result["trace"], result["trace"][1:]):
        if previous["action"] == "refine":
            assert current["point"] == previous["point"]
            assert current["objective_calls_before_query"] == previous["objective_calls_before_query"]
            assert current["eta"] == previous["eta"] / 2.0
        else:
            assert current["eta"] == previous["eta"]

    true_optimum = math.exp(-1.0)
    true_error = result["final_value"] - true_optimum
    assert true_error >= -1e-13
    assert true_error <= result["final_certified_error"] + 1e-13
    assert result["final_certified_error"] <= epsilon
    for row in result["trace"]:
        point_value = math.exp(row["point"][0]) + 100.0 * row["point"][1]
        assert point_value - true_optimum <= row["certified_error"] + 1e-12

    if result["start_x"] == -1.0:
        assert result["accepted_steps"] == 0
        assert result["objective_calls"] == 1
        assert not result["terminal_precision_would_pass"]
        assert result["final_point"] == [-1.0, 0.0]

    result["theoretical_bounds"] = {
        "accuracy_halvings": refinements_bound,
        "accepted_steps": steps_bound,
        "objective_calls": function_bound,
        "curvature": CURVATURE,
        "diameter_squared": DIAMETER_SQUARED,
    }
    result["posthoc_true_error"] = true_error
    result["posthoc_final_gradient_norm"] = math.hypot(math.exp(result["final_point"][0]), 100.0)
    result["verified"] = True


def main():
    runs = []
    for start_x in (-1.0, 0.7):
        for epsilon in (1e-3, 1e-6):
            for eta_initial in (0.3, 1.0):
                result = solve(start_x, epsilon, eta_initial)
                validate(result)
                runs.append(result)
    output = {
        "description": "Fixed deterministic checks of stopping before accuracy refinement",
        "problem": "exp(x) + 100*y on [-1,1] x [0,1]",
        "run_count": len(runs),
        "verified_runs": sum(run["verified"] for run in runs),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numerics": "Python standard-library binary64; analytical oracle error bounds; posthoc tolerance at most 1e-12",
        "true_optimum_used_by_solver": False,
        "runs": runs,
    }
    destination = ROOT / "Evidence" / "directional_checks.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("start_x epsilon eta0 N R T Nf certified_error")
    for run in runs:
        print(
            run["start_x"], run["epsilon"], run["eta_initial"],
            run["accepted_steps"], run["accuracy_halvings"],
            run["directional_queries"], run["objective_calls"],
            f'{run["final_certified_error"]:.9g}',
        )
    print(f"Verified {len(runs)}/{len(runs)} fixed runs; saved {destination}")


if __name__ == "__main__":
    main()
