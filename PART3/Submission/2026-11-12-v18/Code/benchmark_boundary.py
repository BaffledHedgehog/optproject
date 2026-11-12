"""Deterministic boundary-gradient benchmark for the v18 manuscript.

Problem: f_A(x,y)=exp(x)+A*y on [-1,1] x [0,1], start (0.7,0).
All instances are globally convex and (1,1)-smooth. Their Hessian and
all objective values on the visited face y=0 are independent of A.

Compared methods:
  * Equation (7), Algorithm 1 of Vyguzov--Stonyakin, arXiv:2510.16468v5,
    with fixed L0=L1=1, including the factor e in the denominator.
    This is NOT their adaptive Algorithm 2.
  * Ordinary quadratic Frank--Wolfe backtracking, with memory 1,
    initial trial at half the memory and doubling after rejection.

Both use the same exact box LMO, the same objective cache, and stop at
f(x_k)-max_{j<=k}(f(x_j)-G(x_j)) <= 1e-6. No optimum value or gradient
norm threshold enters either stopping decision. All algorithm objective,
gradient and LMO calls are counted, including the final source check.

Run: python Code/benchmark_boundary.py
Optional plots require matplotlib; all data and verification use stdlib.
"""

import argparse
import csv
import json
import math
from pathlib import Path


SOURCE = "https://arxiv.org/html/2510.16468v5#S2.E7"
NORMAL_COEFFICIENTS = (0.0, 1.0, 10.0, 100.0, 1000.0)
TOLERANCE = 1e-6
START = (0.7, 0.0)


def objective(point, normal_coefficient):
    return math.exp(point[0]) + normal_coefficient * point[1]


def gradient(point, normal_coefficient):
    return math.exp(point[0]), normal_coefficient


def box_lmo(grad):
    # At a zero coefficient, select the lower endpoint deterministically.
    return (-1.0 if grad[0] >= 0 else 1.0,
            0.0 if grad[1] >= 0 else 1.0)


def run(normal_coefficient, method, max_updates=1_000_000):
    point = START
    value = objective(point, normal_coefficient)
    n_objective = 1
    n_gradient = n_lmo = updates = rejected = 0
    remembered_curvature = 1.0
    lower_bound = -math.inf
    first_crossing_zero = None
    trace = []

    while True:
        grad = gradient(point, normal_coefficient)
        n_gradient += 1
        atom = box_lmo(grad)
        n_lmo += 1
        direction = tuple(s - x for s, x in zip(atom, point))
        gap = -sum(g * d for g, d in zip(grad, direction))
        lower_bound = max(lower_bound, value - gap)
        certificate = value - lower_bound
        trace_row = {
            "normal_coefficient": normal_coefficient,
            "method": method,
            "iteration": updates,
            "x": point[0], "y": point[1],
            "objective": value, "full_gap": gap,
            "accumulated_certificate": certificate,
            "gradient_norm": math.hypot(*grad),
            "objective_calls": n_objective,
            "gradient_calls": n_gradient, "lmo_calls": n_lmo,
            "alpha": None, "accepted_curvature": None,
            "rejected_trials_at_update": 0,
        }
        trace.append(trace_row)
        if certificate <= TOLERANCE:
            status = "converged"
            break
        if updates >= max_updates:
            status = "budget_exhausted"
            break

        direction_squared = sum(d * d for d in direction)
        if gap <= 0 or direction_squared == 0:
            raise ArithmeticError("Nonterminal source without a descent direction")

        if method == "vyguzov_stonyakin_fixed":
            alpha = min(1.0, gap / (math.e * (1.0 + math.hypot(*grad))
                                    * direction_squared))
            trial = tuple(x + alpha * d for x, d in zip(point, direction))
            new_value = objective(trial, normal_coefficient)
            n_objective += 1
        elif method == "quadratic_backtracking":
            curvature = remembered_curvature / 2.0
            local_rejected = 0
            while True:
                alpha = min(1.0, gap / (curvature * direction_squared))
                trial = tuple(x + alpha * d for x, d in zip(point, direction))
                new_value = objective(trial, normal_coefficient)
                n_objective += 1
                upper_model = value - alpha * gap
                upper_model += curvature * alpha * alpha * direction_squared / 2.0
                if new_value <= upper_model:
                    break
                local_rejected += 1
                curvature *= 2.0
                if local_rejected > 100:
                    raise ArithmeticError("Backtracking did not close")
            rejected += local_rejected
            remembered_curvature = curvature
            trace_row["accepted_curvature"] = curvature
            trace_row["rejected_trials_at_update"] = local_rejected
        else:
            raise ValueError(method)

        if not (-1.0 <= trial[0] <= 1.0 and 0.0 <= trial[1] <= 1.0):
            raise ArithmeticError("Infeasible update")
        if new_value > value + 8e-15 * max(1.0, abs(value)):
            raise ArithmeticError("Objective increased")
        if trial == point:
            raise ArithmeticError("Floating-point stagnation before certificate closure")
        trace_row["alpha"] = alpha
        point, value = trial, new_value
        updates += 1
        if first_crossing_zero is None and point[0] <= 0.0:
            first_crossing_zero = updates

    # Analytic solution is used only for this diagnostic, after stopping.
    true_error = value - math.exp(-1.0)
    if true_error > certificate + 1e-12:
        raise ArithmeticError("The computed certificate does not cover the true error")
    if n_gradient != updates + 1 or n_lmo != updates + 1:
        raise ArithmeticError("Source accounting failed")
    if method == "quadratic_backtracking":
        assert n_objective == 1 + updates + rejected
    else:
        assert n_objective == updates + 1
    crossing_lower_bound = math.ceil(START[0] * math.exp(1.0 - START[0])
                                     * normal_coefficient)
    if method == "vyguzov_stonyakin_fixed" and status == "converged":
        assert first_crossing_zero is not None
        assert first_crossing_zero >= crossing_lower_bound
    row = {
        "normal_coefficient": normal_coefficient, "method": method,
        "status": status, "accepted_updates": updates,
        "objective_calls": n_objective, "gradient_calls": n_gradient,
        "lmo_calls": n_lmo, "rejected_trials": rejected,
        "final_x": point[0], "final_y": point[1],
        "final_gap": gap, "final_certificate": certificate,
        "true_error_diagnostic": true_error,
        "final_gradient_norm": math.hypot(*grad),
        "optimum_gradient_norm": math.hypot(math.exp(-1.0), normal_coefficient),
        "first_crossing_zero": first_crossing_zero,
        "fixed_rule_crossing_lower_bound": crossing_lower_bound,
    }
    return row, trace


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.0), layout="constrained")
    styles = [("vyguzov_stonyakin_fixed", "Fixed (L0,L1) rule", "#b45309", "o"),
              ("quadratic_backtracking", "Quadratic backtracking", "#0369a1", "s")]
    for method, label, color, marker in styles:
        series = [row for row in rows if row["method"] == method]
        axes[0].plot([row["normal_coefficient"] for row in series],
                     [row["accepted_updates"] for row in series],
                     marker=marker, color=color, label=label, linewidth=1.5)
    series = [row for row in rows if row["method"] == "quadratic_backtracking"]
    axes[1].plot([row["normal_coefficient"] for row in series],
                 [row["optimum_gradient_norm"] for row in series],
                 marker="o", color="#334155", linewidth=1.5)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("log")
        ax.set_xticks(NORMAL_COEFFICIENTS, ["0", "1", "10", "100", "1000"])
        ax.set_xlabel("Normal coefficient A")
        ax.grid(alpha=0.2, which="major")
    axes[0].set_ylabel("Accepted updates")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel("Gradient norm at the optimum")
    axes[0].set_title("Same stopping certificate")
    axes[1].set_title("The gradient does not vanish")
    fig.savefig(destination / "boundary_normal_benchmark.pdf", bbox_inches="tight")
    fig.savefig(destination / "boundary_normal_benchmark.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--output", type=Path, default=root / "Evidence")
    parser.add_argument("--plots", type=Path, default=root / "Paper" / "figures")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, all_trace = [], []
    reference_backtracking = None
    for normal_coefficient in NORMAL_COEFFICIENTS:
        for method in ("vyguzov_stonyakin_fixed", "quadratic_backtracking"):
            row, trace = run(normal_coefficient, method)
            rows.append(row)
            all_trace.extend(trace)
            if method == "quadratic_backtracking":
                invariant = [(z["x"], z["y"], z["full_gap"], z["alpha"],
                              z["accepted_curvature"], z["objective_calls"])
                             for z in trace]
                if reference_backtracking is None:
                    reference_backtracking = invariant
                else:
                    assert invariant == reference_backtracking
    protocol = {
        "objective": "exp(x)+A*y", "domain": "[-1,1] x [0,1]",
        "normal_coefficients": NORMAL_COEFFICIENTS, "start": START,
        "tolerance": TOLERANCE, "fixed_parameters": {"L0": 1.0, "L1": 1.0},
        "backtracking_initial_memory": 1.0,
        "backtracking_initial_trial": "memory/2",
        "backtracking_growth_factor": 2.0,
        "stopping": "f(x_k)-max_{j<=k}(f(x_j)-G(x_j)) <= tolerance",
        "source": SOURCE,
        "comparison_scope": "Fixed-parameter Algorithm 1, not adaptive Algorithm 2",
        "gradient_lmo_calls": "One of each per source check, including terminal",
        "objective_calls": "All initial, accepted and rejected algorithm evaluations",
        "randomness": "None",
        "backtracking_trajectories_identical": True,
        "global_generalized_smoothness": "L0=L1=1, since Hessian norm=exp(x)<=gradient norm",
        "curvature_on_domain": "Hessian norm<=e, independent of A",
        "true_optimum_usage": "Only diagnostic verification after the algorithm stops",
        "results": rows,
    }
    write_csv(args.output / "boundary_benchmark.csv", rows)
    write_csv(args.output / "boundary_trace.csv", all_trace)
    (args.output / "boundary_benchmark.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    if not args.no_plots:
        make_plots(rows, args.plots)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
