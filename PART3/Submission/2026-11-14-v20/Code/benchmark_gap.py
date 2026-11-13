"""Fixed numerical protocol for the v19 gap-dependent curvature model.

Run from the release directory:
    python Code/benchmark_gap.py --no-plots
    python Code/benchmark_gap.py

Only the optional figures require matplotlib. All solver arithmetic,
special functions, checks and saved tables use the Python standard library.

The protocol below is fixed before execution. There is no parameter search.
Each solver obtains one objective, gradient and full linear-oracle value at
each visited point, including the initial and terminal point. Both methods
stop using f(x)-max_past(f-G/gamma), without an optimum value. There are no
line searches. Known minima are used only by the separate diagnostic check.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


# Fixed protocol. Do not adjust these parameters using the observed results.
RADII = (1.0, 2.0, 4.0, 6.0, 8.0)
NORMAL_COEFFICIENTS = (1.0, 1000.0)
TOLERANCE = 1e-6
MAX_UPDATES = 1_000_000
MAIN_L0 = 1.0
MAIN_L1 = 1.0
MAIN_GAMMA = 1.0


def exact_box_lmo(gradient, lower, upper):
    return tuple(lo if g >= 0.0 else hi
                 for g, lo, hi in zip(gradient, lower, upper))


def run_solver(objective, gradient, lower, upper, start, diameter,
               L0, L1, gamma, rule, ordinary_L=None):
    """Return a complete trace; this interface has no optimum argument."""
    point = tuple(start)
    lower_certificate = -math.inf
    trace = []
    updates = 0
    while True:
        value = objective(point)
        grad = gradient(point)
        atom = exact_box_lmo(grad, lower, upper)
        direction = tuple(s - x for s, x in zip(atom, point))
        gap = -sum(g * d for g, d in zip(grad, direction))
        if gap < -1e-12:
            raise ArithmeticError("A full box oracle produced a negative gap")
        gap = max(0.0, gap)
        lower_certificate = max(lower_certificate, value - gap / gamma)
        certificate = value - lower_certificate
        record = {
            "iteration": updates,
            "x": point[0],
            "y": point[1] if len(point) > 1 else None,
            "objective": value,
            "full_gap": gap,
            "retained_lower_bound": lower_certificate,
            "accumulated_certificate": certificate,
            "gradient_norm": math.sqrt(sum(g * g for g in grad)),
            "objective_calls": updates + 1,
            "gradient_calls": updates + 1,
            "full_lmo_calls": updates + 1,
            "alpha": None,
            "step_lower_bound": None,
            "direction_norm": math.sqrt(sum(d * d for d in direction)),
        }
        trace.append(record)
        if certificate <= TOLERANCE:
            status = "converged"
            break
        if updates >= MAX_UPDATES:
            status = "budget_exhausted"
            break
        squared_length = sum(d * d for d in direction)
        if gap <= 0.0 or squared_length == 0.0:
            raise ArithmeticError("Nonterminal point without a descent direction")

        if rule == "gap_logarithmic":
            if L1 > 0.0:
                length = math.sqrt(squared_length)
                coefficient = L0 + L1 * gap
                exponent_rate = diameter * L1 * length
                alpha = min(1.0, math.log1p(
                    diameter * L1 * gap / (coefficient * length)
                ) / exponent_rate)
            elif L0 > 0.0:
                alpha = min(1.0, gap / (L0 * squared_length))
            else:
                alpha = 1.0
            if L0 + L1 * gap > 0.0:
                record["step_lower_bound"] = min(
                    1.0, math.log(2.0) * gap /
                    ((L0 + L1 * gap) * diameter * diameter))
        elif rule == "ordinary_best_global_L":
            if ordinary_L is None or ordinary_L <= 0.0:
                raise ValueError("The ordinary rule needs a positive global L")
            alpha = min(1.0, gap / (ordinary_L * squared_length))
        else:
            raise ValueError(rule)

        if not 0.0 < alpha <= 1.0:
            raise ArithmeticError("Invalid step size")
        record["alpha"] = alpha
        # This convex-combination form reaches an oracle endpoint exactly
        # when alpha=1, without subtractive cancellation.
        trial = tuple((1.0 - alpha) * x + alpha * s
                      for x, s in zip(point, atom))
        if trial == point:
            raise ArithmeticError("Stagnation before the certificate closes")
        point = trial
        updates += 1

    return status, trace


def exponential_problem(radius, normal_coefficient, start_at_minimum=False):
    return {
        "id": ("boundary_minimum_start" if start_at_minimum else
               f"exp_R{radius:g}_A{normal_coefficient:g}"),
        "family": "exponential",
        "formula": "f(x,y)=exp(x)+A*y",
        "objective": lambda z: math.exp(z[0]) + normal_coefficient * z[1],
        "gradient": lambda z: (math.exp(z[0]), normal_coefficient),
        "hessian_norm": lambda z: math.exp(z[0]),
        "lower": (0.0, 0.0), "upper": (radius, 1.0),
        "start": (0.0, 0.0) if start_at_minimum else (radius, 0.0),
        "diameter": math.hypot(radius, 1.0),
        "L0": MAIN_L0, "L1": MAIN_L1, "gamma": MAIN_GAMMA,
        "ordinary_L": math.exp(radius),
        "R": radius, "A": normal_coefficient,
        "diagnostic_optimum": (0.0, 0.0),
        "diagnostic_optimum_value": 1.0,
        "diagnostic_optimum_gradient_norm": math.hypot(1.0, normal_coefficient),
        "model_proof": "G=x*exp(x)+A*y; exp(x)<=1+x*exp(x)<=1+G on Q",
        "error_proof": "Convexity gives f(x)-min_Q f <= G(x)",
    }


def additional_problems():
    coefficient = math.sqrt(math.pi) / (2.0 * math.sqrt(2.0))
    return [
        {
            "id": "nonconvex_L0_zero", "family": "erf_nonconvex",
            "formula": "f(x)=sqrt(pi)/(2*sqrt(2))*erf(sqrt(2)*x)",
            "objective": lambda z: coefficient * math.erf(math.sqrt(2.0) * z[0]),
            "gradient": lambda z: (math.exp(-2.0 * z[0] ** 2),),
            "hessian_norm": lambda z: 4.0 * z[0] * math.exp(-2.0 * z[0] ** 2),
            "lower": (0.0,), "upper": (1.0,), "start": (1.0,),
            "diameter": 1.0, "L0": 0.0, "L1": 4.0,
            "gamma": math.exp(-2.0), "ordinary_L": None, "R": None, "A": None,
            "diagnostic_optimum": (0.0,), "diagnostic_optimum_value": 0.0,
            "diagnostic_optimum_gradient_norm": 1.0,
            "model_proof": "f'(x)=exp(-2*x*x)>0; |f''(x)|=4*x*f'(x)=4*G(x)",
            "error_proof": "f(x)<=x and G(x)>=exp(-2)*x on [0,1]",
        },
        {
            "id": "smooth_L1_zero", "family": "quadratic_plus_linear",
            "formula": "f(x)=x+x*x/2",
            "objective": lambda z: z[0] + z[0] ** 2 / 2.0,
            "gradient": lambda z: (1.0 + z[0],),
            "hessian_norm": lambda z: 1.0,
            "lower": (0.0,), "upper": (1.0,), "start": (1.0,),
            "diameter": 1.0, "L0": 1.0, "L1": 0.0, "gamma": 1.0,
            "ordinary_L": None, "R": None, "A": None,
            "diagnostic_optimum": (0.0,), "diagnostic_optimum_value": 0.0,
            "diagnostic_optimum_gradient_norm": 1.0,
            "model_proof": "|f''(x)|=1=L0",
            "error_proof": "Convexity gives f(x)-min_Q f <= G(x)",
        },
        {
            "id": "affine_L0_L1_zero", "family": "affine",
            "formula": "f(x)=x",
            "objective": lambda z: z[0],
            "gradient": lambda z: (1.0,),
            "hessian_norm": lambda z: 0.0,
            "lower": (0.0,), "upper": (1.0,), "start": (1.0,),
            "diameter": 1.0, "L0": 0.0, "L1": 0.0, "gamma": 1.0,
            "ordinary_L": None, "R": None, "A": None,
            "diagnostic_optimum": (0.0,), "diagnostic_optimum_value": 0.0,
            "diagnostic_optimum_gradient_norm": 1.0,
            "model_proof": "f''(x)=0", "error_proof": "f(x)-min_Q f=G(x)=x",
        },
        exponential_problem(4.0, 1000.0, start_at_minimum=True),
    ]


def validate_run(problem, rule, status, trace):
    """Post-run checks using recorded oracle values and known test minima."""
    maximum_descent_violation = 0.0
    maximum_certificate_violation = 0.0
    for index, record in enumerate(trace):
        point = (record["x"],) if record["y"] is None else (record["x"], record["y"])
        assert all(lo <= x <= hi for x, lo, hi in zip(point, problem["lower"], problem["upper"]))
        scale = max(1.0, abs(record["objective"]))
        tolerance = 5e-13 * scale
        true_error = record["objective"] - problem["diagnostic_optimum_value"]
        certificate_violation = true_error - record["accumulated_certificate"]
        maximum_certificate_violation = max(maximum_certificate_violation, certificate_violation)
        assert certificate_violation <= tolerance
        assert record["retained_lower_bound"] <= problem["diagnostic_optimum_value"] + tolerance
        assert record["full_gap"] / problem["gamma"] + tolerance >= true_error
        assert problem["hessian_norm"](point) <= problem["L0"] + problem["L1"] * record["full_gap"] + tolerance
        assert record["objective_calls"] == index + 1
        assert record["gradient_calls"] == index + 1
        assert record["full_lmo_calls"] == index + 1
        if record["step_lower_bound"] is not None:
            assert record["alpha"] + 5e-14 >= record["step_lower_bound"]
        if index + 1 < len(trace):
            following = trace[index + 1]
            violation = (following["objective"] - record["objective"]
                         + record["alpha"] * record["full_gap"] / 2.0)
            maximum_descent_violation = max(maximum_descent_violation, violation)
            assert violation <= tolerance
            assert following["accumulated_certificate"] <= record["accumulated_certificate"] + tolerance
            assert following["retained_lower_bound"] + tolerance >= record["retained_lower_bound"]

    updates = len(trace) - 1
    final = trace[-1]
    assert status == "converged", "A fixed-protocol run exhausted its budget"
    assert final["accumulated_certificate"] <= TOLERANCE
    lower_count = upper_count = None
    if problem["family"] == "exponential" and problem["id"] != "boundary_minimum_start":
        if rule == "gap_logarithmic":
            upper_count = math.ceil(problem["R"] * problem["diameter"] / math.log(2.0))
            assert updates <= upper_count
            # On y=0, the exact update is max(0,x-h(x)); check the
            # independently simplified expression and its uniform decrease.
            for before, after in zip(trace, trace[1:]):
                x = before["x"]
                h = math.log1p(problem["diameter"] / (x + math.exp(-x))) / problem["diameter"]
                expected = max(0.0, x - h)
                assert abs(after["x"] - expected) <= 3e-13 * max(1.0, x)
                assert h + 3e-14 >= math.log(2.0) / problem["diameter"]
        else:
            lower_count = math.ceil(max(0.0,
                (math.exp(problem["R"]) / (1.0 + TOLERANCE) - 1.0) / (math.e - 1.0)))
            assert updates >= lower_count
            potential_increment_bound = (math.e - 1.0) * math.exp(-problem["R"])
            for before, after in zip(trace, trace[1:]):
                increment = math.exp(-after["x"]) - math.exp(-before["x"])
                assert increment <= potential_increment_bound + 5e-14

    if problem["id"] == "nonconvex_L0_zero":
        assert max(z["full_gap"] for z in trace) > trace[0]["full_gap"] * 1.01
        assert abs(final["gradient_norm"] - 1.0) <= 1e-12
        uniform_alpha = min(1.0, math.log(2.0) / (problem["L1"] * problem["diameter"] ** 2))
        assert all(z["alpha"] + 1e-14 >= uniform_alpha for z in trace[:-1])
    if problem["id"] in ("smooth_L1_zero", "affine_L0_L1_zero"):
        assert updates == 1
    if problem["id"] == "boundary_minimum_start":
        assert updates == 0
        assert final["gradient_norm"] > 1000.0

    return {
        "problem": problem["id"], "family": problem["family"], "rule": rule,
        "R": problem["R"], "A": problem["A"],
        "L0": problem["L0"], "L1": problem["L1"], "gamma": problem["gamma"],
        "status": status, "accepted_updates": updates,
        "objective_calls": updates + 1, "gradient_calls": updates + 1,
        "full_lmo_calls": updates + 1,
        "initial_gap": trace[0]["full_gap"],
        "maximum_observed_gap": max(z["full_gap"] for z in trace),
        "final_x": final["x"], "final_y": final["y"],
        "final_gap": final["full_gap"],
        "final_certificate": final["accumulated_certificate"],
        "true_error_diagnostic": final["objective"] - problem["diagnostic_optimum_value"],
        "final_gradient_norm": final["gradient_norm"],
        "optimum_gradient_norm_diagnostic": problem["diagnostic_optimum_gradient_norm"],
        "theoretical_update_lower_bound": lower_count,
        "theoretical_update_upper_bound": upper_count,
        "maximum_descent_violation": maximum_descent_violation,
        "maximum_certificate_violation": maximum_certificate_violation,
    }


def trajectory_signature(trace):
    # Normal gradient coordinates and their norms differ with A. All points,
    # steps, objective values, full gaps, certificates and counts must agree.
    keys = ("x", "y", "objective", "full_gap", "retained_lower_bound",
            "accumulated_certificate", "alpha", "step_lower_bound",
            "direction_norm", "objective_calls", "gradient_calls", "full_lmo_calls")
    return [tuple(row[key] for key in keys) for row in trace]


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
    languages = {
        "en": {
            "log": "Gap-based logarithmic step", "ordinary": "Short step with L = exp(R)",
            "left_title": "Both A = 1 and A = 1000", "left_x": "Interval endpoint R",
            "left_y": "Accepted updates", "right_title": "Gradient at the boundary minimum",
            "right_x": "Normal coefficient A", "right_y": "Gradient norm",
        },
        "ru": {
            "log": "Логарифмический шаг через зазор", "ordinary": "Короткий шаг с L = exp(R)",
            "left_title": "Для A = 1 и A = 1000", "left_x": "Правая граница отрезка R",
            "left_y": "Число принятых шагов", "right_title": "Градиент в граничном минимуме",
            "right_x": "Нормальный коэффициент A", "right_y": "Норма градиента",
        },
    }
    for language, labels in languages.items():
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), layout="constrained")
        for rule, label_key, color, marker in (
            ("gap_logarithmic", "log", "#0369a1", "o"),
            ("ordinary_best_global_L", "ordinary", "#b45309", "s"),
        ):
            series = [z for z in rows if z["family"] == "exponential"
                      and z["problem"] != "boundary_minimum_start"
                      and z["A"] == 1.0 and z["rule"] == rule]
            axes[0].plot([z["R"] for z in series], [z["accepted_updates"] for z in series],
                         marker=marker, color=color, label=labels[label_key], linewidth=1.5)
        axes[0].set_yscale("log")
        axes[0].set_xticks(RADII, [str(int(z)) for z in RADII])
        axes[0].set_xlabel(labels["left_x"])
        axes[0].set_ylabel(labels["left_y"])
        axes[0].set_title(labels["left_title"])
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        axes[0].grid(alpha=0.2)
        values = [math.hypot(1.0, normal) for normal in NORMAL_COEFFICIENTS]
        axes[1].bar([0, 1], values, width=0.5, color=["#64748b", "#334155"])
        axes[1].set_xticks([0, 1], ["1", "1000"])
        axes[1].set_yscale("log")
        axes[1].set_xlabel(labels["right_x"])
        axes[1].set_ylabel(labels["right_y"])
        axes[1].set_title(labels["right_title"])
        axes[1].grid(axis="y", alpha=0.2)
        axes[1].set_axisbelow(True)
        fig.savefig(destination / f"gap_benchmark_{language}.pdf", bbox_inches="tight")
        fig.savefig(destination / f"gap_benchmark_{language}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    release = Path(__file__).resolve().parent.parent
    parser.add_argument("--output", type=Path, default=release / "Evidence")
    parser.add_argument("--plots", type=Path, default=release / "Paper" / "figures")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    problems = [exponential_problem(radius, normal)
                for radius in RADII for normal in NORMAL_COEFFICIENTS]
    problems += additional_problems()
    rows, complete_trace, model_records = [], [], []
    reference_trajectories = {}
    for problem in problems:
        model_records.append({key: value for key, value in problem.items() if not callable(value)})
        rules = (("gap_logarithmic", "ordinary_best_global_L")
                 if problem["family"] == "exponential" and problem["id"] != "boundary_minimum_start"
                 else ("gap_logarithmic",))
        for rule in rules:
            status, trace = run_solver(
                problem["objective"], problem["gradient"], problem["lower"], problem["upper"],
                problem["start"], problem["diameter"], problem["L0"], problem["L1"],
                problem["gamma"], rule, problem["ordinary_L"])
            rows.append(validate_run(problem, rule, status, trace))
            if problem["family"] == "exponential" and problem["id"] != "boundary_minimum_start":
                key = (problem["R"], rule)
                signature = trajectory_signature(trace)
                if key in reference_trajectories:
                    assert signature == reference_trajectories[key]
                else:
                    reference_trajectories[key] = signature
            for record in trace:
                complete_trace.append({"problem": problem["id"], "rule": rule,
                                       "R": problem["R"], "A": problem["A"], **record})

    protocol = {
        "protocol_version": 1,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "radii": RADII, "normal_coefficients": NORMAL_COEFFICIENTS,
        "tolerance": TOLERANCE, "maximum_updates": MAX_UPDATES,
        "stopping": "f(x_k)-max_{j<=k}(f(x_j)-G(x_j)/gamma) <= tolerance",
        "algorithm_calls": "Exactly N+1 objective, N+1 gradient, N+1 full LMO calls",
        "line_searches": "None",
        "optimum_in_solver": False,
        "diagnostics": "Known test minima are used only after each solver run",
        "invariance_checked": "Both rules: all iterates, steps, objectives, gaps, certificates and calls identical across A",
        "parameter_selection": "All constants, grids, starting points and tolerances fixed in source before execution",
        "models": model_records,
        "results": rows,
    }
    write_csv(args.output / "gap_summary.csv", rows)
    write_csv(args.output / "gap_trace.csv", complete_trace)
    (args.output / "gap_summary.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (args.output / "gap_checks.json").write_text(json.dumps({
        "status": "passed", "runs": len(rows),
        "all_runs_converged": all(z["status"] == "converged" for z in rows),
        "complete_source_accounting": True, "descent_alpha_gap_over_two": True,
        "feasibility": True, "certificates_cover_true_error": True,
        "theoretical_iteration_bounds": True, "step_lower_bounds": True,
        "trajectory_invariance_across_A": True,
        "nonconvex_gap_increases_before_termination": True,
        "L1_zero_case": True, "affine_case": True, "boundary_optimum_start": True,
        "results": rows,
    }, indent=2) + "\n", encoding="utf-8")
    if not args.no_plots:
        make_plots(rows, args.plots)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
