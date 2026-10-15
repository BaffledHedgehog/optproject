"""AC-FW, Algorithms 1+2 of arXiv:2605.15512v2, with Eq. (6), delta=1.

Independent implementation, not the authors' software. The mandatory
initial LMO move is charged in every oracle counter. Rejected trials consume
an outer iteration and a function call, but reuse the unchanged gradient.
"""

import math
from time import perf_counter

import numpy as np

from fw import Result
from problems import lmo


def damping(iteration):
    return 1 - 1 / ((iteration + 1) * math.log(iteration + 3)**2)


def local_curvature(x, y, fx, fy, gradient):
    displacement = y - x
    norm_sq = float(displacement @ displacement)
    return 2 * abs(fy - fx - float(gradient @ displacement)) / norm_sq if norm_sq else 0.0


def solve_auto_conditioned(problem, max_iter, tolerance):
    start = perf_counter()
    previous = problem.x0.copy()
    previous_f, previous_g = problem.objective(previous), problem.gradient(previous)
    x = lmo(previous_g, problem.radius)
    fx, g = problem.objective(x), problem.gradient(x)
    curvature = local_curvature(previous, x, previous_f, fx, previous_g)
    # Both initial linearizations are valid; the first is already available
    # from the compulsory initialization and requires no extra oracle call.
    lower = previous_f + float(previous_g @ (x - previous))
    counts = {"function_evals": 2, "gradient_evals": 2, "lmo_evals": 1}
    accepted, rejected, trace = 0, 0, []
    for k in range(max_iter + 1):
        s = lmo(g, problem.radius)
        counts["lmo_evals"] += 1
        d = s - x
        raw_gap = float(-g @ d)
        if raw_gap < -1e-10 * max(1.0, abs(fx)):
            raise ArithmeticError("Negative FW gap")
        gap = max(0.0, raw_gap)
        lower = max(lower, fx - gap)
        upper = max(0.0, fx - lower)
        q2 = float(d @ d)
        trace.append({"iteration": k, "objective": fx, "fw_gap": gap,
                      "lower_bound": lower, "certificate": upper,
                      "direction_norm_sq": q2,
                      "l1_norm": float(np.linalg.norm(x, 1)),
                      "support": int(np.count_nonzero(np.abs(x) > 1e-10)),
                      "accepted_steps": accepted, "rejected_steps": rejected,
                      "elapsed_seconds": perf_counter() - start, **counts})
        if upper <= tolerance or k == max_iter:
            break
        if q2 == 0:
            raise ArithmeticError("Nonzero certificate with zero direction")
        alpha = min(1.0, gap / (curvature * q2)) if curvature > 0 else 1.0
        trial_x = x + alpha * d
        trial_f = problem.objective(trial_x)
        counts["function_evals"] += 1
        next_curvature = max(local_curvature(x, trial_x, fx, trial_f, g), damping(k) * curvature)
        improved = trial_f < fx  # strict acceptance in the cited Algorithm 1
        trace[-1].update(step=alpha, trial_curvature=curvature,
                         next_curvature=next_curvature, trial_accepted=improved,
                         trial_objective=trial_f, line_search_trials=1)
        if improved:
            x, fx = trial_x, trial_f
            g = problem.gradient(x)
            counts["gradient_evals"] += 1
            accepted += 1
        else:
            rejected += 1
        curvature = next_curvature
        if not np.isfinite(trial_f) or not np.all(np.isfinite(g)) or not math.isfinite(curvature):
            raise FloatingPointError("Nonfinite AC-FW oracle output")
    return Result(x, trace, perf_counter() - start, 0)
