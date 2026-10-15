"""Certificate-based FW and explicit, auditable comparison implementations.

The solver never reads a problem's reference solution or reference value.
Function/gradient/LMO calls count actual calls, including line-search trials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from problems import Problem, lmo


METHODS = ("H_gap", "H_memory", "H_nonnegative", "quadratic_memory", "global_L",
           "diminishing", "Armijo", "adaptive_L", "adaptive_L0L1", "secant", "line_search",
           "adaptive_certificate", "adaptive_L_initial", "AC_FW")


def quadratic_step(gap, upper_gap, q, h0, h1):
    if gap <= 0 or q == 0:
        return 0.0
    curvature = (h0 + h1 * upper_gap) * q * q
    return min(1.0, gap / curvature) if curvature > 0 else 1.0


def h_step(gap, upper_gap, q, h0, h1):
    """Stable equivalent of artanh(G / (c * (U + H0/H1))) / c."""
    if gap <= 0 or q == 0:
        return 0.0
    curvature = (h0 + h1 * upper_gap) * q * q
    if curvature == 0:
        return 1.0
    short = gap / curvature
    if h1 == 0:
        return min(1.0, short)
    c = q * math.sqrt(h1)
    ratio = short * c
    if ratio >= math.tanh(c):
        return 1.0
    factor = 1 + ratio * ratio / 3 if ratio < 1e-7 else math.atanh(ratio) / ratio
    return min(1.0, short * factor)


def corrected_lower_bound(f_approx, gradient_approx, x, s, diameter,
                          gradient_error=0.0, lmo_error=0.0, value_error=0.0):
    """Valid for deterministic norm/additive error bounds, not estimated variance.

    ||g_hat-grad f(x)||<=gradient_error, g_hat@s<=min_Q(g_hat@v)+lmo_error,
    and |f_approx-f(x)|<=value_error imply this is a lower bound for min_Q f.
    """
    if min(gradient_error, lmo_error, value_error, diameter) < 0:
        raise ValueError("Error bounds and diameter must be nonnegative")
    return float(f_approx + np.dot(gradient_approx, s - x)
                 - gradient_error * diameter - lmo_error - value_error)


@dataclass
class Result:
    x: np.ndarray
    trace: list[dict]
    elapsed_seconds: float
    line_search_fallbacks: int


def solve(problem: Problem, method: str, max_iter=1000, tolerance=1e-5,
          diameter_bound: float | None = None) -> Result:
    if method not in METHODS:
        raise ValueError(method)
    if max_iter < 0 or tolerance < 0:
        raise ValueError("Invalid budget or tolerance")
    if method == "AC_FW":
        from auto_conditioned import solve_auto_conditioned
        return solve_auto_conditioned(problem, max_iter, tolerance)
    diameter = 2 * problem.radius if diameter_bound is None else diameter_bound
    if method in ("adaptive_certificate", "adaptive_L_initial") and (
        not math.isfinite(diameter) or diameter <= 0 or diameter < 2 * problem.radius
    ):
        raise ValueError("Need a positive, valid upper bound on the l1-ball diameter")
    counts = {"function_evals": 0, "gradient_evals": 0, "lmo_evals": 0}

    def objective(x):
        counts["function_evals"] += 1
        return problem.objective(x)

    def gradient(x):
        counts["gradient_evals"] += 1
        return problem.gradient(x)

    start = perf_counter()
    x = problem.x0.copy()
    fx, g = objective(x), gradient(x)
    lower = problem.known_lower if method == "H_nonnegative" else -math.inf
    if lower is None:
        raise ValueError("This variant requires a valid known lower bound")
    local_l, l0, l1, toggle = 1.0, 1.0, 1.0, 0
    normalized_curvature = 1.0
    fallbacks = 0
    trace = []
    for k in range(max_iter + 1):
        s = lmo(g, problem.radius)
        counts["lmo_evals"] += 1
        d = s - x
        raw_gap = float(-np.dot(g, d))
        if raw_gap < -1e-10 * max(1.0, abs(fx)):
            raise ArithmeticError("Negative FW gap: infeasible iterate or invalid LMO")
        gap = max(0.0, raw_gap)
        lower = max(lower, fx - gap)
        upper = max(0.0, fx - lower)
        trace.append({"iteration": k, "objective": fx, "fw_gap": gap,
                      "lower_bound": lower, "certificate": upper,
                      "direction_norm_sq": float(np.dot(d, d)),
                      "l1_norm": float(np.linalg.norm(x, 1)),
                      "support": int(np.count_nonzero(np.abs(x) > 1e-10)),
                      "elapsed_seconds": perf_counter() - start, **counts})
        if upper <= tolerance or k == max_iter:
            break
        if method == "adaptive_L_initial" and k == 0:
            local_l = upper / (diameter * diameter)
        q2 = float(np.dot(d, d))
        if q2 == 0:
            raise ArithmeticError("Nonzero certificate with zero direction")
        q = math.sqrt(q2)
        next_f, next_g = None, None
        if method.startswith("H_"):
            u = gap if method == "H_gap" else upper
            alpha = h_step(gap, u, q, problem.h0, problem.h1)
        elif method == "quadratic_memory":
            alpha = quadratic_step(gap, upper, q, problem.h0, problem.h1)
        elif method == "global_L":
            alpha = min(1.0, gap / (problem.global_l * q2)) if problem.global_l else 1.0
        elif method == "diminishing":
            alpha = 2.0 / (k + 2)
        elif method == "Armijo":
            alpha = 1.0
            for _ in range(100):
                next_f = objective(x + alpha * d)
                if next_f <= fx - 1e-4 * alpha * gap:
                    break
                alpha *= 0.5
            else:
                raise RuntimeError("Armijo failed")
        elif method == "adaptive_certificate":
            # No H0, H1, global L or optimal-value input. The dimensionless
            # estimate is warm-started and the trial curvature scales with U.
            normalized_curvature *= 0.5
            for trial in range(1024):
                curvature = normalized_curvature * upper / (diameter * diameter)
                if not math.isfinite(curvature) or curvature <= 0:
                    raise FloatingPointError("Certificate curvature is not representable")
                alpha = min(1.0, gap / (curvature * q2))
                next_f = objective(x + alpha * d)
                if next_f <= fx - alpha * gap + 0.5 * curvature * alpha * alpha * q2:
                    break
                normalized_curvature *= 2
            else:
                raise RuntimeError("Certificate backtracking failed")
            trace[-1].update(trial_curvature=curvature,
                             normalized_curvature=normalized_curvature,
                             line_search_trials=trial + 1)
        elif method in ("adaptive_L", "adaptive_L_initial", "adaptive_L0L1"):
            if method in ("adaptive_L", "adaptive_L_initial"):
                local_l *= 0.5
            else:
                # Vyguzov--Stonyakin v5, Algorithm 2, rho=3.
                a = l0 + l1 * np.linalg.norm(g)
                l0, l1 = l0 / (3 + l0 / a), l1 / (3 + l1 * np.linalg.norm(g) / a)
            for trial in range(100):
                a = local_l if method in ("adaptive_L", "adaptive_L_initial") else math.e * (l0 + l1 * np.linalg.norm(g))
                alpha = min(1.0, gap / (a * q2))
                next_f = objective(x + alpha * d)
                if next_f <= fx - alpha * gap + 0.5 * a * alpha * alpha * q2:
                    break
                if method in ("adaptive_L", "adaptive_L_initial"):
                    local_l *= 2
                else:
                    total = l0 + l1 * np.linalg.norm(g)
                    if toggle == 0:
                        l0 *= 3 - l0 / total
                    else:
                        l1 *= 3 - l1 * np.linalg.norm(g) / total
                    toggle = 1 - toggle
            else:
                raise RuntimeError("Adaptive search failed")
            trace[-1].update(trial_curvature=float(a), line_search_trials=trial + 1)
        elif method == "secant":
            # Clipped SLS with a bracket/bisection safeguard. This is explicitly
            # a safeguarded variant, not a bitwise reproduction of the paper.
            endpoint_g = gradient(x + d)
            endpoint_slope = float(np.dot(endpoint_g, d))
            if endpoint_slope <= 0:
                alpha, next_g = 1.0, endpoint_g
            else:
                lo, hi, slo, shi = 0.0, 1.0, -gap, endpoint_slope
                previous, previous_slope = 0.0, -gap
                alpha, slope = 1e-3, None
                for j in range(80):
                    next_g = gradient(x + alpha * d)
                    slope = float(np.dot(next_g, d))
                    if abs(slope) <= 1e-10 * max(1.0, gap) or hi - lo <= 1e-12:
                        break
                    if slope < 0:
                        lo, slo = alpha, slope
                    else:
                        hi, shi = alpha, slope
                    denominator = slope - previous_slope
                    trial = alpha - slope * (alpha - previous) / denominator if denominator else math.nan
                    previous, previous_slope = alpha, slope
                    if not math.isfinite(trial) or not lo < trial < hi or j % 8 == 7:
                        trial = (lo + hi) / 2
                        fallbacks += 1
                    alpha = trial
                else:
                    raise RuntimeError("Safeguarded secant failed")
        elif method == "line_search":
            # Golden section on [0,1] plus endpoints, absolute interval tol 1e-10.
            lo, hi = 0.0, 1.0
            tau = (math.sqrt(5) - 1) / 2
            left, right = hi - tau * (hi - lo), lo + tau * (hi - lo)
            fl, fr = objective(x + left * d), objective(x + right * d)
            while hi - lo > 1e-10:
                if fl <= fr:
                    hi, right, fr = right, left, fl
                    left = hi - tau * (hi - lo)
                    fl = objective(x + left * d)
                else:
                    lo, left, fl = left, right, fr
                    right = lo + tau * (hi - lo)
                    fr = objective(x + right * d)
            endpoint_f = objective(x + d)
            next_f, alpha = min((fx, 0.0), (fl, left), (fr, right), (endpoint_f, 1.0))
        else:
            raise AssertionError(method)
        trace[-1]["step"] = float(alpha)
        trace[-1]["model_upper_gap"] = gap if method == "H_gap" else upper
        x = x + alpha * d
        fx = objective(x) if next_f is None else next_f
        g = gradient(x) if next_g is None else next_g
        if not np.isfinite(fx) or not np.all(np.isfinite(g)):
            raise FloatingPointError("Nonfinite oracle output")
    return Result(x, trace, perf_counter() - start, fallbacks)
