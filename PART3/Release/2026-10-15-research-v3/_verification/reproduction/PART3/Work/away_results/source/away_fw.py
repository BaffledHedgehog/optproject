"""Standard away-step FW with ordinary backtracking or a known-H short step.

The active dictionary is {+/- R e_i}. No optimal face or reference is read.
Accepted function values are cached at the actual reconstructed trial point.
Active-set scans are counted separately from full linear-oracle calls.
"""

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np

from fw import Result
from problems import lmo


@dataclass
class AwayResult(Result):
    weights: dict[int, float]


def reconstruct(weights, dimension, radius):
    x = np.zeros(dimension)
    for atom, weight in weights.items():
        x[atom//2] += radius*weight*(1 if atom % 2 == 0 else -1)
    return x


def decompose(x, radius):
    if not math.isfinite(radius) or radius <= 0 or not np.all(np.isfinite(x)):
        raise ValueError("Need a positive radius and finite initial point")
    weights = {2*j+(value < 0): abs(float(value))/radius
               for j, value in enumerate(x) if value != 0}
    total = math.fsum(weights.values())
    if total > 1+1e-13:
        raise ValueError("Initial point is outside the l1 ball")
    if total < 1:
        for atom in (0, 1):
            weights[atom] = weights.get(atom, 0.0)+(1-total)/2
    total = math.fsum(weights.values())
    return {atom: weight/total for atom, weight in weights.items()}


def trial_state(weights, kind, atom, alpha, cap, dimension, radius):
    if kind == "FW" and alpha == 1:
        return {atom: 1.0}, reconstruct({atom: 1.0}, dimension, radius)
    factor = 1-alpha if kind == "FW" else 1+alpha
    updated = {key: factor*value for key, value in weights.items()}
    updated[atom] = updated.get(atom, 0.0)+(alpha if kind == "FW" else -alpha)
    if kind == "away" and alpha == cap:
        del updated[atom]  # exact combinatorial drop, no positive pruning threshold
    if min(updated.values(), default=0.0) < -1e-13:
        raise ArithmeticError("Negative active weight")
    updated = {key: value for key, value in updated.items() if value > 0}
    total = math.fsum(updated.values())
    if total <= 0:
        raise ArithmeticError("Empty active representation")
    updated = {key: value/total for key, value in updated.items()}
    return updated, reconstruct(updated, dimension, radius)


def solve_away(problem, step="BT", max_iter=1000, tolerance=1e-5, initial_estimate=1.0):
    if step not in ("BT", "Q") or max_iter < 0 or tolerance < 0:
        raise ValueError("Invalid AFW parameters")
    if not math.isfinite(initial_estimate) or initial_estimate <= 0:
        raise ValueError("Initial curvature estimate must be finite and positive")
    if step == "Q" and any(not math.isfinite(value) or value < 0 for value in (problem.h0, problem.h1)):
        raise ValueError("Known-H steps require finite nonnegative curvature constants")
    weights = decompose(problem.x0, problem.radius)
    initial_size = len(weights)
    x = reconstruct(weights, problem.x0.size, problem.radius)
    counts = {"function_evals": 0, "gradient_evals": 0, "lmo_evals": 0,
              "away_scans": 0, "active_atom_scores": 0}
    def objective(point):
        counts["function_evals"] += 1
        return problem.objective(point)
    def gradient(point):
        counts["gradient_evals"] += 1
        return problem.gradient(point)
    started = perf_counter()
    fx, g = objective(x), gradient(x)
    lower, estimate = -math.inf, initial_estimate
    fw_steps = away_steps = drops = numerical_prunes = 0
    trace = []
    for k in range(max_iter+1):
        s = lmo(g, problem.radius)
        counts["lmo_evals"] += 1
        fw_direction = s-x
        raw_gap = float(-g @ fw_direction)
        if raw_gap < -1e-10*max(1.0, abs(fx)):
            raise ArithmeticError("Negative FW certificate gap")
        gap = max(0.0, raw_gap)
        lower = max(lower, fx-gap)
        upper = max(0.0, fx-lower)
        trace.append({"iteration": k, "objective": fx, "fw_gap": gap,
                      "lower_bound": lower, "certificate": upper,
                      "active_size": len(weights), "initial_active_size": initial_size,
                      "minimum_weight": min(weights.values()),
                      "weight_sum": math.fsum(weights.values()),
                      "representation_error": float(np.linalg.norm(x-reconstruct(weights, x.size, problem.radius))),
                      "l1_norm": float(np.linalg.norm(x, 1)),
                      "support": int(np.count_nonzero(np.abs(x) > 1e-10)),
                      "fw_steps": fw_steps, "away_steps": away_steps,
                      "drop_steps": drops, "good_steps": k-drops,
                      "numerical_prunes": numerical_prunes,
                      "elapsed_seconds": perf_counter()-started, **counts})
        if upper <= tolerance or k == max_iter:
            break
        counts["away_scans"] += 1
        counts["active_atom_scores"] += len(weights)
        def atom_product(atom):
            return problem.radius*g[atom//2]*(1 if atom % 2 == 0 else -1)
        away_atom = max(sorted(weights), key=atom_product)
        v = np.zeros_like(x)
        v[away_atom//2] = problem.radius*(1 if away_atom % 2 == 0 else -1)
        away_gap = float(g @ (v-x))
        if gap >= away_gap or len(weights) == 1:
            kind, direction, directional_gap, cap = "FW", fw_direction, gap, 1.0
            j = int(np.flatnonzero(s)[0])
            atom = 2*j+(s[j] < 0)
        else:
            kind, direction, directional_gap = "away", x-v, away_gap
            atom = away_atom
            cap = weights[atom]/(1-weights[atom])
        q2 = float(direction @ direction)
        if q2 == 0 or directional_gap <= 0:
            raise ArithmeticError("Nonterminal AFW has no descent direction")
        curvature = estimate/2 if step == "BT" else problem.h0+problem.h1*upper
        for trial in range(1024):
            alpha = min(cap, directional_gap/(curvature*q2)) if curvature > 0 else cap
            next_weights, trial_x = trial_state(weights, kind, atom, alpha, cap, x.size, problem.radius)
            trial_f = objective(trial_x)
            if step == "Q" or trial_f <= fx-alpha*directional_gap+0.5*curvature*alpha**2*q2:
                break
            curvature *= 2
            if not math.isfinite(curvature) or curvature <= 0:
                raise FloatingPointError("Unrepresentable AFW curvature")
        else:
            raise RuntimeError("AFW backtracking failed")
        drop = kind == "away" and alpha == cap
        unexpected = len(set(weights)-set(next_weights)) if not (drop or (kind == "FW" and alpha == 1)) else 0
        numerical_prunes += unexpected
        trace[-1].update(direction=kind, directional_gap=directional_gap, away_gap=away_gap,
                         direction_norm_sq=q2, step=alpha, step_cap=cap, drop_step=drop,
                         trial_curvature=curvature, line_search_trials=trial+1,
                         affine_roundoff=float(np.linalg.norm(trial_x-(x+alpha*direction))))
        fw_steps += kind == "FW"
        away_steps += kind == "away"
        drops += drop
        weights, x, fx = next_weights, trial_x, trial_f
        if step == "BT":
            estimate = curvature
        g = gradient(x)
        if not math.isfinite(fx) or not np.all(np.isfinite(g)):
            raise FloatingPointError("Nonfinite AFW oracle output")
    return AwayResult(x, trace, perf_counter()-started, 0, weights)
