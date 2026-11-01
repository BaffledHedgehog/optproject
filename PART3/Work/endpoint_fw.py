"""Endpoint-probe Frank--Wolfe replay for the recovered Appendix G protocol.

Only deterministic objective values queried at signed l1-ball vertices are
cached. Gradients, LMOs, and all nonvertex function values are always paid.
The solver receives no reference optimum/value, Hessian, or curvature bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np

from problems import Problem, lmo


PROBES = ("ordinary", "dense", "dyadic")


class VertexObjectiveCache:
    def __init__(self, problem: Problem, enabled: bool = True):
        self.problem = problem
        self.enabled = enabled
        self.values: dict[bytes, float] = {}
        self.requests = 0
        self.paid = 0
        self.hits = 0

    def _key(self, x: np.ndarray):
        if not self.enabled:
            return None
        x = np.asarray(x, dtype=np.float64)
        nz = np.flatnonzero(x)
        if len(nz) != 1 or abs(float(x[nz[0]])) != self.problem.radius:
            return None
        # Canonical exact coordinate/sign identity; robust to signed zero elsewhere.
        return f"{x.size}:{int(nz[0])}:{1 if x[nz[0]] > 0 else -1}".encode("ascii")

    def __call__(self, x: np.ndarray) -> float:
        self.requests += 1
        key = self._key(x)
        if key is not None and key in self.values:
            self.hits += 1
            return self.values[key]
        value = float(self.problem.objective(x))
        self.paid += 1
        if key is not None:
            self.values[key] = value
        return value


@dataclass
class EndpointResult:
    x: np.ndarray
    trace: list[dict]
    elapsed_seconds: float
    cache: VertexObjectiveCache


def endpoint_due(probe: str, iteration: int) -> bool:
    if probe == "ordinary":
        return False
    if probe == "dense":
        return True
    if probe == "dyadic":
        return (iteration + 1) & iteration == 0
    raise ValueError(probe)


def solve_endpoint(problem: Problem, probe: str = "ordinary", max_iter: int = 5000,
                   tolerance: float = 1e-5, objective_budget: int = 1000,
                   endpoint_sigma: float = 0.5) -> EndpointResult:
    if probe not in PROBES:
        raise ValueError(probe)
    if max_iter < 0 or tolerance < 0 or objective_budget < 1 or not 0 < endpoint_sigma < 1:
        raise ValueError("Invalid iteration, accuracy, budget, or Armijo coefficient")
    cache = VertexObjectiveCache(problem, enabled=True)
    counts = {"gradient_evals": 0, "lmo_evals": 0}

    def grad(x):
        counts["gradient_evals"] += 1
        return np.asarray(problem.gradient(x), dtype=float)

    started = perf_counter()
    x = np.asarray(problem.x0, dtype=float).copy()
    fx = cache(x)
    g = grad(x)
    lower = -math.inf
    diameter = 2.0 * problem.radius
    initial_curvature = None
    previous_curvature = None
    trace: list[dict] = []

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
        if initial_curvature is None:
            initial_curvature = upper / (diameter * diameter)
            previous_curvature = initial_curvature
        row = {"iteration": k, "objective": fx, "fw_gap": gap, "lower_bound": lower,
               "certificate": upper, "probe": probe, "probe_due": endpoint_due(probe, k),
               "l1_norm": float(np.linalg.norm(x, 1)), "support": int(np.count_nonzero(np.abs(x) > 1e-10)),
               "endpoint_tested": False, "endpoint_accepted": False, "endpoint_value": math.nan,
               "step": 0.0, "accepted_curvature": math.nan,
               "line_search_trials": 0, "function_requests": cache.requests,
               "function_evals": cache.paid, "cache_hits": cache.hits,
               "gradient_evals": counts["gradient_evals"], "lmo_evals": counts["lmo_evals"],
               "cache_vertices": len(cache.values), "source_paid_evals": cache.paid,
               "source_function_requests": cache.requests, "source_cache_hits": cache.hits}
        trace.append(row)
        if upper <= tolerance or k == max_iter or cache.paid >= objective_budget:
            break

        # Sparse/dense endpoint Armijo probe is separate from the unchanged
        # ordinary quadratic-backtracking fallback.
        if endpoint_due(probe, k):
            row["endpoint_tested"] = True
            fs = cache(s)
            row["endpoint_value"] = fs
            if fs <= fx - endpoint_sigma * gap:
                row["endpoint_accepted"] = True
                row["step"] = 1.0
                row["line_search_trials"] = 1
                x, fx = s.copy(), fs
                g = grad(x)
                row.update(function_requests=cache.requests, function_evals=cache.paid,
                           cache_hits=cache.hits, gradient_evals=counts["gradient_evals"],
                           lmo_evals=counts["lmo_evals"], cache_vertices=len(cache.values))
                continue

        q2 = float(np.dot(d, d))
        if q2 == 0:
            break
        curvature = previous_curvature / 2.0
        trials = 0
        while True:
            trials += 1
            alpha = min(1.0, gap / (curvature * q2)) if curvature > 0 else 1.0
            candidate = x + alpha * d
            fc = cache(candidate)
            if fc <= fx - alpha * gap + 0.5 * curvature * alpha * alpha * q2:
                break
            curvature *= 2.0
            if trials >= 1024 or not math.isfinite(curvature):
                raise RuntimeError("Endpoint FW backtracking failed")
        row["step"] = float(alpha)
        row["accepted_curvature"] = float(curvature)
        row["line_search_trials"] = trials
        x, fx = candidate, fc
        g = grad(x)
        previous_curvature = curvature
        row.update(function_requests=cache.requests, function_evals=cache.paid,
                   cache_hits=cache.hits, gradient_evals=counts["gradient_evals"],
                   lmo_evals=counts["lmo_evals"], cache_vertices=len(cache.values))
    return EndpointResult(x, trace, perf_counter() - started, cache)
