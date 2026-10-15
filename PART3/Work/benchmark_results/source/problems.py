"""Benchmark problems with analytically valid curvature bounds on an l1 ball."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Problem:
    name: str
    seed: int
    radius: float
    x0: np.ndarray
    objective: Callable
    gradient: Callable
    hessian: Callable
    h0: float
    h1: float
    global_l: float
    known_lower: float | None
    reference_lower: float
    reference_upper: float
    reference_x: np.ndarray
    metadata: dict


SCENARIOS = (
    "interior", "boundary", "active", "active_scaled", "active_highdim",
    "breast_cancer",
)


def lmo(gradient: np.ndarray, radius: float) -> np.ndarray:
    vertex = np.zeros_like(gradient)
    j = int(np.argmax(np.abs(gradient)))
    vertex[j] = -radius if gradient[j] >= 0 else radius
    return vertex


def make_cosh(seed: int, scenario: str) -> Problem:
    """The active cases have a strictly positive, planted KKT multiplier.

    On Q, -multiplier * normal @ (x - optimum) >= 0. Consequently the
    usual cosh curvature bound also holds relative to the constrained minimum.
    The normal and optimum are construction data, never solver inputs.
    """
    if scenario not in SCENARIOS[:-1]:
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    dim, rows = (60, 40) if scenario == "active_highdim" else (12, 10)
    scale = 0.9 if scenario == "active_scaled" else 0.22
    a = rng.normal(0, scale, size=(rows, dim))
    radius, mu = 1.5, 0.08
    optimum = np.zeros(dim)
    optimum[[1, 5, 9]] = [0.3, -0.25, 0.2] if scenario == "interior" else [0.6, -0.5, 0.4]
    multiplier = 0.15 if scenario.startswith("active") else 0.0
    normal = rng.uniform(-0.6, 0.6, dim)
    normal[[1, 5, 9]] = np.sign(optimum[[1, 5, 9]])
    x0 = np.zeros(dim)
    x0[0] = radius
    h0 = float(mu + np.linalg.eigvalsh(a.T @ a / rows)[-1])
    h1 = float(np.max(np.sum(a * a, axis=1)))
    z_bound = radius * np.max(np.abs(a), axis=1) + np.abs(a @ optimum)
    global_l = float(mu + np.linalg.eigvalsh(a.T @ (np.cosh(z_bound)[:, None] * a) / rows)[-1])

    def objective(x):
        diff = x - optimum
        # 2*sinh(z/2)^2 avoids cancellation in cosh(z)-1 near the optimum.
        return float(np.mean(2 * np.sinh((a @ diff) / 2) ** 2)
                     + 0.5 * mu * np.dot(diff, diff) - multiplier * np.dot(normal, diff))

    def gradient(x):
        return a.T @ np.sinh(a @ (x - optimum)) / rows + mu * (x - optimum) - multiplier * normal

    def hessian(x):
        return a.T @ (np.cosh(a @ (x - optimum))[:, None] * a) / rows + mu * np.eye(dim)

    return Problem(scenario, seed, radius, x0, objective, gradient, hessian,
                   h0, h1, global_l, 0.0, 0.0, 0.0, optimum,
                   {"dimension": dim, "rows": rows, "scale": scale, "mu": mu,
                    "kkt_multiplier": multiplier,
                    "reference_gradient_norm": float(np.linalg.norm(gradient(optimum)))})


def make_breast_cancer(seed: int) -> Problem:
    """Real-data, globally smooth control; H1=0, not evidence for H1 gains.

    Uses sklearn's bundled Wisconsin Diagnostic Breast Cancer dataset.
    Only the training partition enters the optimization problem. The reference
    interval is computed independently and is unavailable to the FW solver.
    """
    from scipy.optimize import minimize
    from scipy.special import expit
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    train_x, _, train_y, _ = train_test_split(
        data.data, data.target, test_size=0.2, stratify=data.target, random_state=seed,
    )
    a = StandardScaler().fit_transform(train_x)
    y = train_y.astype(float)
    rows, dim = a.shape
    radius, mu = 2.0, 0.01
    global_l = float(0.25 * np.linalg.eigvalsh(a.T @ a / rows)[-1] + mu)

    def objective(x):
        z = a @ x
        return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * mu * np.dot(x, x))

    def gradient(x):
        return a.T @ (expit(a @ x) - y) / rows + mu * x

    def hessian(x):
        p = expit(a @ x)
        return a.T @ ((p * (1 - p))[:, None] * a) / rows + mu * np.eye(dim)

    # x = positive - negative, with sum(positive + negative) <= radius.
    def split_gradient(z):
        g = gradient(z[:dim] - z[dim:])
        return np.concatenate((g, -g))

    result = minimize(
        lambda z: objective(z[:dim] - z[dim:]), np.zeros(2 * dim),
        jac=split_gradient, method="SLSQP", bounds=[(0, None)] * (2 * dim),
        constraints={"type": "ineq", "fun": lambda z: radius - z.sum(),
                     "jac": lambda z: -np.ones_like(z)},
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    reference_x = result.x[:dim] - result.x[dim:]
    # Remove any numerical feasibility violation before using a primal upper bound.
    reference_x *= min(1.0, radius / max(np.linalg.norm(reference_x, 1), radius))
    reference_upper = objective(reference_x)
    g = gradient(reference_x)
    reference_gap = max(0.0, float(np.dot(g, reference_x - lmo(g, radius))))
    if not result.success or reference_gap > 2e-6:
        raise RuntimeError(f"Unreliable logistic reference: {result.message}; gap={reference_gap}")
    x0 = np.zeros(dim)
    x0[0] = radius
    return Problem("breast_cancer", seed, radius, x0, objective, gradient, hessian,
                   global_l, 0.0, global_l, 0.0,
                   reference_upper - reference_gap, reference_upper, reference_x,
                   {"dimension": dim, "rows": rows, "mu": mu,
                    "reference_solver": "SLSQP, split variables",
                    "reference_gap": reference_gap,
                    "data": "sklearn.load_breast_cancer; stratified 80% train; train-only scaling"})


def make_problem(seed: int, scenario: str) -> Problem:
    return make_breast_cancer(seed) if scenario == "breast_cancer" else make_cosh(seed, scenario)
