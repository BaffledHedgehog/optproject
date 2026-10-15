"""Reproducible l1-ball Frank--Wolfe experiment for the research note.

Run: python PART3/Work/experiment.py
Only NumPy is required. Outputs are written next to this script.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RADIUS = 1.5
DIM = 12
ROWS = 10
MU = 0.08
MAX_ITER = 600
TOL = 1e-7


def make_problem(seed: int, scenario: str):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.22, size=(ROWS, DIM))
    x_star = np.zeros(DIM)
    if scenario == "interior":
        x_star[[1, 5, 9]] = [0.3, -0.25, 0.2]
    elif scenario == "boundary":
        x_star[[1, 5, 9]] = [0.6, -0.5, 0.4]
    else:
        raise ValueError(scenario)
    x0 = np.zeros(DIM)
    x0[0] = RADIUS

    h0 = MU + np.linalg.eigvalsh(a.T @ a / ROWS)[-1]
    h1 = np.max(np.sum(a * a, axis=1))
    z_bound = RADIUS * np.max(np.abs(a), axis=1) + np.abs(a @ x_star)
    l_global = MU + np.linalg.eigvalsh(
        a.T @ (np.cosh(z_bound)[:, None] * a) / ROWS
    )[-1]

    def objective(x):
        z = a @ (x - x_star)
        return float(np.mean(np.cosh(z) - 1) + 0.5 * MU * np.dot(x - x_star, x - x_star))

    def gradient(x):
        z = a @ (x - x_star)
        return a.T @ np.sinh(z) / ROWS + MU * (x - x_star)

    return objective, gradient, x0, h0, h1, l_global


def h_step(gap: float, upper_gap: float, direction_norm: float, h0: float, h1: float) -> float:
    if gap <= 0 or direction_norm == 0:
        return 0.0
    if h1 == 0:
        return min(1.0, gap / (h0 * direction_norm**2)) if h0 else 1.0
    c = direction_norm * math.sqrt(h1)
    b = upper_gap + h0 / h1
    ratio = gap / (b * c)
    return 1.0 if ratio >= 1 else min(1.0, math.atanh(ratio) / c)


def lmo_l1_ball(grad):
    s = np.zeros_like(grad)
    j = int(np.argmax(np.abs(grad)))
    s[j] = -RADIUS * np.sign(grad[j])
    return s


def run(seed: int, method: str, scenario: str):
    objective, gradient, x, h0, h1, l_global = make_problem(seed, scenario)
    values = []
    evals = 0
    lower_bound = 0.0 if method == "H_nonnegative" else -math.inf
    for k in range(MAX_ITER + 1):
        fx = objective(x)
        evals += 1
        grad = gradient(x)
        s = lmo_l1_ball(grad)
        d = s - x
        gap = float(-np.dot(grad, d))
        lower_bound = max(lower_bound, fx - gap)
        upper_gap = fx - lower_bound
        values.append((k, fx, gap, evals))
        if fx <= TOL or k == MAX_ITER:
            break
        norm = float(np.linalg.norm(d))
        if method in ("H_gap", "H_memory", "H_nonnegative"):
            u = gap if method == "H_gap" else upper_gap
            alpha = h_step(gap, u, norm, h0, h1)
        elif method == "global_L":
            alpha = min(1.0, gap / (l_global * norm**2))
        elif method == "diminishing":
            alpha = 2.0 / (k + 2)
        elif method == "Armijo":
            alpha = 1.0
            while objective(x + alpha * d) > fx - 1e-4 * alpha * gap:
                alpha *= 0.5
                evals += 1
                if alpha < 2**-50:
                    raise RuntimeError("Armijo failed to find a step")
            evals += 1
        else:
            raise ValueError(method)
        x = x + alpha * d
    return values, (h0, h1, l_global)


def main():
    methods = ("H_gap", "H_memory", "H_nonnegative", "global_L", "diminishing", "Armijo")
    rows = []
    summary = []
    for scenario in ("interior", "boundary"):
        for seed in range(5):
            for method in methods:
                trace, constants = run(seed, method, scenario)
                for k, fx, gap, evals in trace:
                    rows.append((scenario, seed, method, k, fx, gap, evals))
                k, fx, gap, evals = trace[-1]
                summary.append((scenario, seed, method, k, fx, gap, evals, *constants))

    with (ROOT / "trace.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("scenario", "seed", "method", "iteration", "f_minus_fstar", "fw_gap", "function_evals"))
        writer.writerows(rows)
    with (ROOT / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("scenario", "seed", "method", "iterations", "final_error", "final_fw_gap", "function_evals", "H0", "H1", "L_global_bound"))
        writer.writerows(summary)
    for row in summary:
        print(f"{row[0]:8} seed={row[1]} {row[2]:13} k={row[3]:3} f={row[4]:.2e} G={row[5]:.2e} evals={row[6]:4}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    plotted = ("H_nonnegative", "global_L", "diminishing", "Armijo")
    labels = {"H_nonnegative": "H-шаг, f ≥ 0", "global_L": "глобальная L", "diminishing": "шаг 2/(k+2)", "Armijo": "Армихо"}
    for ax, scenario in zip(axes, ("interior", "boundary")):
        for method in plotted:
            curves = []
            for seed in range(5):
                vals = np.array([r[4] for r in rows if r[0] == scenario and r[1] == seed and r[2] == method])
                curves.append(np.pad(vals, (0, MAX_ITER + 1 - len(vals)), mode="edge"))
            ax.semilogy(np.arange(MAX_ITER + 1), np.median(curves, axis=0), label=labels[method])
        ax.set_title("Оптимум внутри шара" if scenario == "interior" else "Оптимум на границе")
        ax.set_xlabel("Итерации Франка–Вульфа")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Медиана f(x) − f*")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "convergence.png", dpi=180)


if __name__ == "__main__":
    main()
