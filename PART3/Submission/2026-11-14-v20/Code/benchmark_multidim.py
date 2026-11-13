"""Reproducible v20 Poisson/ball comparison; no optimizer receives f* or x*.

The protocol is written before any solver runs. Historical v19 controls are
kept in benchmark_gap.py. Timing repetitions are dependent measurements,
not additional instances. Every objective/gradient/LMO call is instrumented.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

METHODS = ("gap_log", "global_L", "open_loop", "armijo", "backtracking",
           "norm_smooth", "line_bisection", "auto_conditioned")
LABELS = ("Gap-log", "Global-L", "Open-loop", "Armijo", "Backtracking",
          "Norm-based", "Line bisection", "AC-FW")
PROTOCOL = {
    "version": "v20", "dimensions": [4, 12], "rows_per_dimension": 12,
    "seeds": [11, 29, 47], "radii": [0.5, 2.0, 4.0],
    "starts": ["zero", "plus_axis", "minus_axis"],
    "epsilon": 1e-4, "max_attempts": 20000, "timing_repetitions": 3,
    "methods": list(METHODS), "latent_norm": 1.2,
    "armijo_beta": 0.5, "armijo_sigma": 0.5,
    "backtracking_initial_M": 1.0, "backtracking_eta": 0.5,
    "backtracking_tau": 2.0, "line_bisection_width": 1e-12,
    "ac_delta": 1.0, "gamma": 1.0,
    "note": "Synthetic data; three starts/radii share each generated dataset. No parameter tuning.",
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class PoissonBall:
    A: np.ndarray
    y: np.ndarray
    radius: float

    def __post_init__(self):
        self.m, self.n = self.A.shape
        self.B = float(np.max(np.linalg.norm(self.A, axis=1)))
        self.L0 = self.B**2 * (1 + float(np.mean(self.y**2)) / 2)
        self.L1 = self.B**2
        self.diameter = 2 * self.radius
        self.norm_L0, self.norm_L1 = self.L0, self.L1 * self.radius
        self.global_L = math.exp(self.B * self.radius) * float(np.linalg.norm(self.A, 2))**2 / self.m

    def objective(self, x):
        t = self.A @ x
        return float(np.mean(np.exp(t) - self.y * t))

    def gradient(self, x):
        return self.A.T @ (np.exp(self.A @ x) - self.y) / self.m

    def hessian(self, x):
        return self.A.T @ (np.exp(self.A @ x)[:, None] * self.A) / self.m

    def lmo(self, g):
        length = float(np.linalg.norm(g))
        return -self.radius * g / length if length else np.zeros_like(g)


class Oracles:
    def __init__(self, problem):
        self.problem = problem
        self.counts = {"f_calls": 0, "g_calls": 0, "lmo_calls": 0}

    def f(self, x):
        self.counts["f_calls"] += 1
        return self.problem.objective(x)

    def g(self, x):
        self.counts["g_calls"] += 1
        return self.problem.gradient(x)

    def lmo(self, g):
        self.counts["lmo_calls"] += 1
        return self.problem.lmo(g)


def logarithmic_step(gap, length, L0, L1, diameter):
    if gap <= 0:
        return 0.0
    if L1 == 0:
        return min(1.0, gap / (L0 * length**2)) if L0 else 1.0
    return min(1.0, math.log1p(L1 * diameter * gap / ((L0 + L1 * gap) * length))
               / (L1 * diameter * length))


def solve(problem, start, method, epsilon=1e-4, max_attempts=20000, record=True, oracle=None):
    """Only data, bounds, the starting point and tolerance enter this solver."""
    oracle = oracle or Oracles(problem)
    tic = perf_counter()
    x = np.array(start, dtype=float, copy=True)
    f, g = oracle.f(x), oracle.g(x)
    s = oracle.lmo(g)
    lower = -math.inf
    updates = rejections = attempts = 0
    M = 1.0
    trace = []
    while True:
        d = s - x
        q2 = float(d @ d)
        raw_gap = float(-g @ d)
        if raw_gap < -1e-11:
            raise ArithmeticError("Negative exact-oracle gap")
        gap = max(0.0, raw_gap)
        lower = max(lower, f - gap)
        cert = f - lower
        row = {"attempt": attempts, "updates": updates, "rejections": rejections,
               "objective": f, "gap": gap, "lower": lower, "certificate": cert,
               "norm": float(np.linalg.norm(x)), "direction_norm": math.sqrt(q2),
               "gradient_norm": float(np.linalg.norm(g)), **oracle.counts,
               "alpha": None, "trials": 0, "accepted": None, "initialization": False,
               "trial_objective": None, "curvature_before": M,
               "curvature_after": None}
        if record:
            row["x"] = x.copy()
            trace.append(row)
        if cert <= epsilon or attempts >= max_attempts:
            break
        if gap <= 0 or q2 <= 0:
            raise ArithmeticError("Nonterminal zero gap/direction")
        alpha = 1.0
        trial_f = trial_g = None
        accepted = True
        initialization = method == "auto_conditioned" and attempts == 0
        trials = 0
        if initialization:
            # A can be the whole closed ball, so the zero start is permitted.
            # This compulsory endpoint move may increase f and is fully charged.
            trial_f = oracle.f(s)
            trials = 1
            M = max(2 * abs(trial_f - f + gap) / q2, np.finfo(float).tiny)
        elif method == "gap_log":
            alpha = logarithmic_step(gap, math.sqrt(q2), problem.L0, problem.L1, problem.diameter)
        elif method == "global_L":
            alpha = min(1.0, gap / (problem.global_L * q2))
        elif method == "open_loop":
            alpha = 2 / (updates + 2)
        elif method == "norm_smooth":
            coefficient = problem.norm_L0 + problem.norm_L1 * float(np.linalg.norm(g))
            alpha = min(1.0, gap / (math.e * coefficient * q2))
        elif method == "line_bisection":
            trial_g = oracle.g(s)
            if float(trial_g @ d) > 0:
                lo, hi = 0.0, 1.0
                while hi - lo > 1e-12:
                    mid = (lo + hi) / 2
                    gm = oracle.g((1 - mid) * x + mid * s)
                    if float(gm @ d) > 0:
                        hi = mid
                    else:
                        lo = mid
                alpha = (lo + hi) / 2
                trial_g = oracle.g((1 - alpha) * x + alpha * s)
        elif method in ("armijo", "backtracking"):
            if method == "backtracking":
                M *= 0.5
            for _ in range(100):
                if method == "backtracking":
                    alpha = min(1.0, gap / (M * q2))
                trial_f = oracle.f((1 - alpha) * x + alpha * s)
                trials += 1
                threshold = (f - 0.5 * alpha * gap if method == "armijo"
                             else f - alpha * gap + 0.5 * M * alpha**2 * q2)
                if trial_f <= threshold:
                    break
                if method == "armijo":
                    alpha *= 0.5
                else:
                    M *= 2
            else:
                raise ArithmeticError("Line search did not terminate")
        elif method == "auto_conditioned":
            alpha = min(1.0, gap / (M * q2))
            trial_f = oracle.f((1 - alpha) * x + alpha * s)
            trials = 1
            k = attempts - 1
            damping = 1 - 1 / ((k + 1) * math.log(k + 3)**2)
            M = max(2 * abs(trial_f - f + alpha * gap) / (alpha**2 * q2), damping * M)
            accepted = trial_f < f
        else:
            raise ValueError(method)
        trial = (1 - alpha) * x + alpha * s
        if trial_f is None:
            trial_f = oracle.f(trial)
            trials = 1
        row.update(alpha=alpha, trials=trials, accepted=accepted,
                   initialization=initialization, trial_objective=trial_f, curvature_after=M)
        attempts += 1
        if accepted:
            x, f = trial, trial_f
            g = trial_g if trial_g is not None else oracle.g(x)
            s = oracle.lmo(g)
            updates += 1
        else:
            rejections += 1
    elapsed = perf_counter() - tic
    result = {"method": method, "status": "converged" if cert <= epsilon else "budget_exhausted",
              "attempts": attempts, "updates": updates, "rejections": rejections,
              "objective": f, "gap": gap, "certificate": cert, "lower": lower,
              "seconds": elapsed, **oracle.counts}
    return result, trace, x


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(12 * n, n))
    A /= np.linalg.norm(A, axis=1)[:, None]
    beta = rng.normal(size=n)
    beta *= 1.2 / np.linalg.norm(beta)
    y = rng.poisson(np.exp(A @ beta)).astype(float)
    return A, y


def reference_interval(problem):
    """Post-hoc diagnostic; never passed to solve()."""
    def constraint(x):
        return problem.radius**2 - float(x @ x)
    result = minimize(problem.objective, np.zeros(problem.n), jac=problem.gradient,
                      method="SLSQP", constraints={"type": "ineq", "fun": constraint,
                      "jac": lambda x: -2 * x}, options={"ftol": 1e-14, "maxiter": 3000})
    x = result.x
    norm = float(np.linalg.norm(x))
    if norm > problem.radius:
        x = x * (problem.radius / norm)
    # Report the actual first-order residual instead of trusting solver status.
    g = problem.gradient(x)
    gap = max(0.0, float(g @ (x - problem.lmo(g))))
    upper = problem.objective(x)
    return {"upper": upper, "lower": upper - gap, "gap": gap,
            "norm": float(np.linalg.norm(x)), "gradient_norm": float(np.linalg.norm(g)),
            "slsqp_success": bool(result.success), "slsqp_message": str(result.message),
            "active": bool(np.linalg.norm(x) >= problem.radius * (1 - 1e-6))}


def audit_trace(problem, method, result, trace, reference, epsilon):
    max_descent_violation = max_model_violation = 0.0
    for i, row in enumerate(trace):
        assert row["norm"] <= problem.radius + 1e-10
        assert row["lower"] <= reference["upper"] + 1e-9
        assert row["objective"] >= reference["lower"] - 1e-9
        if i == len(trace) - 1:
            continue
        nxt = trace[i + 1]
        assert nxt["lower"] >= row["lower"] - 1e-12
        if method != "open_loop" and not row["initialization"]:
            assert nxt["certificate"] <= row["certificate"] + 1e-10
        if method in ("gap_log", "global_L", "norm_smooth", "armijo", "backtracking"):
            residual = nxt["objective"] - row["objective"] + row["alpha"] * row["gap"] / 2
            max_descent_violation = max(max_descent_violation, residual)
            assert residual <= 1e-10, (method, residual)
        if method in ("gap_log", "armijo"):
            bound = min(1.0, math.log(2) * row["gap"] /
                        ((problem.L0 + problem.L1 * row["gap"]) * problem.diameter**2))
            factor = 0.5 if method == "armijo" else 1.0
            assert row["alpha"] + 1e-12 >= factor * bound
            if method == "armijo":
                logalpha = logarithmic_step(row["gap"], row["direction_norm"], problem.L0,
                                           problem.L1, problem.diameter)
                assert row["alpha"] + 1e-12 >= 0.5 * logalpha
    # Spectral Hessian checks at up to 25 deterministic points per trajectory.
    for i in np.unique(np.linspace(0, len(trace) - 1, min(25, len(trace)), dtype=int)):
        row = trace[i]
        hnorm = float(np.linalg.eigvalsh(problem.hessian(row["x"]))[-1])
        violation = hnorm - problem.L0 - problem.L1 * row["gap"]
        max_model_violation = max(max_model_violation, violation)
        assert violation <= 1e-10
        assert hnorm <= problem.norm_L0 + problem.norm_L1 * row["gradient_norm"] + 1e-10
        assert hnorm <= problem.global_L + 1e-10
    if method in ("gap_log", "global_L", "open_loop", "norm_smooth"):
        assert result["f_calls"] == result["g_calls"] == result["lmo_calls"] == result["updates"] + 1
    else:
        assert result["lmo_calls"] == result["updates"] + 1
        assert result["f_calls"] == 1 + sum(r["trials"] for r in trace)
    if method in ("gap_log", "armijo") and trace[0]["gap"] > epsilon:
        J = math.ceil(math.log2(trace[0]["gap"] / epsilon))
        bound = max(4, 4 * problem.L1 * problem.diameter**2 / math.log(2)) * J
        bound += 16 * problem.L0 * problem.diameter**2 / (math.log(2) * epsilon)
        if method == "armijo":
            bound /= 0.5
        assert result["updates"] <= math.ceil(bound)
    return {"points": len(trace), "max_descent_violation": max_descent_violation,
            "max_sampled_model_violation": max_model_violation, "passed": True}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "Evidence/multidim")
    parser.add_argument("--smoke", action="store_true", help="Small implementation check, not publication evidence")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL)
    if args.smoke:
        protocol.update(dimensions=[4], seeds=[11], radii=[0.5], starts=["zero"], timing_repetitions=1)
    dump(output / "protocol.json", protocol)
    dump(output / "environment.json", {"python": sys.version, "platform": platform.platform(),
         "numpy": np.__version__, "blas_threads": {k: os.environ.get(k) for k in
         ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")},
         "timing_note": "Sequential runs; median of untraced repetitions; no validation/serialization in timed region."})
    summaries, audits, references, metadata = [], [], {}, []
    trace_path = output / "traces.csv.gz"
    with gzip.open(trace_path, "wt", newline="", encoding="utf-8") as handle:
        writer = None
        for n in protocol["dimensions"]:
            for seed in protocol["seeds"]:
                A, y = make_data(n, seed)
                for radius in protocol["radii"]:
                    problem = PoissonBall(A, y, radius)
                    pid = f"poisson_n{n}_s{seed}_R{radius:g}"
                    dump(output / "data" / (pid + ".json"), {"A": A.tolist(), "y": y.tolist(), "radius": radius})
                    reference = reference_interval(problem)
                    references[pid] = reference
                    metadata.append({"problem": pid, "n": n, "m": problem.m, "radius": radius,
                                     "seed": seed, "B": problem.B, "L0": problem.L0, "L1": problem.L1,
                                     "global_L_bound": problem.global_L, "norm_L0": problem.norm_L0,
                                     "norm_L1": problem.norm_L1, "reference_active": reference["active"]})
                    for start_name in protocol["starts"]:
                        start = np.zeros(n)
                        if start_name != "zero":
                            start[0] = radius if start_name == "plus_axis" else -radius
                        for method in METHODS:
                            result, trace, end = solve(problem, start, method, protocol["epsilon"], protocol["max_attempts"])
                            audit = audit_trace(problem, method, result, trace, reference, protocol["epsilon"])
                            times = []
                            for _ in range(protocol["timing_repetitions"]):
                                rerun, _, other_end = solve(problem, start, method, protocol["epsilon"], protocol["max_attempts"], record=False)
                                assert all(rerun[k] == result[k] for k in result if k != "seconds")
                                assert np.array_equal(end, other_end)
                                times.append(rerun["seconds"])
                            result["seconds"] = float(np.median(times))
                            result.update(problem=pid, n=n, radius=radius, seed=seed, start=start_name,
                                          reference_active=reference["active"],
                                          error_lower=max(0.0, result["objective"] - reference["upper"]),
                                          error_upper=max(0.0, result["objective"] - reference["lower"]))
                            summaries.append(result)
                            audits.append({"problem": pid, "start": start_name, "method": method, **audit})
                            for row in trace:
                                record = {"problem": pid, "start": start_name, "method": method, **row}
                                record["x"] = json.dumps(row["x"].tolist(), separators=(",", ":"))
                                if writer is None:
                                    writer = csv.DictWriter(handle, fieldnames=list(record)); writer.writeheader()
                                writer.writerow(record)
                            dump(output / "summary.json", summaries)
                    print(pid, "completed; reference gap", f"{reference['gap']:.2g}", flush=True)
    write_csv(output / "summary.csv", summaries)
    dump(output / "audit.json", audits)
    dump(output / "references.json", references)
    dump(output / "problems.json", metadata)
    dump(output / "data_hashes.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((output / "data").glob("*.json"))})
    print(json.dumps({"runs": len(summaries), "converged": sum(r["status"] == "converged" for r in summaries),
                      "audited_points": sum(a["points"] for a in audits)}, indent=2))


if __name__ == "__main__":
    main()
