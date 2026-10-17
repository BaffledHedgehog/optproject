"""Finite checks of the repaired research-note constructions, not a formal proof.

python PART3/Work/lower_bound_audit.py --output PART3/Work/lower_bound_audit_results
"""

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def chain_matrix(m):
    return 2 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)


def chain_minimizer(m):
    return 1 - np.arange(1, m + 1) / (m + 1)


def exact_chain_floor(m, n):
    if not 0 <= n < m:
        raise ValueError("Need 0 <= n < m")
    return Fraction(m - n, 8 * (m + 1) * (n + 1))


def rational_chain_minimum(n):
    """Independent Fraction Gaussian solve of T_n y = e_1 and evaluation."""
    if n == 0:
        return Fraction(0)
    matrix = [[Fraction(2 if i == j else -1 if abs(i-j) == 1 else 0)
               for j in range(n)] + [Fraction(i == 0)] for i in range(n)]
    for k in range(n):
        pivot = matrix[k][k]
        matrix[k] = [value / pivot for value in matrix[k]]
        for i in range(n):
            if i == k:
                continue
            factor = matrix[i][k]
            matrix[i] = [a - factor*b for a, b in zip(matrix[i], matrix[k])]
    y = [row[-1] for row in matrix]
    quadratic = sum(v*v for v in y) - sum(a*b for a, b in zip(y, y[1:]))
    return (quadratic-y[0]) / 4


@dataclass
class LiftedChain:
    basis: np.ndarray
    h0: float
    radius: float

    @property
    def m(self):
        return self.basis.shape[1]

    @property
    def beta(self):
        return float(np.linalg.norm(chain_minimizer(self.m))) / self.radius

    @property
    def alpha(self):
        return self.h0 / self.beta**2

    @property
    def f0(self):
        return self.alpha * self.m / (8 * (self.m + 1))

    @property
    def optimum(self):
        return self.basis @ chain_minimizer(self.m) / self.beta

    @property
    def hessian(self):
        projector = self.basis @ self.basis.T
        return (self.h0/4 * self.basis @ chain_matrix(self.m) @ self.basis.T
                + self.h0/2 * (np.eye(len(projector)) - projector))

    def value_gradient(self, x):
        y = self.beta * (self.basis.T @ x)
        difference = y - chain_minimizer(self.m)
        orthogonal = x - self.basis @ (self.basis.T @ x)
        matrix = chain_matrix(self.m)
        value = self.alpha/8 * float(difference @ matrix @ difference)
        value += self.h0/4 * float(orthogonal @ orthogonal)
        gradient = self.alpha*self.beta/4 * self.basis @ (matrix @ difference)
        gradient += self.h0/2 * orthogonal
        return value, gradient


class ResistingOracle:
    """An executable version of the online answers in M2, including P_perp."""

    def __init__(self, budget, h0=1.0, radius=1.0, seed=0):
        if budget < 0 or h0 <= 0 or radius <= 0:
            raise ValueError("Invalid construction parameters")
        self.budget = budget
        self.m, self.dimension = 2*budget+3, 3*budget+4
        self.h0, self.radius = h0, radius
        self.beta = float(np.linalg.norm(chain_minimizer(self.m))) / radius
        self.alpha = h0 / self.beta**2
        self.rng = np.random.default_rng(seed)
        self.columns = [np.eye(self.dimension)[0]]
        self.history = []
        self.finished = False

    def _orthogonal_column(self, constraints):
        # Full SVD handles repeated queries and linearly dependent constraints.
        if constraints:
            matrix = np.array(constraints)
            _, singular, right = np.linalg.svd(matrix, full_matrices=True)
            threshold = 1e-11 * max(1.0, float(singular[0]))
            rank = int(np.count_nonzero(singular > threshold))
            nullspace = right[rank:].T
        else:
            nullspace = np.eye(self.dimension)
        if nullspace.shape[1] == 0:
            raise ArithmeticError("No remaining orthogonal direction")
        column = nullspace @ self.rng.normal(size=nullspace.shape[1])
        return column / np.linalg.norm(column)

    def query(self, x):
        if self.finished or len(self.history) >= self.budget:
            raise ValueError("Oracle already completed or budget exhausted")
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError("Invalid query dimension")
        t = len(self.history) + 1
        old_queries = [record[0] for record in self.history]
        self.columns.append(self._orthogonal_column([*old_queries, x, *self.columns]))
        known = np.column_stack(self.columns[:t])
        coordinates = known.T @ x
        y = np.zeros(self.m)
        y[:t] = self.beta * coordinates
        g = chain_matrix(self.m) @ y
        g[0] -= 1
        value = self.alpha * ((float(y @ chain_matrix(self.m) @ y)/2-y[0])/4
                              + self.m/(8*(self.m+1)))
        value += self.h0/4 * (float(x @ x) - float(coordinates @ coordinates))
        gradient = self.alpha*self.beta/4 * np.column_stack(self.columns) @ g[:t+1]
        gradient += self.h0/2 * (x-known @ coordinates)
        self.history.append((x.copy(), value, gradient.copy()))
        return value, gradient

    def finish(self, output):
        if self.finished or len(self.history) != self.budget:
            raise ValueError("Finish requires a complete query history")
        constraints = [record[0] for record in self.history] + [np.asarray(output)]
        while len(self.columns) < self.m:
            self.columns.append(self._orthogonal_column([*constraints, *self.columns]))
        self.finished = True
        return LiftedChain(np.column_stack(self.columns), self.h0, self.radius)


def carrier_value(z, h0, h1):
    if h0 <= 0 or h1 < 0:
        raise ValueError("Carrier needs H0>0 and H1>=0")
    if h1 == 0:
        return h0*z*z/2
    # Avoid subtracting nearly equal cosh and 1 at the origin.
    return 2*h0/h1 * math.sinh(math.sqrt(h1)*z/2)**2


def match_carrier(h0, h1, radius, budget, f0):
    if radius <= 0 or budget < 0:
        raise ValueError("Invalid radius or budget")
    gamma = 3*h0/(4*(4*budget+7))
    low, high = gamma*radius**2, carrier_value(radius, h0, h1)
    if not low <= f0 <= high:
        raise ValueError("Exact initial gap outside this carrier construction")
    if f0 == low:
        z = 0.0
    elif f0 == high:
        z = radius
    else:
        left, right = 0.0, radius
        for _ in range(100):
            z = (left+right)/2
            if gamma*(radius**2-z**2)+carrier_value(z, h0, h1) < f0:
                left = z
            else:
                right = z
        z = (left+right)/2
    remaining = max(0.0, (radius-z)*(radius+z))
    return {"z": z, "chain_radius_sq": remaining,
            "matched_f0": gamma*remaining+carrier_value(z, h0, h1),
            "floor": 3*h0*remaining/(4*(2*budget+3)*(4*budget+7))}


def bisect_observable(objective, derivative, x0, radius, tolerance, max_queries=1000):
    if radius < 0 or tolerance <= 0:
        raise ValueError("Invalid bracket or accuracy")
    if radius == 0:
        return x0, []
    left, right = x0-radius, x0+radius
    history = []
    for _ in range(max_queries):
        x = (left+right)/2
        value, slope = objective(x), derivative(x)
        certificate = abs(slope)*(right-left)/2
        history.append({"x": x, "value": value, "slope": slope,
                        "left": left, "right": right, "certificate": certificate})
        if certificate <= tolerance:
            return x, history
        if slope > 0:
            right = x
        else:
            left = x
    raise RuntimeError("Diagnostic query cap reached; this is not a uniform class bound")


def run_audit():
    rational = 0
    for m in range(1, 14):
        full = rational_chain_minimum(m)
        for n in range(m):
            assert rational_chain_minimum(n)-full == exact_chain_floor(m, n)
            rational += 1
    runs = []
    for budget in (0, 1, 3, 6):
        for seed in range(9):
            oracle = ResistingOracle(budget, h0=1.3, radius=1.7, seed=seed)
            rng = np.random.default_rng(1000+seed)
            direction = rng.normal(size=oracle.dimension)
            last_gradient = np.zeros(oracle.dimension)
            for t in range(budget):
                # Repeated, dense and gradient-adaptive directions exercise
                # the complement term which was missing in the old proof.
                query = direction if seed % 3 == 0 else (
                    rng.normal(size=oracle.dimension)-.2*last_gradient)
                _, last_gradient = oracle.query(query)
            output = rng.normal(size=oracle.dimension)-.3*last_gradient
            problem = oracle.finish(output)
            errors = []
            for query, value, gradient in oracle.history:
                true_value, true_gradient = problem.value_gradient(query)
                errors.append(max(abs(true_value-value), np.linalg.norm(true_gradient-gradient)))
            error = max(errors, default=0.0)
            spectrum = np.linalg.eigvalsh(problem.hessian)
            floor = 3*problem.h0*problem.radius**2/(4*(2*budget+3)*(4*budget+7))
            value = problem.value_gradient(output)[0]
            assert error < 2e-10 and spectrum[0] > 0 and spectrum[-1] <= problem.h0+1e-10
            assert value >= floor-1e-10
            assert abs(np.linalg.norm(problem.optimum)-problem.radius) < 1e-10
            runs.append({"N": budget, "seed": seed, "dimension": oracle.dimension,
                         "consistency_error": float(error), "min_eigenvalue": float(spectrum[0]),
                         "output_gap": value, "guaranteed_floor": floor})
    carriers = []
    for budget in (0, 3, 8):
        for h1 in (0.0, .2, 2.0):
            for rho in (.25, .7, .9):
                h0, radius = 1.0, 2.0
                gamma = 3*h0/(4*(4*budget+7))
                f0 = gamma*(1-rho*rho)*radius**2+carrier_value(rho*radius, h0, h1)
                record = match_carrier(h0, h1, radius, budget, f0)
                assert record["chain_radius_sq"] >= (1-rho*rho)*radius**2-1e-10
                assert abs(record["matched_f0"]-f0) < 1e-10
                carriers.append({"N": budget, "H1": h1, "rho": rho, **record})
    return {"scope": "Finite internal checks of the written proofs; no universal H1 lower bound or bibliographic priority",
            "rational_schur_checks": rational, "dense_oracle_runs": runs,
            "carrier_cases": carriers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit()
    args.output.mkdir(parents=True, exist_ok=True)
    notes = Path(__file__).resolve().parents[1]/"some md research"
    inputs = [Path(__file__).resolve(), Path(__file__).with_name("test_lower_bound_audit.py"),
              *sorted(notes.glob("*.md"))]
    report["input_sha256"] = {str(p.relative_to(notes.parent)).replace("\\", "/"):
                              hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (args.output/"checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8", newline="\n")
    print(f"Lower-bound audit: {report['rational_schur_checks']} exact Schur checks, "
          f"{len(report['dense_oracle_runs'])} dense oracle histories, {len(report['carrier_cases'])} carrier cases")


if __name__ == "__main__":
    main()
