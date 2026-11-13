"""Independent numerical checks of the new class, oracles, and comparators."""
import math
import unittest
from unittest.mock import Mock

import numpy as np

from benchmark_multidim import METHODS, PoissonBall, logarithmic_step, make_data, solve


class RevisionChecks(unittest.TestCase):
    def setUp(self):
        A, y = make_data(4, 11)
        self.problem = PoissonBall(A, y, 2.0)

    def test_gradient_and_hessian_against_finite_differences(self):
        p = self.problem
        x = np.array([0.2, -0.3, 0.1, 0.4])
        h = 1e-5
        basis = np.eye(4)
        numerical_g = np.array([(p.objective(x + h * e) - p.objective(x - h * e)) / (2 * h) for e in basis])
        numerical_H = np.column_stack([(p.gradient(x + h * e) - p.gradient(x - h * e)) / (2 * h) for e in basis])
        np.testing.assert_allclose(p.gradient(x), numerical_g, rtol=2e-8, atol=2e-10)
        np.testing.assert_allclose(p.hessian(x), numerical_H, rtol=2e-8, atol=2e-10)

    def test_poisson_scalar_inequality_on_both_half_lines(self):
        # Relative residuals keep the check stable for large positive t.
        for y in (0.0, 0.1, 1.0, 3.0, 20.0):
            for t in np.linspace(-40, 40, 1601):
                lhs = (1 - t) * math.exp(t) + y * t
                self.assertLessEqual(lhs, 1 + y**2 / 2 + 1e-12)

    def test_global_class_bounds_and_transport_under_rescaling(self):
        rng = np.random.default_rng(113)
        for scale in (0.2, 1.0, 3.0):
            p = PoissonBall(scale * self.problem.A, self.problem.y, 1.3)
            for _ in range(20):
                x = rng.normal(size=4); x *= p.radius * rng.random() / np.linalg.norm(x)
                g = p.gradient(x); s = p.lmo(g); d = s - x
                gap = float(-g @ d); a = p.L0 + p.L1 * gap
                for t in (0.0, 0.1, 0.5, 1.0):
                    z = x + t * d; gz = p.gradient(z)
                    gapz = float(gz @ (z - p.lmo(gz)))
                    h = float(np.linalg.eigvalsh(p.hessian(z))[-1])
                    self.assertLessEqual(h, p.L0 + p.L1 * gapz + 1e-9)
                    self.assertLessEqual(h, p.norm_L0 + p.norm_L1 * np.linalg.norm(gz) + 1e-9)
                    self.assertLessEqual(h, p.global_L + 1e-9)
                    self.assertLessEqual(p.L0 + p.L1 * gapz, a * math.exp(p.L1 * p.diameter * np.linalg.norm(d) * t) + 1e-9)

    def test_actual_oracle_invocations_for_every_comparator(self):
        for method in METHODS:
            p = PoissonBall(self.problem.A, self.problem.y, 2.0)
            p.objective = Mock(wraps=p.objective)
            p.gradient = Mock(wraps=p.gradient)
            p.lmo = Mock(wraps=p.lmo)
            result, trace, _ = solve(p, np.zeros(4), method, epsilon=1e-3, max_attempts=1000)
            self.assertEqual(p.objective.call_count, result["f_calls"], method)
            self.assertEqual(p.gradient.call_count, result["g_calls"], method)
            self.assertEqual(p.lmo.call_count, result["lmo_calls"], method)
            self.assertEqual(len(trace), result["attempts"] + 1)

    def test_all_methods_stop_at_an_initial_minimum(self):
        p = PoissonBall(self.problem.A, np.ones(self.problem.m), 2.0)
        for method in METHODS:
            result, _, end = solve(p, np.zeros(4), method)
            self.assertEqual(result["updates"], 0)
            self.assertEqual((result["f_calls"], result["g_calls"], result["lmo_calls"]), (1, 1, 1))
            np.testing.assert_array_equal(end, np.zeros(4))

    def test_armijo_grid_guarantee_and_trial_accounting(self):
        p = self.problem
        result, trace, _ = solve(p, np.array([2., 0, 0, 0]), "armijo", epsilon=1e-6)
        for row in trace[:-1]:
            a = logarithmic_step(row["gap"], row["direction_norm"], p.L0, p.L1, p.diameter)
            self.assertGreaterEqual(row["alpha"] + 1e-14, 0.5 * a)
            self.assertLessEqual(row["trials"], 1 + math.ceil(math.log(1 / a, 2)))
        self.assertEqual(result["f_calls"], 1 + sum(r["trials"] for r in trace))

    def test_bisection_directional_optimality(self):
        p = self.problem
        _, trace, _ = solve(p, np.array([2., 0, 0, 0]), "line_bisection", epsilon=1e-6)
        for row in trace[:-1]:
            x = row["x"]; s = p.lmo(p.gradient(x)); d = s - x
            z = (1 - row["alpha"]) * x + row["alpha"] * s
            derivative = float(p.gradient(z) @ d)
            if row["alpha"] == 1:
                self.assertLessEqual(derivative, 1e-10)
            else:
                self.assertLessEqual(abs(derivative), 1e-9)

    def test_poisson_oracle_endpoint_is_not_the_solution(self):
        p = self.problem
        s = p.lmo(p.gradient(np.zeros(4)))
        gs = p.gradient(s)
        self.assertGreater(float(gs @ (s - p.lmo(gs))), 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
