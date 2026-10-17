"""Independent checks of repaired lower-bound and observable-bisection claims."""

from fractions import Fraction
import math
import unittest

import numpy as np

from lower_bound_audit import (ResistingOracle, bisect_observable, carrier_value,
                               exact_chain_floor, match_carrier, rational_chain_minimum,
                               run_audit)


class LowerBoundChecks(unittest.TestCase):
    def test_exact_schur_floor_by_rational_linear_solves(self):
        for m in range(1, 14):
            for n in range(m):
                self.assertEqual(rational_chain_minimum(n)-rational_chain_minimum(m),
                                 exact_chain_floor(m, n))
        for n in range(101):
            exact = Fraction(3, 4*(2*n+3)*(4*n+7))
            self.assertGreaterEqual(exact, Fraction(3, 32*(n+2)**2))

    def test_dense_repeated_and_adaptive_queries_match_one_coercive_function(self):
        report = run_audit()
        self.assertEqual(len(report["dense_oracle_runs"]), 36)
        self.assertTrue(all(r["min_eigenvalue"] > 0 for r in report["dense_oracle_runs"]))

    def test_output_need_not_be_a_query_or_lie_in_gradient_span(self):
        oracle = ResistingOracle(5, seed=27)
        rng = np.random.default_rng(19)
        for _ in range(5):
            oracle.query(rng.normal(size=oracle.dimension))
        output = rng.normal(size=oracle.dimension)
        problem = oracle.finish(output)
        hidden = problem.basis[:, 6:].T @ output
        np.testing.assert_allclose(hidden, 0, atol=1e-12)
        value, _ = problem.value_gradient(output)
        self.assertGreaterEqual(value, problem.alpha/112-1e-12)
        np.testing.assert_allclose(problem.value_gradient(problem.optimum)[1], 0, atol=1e-12)

    def test_exact_gap_carrier_preserves_a_proven_radius_fraction(self):
        for h1 in (0.0, .5, 2.0):
            n, h0, radius, rho = 7, 1.2, 2.0, .6
            gamma = 3*h0/(4*(4*n+7))
            f0 = gamma*(1-rho*rho)*radius**2+carrier_value(rho*radius, h0, h1)
            result = match_carrier(h0, h1, radius, n, f0)
            self.assertAlmostEqual(result["matched_f0"], f0, places=11)
            self.assertGreaterEqual(result["chain_radius_sq"], (1-rho*rho)*radius**2-1e-11)
            # Block-diagonal membership at points independent of the start.
            for z in np.linspace(-3, 3, 15):
                h = carrier_value(z, h0, h1)
                hessian = h0 if h1 == 0 else h0*math.cosh(math.sqrt(h1)*z)
                for chain_gap in (0.0, .2, 10.0):
                    self.assertLessEqual(max(h0, hessian), h0+h1*(chain_gap+h)+1e-10)

    def test_carrier_ceiling_destroys_the_unqualified_lower_bound(self):
        h0, h1, radius, n = 1.0, 2.0, 2.0, 8
        ceiling = carrier_value(radius, h0, h1)
        endpoint = match_carrier(h0, h1, radius, n, ceiling)
        self.assertEqual(endpoint["chain_radius_sq"], 0)
        self.assertEqual(endpoint["floor"], 0)
        previous = float("inf")
        for fraction in (.9, .99, .999, .9999):
            result = match_carrier(h0, h1, radius, n, fraction*ceiling)
            self.assertLess(result["chain_radius_sq"], previous)
            previous = result["chain_radius_sq"]
        with self.assertRaises(ValueError):
            match_carrier(h0, h1, radius, n, 2*ceiling)

    def test_unknown_h1_bisection_has_a_finite_correct_certificate(self):
        for kind in ("quadratic", "cosh", "quartic", "constant"):
            optimum = .371
            if kind == "quadratic":
                f, g = lambda x: (x-optimum)**2/2, lambda x: x-optimum
            elif kind == "cosh":
                f = lambda x: 2*math.sinh(6.5*(x-optimum)/2)**2/6.5**2
                g = lambda x: math.sinh(6.5*(x-optimum))/6.5
            elif kind == "quartic":
                f, g = lambda x: (x-optimum)**4, lambda x: 4*(x-optimum)**3
            else:
                f, g = lambda x: 0.0, lambda x: 0.0
            x, history = bisect_observable(f, g, 0, 1, 1e-10)
            self.assertLessEqual(f(x), 1e-10)
            for row in history:
                self.assertLessEqual(row["value"], row["certificate"]+1e-12)
            self.assertLess(len(history), 100)
        self.assertEqual(bisect_observable(lambda x: 0, lambda x: 0, 0, 0, 1e-5), (0, []))

    def test_radius_types_and_two_term_composition_are_distinct_issues(self):
        for mu in (1.0, 1e-2, 1e-6):
            x0 = np.array([1.0, 0.0])
            endpoint = np.array([0.0, 1/math.sqrt(mu)])
            self.assertAlmostEqual((endpoint[0]**2+mu*endpoint[1]**2)/2, .5)
            self.assertEqual(np.linalg.norm(x0), 1)
            self.assertAlmostEqual(np.linalg.norm(endpoint), 1/math.sqrt(mu))
        for a in (0, 1, 1e4):
            for b in (0, 2, 1e-3):
                self.assertLessEqual(max(a, b), a+b)
                self.assertLessEqual(a+b, 2*max(a, b))


if __name__ == "__main__":
    unittest.main()
