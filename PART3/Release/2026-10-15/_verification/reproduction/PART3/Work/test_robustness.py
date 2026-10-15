"""Controls that would catch invalid radius references or free initialization."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from away_fw import solve_away
from problems import lmo, make_bike_poisson, make_breast_cancer, make_problem
from robustness_benchmark import audit_run, run_method, scale_problem


class RobustnessChecks(unittest.TestCase):
    def test_new_radii_recompute_reference_and_keep_the_same_objective(self):
        for maker, radii in ((make_breast_cancer, (1.0, 4.0)), (make_bike_poisson, (0.75, 3.0))):
            small, large = [maker(0, radius=r) for r in radii]
            self.assertLessEqual(large.reference_upper, small.reference_upper+1e-9)
            for p in (small, large):
                self.assertLessEqual(np.linalg.norm(p.reference_x, 1), p.radius+1e-12)
                g = p.gradient(p.reference_x)
                gap = float(g @ (p.reference_x-lmo(g, p.radius)))
                self.assertAlmostEqual(p.reference_upper-p.reference_lower, max(0, gap), places=12)
                rng = np.random.default_rng(812)
                for _ in range(12):
                    x = rng.normal(size=p.x0.size)
                    x *= p.radius/np.linalg.norm(x, 1)
                    self.assertEqual(small.objective(x), large.objective(x))
                    local = p.h0+p.h1*(p.objective(x)-p.reference_upper)
                    self.assertLessEqual(np.linalg.eigvalsh(p.hessian(x))[-1], local+1e-10)
                    self.assertLessEqual(np.linalg.eigvalsh(p.hessian(x))[-1], p.global_l+1e-10)
            for radius in (0, -1, math.inf, math.nan):
                with self.assertRaises(ValueError):
                    maker(0, radius=radius)

    def test_objective_scale_transforms_constants_and_references(self):
        p = make_problem(0, "active_scaled")
        for a in (1e-3, 1e3):
            q = scale_problem(p, a)
            self.assertEqual(q.h1, p.h1)
            self.assertEqual(q.h0, a*p.h0)
            self.assertEqual(q.metadata["mu"], a*p.metadata["mu"])
            self.assertEqual(q.reference_upper, a*p.reference_upper)
            self.assertEqual(q.objective(q.x0), a*p.objective(p.x0))
            np.testing.assert_array_equal(q.gradient(q.x0), a*p.gradient(p.x0))
            np.testing.assert_array_equal(q.hessian(q.x0), a*p.hessian(p.x0))

    def test_matched_units_have_the_same_trajectory_in_binary_scaling(self):
        p = make_problem(1, "active_scaled")
        for method in ("FW_BT_certificate", "AFW_BT_certificate"):
            original = run_method(p, method, 100, 1e-5)
            for a in (2.0**-10, 2.0**10):
                q = scale_problem(p, a)
                result = run_method(q, method, 100, a*1e-5)
                np.testing.assert_array_equal(result.x, original.x)
                self.assertEqual(len(result.trace), len(original.trace))
                for row, baseline in zip(result.trace, original.trace):
                    for key in ("function_evals", "gradient_evals", "lmo_evals", "step", "line_search_trials"):
                        self.assertEqual(row.get(key), baseline.get(key))
                    self.assertEqual(row["certificate"]/a, baseline["certificate"])
                audit_run(result, q, method, a*1e-5)

    def test_certificate_initialization_pays_oracles_and_ignores_reference(self):
        p = make_problem(2, "active_scaled")
        counters = {"f": 0, "g": 0}
        def f(x):
            counters["f"] += 1
            return p.objective(x)
        def g(x):
            counters["g"] += 1
            return p.gradient(x)
        hidden = replace(p, objective=f, gradient=g, h0=math.nan, h1=math.nan,
                         known_lower=None, reference_x=None,
                         reference_lower=math.nan, reference_upper=math.nan)
        with patch("away_fw.lmo", side_effect=lmo) as oracle:
            result = solve_away(hidden, max_iter=50, initial_estimate="certificate")
        last = result.trace[-1]
        self.assertEqual(last["function_evals"], counters["f"])
        self.assertEqual(last["gradient_evals"], counters["g"])
        self.assertEqual(last["lmo_evals"], oracle.call_count)
        np.testing.assert_array_equal(result.x, solve_away(p, max_iter=50, initial_estimate="certificate").x)
        terminal = solve_away(p, tolerance=1e20, initial_estimate="certificate").trace[-1]
        self.assertEqual([terminal[key] for key in ("function_evals", "gradient_evals", "lmo_evals")], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
