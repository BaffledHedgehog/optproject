"""Checks of new count-data protocols, curvature and independent references."""

from dataclasses import replace
import math
import unittest

import numpy as np

from count_problems import load_training, make_count_problem, unconstrained_reference
from multidata_benchmark import METHODS, audit, run, vertex


class CountDataChecks(unittest.TestCase):
    def test_training_transforms_and_data_dimensions(self):
        for dataset, shape, raw_rows in (("seoul", (7008, 15), 8760), ("metro", (38563, 12), 48204)):
            a, target, constants, metadata = load_training(dataset)
            self.assertEqual(a.shape, shape)
            self.assertEqual(metadata["raw_rows"], raw_rows)
            np.testing.assert_array_equal(a[:, 0], np.ones(a.shape[0]))
            self.assertLessEqual(float(np.max(np.abs(a[:, 1:]))), 1)
            self.assertAlmostEqual(float(target.mean()), 1)
            self.assertTrue(np.all(np.isfinite(constants)))
            self.assertFalse(a.flags.writeable)
            self.assertTrue(metadata["training_first_date"] < metadata["training_last_date"])

    def test_derivatives_references_and_curvature(self):
        rng = np.random.default_rng(3891)
        for dataset in ("seoul", "metro"):
            unc, residual = unconstrained_reference(dataset)
            self.assertLess(residual, 1e-10)
            critical = np.linalg.norm(unc, 1)
            for ratio in (0.5, 2):
                p = make_count_problem(dataset, ratio*critical)
                self.assertLessEqual(np.linalg.norm(p.reference_x, 1), p.radius+1e-12)
                self.assertLess(p.reference_upper-p.reference_lower, 2e-7)
                if ratio == 0.5:
                    self.assertAlmostEqual(np.linalg.norm(p.reference_x, 1), p.radius, places=9)
                    self.assertGreater(np.linalg.norm(p.gradient(p.reference_x)), 1e-3)
                else:
                    np.testing.assert_array_equal(p.reference_x, unc)
                x = rng.normal(size=unc.size)
                x *= min(0.5, p.radius/np.linalg.norm(x, 1))
                v = rng.normal(size=x.size)
                v /= np.linalg.norm(v)
                epsilon = 1e-5
                self.assertAlmostEqual((p.objective(x+epsilon*v)-p.objective(x-epsilon*v))/(2*epsilon), float(p.gradient(x)@v), places=8)
                np.testing.assert_allclose((p.gradient(x+epsilon*v)-p.gradient(x-epsilon*v))/(2*epsilon), p.hessian(x)@v, atol=1e-9)
                for _ in range(10):
                    y = rng.normal(size=x.size)
                    y *= p.radius*rng.random()/np.linalg.norm(y, 1)
                    norm_h = np.linalg.eigvalsh(p.hessian(y))[-1]
                    self.assertLessEqual(norm_h, p.h0+p.h1*(p.objective(y)-p.reference_upper)+1e-10)
                    self.assertLessEqual(norm_h, p.global_l+1e-10)

    def test_all_vertices_are_distinct_and_feasible(self):
        for dimension in (12, 15):
            vertices = [vertex(k, dimension, 2.3) for k in range(2*dimension)]
            self.assertEqual(len({v.tobytes() for v in vertices}), 2*dimension)
            for v in vertices:
                self.assertAlmostEqual(np.linalg.norm(v, 1), 2.3)
                self.assertEqual(np.count_nonzero(v), 1)
            with self.assertRaises(ValueError):
                vertex(2*dimension, dimension, 2.3)

    def test_oracle_counts_and_no_reference_inputs(self):
        unc, _ = unconstrained_reference("seoul")
        p = make_count_problem("seoul", np.linalg.norm(unc, 1)/2)
        for method in METHODS:
            counts = {"f": 0, "g": 0}
            def f(x):
                counts["f"] += 1
                return p.objective(x)
            def g(x):
                counts["g"] += 1
                return p.gradient(x)
            hidden = replace(p, objective=f, gradient=g, reference_x=None,
                             reference_lower=math.nan, reference_upper=math.nan, known_lower=None)
            if not method.endswith("Q"):
                hidden = replace(hidden, h0=math.nan, h1=math.nan, global_l=math.nan)
            result = run(hidden, method, 20, 1e-5)
            self.assertEqual(result.trace[-1]["function_evals"], counts["f"])
            self.assertEqual(result.trace[-1]["gradient_evals"], counts["g"])
            np.testing.assert_array_equal(result.x, run(p, method, 20, 1e-5).x)
            audit(result, p, method, 1e-5)


if __name__ == "__main__":
    unittest.main()
