import unittest
from unittest.mock import patch

import numpy as np

from endpoint_fw import VertexObjectiveCache, endpoint_due, solve_endpoint
from problems import Problem, lmo


def quadratic_problem():
    def f(x):
        return float((x[0] - 0.2) ** 2)
    def g(x):
        return np.array([2.0 * (x[0] - 0.2)])
    return Problem("test", 0, 1.0, np.array([1.0]), f, g,
                   lambda x: np.array([[2.0]]), 2.0, 0.0, 2.0,
                   None, 0.0, 0.0, np.array([0.2]), {})


class EndpointFWTests(unittest.TestCase):
    def test_dyadic_schedule(self):
        self.assertEqual([k for k in range(12) if endpoint_due("dyadic", k)], [0, 1, 3, 7])
        self.assertFalse(endpoint_due("ordinary", 0))
        self.assertTrue(all(endpoint_due("dense", k) for k in range(12)))

    def test_cache_only_deduplicates_vertices(self):
        calls = []
        p = quadratic_problem()
        p.objective = lambda x: calls.append(float(x[0])) or float(x[0] ** 2)
        cache = VertexObjectiveCache(p)
        cache(np.array([1.0]))
        cache(np.array([1.0]))
        cache(np.array([0.5]))
        cache(np.array([0.5]))
        self.assertEqual(cache.paid, 3)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.requests, 4)

    def test_near_vertex_is_not_merged_with_exact_vertex(self):
        p = quadratic_problem()
        calls = []
        p.objective = lambda x: calls.append(x.copy()) or float(x[0] ** 2)
        cache = VertexObjectiveCache(p)
        near = np.array([np.nextafter(p.radius, 0.0)])
        exact = np.array([p.radius])
        cache(near)
        cache(exact)
        cache(near)
        cache(exact)
        self.assertEqual(cache.paid, 3)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(len(calls), 3)

    def test_rejected_endpoint_preserves_fallback_trajectory(self):
        p = quadratic_problem()
        ordinary = solve_endpoint(p, "ordinary", max_iter=4, objective_budget=100)
        dense = solve_endpoint(p, "dense", max_iter=4, objective_budget=100)
        self.assertEqual(sum(r["endpoint_accepted"] for r in dense.trace), 0)
        self.assertEqual([r["step"] for r in ordinary.trace], [r["step"] for r in dense.trace])
        np.testing.assert_allclose([r["objective"] for r in ordinary.trace],
                                   [r["objective"] for r in dense.trace], rtol=0, atol=0)
        self.assertGreater(dense.cache.hits, ordinary.cache.hits)

    def test_endpoint_armijo_accepts_vertex_and_counts_lmo(self):
        def f(x): return float(-x[0])
        def g(x): return np.array([-1.0])
        p = Problem("linear", 0, 1.0, np.array([-1.0]), f, g,
                    lambda x: np.zeros((1, 1)), 0.0, 0.0, 0.0,
                    None, -1.0, -1.0, np.array([1.0]), {})
        result = solve_endpoint(p, "dense", max_iter=3, objective_budget=20)
        self.assertTrue(result.trace[0]["endpoint_accepted"])
        self.assertEqual(result.trace[0]["step"], 1.0)
        self.assertEqual(result.trace[-1]["certificate"], 0.0)
        self.assertEqual(result.trace[-1]["lmo_evals"], 2)

    def test_material_negative_gap_raises(self):
        p = quadratic_problem()
        p.x0 = np.array([1.0])
        p.gradient = lambda x: np.array([-1.0])
        with patch("endpoint_fw.lmo", return_value=np.array([-1.0])):
            with self.assertRaisesRegex(ArithmeticError, "Negative FW gap"):
                solve_endpoint(p, "ordinary", max_iter=1, objective_budget=10)


if __name__ == "__main__":
    unittest.main()
