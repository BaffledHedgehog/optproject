"""Verify the independent radius threshold and the boundary/interior control."""

import unittest

import numpy as np

from constraint_transition import critical_radius, reference_geometry
from problems import make_bike_poisson
from report_constraint import seed_atom


class TransitionChecks(unittest.TestCase):
    def test_deduplication_uses_the_actual_starting_vertices(self):
        for seed in range(10):
            p = make_bike_poisson(seed)
            coordinate, sign = seed_atom(seed, p.x0.size)
            expected = np.zeros_like(p.x0)
            expected[coordinate] = sign*p.radius
            np.testing.assert_array_equal(expected, p.x0)

    def test_activity_changes_at_stationary_reference_norm(self):
        interior = make_bike_poisson(0)
        critical = critical_radius(interior)
        self.assertLess(critical["residual_radius_bound"], 1e-11)
        active = make_bike_poisson(0, 0.5*critical["estimate"])
        inactive = make_bike_poisson(0, 1.1*critical["estimate"])
        a, b = [reference_geometry(p, critical) for p in (active, inactive)]
        self.assertEqual(a["regime"], "active")
        self.assertEqual(b["regime"], "interior")
        self.assertGreater(a["estimated_multiplier"], 0.01)
        self.assertGreater(a["inactive_margin"], 0.001)
        self.assertLess(a["kkt_residual_inf"], 1e-6)
        self.assertAlmostEqual(np.linalg.norm(active.reference_x, 1), active.radius, places=10)
        np.testing.assert_allclose(inactive.reference_x, interior.reference_x, atol=1e-12)
        self.assertGreater(active.reference_lower, inactive.reference_upper)
        self.assertEqual(reference_geometry(make_bike_poisson(0, critical["estimate"]), critical)["regime"], "transition")

    def test_radius_bound_scales_with_objective_units(self):
        from robustness_benchmark import scale_problem
        p = make_bike_poisson(0)
        reference = critical_radius(p)
        for a in (1e-3, 1e3):
            scaled = critical_radius(scale_problem(p, a))
            self.assertEqual(scaled["estimate"], reference["estimate"])
            self.assertAlmostEqual(scaled["residual_radius_bound"], reference["residual_radius_bound"], places=26)


if __name__ == "__main__":
    unittest.main()
