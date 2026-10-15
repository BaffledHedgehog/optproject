"""Validity, paid LMOs and implementability of causal certificate prefixes."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from certificate_benchmark import METHODS, MonitorBank, observed_solve
from problems import lmo, make_problem
from tangent_monitor import PROFILES, TangentMonitor


class TangentChecks(unittest.TestCase):
    def test_block_bounds_can_be_stronger_or_weaker_than_scalar(self):
        # f=x^2/2 on [-1,1]; eight sources repeat each two-point history.
        for history, expected, scalar in (((-1., 1.), -.5, -1.5),
                                          ((-1., -.5), -17/16, -5/8)):
            monitor = TangentMonitor(1., "block_8")
            lower = -math.inf
            with patch("tangent_monitor.lmo", side_effect=lmo) as oracle:
                for j, value in enumerate(history*4):
                    x = np.array([value])
                    lower = max(lower, -.5*value**2-abs(value))
                    row = monitor.observe(x, .5*value**2, x, lower)
                    self.assertEqual(row["certificate_lmo_evals"], (j+1)//8)
                self.assertEqual(oracle.call_count, 1)
            self.assertAlmostEqual(row["candidate_lower_bound"], expected)
            self.assertAlmostEqual(row["lower_bound"], max(expected, scalar))
            self.assertLessEqual(row["lower_bound"], 0)

    def test_exact_block_schedule_and_causal_prefixes(self):
        rng = np.random.default_rng(24)
        xs = rng.uniform(-1, 1, (300, 3)) / 3
        for profile in PROFILES:
            monitor = TangentMonitor(1., profile, mu=1.)
            short = TangentMonitor(1., profile, mu=1.)
            lower, rows = -math.inf, []
            with patch("tangent_monitor.lmo", side_effect=lmo) as oracle:
                for j, x in enumerate(xs):
                    fx = float(x@x/2)
                    lower = max(lower, float(fx+x@(lmo(x, 1.)-x)))
                    row = monitor.observe(x, fx, x, lower)
                    rows.append(row)
                    self.assertLessEqual(row["lower_bound"], 1e-15)
                    self.assertGreaterEqual(row["certificate"]+1e-15, fx)
                    self.assertLessEqual(row["certificate"], row["scalar_certificate"]+1e-15)
                size = monitor.block_size
                self.assertEqual(oracle.call_count, len(xs)//size if size else 0)
            lower = -math.inf
            for j, x in enumerate(xs[:73]):
                fx = float(x@x/2)
                lower = max(lower, float(fx+x@(lmo(x, 1.)-x)))
                self.assertEqual(short.observe(x, fx, x, lower), rows[j])

    def test_known_mu_validity_and_objective_units(self):
        x, gradient = np.array([.4, -.1]), np.array([.3, -.2])
        # Quadratic centred at (.1,.1), mu=1, f*=7.
        fx = 7+float(gradient@gradient)/2
        lower = fx+float(gradient@(lmo(gradient, 1.)-x))
        first = TangentMonitor(1., "known_mu", mu=1.).observe(x, fx, gradient, lower)
        self.assertAlmostEqual(first["lower_bound"], 7.)
        for scale, shift in ((.01, -2.), (9., 130.)):
            scaled = TangentMonitor(1., "known_mu", mu=scale).observe(
                x, scale*fx+shift, scale*gradient, scale*lower+shift)
            self.assertAlmostEqual(scaled["lower_bound"], 7*scale+shift)
            self.assertAlmostEqual(scaled["certificate"], scale*first["certificate"])
        with self.assertRaises(ValueError):
            TangentMonitor(1., "known_mu")

    def test_online_stopping_matches_attainable_prefix_and_actual_costs(self):
        p = make_problem(0, "interior")
        hidden = replace(p, h0=math.nan, h1=math.nan, global_l=math.nan,
                         reference_x=None, reference_lower=math.nan,
                         reference_upper=math.nan, known_lower=None)
        tolerance, cap = 1e-4, 250
        early_stops = 0
        for method in METHODS:
            bank = MonitorBank(p.radius, .08)
            observed_solve(hidden, method, cap, tolerance, bank.observe)
            for profile in PROFILES:
                expected = next((r for r in bank.rows[profile] if r["certificate"] <= tolerance),
                                bank.rows[profile][-1])
                monitor = TangentMonitor(p.radius, profile, mu=.08)
                counts = {"f": 0, "g": 0}
                def objective(x):
                    counts["f"] += 1
                    return p.objective(x)
                def gradient(x):
                    counts["g"] += 1
                    return p.gradient(x)
                observed = []
                def stop(x, fx, grad, atom, original):
                    row = monitor.observe(x, fx, grad, original["lower_bound"])
                    observed.append((x, row))
                    # Mutating callback arguments must not change the solver.
                    x[:] = 123
                    grad[:] = 456
                    atom[:] = 789
                    original["objective"] = -999
                    return row["certificate"] <= tolerance
                module = "away_fw" if method.startswith("AFW") else "fw"
                with patch(module+".lmo", side_effect=lmo) as directions, \
                        patch("tangent_monitor.lmo", side_effect=lmo) as extras:
                    result = observed_solve(replace(hidden, objective=objective, gradient=gradient),
                                            method, cap, tolerance, stop)
                last = result.trace[-1]
                self.assertEqual(last["iteration"], expected["iteration"])
                self.assertAlmostEqual(last["objective"], expected["objective"])
                self.assertAlmostEqual(observed[-1][1]["certificate"], expected["certificate"])
                self.assertEqual(counts["f"], expected["function_evals"])
                self.assertEqual(counts["g"], expected["gradient_evals"])
                self.assertEqual(directions.call_count, expected["direction_lmo_evals"])
                self.assertEqual(extras.call_count, expected["certificate_lmo_evals"])
                self.assertEqual(directions.call_count+extras.call_count, expected["lmo_evals"])
                early_stops += last["iteration"] < bank.rows["scalar"][-1]["iteration"]
        self.assertGreater(early_stops, 0)


if __name__ == "__main__":
    unittest.main()
