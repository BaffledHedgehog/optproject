"""Checks of the active-set algebra, oracle accounting and phase arguments."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from away_fw import decompose, reconstruct, solve_away, trial_state
from certificate_bounds import certificate_threshold, gap_envelope
from problems import Problem, lmo, make_problem


class AwayChecks(unittest.TestCase):
    def test_active_representation_and_feasible_away_caps(self):
        rng = np.random.default_rng(42017)
        for _ in range(80):
            x = rng.normal(size=12)
            x *= 1.5*rng.random()/np.linalg.norm(x, 1)
            weights = decompose(x, 1.5)
            np.testing.assert_allclose(reconstruct(weights, 12, 1.5), x, atol=1e-15)
            self.assertAlmostEqual(math.fsum(weights.values()), 1)
            atom = max(weights, key=weights.get)
            cap = weights[atom]/(1-weights[atom])
            v = reconstruct({atom: 1.0}, 12, 1.5)
            for alpha in (0.2*cap, cap):
                new, y = trial_state(weights, "away", atom, alpha, cap, 12, 1.5)
                np.testing.assert_allclose(y, x+alpha*(x-v), atol=2e-15)
                self.assertGreater(min(new.values()), 0)
                self.assertLessEqual(np.linalg.norm(y, 1), 1.5+1e-14)
                if alpha == cap:
                    self.assertNotIn(atom, new)
        with self.assertRaises(ValueError):
            decompose(np.array([2.0]), 1.5)

    def test_warm_curvature_identity_drop_budget_and_progress(self):
        observed_drops = 0
        for scenario in ("active", "active_scaled", "active_highdim", "bike_poisson"):
            p = make_problem(1, scenario)
            r = solve_away(p, max_iter=160, tolerance=1e-6)
            previous = 1.0
            trials = 0
            for row, following in zip(r.trace, r.trace[1:]):
                current = row["trial_curvature"]
                self.assertAlmostEqual(math.log2(current/previous), row["line_search_trials"]-2, places=10)
                self.assertLessEqual(current, max(previous/2, 2*(p.h0+p.h1*row["certificate"]))*(1+1e-12))
                self.assertGreaterEqual(row["directional_gap"]+1e-12, row["fw_gap"])
                self.assertLessEqual(following["objective"], row["objective"]-0.5*row["step"]*row["directional_gap"]+1e-11)
                self.assertEqual(following["active_size"] < row["active_size"],
                                 row["drop_step"] or (row["direction"] == "FW" and row["step"] == 1 and row["active_size"] > 1))
                previous = current
                trials += row["line_search_trials"]
                observed_drops += row["drop_step"]
            last = r.trace[-1]
            self.assertEqual(last["numerical_prunes"], 0)
            self.assertLessEqual(last["drop_steps"], last["fw_steps"]+last["initial_active_size"]-1)
            self.assertEqual(last["function_evals"], trials+1)
            self.assertAlmostEqual(trials, 2*last["iteration"]+math.log2(previous), places=10)
            self.assertLessEqual(last["representation_error"], 1e-13)
        self.assertGreater(observed_drops, 0)

    def test_actual_oracle_calls_and_unknown_curvature_inputs(self):
        p = make_problem(0, "active_scaled")
        counts = {"f": 0, "g": 0}
        def f(x):
            counts["f"] += 1
            return p.objective(x)
        def g(x):
            counts["g"] += 1
            return p.gradient(x)
        hidden = replace(p, objective=f, gradient=g, h0=math.nan, h1=math.nan,
                         global_l=math.nan, reference_x=None,
                         reference_lower=math.nan, reference_upper=math.nan, known_lower=None)
        with patch("away_fw.lmo", side_effect=lmo) as oracle:
            r = solve_away(hidden, max_iter=35)
        last = r.trace[-1]
        self.assertEqual(last["function_evals"], counts["f"])
        self.assertEqual(last["gradient_evals"], counts["g"])
        self.assertEqual(last["lmo_evals"], oracle.call_count)
        self.assertEqual(last["away_scans"], last["iteration"])
        np.testing.assert_array_equal(r.x, solve_away(p, max_iter=35).x)

    def test_known_h_steps_decrease_and_use_valid_certificates(self):
        for scenario in ("active", "active_scaled", "bike_poisson"):
            p = make_problem(0, scenario)
            r = solve_away(p, "Q", max_iter=100)
            self.assertEqual(r.trace[-1]["function_evals"], len(r.trace))
            for row in r.trace:
                self.assertLessEqual(row["lower_bound"], p.reference_upper+1e-10)
                self.assertLessEqual(row["l1_norm"], p.radius+1e-12)
            for row, following in zip(r.trace, r.trace[1:]):
                self.assertLessEqual(following["objective"], row["objective"]-0.5*row["step"]*row["directional_gap"]+1e-11)

    def test_strongly_convex_certificate_bridge_and_geometric_phase_bound(self):
        # A one-dimensional cosh loss has interval width W=2R exactly.
        # H0=mu=1 and H1=1; the algorithm is not given these constants.
        radius, optimum = 1.0, 0.2
        p = Problem("interval_cosh", 0, radius, np.array([-0.8]),
                    lambda x: float(2*np.sinh((x[0]-optimum)/2)**2),
                    lambda x: np.array([np.sinh(x[0]-optimum)]),
                    lambda x: np.array([[np.cosh(x[0]-optimum)]]),
                    1.0, 1.0, float(np.cosh(1.2)), None, 0, 0, np.array([optimum]), {})
        tolerance = 1e-6
        for initial in (1.0, 1e4):
            r = solve_away(p, max_iter=400, tolerance=tolerance, initial_estimate=initial)
            u0, d2, kappa, mu = r.trace[0]["certificate"], 4*radius**2, radius**2, 1.0
            local_max = p.h0+p.h1*u0
            eta = certificate_threshold(tolerance, p.h0, p.h1, 2*radius)
            phases = math.ceil(math.log2(u0/eta))
            potential = math.ceil(math.log2(max(1, initial/(2*local_max))))
            bound = 2*math.ceil(potential+(6+4*p.h0*d2/kappa)*phases+8*p.h1*d2*u0/kappa)+1
            self.assertLessEqual(r.trace[-1]["certificate"], tolerance)
            self.assertLessEqual(r.trace[-1]["iteration"], bound)
            for row in r.trace:
                delta = row["objective"]
                self.assertLessEqual(row["fw_gap"], delta+math.sqrt(d2)*local_max*math.sqrt(2*delta/mu)+1e-11)
                self.assertLessEqual(row["fw_gap"], gap_envelope(delta, p.h0, p.h1, 2*radius)+1e-11)
                if "directional_gap" in row:
                    self.assertGreaterEqual(row["directional_gap"]**2+1e-12, 2*kappa*delta)


if __name__ == "__main__":
    unittest.main()
