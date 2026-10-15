"""Boundary cases and attainable examples for the refined proof constants."""

import math
import unittest

import numpy as np

from away_fw import solve_away
from certificate_bounds import (certificate_threshold, gap_envelope,
                                sharp_certificate_threshold, sharp_gap_envelope)
from problems import Problem


class ProofAuditChecks(unittest.TestCase):
    def test_quadratics_attain_both_envelope_branches(self):
        for h0 in (.1, 1., 9.):
            for diameter in (.2, 1., 4.):
                cap = h0*diameter**2
                radius = diameter/2
                for ratio in (.001, .1, .5, 1., 3.):
                    delta = ratio*cap
                    center = (radius-math.sqrt(2*delta/h0) if ratio <= .5
                              else -delta/(h0*diameter))
                    optimum = max(-radius, center)
                    actual_error = .5*h0*((radius-center)**2-(optimum-center)**2)
                    gap = h0*(radius-center)*diameter
                    self.assertAlmostEqual(actual_error, delta)
                    self.assertAlmostEqual(gap, sharp_gap_envelope(delta, h0, 0, diameter))
        for delta in (0., .1, 3.):
            self.assertEqual(sharp_gap_envelope(delta, 0, 0, 2), delta)

    def test_threshold_inverse_continuity_and_conservative_comparison(self):
        for h0, h1 in ((0, 0), (1, 0), (0, .1), (0, 2), (1e-6, 10), (3, .01)):
            for diameter in (.1, 2., 10.):
                for epsilon in (1e-8, 1e-3, 10.):
                    eta = sharp_certificate_threshold(epsilon, h0, h1, diameter)
                    self.assertGreaterEqual(eta*(1+1e-14), certificate_threshold(epsilon, h0, h1, diameter))
                    self.assertAlmostEqual(sharp_gap_envelope(eta, h0, h1, diameter)/epsilon, 1.)
                    self.assertGreater(sharp_gap_envelope(eta*(1+1e-6), h0, h1, diameter), epsilon)
                    self.assertLessEqual(sharp_gap_envelope(eta, h0, h1, diameter), gap_envelope(eta, h0, h1, diameter))
        # The branch junction is epsilon=2A/(2-B).
        a, b = 3., .5
        epsilon = 2*a/(2-b)
        for direction in (1-1e-8, 1., 1+1e-8):
            eta = sharp_certificate_threshold(epsilon*direction, a, b, 1.)
            self.assertAlmostEqual(eta, epsilon/2, places=6)

    def test_square_root_gap_cannot_be_replaced_by_linear_error(self):
        # Strongly convex f=x^2/2, Q=[-1,1], first source x=t.
        # U=G=t(1+t), delta=t^2/2: U/delta diverges, even with mu=1.
        ratios = []
        for t in (1e-1, 1e-3, 1e-6):
            delta, gap = t*t/2, t*(1+t)
            self.assertAlmostEqual(gap/math.sqrt(2*delta), 1+t)
            ratios.append(gap/delta)
        self.assertGreater(ratios[-1], 1e6)
        self.assertTrue(all(b > a for a, b in zip(ratios, ratios[1:])))

    def test_geometric_theorem_includes_zero_h0_nonaffine_objective(self):
        # t=.5-x in [0,1], f=t+t^3: f''=6t<=6f and min Hessian=0.
        # At nonoptimal sources selected gap >= 1/2 and delta<=2 imply
        # g^2>=delta/8 (kappa=1/16); zero error is immediate.
        def objective(x):
            t = .5-x[0]
            return float(t+t**3)
        def gradient(x):
            return np.array([-1-3*(.5-x[0])**2])
        p = Problem("zero_h0_cubic", 0, .5, np.array([-.5]), objective, gradient,
                    lambda x: np.array([[6*(.5-x[0])]]), 0., 6., 6., None,
                    0., 0., np.array([.5]), {})
        epsilon, kappa, diameter = 1e-6, 1/16, 1.
        eta = sharp_certificate_threshold(epsilon, p.h0, p.h1, diameter)
        saw_lag = False
        for initial in (1., 1e6):
            result = solve_away(p, max_iter=150, tolerance=epsilon, initial_estimate=initial)
            u0 = result.trace[0]["certificate"]
            phases = math.ceil(math.log2(u0/eta))
            potential = math.ceil(math.log2(max(1, initial/(2*(p.h0+p.h1*u0)))))
            bound = 2*math.ceil(potential+6*phases+8*p.h1*diameter**2*u0/kappa)
            self.assertLessEqual(result.trace[-1]["certificate"], epsilon)
            self.assertLessEqual(result.trace[-1]["iteration"], bound)
            for row, following in zip(result.trace, result.trace[1:]):
                delta = row["objective"]
                self.assertLessEqual(row["fw_gap"], sharp_gap_envelope(delta, 0, 6, 1)+1e-12)
                self.assertGreaterEqual(row["directional_gap"]**2, 2*kappa*delta)
                self.assertLessEqual(following["objective"], delta-.5*row["step"]*row["directional_gap"]+1e-12)
                saw_lag |= row["trial_curvature"] > 2*(p.h0+p.h1*u0)
        self.assertTrue(saw_lag)


if __name__ == "__main__":
    unittest.main()
