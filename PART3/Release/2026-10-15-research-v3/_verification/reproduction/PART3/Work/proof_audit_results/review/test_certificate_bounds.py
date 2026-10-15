"""Independent checks of the local FW-gap bound and its limiting cases."""

import math
import unittest

import numpy as np

from away_fw import solve_away
from certificate_bounds import certificate_threshold, gap_envelope, sharp_gap_envelope
from problems import Problem, lmo, make_problem


class CertificateBoundChecks(unittest.TestCase):
    def test_threshold_solves_both_inequalities_and_limits(self):
        for h0, h1 in ((0, 0), (1, 0), (0, 2), (1e-6, 10), (3, 0.01)):
            for diameter in (0.1, 2, 10):
                for epsilon in (1e-8, 1e-3, 10):
                    eta = certificate_threshold(epsilon, h0, h1, diameter)
                    self.assertGreater(eta, 0)
                    self.assertLessEqual(gap_envelope(eta, h0, h1, diameter), epsilon*(1+5e-15))
                    self.assertGreater(gap_envelope(eta*(1+1e-6), h0, h1, diameter), epsilon)
        self.assertEqual(certificate_threshold(1, 0, 0, 2), 0.5)
        self.assertEqual(certificate_threshold(0.1, 2, 0, 2), 0.1**2/16)
        self.assertAlmostEqual(certificate_threshold(0.1, 0, 2, 2), 0.025)

    def test_pointwise_envelope_against_known_and_independent_minima(self):
        rng = np.random.default_rng(19307)
        for scenario in ("interior", "active", "active_scaled", "breast_cancer", "bike_poisson"):
            p = make_problem(1, scenario)
            for _ in range(40):
                x = rng.normal(size=p.x0.size)
                x *= p.radius*rng.random()/np.linalg.norm(x, 1)
                g = p.gradient(x)
                gap = float(g @ (x-lmo(g, p.radius)))
                delta_upper = max(0, p.objective(x)-p.reference_lower)
                self.assertLessEqual(gap, gap_envelope(delta_upper, p.h0, p.h1, 2*p.radius)+1e-10)
                self.assertLessEqual(gap, sharp_gap_envelope(delta_upper, p.h0, p.h1, 2*p.radius)+1e-10)

    def test_flat_minimizer_set_satisfies_geometric_phase_bound(self):
        # Hessian diag(cosh(x1),0): there is no positive strong-convexity mu.
        # G=x1*sinh(x1)+abs(sinh(x1)), so G^2 >= 2*(cosh(x1)-1).
        p = Problem("flat_cosh", 0, 1, np.array([0.8, 0.2]),
                    lambda x: float(2*np.sinh(x[0]/2)**2),
                    lambda x: np.array([np.sinh(x[0]), 0.0]),
                    lambda x: np.diag([np.cosh(x[0]), 0.0]),
                    1, 1, math.cosh(1), None, 0, 0, np.zeros(2), {})
        epsilon, diameter, kappa = 1e-7, 2, 1
        eta = certificate_threshold(epsilon, p.h0, p.h1, diameter)
        for initial in (1.0, 1e4):
            result = solve_away(p, max_iter=500, tolerance=epsilon, initial_estimate=initial)
            u0 = result.trace[0]["certificate"]
            phases = math.ceil(math.log2(u0/eta))
            potential = math.ceil(math.log2(max(1, initial/(2*(p.h0+p.h1*u0)))))
            bound = 2*math.ceil(potential+(6+4*p.h0*diameter**2/kappa)*phases
                                +8*p.h1*diameter**2*u0/kappa)+result.trace[0]["initial_active_size"]-1
            self.assertLessEqual(result.trace[-1]["certificate"], epsilon)
            self.assertLessEqual(result.trace[-1]["iteration"], bound)
            for row in result.trace:
                delta = row["objective"]
                self.assertLessEqual(row["fw_gap"], gap_envelope(delta, 1, 1, diameter)+1e-12)
                if "directional_gap" in row:
                    self.assertGreaterEqual(row["directional_gap"]**2+1e-12, 2*kappa*delta)

    def test_affine_and_boundary_minimum_cases(self):
        # Affine G=delta, including zero curvature.
        for x in np.linspace(-1, 1, 30):
            delta, gap = 3*(1-x), 3*(1-x)
            self.assertLessEqual(gap, gap_envelope(delta, 0, 0, 2))
        # f=cosh(x)-1 on [1,2], constrained minimizer at 1. A valid H0
        # must cover its nonzero Hessian at that minimizer.
        for x in np.linspace(1, 2, 40):
            delta = math.cosh(x)-math.cosh(1)
            gap = math.sinh(x)*(x-1)
            self.assertLessEqual(gap, gap_envelope(delta, math.cosh(1), 1, 1)+1e-12)


if __name__ == "__main__":
    unittest.main()
