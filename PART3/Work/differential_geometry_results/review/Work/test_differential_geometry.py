"""Independent extremizers, real objective gaps, and rational endpoint checks."""

from decimal import Decimal as Dec, localcontext
import itertools
import math
import unittest

import numpy as np

from away_fw import solve_away
from certificate_bounds import (optimal_certificate_threshold, optimal_gap_envelope,
                                sharp_certificate_threshold, sharp_gap_envelope, zero_base_finite_bound)
from differential_geometry_audit import (F, audit_problem, dot, exact_afw, point,
                                         positive_base_counterexample, rational_finite_audit)
from problems import Problem, lmo, make_problem


class DifferentialGeometryChecks(unittest.TestCase):
    def test_inverse_and_improvement_over_frozen_curvature(self):
        for h0, h1, diameter, epsilon in itertools.product(
                (0, 1e-8, 1, 10), (0, 1e-8, .2, 10), (.01, 1, 10), (1e-10, 1e-3, 1, 100)):
            eta = optimal_certificate_threshold(epsilon, h0, h1, diameter)
            value = optimal_gap_envelope(eta, h0, h1, diameter)
            self.assertTrue(math.isclose(value, epsilon, rel_tol=3e-13), (h0, h1, diameter, epsilon, eta, value))
            self.assertGreater(optimal_gap_envelope(eta*(1+1e-5), h0, h1, diameter), epsilon)
            old_eta = sharp_certificate_threshold(epsilon, h0, h1, diameter)
            self.assertGreaterEqual(eta, old_eta*(1-3e-13))
            for delta in (1e-6, .1, 100):
                self.assertLessEqual(optimal_gap_envelope(delta, h0, h1, diameter),
                                     sharp_gap_envelope(delta, h0, h1, diameter)*(1+3e-13))
        self.assertLess(optimal_gap_envelope(1, 0, 1, 10),
                        .72*sharp_gap_envelope(1, 0, 1, 10))

    def test_attaining_objectives_using_independent_decimal_arithmetic(self):
        # Construct the functions from their ODE, using 70-digit arithmetic;
        # compare their actual derivative/LMO gap with the production formula.
        with localcontext() as context:
            context.prec = 70
            def sinh(v):
                return (v.exp()-(-v).exp())/2
            def cosh(v):
                return (v.exp()+(-v).exp())/2
            branches = set()
            for h0s, h1s, ds, delta_s in itertools.product(
                    ('0', '.1', '1'), ('.01', '1', '4'), ('.2', '1', '3'), ('.001', '.1', '1', '20')):
                h0, h1, diameter, delta = map(Dec, (h0s, h1s, ds, delta_s))
                k, c = h1.sqrt(), h0/h1
                a = k*diameter
                if c and delta <= c*(cosh(a)-1):
                    z = 1+delta/c
                    r = (z+(z*z-1).sqrt()).ln()/k
                    def f(u):
                        return c*(cosh(k*(u-r))-1)
                    def grad(u):
                        return c*k*sinh(k*(u-r))
                    optimum = r
                    branches.add('interior')
                else:
                    coefficient = ((delta+c)*cosh(a)-c)/sinh(a)
                    def f(u):
                        return (delta+c)*cosh(k*u)-coefficient*sinh(k*u)-c
                    def grad(u):
                        return k*((delta+c)*sinh(k*u)-coefficient*cosh(k*u))
                    optimum = diameter
                    branches.add('boundary_zero' if not c else 'boundary_positive')
                self.assertLess(abs(f(optimum)), Dec('1e-60'))
                self.assertLess(abs(f(Dec(0))-delta), Dec('1e-60'))
                actual_gap = -diameter*grad(Dec(0))
                predicted = optimal_gap_envelope(float(delta), float(h0), float(h1), float(diameter))
                self.assertTrue(math.isclose(float(actual_gap), predicted, rel_tol=3e-13))
                for j in range(9):
                    value = f(diameter*j/8)
                    self.assertGreaterEqual(value, -Dec('1e-60'))
                    self.assertGreaterEqual(h0+h1*value, -Dec('1e-60'))
            self.assertEqual(branches, {'interior', 'boundary_positive', 'boundary_zero'})

    def test_actual_gaps_on_synthetic_and_regression_objectives(self):
        rng = np.random.default_rng(604917)
        for scenario in ('interior', 'active', 'active_scaled', 'breast_cancer', 'bike_poisson'):
            problem = make_problem(1, scenario)
            for _ in range(40):
                x = rng.normal(size=problem.x0.size)
                x *= problem.radius*rng.uniform(.05, 1)/np.linalg.norm(x, 1)
                grad = problem.gradient(x)
                actual_gap = float(grad @ (x-lmo(grad, problem.radius)))
                delta_upper = max(0, problem.objective(x)-problem.reference_lower)
                bound = optimal_gap_envelope(delta_upper, problem.h0, problem.h1, 2*problem.radius)
                self.assertLessEqual(actual_gap, bound+1e-10)

    def test_zero_base_nonlinear_face_and_gradient_control(self):
        diameter, h1 = math.sqrt(2), 10
        for t, y in itertools.product(np.linspace(0, 1, 31), np.linspace(-.5, .5, 21)):
            delta = t+t**3*(1+y*y)
            grad = np.array([1+3*t*t*(1+y*y), 2*t**3*y])
            hessian = np.array([[6*t*(1+y*y), 6*t*t*y], [6*t*t*y, 2*t**3]])
            eigenvalues = np.linalg.eigvalsh(hessian)
            self.assertGreaterEqual(eigenvalues[0], -1e-14)
            self.assertLessEqual(eigenvalues[-1], h1*delta+1e-13)
            self.assertLessEqual(np.linalg.norm(grad-[1, 0]), h1*diameter*delta/2+1e-13)
            if t == 0:
                self.assertEqual(delta, 0)
                np.testing.assert_array_equal(grad, [1, 0])
            else:
                self.assertGreater(delta, 0)

    def test_exact_rational_finite_termination_and_counterexample(self):
        checks, records = rational_finite_audit()
        self.assertEqual(checks['global_runs'], 24)
        self.assertEqual(checks['planted_local_runs'], 6)
        self.assertGreaterEqual(checks['checked_local_maximal_drops'], 12)
        self.assertTrue(all(r['exact_objective'] == r['exact_certificate'] == '0' for r in records))
        for row in records:
            atoms, objective, gradient, h1, margin, face = audit_problem(row['family'], F(row['beta']))
            x = point(atoms, {i: F(1, 4) for i in range(4)})
            grad = gradient(x)
            u0 = dot(grad, x)-min(dot(grad, a) for a in atoms)
            bound = zero_base_finite_bound(float(u0), float(h1), math.sqrt(2), float(margin),
                                          float(F(row['initial'])), 4)
            self.assertLessEqual(row['iterations'], bound['update_bound'])
            self.assertLessEqual(row['function_evals'], bound['function_call_bound'])
            self.assertGreater(bound['threshold'], 0)
        self.assertEqual(positive_base_counterexample()['exact_positive_base_trial_checks'], 180)
        # Two further local paths start with an oversized inherited estimate.
        # They exercise nonmaximal away steps and the separate halving potential.
        for family in ('simplex', 'box'):
            atoms, objective, gradient, _, _, face = audit_problem(family, F(0))
            theta = F(1, 100)
            weights = {0: (1-theta)/2, 1: (1-theta)/2, 2: theta/3, 3: 2*theta/3}
            bound = zero_base_finite_bound(1, 0, math.sqrt(2), 1, 65536, 4)
            result = exact_afw(atoms, weights, objective, gradient, F(65536))
            self.assertGreater(result['good_steps'], 0)
            self.assertLessEqual(result['good_steps'], bound['local_lag_bound'])
            previous = F(65536)
            for row in result['trace'][:-1]:
                self.assertLessEqual(float(row['certificate']), bound['threshold'])
                self.assertEqual(row['kind'], 'away')
                if float(row['curvature']) > bound['local_curvature_threshold']:
                    self.assertEqual(row['curvature'], previous/2)
                else:
                    self.assertTrue(row['drop'])
                    self.assertNotIn(row['atom'], face)
                previous = row['curvature']
            self.assertEqual(result['trace'][-1]['certificate'], 0)

    def test_positive_base_never_hits_optimum_on_production_path(self):
        for h in (1, 1e-6):
            problem = Problem('positive_base_counterexample', 0, 1, np.array([.5]),
                              lambda x: float(h*x[0]**2/2), lambda x: np.array([h*x[0]]),
                              lambda x: np.array([[h]]), h, 0, h, None, 0, 0, np.zeros(1), {})
            result = solve_away(problem, max_iter=12, tolerance=0, initial_estimate=3*h)
            self.assertEqual(result.trace[-1]['iteration'], 12)
            self.assertAlmostEqual(result.x[0], .5/3**12, delta=2e-16)
            self.assertGreater(result.trace[-1]['objective'], 0)
            self.assertGreater(result.trace[-1]['certificate'], 0)
            for row in result.trace[:-1]:
                self.assertEqual(row['direction'], 'FW')
                self.assertAlmostEqual(row['trial_curvature']/h, 1.5)

    def test_large_hyperbolic_parameters_and_zero_curvature_limits(self):
        for h0 in (0, 1):
            eta = optimal_certificate_threshold(1, h0, 1e6, 1)
            self.assertTrue(math.isfinite(eta) and eta > 0)
            self.assertAlmostEqual(optimal_gap_envelope(eta, h0, 1e6, 1), 1, places=13)
        self.assertEqual(optimal_gap_envelope(.3, 0, 0, 2), .3)
        self.assertEqual(optimal_certificate_threshold(.3, 0, 0, 2), .3)
        for delta in (.01, 1, 10):
            self.assertAlmostEqual(optimal_gap_envelope(delta, 1, 1e-24, 1),
                                   sharp_gap_envelope(delta, 1, 0, 1), places=13)


if __name__ == '__main__':
    unittest.main()
