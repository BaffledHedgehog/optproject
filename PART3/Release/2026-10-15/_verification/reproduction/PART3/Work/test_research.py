"""Independent numerical checks of the claims used in research.md.

Run: python -m unittest discover -s PART3/Work -p "test_*.py" -v
"""

from dataclasses import replace
import math
import unittest

import numpy as np
from scipy.optimize import minimize_scalar

from fw import METHODS, corrected_lower_bound, h_step, solve
from problems import lmo, make_problem


class ResearchChecks(unittest.TestCase):
    def test_scalar_history_and_averaged_gradient_duals_have_no_general_order(self):
        # f(x)=x^2/2, Q=[-1,1]: f*(u)=u^2/2, sigma_Q(-u)=|u|.
        # Arbitrary tangent histories, not claimed algorithm trajectories.
        def scalar_lower(x):
            point, g = np.array([x]), np.array([x])
            return corrected_lower_bound(0.5*x*x, g, point, lmo(g, 1), 2)
        def averaged_dual(points):
            u = float(np.mean(points))
            return -0.5*u*u-abs(u)
        self.assertGreater(averaged_dual([-1.0, 1.0]), max(map(scalar_lower, [-1.0, 1.0])))
        self.assertLess(averaged_dual([-1.0, -0.5]), max(map(scalar_lower, [-1.0, -0.5])))

    def test_analytic_h_step_minimizes_independent_scalar_model(self):
        rng = np.random.default_rng(714)
        for _ in range(120):
            gap, upper, q, h0, h1 = np.exp(rng.uniform(-3, 1, 5))
            c = q * math.sqrt(h1)
            def model(t):
                return (upper + h0 / h1) * 2 * math.sinh(c * t / 2)**2 - gap * math.sinh(c * t) / c
            alpha = h_step(gap, upper, q, h0, h1)
            reference = minimize_scalar(model, bounds=(0, 1), method="bounded",
                                        options={"xatol": 1e-13})
            best = min(model(0), model(1), reference.fun)
            self.assertLessEqual(model(alpha), best + 2e-9 * max(1, abs(best)))
            predicted = 0.5 * gap * min(1, gap / ((h0 + h1 * upper) * q*q))
            self.assertLessEqual(model(alpha), -predicted + 2e-9 * max(1, abs(best)))

    def test_zero_and_small_curvature_limits(self):
        self.assertEqual(h_step(0, 1, 2, 1, 1), 0)
        self.assertEqual(h_step(1, 1, 0, 1, 1), 0)
        self.assertEqual(h_step(1, 1, 2, 0, 0), 1)
        self.assertEqual(h_step(1, 1, 2, 2, 0), 0.125)
        for tiny in (1e-30, 1e-100, 1e-300):
            self.assertAlmostEqual(h_step(1, 1, 2, 2, tiny), 0.125, places=14)
        self.assertTrue(0 < h_step(1, 2, 3, 0, 4) <= 1)

    def test_active_optimum_kkt_and_curvature(self):
        rng = np.random.default_rng(314)
        for scenario in ("interior", "boundary", "active", "active_scaled", "active_highdim"):
            p = make_problem(2, scenario)
            self.assertAlmostEqual(p.objective(p.reference_x), 0, places=13)
            g = p.gradient(p.reference_x)
            self.assertAlmostEqual(float(np.dot(g, p.reference_x - lmo(g, p.radius))), 0, places=12)
            if scenario.startswith("active"):
                self.assertGreater(np.linalg.norm(g), 0.1)
            for _ in range(15):
                x = rng.normal(size=p.x0.size)
                x *= p.radius * rng.random() / np.linalg.norm(x, 1)
                eig = np.linalg.eigvalsh(p.hessian(x))[-1]
                self.assertLessEqual(eig, p.h0 + p.h1 * p.objective(x) + 1e-11)
                self.assertLessEqual(eig, p.global_l + 1e-11)
                self.assertGreaterEqual(p.objective(x), -1e-13)

    def test_gradient_and_hessian_finite_differences(self):
        for scenario in ("active", "active_scaled"):
            p = make_problem(3, scenario)
            x, eps = p.x0 * 0.4, 1e-5
            numerical_g = np.empty_like(x)
            numerical_h = np.empty((x.size, x.size))
            for j in range(x.size):
                e = np.zeros_like(x)
                e[j] = eps
                numerical_g[j] = (p.objective(x + e) - p.objective(x - e)) / (2 * eps)
                numerical_h[:, j] = (p.gradient(x + e) - p.gradient(x - e)) / (2 * eps)
            np.testing.assert_allclose(p.gradient(x), numerical_g, rtol=2e-7, atol=1e-8)
            np.testing.assert_allclose(p.hessian(x), numerical_h, rtol=2e-7, atol=1e-8)

    def test_certificates_descent_and_local_progress(self):
        for scenario in ("interior", "boundary", "active", "active_scaled"):
            p = make_problem(0, scenario)
            for method in ("H_gap", "H_memory", "H_nonnegative", "quadratic_memory"):
                r = solve(p, method, max_iter=80, tolerance=0)
                for row in r.trace:
                    self.assertLessEqual(row["lower_bound"], 1e-11)
                    self.assertGreaterEqual(row["certificate"] + 1e-11, row["objective"])
                    self.assertLessEqual(row["certificate"], row["fw_gap"] + 1e-11)
                    self.assertLessEqual(row["l1_norm"], p.radius + 1e-11)
                for row, following in zip(r.trace, r.trace[1:]):
                    g, u, q2 = row["fw_gap"], row["model_upper_gap"], row["direction_norm_sq"]
                    denominator = (p.h0 + p.h1 * u) * q2
                    progress = 0.5 * g * min(1, g / denominator) if denominator else 0.5 * g
                    self.assertLessEqual(following["objective"], row["objective"] - progress + 2e-11)
                    self.assertLessEqual(following["certificate"], row["certificate"] + 2e-11)

    def test_every_baseline_feasible_and_certified(self):
        p = make_problem(1, "active_scaled")
        for method in METHODS:
            result = solve(p, method, max_iter=30)
            self.assertLessEqual(np.linalg.norm(result.x, 1), p.radius + 1e-11)
            for row in result.trace:
                self.assertLessEqual(row["lower_bound"], 2e-10)
            self.assertEqual(result.trace[-1]["lmo_evals"], len(result.trace) + (method == "AC_FW"))

    def test_real_oracle_call_accounting(self):
        p = make_problem(0, "active")
        for method in METHODS:
            observed = {"f": 0, "g": 0}
            def f(x):
                observed["f"] += 1
                return p.objective(x)
            def g(x):
                observed["g"] += 1
                return p.gradient(x)
            r = solve(replace(p, objective=f, gradient=g), method, max_iter=12)
            self.assertEqual(r.trace[-1]["function_evals"], observed["f"])
            self.assertEqual(r.trace[-1]["gradient_evals"], observed["g"])

    def test_reference_values_are_never_solver_inputs(self):
        p = make_problem(0, "active")
        poisoned = replace(p, reference_x=None, reference_upper=math.nan, reference_lower=math.nan)
        for method in METHODS:
            first, second = solve(p, method, max_iter=8), solve(poisoned, method, max_iter=8)
            np.testing.assert_array_equal(first.x, second.x)

    def test_objective_shift_preserves_unknown_optimum_variants(self):
        p = make_problem(0, "active")
        shifted = replace(p, objective=lambda x: p.objective(x) + 1000,
                          known_lower=None)
        for method in ("H_gap", "H_memory", "quadratic_memory"):
            first, second = solve(p, method, max_iter=40), solve(shifted, method, max_iter=40)
            np.testing.assert_allclose(first.x, second.x, atol=2e-10, rtol=1e-9)

    def test_h0_zero_certificate_complexity(self):
        # f(t)=t+t^3 on [0,1], f''<=6*f; genuinely H0=0.
        from problems import Problem
        p = Problem("h0_zero", 0, 0.5, np.array([-0.5]),
                    lambda x: (0.5-x[0]) + (0.5-x[0])**3,
                    lambda x: np.array([-1-3*(0.5-x[0])**2]),
                    lambda x: np.array([[6*(0.5-x[0])]]),
                    0.0, 6.0, 6.0, None, 0, 0, np.array([0.5]), {})
        for method in ("H_gap", "H_memory", "quadratic_memory"):
            r = solve(p, method, max_iter=1000, tolerance=1e-8)
            self.assertLessEqual(r.trace[-1]["certificate"], 1e-8)
            initial = r.trace[0]["certificate"]
            bound = (5 + 4*p.h1) * math.ceil(math.log2(initial / 1e-8))
            self.assertLessEqual(r.trace[-1]["iteration"], bound)

    def test_corrected_inexact_certificate_and_unsafe_example(self):
        # f(x)=x^2/2 on [-1,1]. At x=1 a zero gradient estimate produces
        # the false uncorrected lower bound 1/2 > f*=0.
        x, s, ghat = np.array([1.0]), np.array([1.0]), np.array([0.0])
        self.assertGreater(0.5 + np.dot(ghat, s-x), 0)
        self.assertLessEqual(corrected_lower_bound(0.5, ghat, x, s, 2, gradient_error=1), 0)
        rng = np.random.default_rng(11)
        p = make_problem(0, "active")
        for _ in range(60):
            x = rng.normal(size=p.x0.size)
            x *= p.radius * rng.random() / np.linalg.norm(x, 1)
            noise = rng.normal(size=x.size) * 0.03
            ghat = p.gradient(x) + noise
            exact = lmo(ghat, p.radius)
            approximate = 0.7 * exact + 0.3 * x
            eta = float(np.dot(ghat, approximate-exact))
            lower = corrected_lower_bound(p.objective(x)+0.01, ghat, x, approximate,
                                           2*p.radius, np.linalg.norm(noise), eta, 0.01)
            self.assertLessEqual(lower, 1e-11)

    def test_logistic_reference_interval(self):
        p = make_problem(0, "breast_cancer")
        self.assertGreaterEqual(p.reference_upper, p.reference_lower)
        self.assertLess(p.reference_upper-p.reference_lower, 2e-6)
        self.assertGreater(np.linalg.norm(p.gradient(p.reference_x)), 1e-3)
        for method in ("H_memory", "adaptive_L", "secant"):
            r = solve(p, method, max_iter=25)
            self.assertLessEqual(r.trace[-1]["lower_bound"], p.reference_upper + 1e-10)
            self.assertGreaterEqual(r.trace[-1]["objective"], p.reference_lower - 1e-10)

    def test_adaptive_certificate_needs_no_curvature_or_reference_data(self):
        p = make_problem(2, "active_scaled")
        poisoned = replace(p, h0=math.nan, h1=math.nan, global_l=math.nan,
                           known_lower=None, reference_lower=math.nan,
                           reference_upper=math.nan, reference_x=None)
        first = solve(p, "adaptive_certificate", max_iter=100)
        second = solve(poisoned, "adaptive_certificate", max_iter=100)
        np.testing.assert_array_equal(first.x, second.x)

    def test_adaptive_envelope_progress_and_trial_accounting(self):
        for scenario in ("interior", "active", "active_scaled", "active_highdim"):
            p = make_problem(1, scenario)
            for factor in (1.0, 3.0):
                diameter = 2 * p.radius * factor
                # Below this certificate scale the active planted objective can
                # subtract quantities at machine precision; exact-arithmetic
                # curvature bounds then cannot be checked by f differences.
                r = solve(p, "adaptive_certificate", max_iter=120,
                          tolerance=1e-6, diameter_bound=diameter)
                trials = 0
                for row, following in zip(r.trace, r.trace[1:]):
                    k, u, g = row["iteration"], row["certificate"], row["fw_gap"]
                    curvature = row["trial_curvature"]
                    envelope = max(u / (2**(k+1) * diameter**2), 2*(p.h0+p.h1*u))
                    self.assertLessEqual(curvature, envelope * (1+1e-12))
                    progress = 0.5 * g * row["step"]
                    self.assertLessEqual(following["objective"], row["objective"]-progress+1e-11)
                    self.assertLessEqual(following["certificate"], u+1e-11)
                    trials += row["line_search_trials"]
                n = len(r.trace)-1
                self.assertEqual(r.trace[-1]["function_evals"], trials+1)
                self.assertEqual(r.trace[-1]["gradient_evals"], n+1)
                # Exact telescoping identity, independent of any curvature bound.
                normalized = r.trace[-2]["normalized_curvature"]
                self.assertAlmostEqual(trials, 2*n+math.log2(normalized), places=10)
                eps_source = r.trace[-2]["certificate"]
                overhead = max(0, math.log2(max(1, 2*diameter**2*(p.h0/eps_source+p.h1))))
                self.assertLessEqual(trials, 2*n+overhead+1e-10)

    def test_adaptive_objective_rescaling_and_shift(self):
        p = make_problem(3, "active")
        control = solve(p, "adaptive_certificate", max_iter=45, tolerance=0)
        for scale, offset in ((1e-6, 0.0), (1e6, 0.0), (1.0, 1000.0)):
            transformed = replace(p, objective=lambda x: scale*p.objective(x)+offset,
                                  gradient=lambda x: scale*p.gradient(x),
                                  known_lower=None, h0=math.nan, h1=math.nan)
            result = solve(transformed, "adaptive_certificate", max_iter=45, tolerance=0)
            np.testing.assert_allclose(control.x, result.x, rtol=1e-8, atol=1e-8)

    def test_adaptive_h0_zero_and_affine_cases(self):
        from problems import Problem
        p = Problem("h0_zero", 0, 0.5, np.array([-0.5]),
                    lambda x: (0.5-x[0])+(0.5-x[0])**3,
                    lambda x: np.array([-1-3*(0.5-x[0])**2]),
                    lambda x: np.array([[6*(0.5-x[0])]]),
                    math.nan, math.nan, math.nan, None, 0, 0, np.array([0.5]), {})
        result = solve(p, "adaptive_certificate", max_iter=200, tolerance=1e-10)
        self.assertLessEqual(result.trace[-1]["certificate"], 1e-10)
        j = math.ceil(math.log2(result.trace[0]["certificate"]/1e-10))
        self.assertLessEqual(result.trace[-1]["iteration"], (7+8*6)*j)
        linear = replace(p, objective=lambda x: 0.5-x[0],
                         gradient=lambda x: np.array([-1.0]))
        result = solve(linear, "adaptive_certificate", max_iter=2)
        self.assertEqual(result.trace[-1]["iteration"], 1)
        self.assertEqual(result.trace[-1]["function_evals"], 2)

    def test_adaptive_rejects_invalid_diameter_bounds(self):
        p = make_problem(0, "active")
        for invalid in (0, -1, math.inf, math.nan, p.radius):
            with self.assertRaises(ValueError):
                solve(p, "adaptive_certificate", diameter_bound=invalid)

    def test_poisson_real_data_derivatives_reference_and_curvature(self):
        p = make_problem(0, "bike_poisson")
        self.assertEqual(p.metadata["rows"], 584)
        self.assertLess(p.reference_upper-p.reference_lower, 1e-9)
        self.assertGreater(p.h1, 0)
        self.assertGreater(p.reference_lower, 0)
        rng = np.random.default_rng(28)
        for _ in range(20):
            x = rng.normal(size=p.x0.size)
            x *= p.radius*rng.random()/np.linalg.norm(x, 1)
            eig = np.linalg.eigvalsh(p.hessian(x))[-1]
            self.assertLessEqual(eig, p.h0+p.h1*(p.objective(x)-p.reference_upper))
            self.assertLessEqual(eig, p.global_l)
        x, eps = p.x0*0.2, 1e-5
        for j in range(x.size):
            delta = np.zeros_like(x)
            delta[j] = eps
            difference = (p.objective(x+delta)-p.objective(x-delta))/(2*eps)
            self.assertAlmostEqual(difference, p.gradient(x)[j], delta=1e-8)
        for method in ("adaptive_certificate", "H_memory", "adaptive_L"):
            r = solve(p, method, max_iter=30)
            self.assertLessEqual(r.trace[-1]["lower_bound"], p.reference_upper+1e-10)
            self.assertGreaterEqual(r.trace[-1]["objective"], p.reference_lower-1e-10)

    def test_initialization_ablation_shares_the_first_trial_scale(self):
        for scenario in ("interior", "active_scaled", "bike_poisson"):
            p = make_problem(0, scenario)
            a = solve(p, "adaptive_certificate", max_iter=1)
            b = solve(p, "adaptive_L_initial", max_iter=1)
            np.testing.assert_array_equal(a.x, b.x)
            self.assertEqual(a.trace[-1]["function_evals"], b.trace[-1]["function_evals"])

    def test_ordinary_backtracking_phase_lag_and_trial_bounds(self):
        from backtracking_audit import audit_run
        for scenario in ("interior", "active_scaled", "active_highdim", "bike_poisson"):
            p = make_problem(1, scenario)
            for method in ("adaptive_L", "adaptive_L_initial"):
                result = solve(p, method, max_iter=150, tolerance=1e-6)
                audit_run(result, p.h0, p.h1, 2*p.radius, method, 1e-6)

    def test_phase_lag_potential_for_abrupt_certificate_drops(self):
        # Stress the abstract lag lemma, independently of a particular loss:
        # arbitrary accepted thresholds below H0+H1*U, skipped phases,
        # and initial estimates spanning many orders of magnitude.
        rng = np.random.default_rng(5831)
        for _ in range(500):
            h0, h1, diameter, u0 = np.exp2(rng.uniform(-12, 12, 4))
            previous = np.exp2(rng.uniform(-30, 30))
            p = math.ceil(math.log2(max(1, previous*diameter**2/u0)))
            lag, last_phase = 0, 0
            u = u0
            for k in range(80):
                if k:
                    u *= np.exp2(-rng.uniform(0, 4))
                phase = math.floor(math.log2(u0/u))
                b = u0 / 2**phase
                cap = max(2*(h0+h1*b), b/(2*diameter**2))
                # Any lower acceptance threshold is consistent with the
                # sufficient upper threshold in the mathematical argument.
                threshold = (h0+h1*u) * rng.uniform(0.001, 1)
                accepted = previous / 2
                while accepted < threshold:
                    accepted *= 2
                lag += accepted > cap
                previous, last_phase = accepted, phase
                self.assertLessEqual(lag, p + last_phase + 1)

    def test_ordinary_backtracking_overestimated_affine_curvature(self):
        from backtracking_audit import audit_run
        from problems import Problem
        scale = 1e-8
        p = Problem("affine", 0, 1.0, np.array([-1.0]),
                    lambda x: -scale*x[0], lambda x: np.array([-scale]),
                    lambda x: np.zeros((1, 1)), 0, 0, 0, None,
                    -scale, -scale, np.array([1.0]), {})
        result = solve(p, "adaptive_L", max_iter=100, tolerance=scale*1e-6)
        checks = audit_run(result, 0, 0, 2, "adaptive_L", scale*1e-6)
        self.assertGreater(checks["lag_steps"], 20)
        self.assertLessEqual(result.trace[-1]["certificate"], scale*1e-6)
        predicted = math.ceil(math.log2(1+(1/scale)*(1-0.5e-6)))
        self.assertEqual(result.trace[-1]["iteration"], predicted)
        for row in result.trace:
            expected = max(0, 2*scale-2*scale**2*(2**row["iteration"]-1))
            self.assertAlmostEqual(row["certificate"], expected, delta=1e-20)
        matched = solve(p, "adaptive_L_initial", max_iter=2, tolerance=scale*1e-6)
        self.assertEqual(matched.trace[-1]["iteration"], 1)

    def test_ac_fw_manual_quadratic_step_and_rejection(self):
        from auto_conditioned import damping, local_curvature
        from problems import Problem
        # Initial atom (-1,0) goes to (1,0). L0=1. The following direction
        # selects the high-curvature second axis and the first trial fails.
        hessian = np.diag([1.0, 100.0])
        linear = np.array([-0.8, 1.0])
        f = lambda x: float(0.5*x @ hessian @ x + linear @ x)
        g = lambda x: hessian @ x + linear
        p = Problem("quadratic", 0, 1.0, np.array([-1.0, 0.0]), f, g,
                    lambda x: hessian, 100, 0, 100, None,
                    -math.inf, math.inf, None, {})
        # Compute initialization and the first trial independently.
        initial = lmo(g(p.x0), 1.0)
        initial_l = local_curvature(p.x0, initial, f(p.x0), f(initial), g(p.x0))
        direction = lmo(g(initial), 1.0) - initial
        gap = float(-g(initial) @ direction)
        alpha = min(1.0, gap/(initial_l*float(direction @ direction)))
        trial = initial + alpha*direction
        expected = trial if f(trial) < f(initial) else initial
        self.assertGreater(f(trial), f(initial))
        result = solve(p, "AC_FW", max_iter=1, tolerance=0)
        np.testing.assert_allclose(result.x, expected, atol=1e-15)
        self.assertEqual(result.trace[-1]["function_evals"], 3)
        self.assertEqual(result.trace[-1]["lmo_evals"], 3)
        self.assertEqual(result.trace[-1]["gradient_evals"], 2 + int(f(trial) < f(initial)))
        result = solve(p, "AC_FW", max_iter=40)
        self.assertGreater(result.trace[-1]["rejected_steps"], 0)
        for row, following in zip(result.trace, result.trace[1:]):
            self.assertEqual(following["function_evals"], row["function_evals"]+1)
            if not row["trial_accepted"]:
                self.assertEqual(following["objective"], row["objective"])
                self.assertEqual(following["gradient_evals"], row["gradient_evals"])
        self.assertTrue(0 < damping(0) < damping(1) < damping(1000) < 1)


if __name__ == "__main__":
    unittest.main()
