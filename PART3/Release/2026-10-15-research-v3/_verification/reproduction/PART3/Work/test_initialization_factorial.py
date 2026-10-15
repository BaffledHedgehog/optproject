"""Independent controls for the starting-point/update-rule ablation."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from fw import solve
from initialization_factorial import CELLS, ORACLES, budget_prefix, solve_control
from problems import SCENARIOS, lmo, make_problem


class InitializationControls(unittest.TestCase):
    def test_native_cell_matches_existing_ac_fw(self):
        for scenario in SCENARIOS:
            p = make_problem(0, scenario)
            native = solve(p, "AC_FW", max_iter=80)
            control = solve_control(p, "AC_vertex", max_iter=80)
            np.testing.assert_array_equal(native.x, control.x)
            for a, b in zip(native.trace, control.trace, strict=True):
                for key in (*ORACLES, "objective", "certificate", "fw_gap", "lower_bound"):
                    self.assertEqual(a[key], b[key], (scenario, key))

    def test_common_information_and_initial_trial(self):
        for scenario in ("active", "active_scaled", "bike_poisson"):
            p = make_problem(1, scenario)
            first = {cell: solve_control(p, cell, max_iter=1, tolerance=0).trace[0] for cell in CELLS}
            self.assertEqual(len({row["lower_bound"] for row in first.values()}), 1)
            self.assertEqual(len({row["first_trial_curvature"] for row in first.values()}), 1)
            for row in first.values():
                for oracle in ORACLES:
                    self.assertEqual(row[oracle], 2)
            for start in ("original", "vertex"):
                for key in ("objective", "certificate", "first_trial_step", "first_trial_objective"):
                    self.assertEqual(first[f"AC_{start}"][key], first[f"BT_{start}"][key])

    def test_actual_counts_feasibility_and_certificate(self):
        p = make_problem(2, "active_scaled")
        for cell in CELLS:
            counts = {"f": 0, "g": 0}
            def f(x):
                counts["f"] += 1
                return p.objective(x)
            def g(x):
                counts["g"] += 1
                return p.gradient(x)
            with patch("initialization_factorial.lmo", side_effect=lmo) as observed_lmo:
                r = solve_control(replace(p, objective=f, gradient=g), cell, max_iter=60)
            last = r.trace[-1]
            self.assertEqual(last["function_evals"], counts["f"])
            self.assertEqual(last["gradient_evals"], counts["g"])
            self.assertEqual(last["lmo_evals"], observed_lmo.call_count)
            self.assertEqual(last["lmo_evals"], last["iteration"]+2)
            self.assertEqual(last["gradient_evals"], last["accepted_steps"]+2)
            if cell.startswith("AC"):
                self.assertEqual(last["function_evals"], last["iteration"]+2)
            else:
                self.assertEqual(last["function_evals"], 2+sum(row["line_search_trials"] for row in r.trace[:-1]))
            self.assertLessEqual(np.linalg.norm(r.x, 1), p.radius+1e-12)
            for row in r.trace:
                self.assertLessEqual(row["lower_bound"], p.reference_upper+1e-11)
                self.assertGreaterEqual(row["objective"], p.reference_lower-1e-11)
            for a, b in zip(r.trace, r.trace[1:]):
                self.assertLessEqual(b["certificate"], a["certificate"]+1e-12)

    def test_budget_prefix_is_last_affordable_certificate(self):
        p = make_problem(0, "interior")
        for cell in CELLS:
            r = solve_control(p, cell, max_iter=150)
            for oracle in ORACLES:
                for budget in (2, 5, 20, 100, 500):
                    row = budget_prefix(r.trace, oracle, budget)
                    affordable = [item for item in r.trace if item[oracle] <= budget]
                    self.assertIs(row, affordable[-1])
                    self.assertLessEqual(row[oracle], budget)
                    index = r.trace.index(row)
                    if index+1 < len(r.trace):
                        self.assertGreater(r.trace[index+1][oracle], budget)

    def test_unknown_curvature_and_reference_values(self):
        p = make_problem(0, "active")
        hidden = replace(p, h0=math.nan, h1=math.nan, global_l=math.nan,
                         reference_x=None, reference_lower=math.nan, reference_upper=math.nan,
                         known_lower=None)
        for cell in CELLS:
            first = solve_control(p, cell, max_iter=20)
            second = solve_control(hidden, cell, max_iter=20)
            np.testing.assert_array_equal(first.x, second.x)


if __name__ == "__main__":
    unittest.main()
