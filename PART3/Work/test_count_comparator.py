"""Independent instrumentation of the new count-data AC-FW comparator."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from count_comparator import audit
from count_problems import make_count_problem, unconstrained_reference
from fw import solve
from problems import lmo


class CountComparatorChecks(unittest.TestCase):
    def test_paid_preprocessing_actual_oracles_and_hidden_parameters(self):
        for dataset in ("seoul", "metro"):
            unc, _ = unconstrained_reference(dataset)
            p = make_count_problem(dataset, 2*float(np.linalg.norm(unc, 1)))
            counts = {"f": 0, "g": 0}
            def objective(x):
                counts["f"] += 1
                return p.objective(x)
            def gradient(x):
                counts["g"] += 1
                return p.gradient(x)
            hidden = replace(p, objective=objective, gradient=gradient, h0=math.nan,
                             h1=math.nan, global_l=math.nan, known_lower=None,
                             reference_x=None, reference_lower=math.nan, reference_upper=math.nan)
            with patch("auto_conditioned.lmo", side_effect=lmo) as oracle:
                result = solve(hidden, "AC_FW", max_iter=12, tolerance=1e-5)
            last = result.trace[-1]
            self.assertEqual(last["function_evals"], counts["f"])
            self.assertEqual(last["gradient_evals"], counts["g"])
            self.assertEqual(last["lmo_evals"], oracle.call_count)
            self.assertEqual(result.trace[0]["function_evals"], 2)
            self.assertEqual(result.trace[0]["gradient_evals"], 2)
            self.assertEqual(result.trace[0]["lmo_evals"], 2)
            audit(result, p)
            np.testing.assert_array_equal(result.x, solve(p, "AC_FW", 12, 1e-5).x)


if __name__ == "__main__":
    unittest.main()
