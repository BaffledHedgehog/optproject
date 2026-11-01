import Std

/-!
  Limited reconstruction of exact algebraic checks for PART3 Sections 5 and 6.

  This file proves polynomial identities underlying the normalized atom-weight
  update after denominators are cleared, and closed integer cross-multiplication
  checks for the displayed rational constants and examples. It does not prove
  the analytic or algorithmic theorems in the manuscript.
-/

namespace PART3.Recovery

section NormalizedWeights

variable {R : Type} [Lean.Grind.CommRing R]

/-- Clearing the common denominator `(1-a) * t` for a nonselected atom:
its weight and the residual mass are both multiplied by `(1-a)`. -/
theorem normalized_other_atom_clear
    (a w t : R) : ((1 - a) * w) * t = w * ((1 - a) * t) := by
  grind

/-- Clearing the common denominator for the selected atom, whose raw weight
changes from `w` to `(1-a) * w + a`. This is only the algebraic update
identity; positivity and nonzero-denominator conditions are not asserted. -/
theorem normalized_selected_atom_clear
    (a w t : R) :
    (((1 - a) * w + a) * t) = w * ((1 - a) * t) + a * t := by
  grind

end NormalizedWeights

/-! The following equalities encode exact rationals by numerator and positive
denominator, then clear denominators. They are closed kernel-checked `decide`
proofs over Nat; there is no floating-point arithmetic.
-/

section EndpointExampleConstants

/-- The gap coefficient in the half-step trajectory for `c = 81/476`:
`(119/36) * (81/476) = 9/16`. -/
theorem endpoint_gap_coefficient_crossmul :
    119 * 81 * 16 = 9 * 36 * 476 := by
  decide

/-- The cubic coefficient of `f(t, plus-or-minus 1/3)` is `333/1904`. -/
theorem endpoint_value_coefficient_crossmul :
    37 * 81 * 1904 = 333 * 36 * 476 := by
  decide

/-- The cubic coefficient in the acceptance slack is `189/15232`:
`7*(333/1904)/8 = 9/64 + 189/15232`. -/
theorem endpoint_acceptance_slack_crossmul :
    7 * 333 * 64 = 9 * 15232 + 189 * 64 := by
  decide

end EndpointExampleConstants

section IntegratedVertexExample

/-- At `x = 9/20`, the derivative of `x + x^3` is `643/400`. -/
theorem vertex_gradient_numerator : 400 + 3 * 9 * 9 = 643 := by
  decide

/-- The exact FW gap at `x = 9/20` for `f(x) = x + x^3` is `5787/8000`. -/
theorem vertex_gap_numerator : 9 * 643 = 5787 := by
  decide

/-- The integrated-profile sufficient condition passes:
`(643/400)^2 > 3 * (5787/8000)^2`. -/
theorem integrated_vertex_certificate_crossmul :
    643 * 643 * 8000 * 8000 > 3 * 5787 * 5787 * 400 * 400 := by
  decide

/-- The endpoint-profile sufficient condition fails:
`(643/400)^2 < 6 * (5787/8000)^2`. -/
theorem endpoint_profile_certificate_crossmul :
    643 * 643 * 8000 * 8000 < 6 * 5787 * 5787 * 400 * 400 := by
  decide

end IntegratedVertexExample

end PART3.Recovery

#print axioms PART3.Recovery.normalized_other_atom_clear
#print axioms PART3.Recovery.normalized_selected_atom_clear
#print axioms PART3.Recovery.endpoint_gap_coefficient_crossmul
#print axioms PART3.Recovery.endpoint_value_coefficient_crossmul
#print axioms PART3.Recovery.endpoint_acceptance_slack_crossmul
#print axioms PART3.Recovery.vertex_gradient_numerator
#print axioms PART3.Recovery.vertex_gap_numerator
#print axioms PART3.Recovery.integrated_vertex_certificate_crossmul
#print axioms PART3.Recovery.endpoint_profile_certificate_crossmul
