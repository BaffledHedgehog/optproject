"""Certified v16 envelopes and a separate algebraic source-error bound.

Joint public calls are keyword-only: ``certify_joint(g=..., P=...)``.
The immutable private kernel is captured from v15; no live legacy file
is imported.  A scalar enclosure and a source-error upper bound have
different result types and deliberately different guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

from ._legacy_core import (
    Certificate, ExactInput, Model, Precision, ResourceLimits,
    _Engine, _Refusal, _json_exact, _mpmath_version,
)

__version__ = "0.2.0"


@dataclass(frozen=True)
class UpperBound:
    """A source-error upper bound, with no relative-envelope guarantee.

    ``source_error_upper`` is an outward rational bound under the model
    assumptions and the supplied norm/gap caps.  It is NOT an estimate
    of the observed error, an envelope lower bound, or a multiplicative
    approximation of the sharp envelope.  No factor is requested here.
    """

    status: Literal["certified", "refused"]
    source_error_upper: Fraction | None
    bound_type: str
    reason: str | None
    message: str | None
    inputs: dict[str, Any]
    precision: dict[str, int]
    plan: dict[str, Any]
    work: dict[str, int]
    elementary_evaluations: dict[str, int]
    refusal_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_exact({
            "status": self.status,
            "source_error_upper": self.source_error_upper,
            "bound_type": self.bound_type,
            "reason": self.reason,
            "message": self.message,
            "inputs": self.inputs,
            "precision": self.precision,
            "plan": self.plan,
            "work": self.work,
            "elementary_evaluations": self.elementary_evaluations,
            "refusal_details": self.refusal_details,
            "bound_semantics": "Only a source-error upper bound under the model assumptions and input caps; no lower error bound or requested relative-envelope factor.",
            "cost_semantics": "Named started evaluations and comparisons; not all arithmetic operations or bit cost.",
        })


class _V16Engine(_Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.work.update({
            "regime_evaluations": 0,
            "regime_sign_queries": 0,
            "regime_nonpositive_decisions": 0,
            "regime_positive_decisions": 0,
            "regime_unresolved_queries": 0,
            "regime_order_predicates": 0,
            "working_gap_minimum_comparisons": 0,
            "algebraic_bound_evaluations": 0,
        })

    def zero_input(self) -> bool:
        return ((self.envelope == "E" and self.P == 0)
                or (self.envelope == "B" and self.g == 0)
                or (self.envelope == "J" and (self.P == 0 or self.g == 0)))

    def working_gap(self) -> Fraction:
        assert self.P is not None and self.g is not None
        self.work["working_gap_minimum_comparisons"] += 1
        norm_gap = self.P * self.P / self.m
        return self.g if self.g <= norm_gap else norm_gap

    def regime(self, gap: Fraction) -> tuple[bool, Fraction, Fraction]:
        """Certify S(gap)<=0 or S(gap)>0; refuse an unresolved sign.

        The dependency and scale guard are here as well as in bounds():
        the exact-gap and algebraic branches may return before bounds().
        """
        self.require(self.h1 > 0 and gap > 0, "invalid_regime_input",
                     "The affine regime test needs H1>0 and a positive working gap.")
        self.require(_mpmath_version == "1.3.0", "unsupported_dependency",
                     "The interval regime kernel is pinned to mpmath 1.3.0.",
                     found_version=_mpmath_version)
        self.cap(self.h1 * gap / self.m, self.scale_cap, "scaled_radius_square")
        self.begin_scalar("regime_evaluations")
        k = self.kernel(self.precisions["work_dps"])
        m, a, b = map(k.rational, (self.m, self.h0, self.h1 * gap))
        tau = (a + m + b / 2) / (a + k.elementary(
            "sqrt", a * a + b * (a + m) + b * b / 2))
        coefficient = a * (k.rational(Fraction(1, 2)) - tau * tau) \
            + b / 3 * (1 - 2 * tau**3)
        tlo, thi = k.endpoints(tau)
        slo, shi = k.endpoints(coefficient)
        self.plan["regime"] = {
            "working_gap": gap, "tau_interval": [tlo, thi],
            "S_interval": [slo, shi], "decision": "pending",
        }
        self.work["regime_sign_queries"] += 1
        self.work["regime_order_predicates"] += 1
        if shi <= 0:
            self.work["regime_nonpositive_decisions"] += 1
            self.plan["regime"]["decision"] = "nonpositive"
            return True, slo, shi
        self.work["regime_order_predicates"] += 1
        if slo > 0:
            self.work["regime_positive_decisions"] += 1
            self.plan["regime"]["decision"] = "positive"
            return False, slo, shi
        self.work["regime_unresolved_queries"] += 1
        self.plan["regime"]["decision"] = "unresolved"
        raise _Refusal(
            "unresolved_regime_sign",
            "The fixed-precision S interval certifies neither S<=0 nor S>0.",
            S_interval=[slo, shi], tau_interval=[tlo, thi],
        )

    def one_node_joint(self, P: Fraction, gap: Fraction) -> tuple[Fraction, Fraction]:
        """Exact joint envelope at its left radius in the certified regime."""
        assert self.internal is not None
        rho = gap * gap / (P * P)
        self.plan.update({
            "branch": "affine_joint_regime_one_node", "working_gap": gap,
            "fixed_radius": rho, "radius_interval": [rho, gap / self.m],
            "grid_intervals": 0, "limiting_zero_radius_nodes": 0,
        })
        K, _ = self.bounds(gap)
        # No mesh error: the full internal allowance is available for
        # the root.  rho<=gap/m and J(g,P)>=gap/2 justify this budget.
        steps = self.ceil_log2_at_least_one(
            4 * K / (self.m * (self.internal - 1)))
        self.preflight_nodes(1, steps)
        lower, upper, rounding = self.node(gap, rho, steps, 0)
        inner = 2 * rho * K / (1 << steps)
        return self.finish_affine(gap / 2, lower, upper, Fraction(0), inner, rounding)

    def affine_joint(self, P: Fraction, gap: Fraction) -> tuple[Fraction, Fraction]:
        """General fallback, using the actual v16 restricted interval width."""
        assert self.internal is not None
        K, L = self.bounds(gap)
        a, b = gap * gap / (P * P), gap / self.m
        width = b - a
        self.require(width > 0, "invalid_joint_branch",
                     "The restricted fallback needs a positive radius interval.")
        J = self.maximum(1, self.ceil_fraction(
            2 * L * width / ((self.internal - 1) * gap)),
            "budget_setup_maximum_comparisons")
        steps = self.ceil_log2_at_least_one(
            8 * K / (self.m * (self.internal - 1)))
        spacing = width / J
        self.plan.update({
            "branch": "affine_joint_restricted_grid", "working_gap": gap,
            "grid_intervals": J, "radius_interval": [a, b],
            "grid_spacing": spacing, "limiting_zero_radius_nodes": 0,
        })
        self.preflight_nodes(J + 1, steps)
        lower, upper, rounding = gap / 2, None, Fraction(0)
        for j in range(J + 1):
            lo, hi, value_width = self.node(gap, a + j * spacing, steps, j)
            upper = hi if upper is None else self.maximum(
                upper, hi, "certificate_maximum_comparisons")
            lower = self.maximum(lower, lo, "witness_maximum_comparisons")
            rounding = self.maximum(rounding, value_width, "rounding_maximum_comparisons")
        assert upper is not None
        mesh = L * spacing / 2
        inner = 2 * b * K / (1 << steps)
        return self.finish_affine(gap / 2, lower, upper + mesh, mesh, inner, rounding)

    def calculate(self) -> tuple[Fraction, Fraction]:
        self.parse()
        if self.zero_input():
            self.plan["branch"] = "zero"
            self.work["exact_formula_evaluations"] += 1
            return Fraction(0), Fraction(0)
        if self.h1 == 0:
            value = self.exact_envelope()
            return value, value
        if self.envelope == "E":
            assert self.P is not None
            return self.affine_norm(self.P, "affine_norm")
        if self.envelope == "B":
            assert self.g is not None
            allowed, _, _ = self.regime(self.g)
            if allowed:
                self.plan.update({"branch": "affine_gap_regime_exact",
                                  "working_gap": self.g, "positive_nodes": 0,
                                  "grid_intervals": 0, "planned_bisections": 0})
                self.work["exact_formula_evaluations"] += 1
                return self.g, self.g
            return self.affine_gap(self.g)
        assert self.P is not None and self.g is not None
        if self.g >= self.P * self.P / self.m:
            return self.affine_norm(self.P, "affine_joint_norm_branch")
        allowed, _, _ = self.regime(self.g)
        if allowed:
            return self.one_node_joint(self.P, self.g)
        return self.affine_joint(self.P, self.g)

    def calculate_algebraic(self) -> Fraction:
        """Only an upper source-error bound; no relative-factor contract."""
        self.parse()
        if self.zero_input():
            self.plan["branch"] = "algebraic_zero"
            self.work["exact_formula_evaluations"] += 1
            return Fraction(0)
        if self.h1 == 0:
            value = self.exact_envelope()
            self.plan["branch"] = "algebraic_constant_curvature_joint"
            return value
        assert self.P is not None and self.g is not None
        h = self.working_gap()
        allowed, slo, shi = self.regime(h)
        self.require(allowed, "outside_algebraic_regime",
                     "The algebraic shortcut requires a certified nonpositive S(h).",
                     S_interval=[slo, shi], working_gap=h)
        self.begin_scalar("algebraic_bound_evaluations")
        rho = h * h / (self.P * self.P)
        # S_upper is an exact rational outward endpoint.  Applying the
        # final affine formula with Fraction arithmetic adds no rounding.
        upper = h + rho * shi
        self.plan.update({
            "branch": "algebraic_joint_regime", "working_gap": h,
            "left_radius": rho, "used_S_upper": shi,
            "positive_nodes": 0, "grid_intervals": 0,
            "planned_bisections": 0,
            "guarantee": "source_error_upper_only_no_relative_factor",
        })
        self.require(h / 2 <= upper <= h, "invalid_algebraic_bound",
                     "The outward algebraic bound is outside its proved range.")
        return upper

    def run_algebraic(self) -> UpperBound:
        upper = None
        status: Literal["certified", "refused"] = "refused"
        reason = message = None
        details: dict[str, Any] = {}
        try:
            upper = self.calculate_algebraic()
            status = "certified"
        except _Refusal as exc:
            reason, message, details = exc.reason, exc.message, exc.details
        except Exception as exc:
            reason = "internal_error"
            message = "An unexpected implementation exception prevented certification."
            details = {"exception_type": type(exc).__name__, "exception_message": str(exc)}
        return UpperBound(
            status=status, source_error_upper=upper if status == "certified" else None,
            bound_type="joint_algebraic", reason=reason, message=message,
            inputs=dict(self.inputs), precision=dict(self.precisions),
            plan=dict(self.plan), work=dict(self.work),
            elementary_evaluations=dict(self.elementary), refusal_details=dict(details),
        )


def certify_norm(P: ExactInput, *, model: Model, factor: ExactInput = "2",
                 precision: Precision | None = None,
                 limits: ResourceLimits | None = None) -> Certificate:
    """Relative enclosure of E(P); only envelope_upper bounds source error."""
    return _V16Engine("E", model, P, None, factor,
                      Precision() if precision is None else precision,
                      ResourceLimits() if limits is None else limits).run()


def certify_gap(g: ExactInput, *, model: Model, factor: ExactInput = "2",
                precision: Precision | None = None,
                limits: ResourceLimits | None = None) -> Certificate:
    """Relative enclosure of B(g), exact when S(g)<=0 is certified."""
    return _V16Engine("B", model, None, g, factor,
                      Precision() if precision is None else precision,
                      ResourceLimits() if limits is None else limits).run()


def certify_joint(*, g: ExactInput, P: ExactInput, model: Model,
                  factor: ExactInput = "2", precision: Precision | None = None,
                  limits: ResourceLimits | None = None) -> Certificate:
    """Relative enclosure of mathematical J(g,P); joint arguments are named."""
    return _V16Engine("J", model, P, g, factor,
                      Precision() if precision is None else precision,
                      ResourceLimits() if limits is None else limits).run()


def algebraic_joint_bound(*, g: ExactInput, P: ExactInput, model: Model,
                          precision: Precision | None = None,
                          limits: ResourceLimits | None = None) -> UpperBound:
    """Upper error bound, with no requested multiplicative envelope factor.

    For H1>0 it requires the certified sign S(min(g,P**2/m))<=0.
    An unresolved or positive sign refuses this optional shortcut.
    No switch root or radius grid is evaluated.
    """
    # The inherited parser has a factor field.  Its fixed internal value
    # is not used by calculate_algebraic and is absent from UpperBound.
    return _V16Engine("J", model, P, g, "2",
                      Precision() if precision is None else precision,
                      ResourceLimits() if limits is None else limits).run_algebraic()
