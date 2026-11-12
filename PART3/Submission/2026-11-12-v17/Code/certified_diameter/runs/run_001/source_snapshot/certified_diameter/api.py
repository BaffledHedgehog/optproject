"""Certified diameter envelopes, using immutable reviewed v16 primitives.

All observations are upper bounds. In particular, g>P*D is a valid
redundant gap cap, not an input error. The returned interval encloses
the sharp mathematical envelope; only its upper endpoint bounds a
source error under the model assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from fractions import Fraction
from typing import Any, Literal

from ._legacy_core import Certificate, ExactInput, _json_exact
from ._v16_api import Model, Precision, ResourceLimits, _V16Engine

__version__ = "0.1.0"


@dataclass(frozen=True)
class DiameterCertificate:
    """A relative enclosure of J_D or E_D, or a costed refusal.

    The lower endpoint is not a lower bound on an observed source error.
    Refusals have no completed bounds or achieved factor. Partial trace
    entries are diagnostic only. Exact zero has achieved factor one.
    """

    status: Literal["certified", "refused"]
    envelope: Literal["J_D", "E_D"]
    envelope_lower: Fraction | None
    envelope_upper: Fraction | None
    achieved_factor: Fraction | None
    requested_factor: Fraction | None
    internal_factor: Fraction | None
    reason: str | None
    message: str | None
    inputs: dict[str, Any]
    precision: dict[str, int]
    plan: dict[str, Any]
    work: dict[str, int]
    elementary_evaluations: dict[str, int]
    trace: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    refusal_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        data.update({
            "bound_semantics": "The interval encloses the sharp diameter-restricted mathematical envelope. Only envelope_upper bounds source error under the model assumptions and input caps.",
            "cost_semantics": "Named started scalar and elementary evaluations, sign decisions and explicit comparisons; not all arithmetic operations or bit cost.",
        })
        return _json_exact(data)


class _DiameterEngine(_V16Engine):
    def __init__(self, kind: Literal["joint", "norm"], model: Model,
                 P: ExactInput, g: ExactInput | None, D: ExactInput,
                 factor: ExactInput, precision: Precision,
                 limits: ResourceLimits):
        super().__init__("J" if kind == "joint" else "E",
                         model, P, g, factor, precision, limits)
        self.kind = kind
        self.D_raw = D
        self.public_envelope = "J_D" if kind == "joint" else "E_D"
        self.work.update({
            "diameter_cap_minimum_comparisons": 0,
            "diameter_radius_minimum_comparisons": 0,
            "diameter_collapse_comparisons": 0,
            "diameter_exact_endpoint_comparisons": 0,
        })

    def parse(self) -> None:
        super().parse()
        self.D = self.exact(self.D_raw, "D")
        self.inputs["D"] = self.D
        self.require(self.D > 0, "invalid_input",
                     "The supplied diameter bound D must be strictly positive.",
                     parameter="D", value=self.D)
        assert self.P is not None
        if self.kind == "norm":
            # This is a derived valid gap cap, not another observation.
            self.g = self.P * self.D
            self.inputs["derived_gap_cap"] = self.g
        self.plan["envelope_family"] = "diameter_restricted"

    def minimum(self, a: Fraction, b: Fraction, counter: str) -> Fraction:
        self.work[counter] += 1
        return a if a <= b else b

    def geometry(self) -> tuple[Fraction, Fraction, Fraction]:
        assert self.P is not None and self.g is not None
        norm_gap = self.P * self.P / self.m
        diameter_gap = self.P * self.D
        h = self.minimum(self.g, norm_gap, "diameter_cap_minimum_comparisons")
        h = self.minimum(h, diameter_gap, "diameter_cap_minimum_comparisons")
        lower = h * h / (self.P * self.P)
        upper = self.minimum(h / self.m, self.D * self.D,
                             "diameter_radius_minimum_comparisons")
        self.require(0 < lower <= upper, "invalid_diameter_interval",
                     "The derived positive radius interval is not ordered.")
        self.plan.update({
            "working_gap": h,
            "working_gap_semantics": "Effective radial parameter h; not a claim that the observed full gap is at most P^2/m.",
            "norm_gap_cap": norm_gap, "diameter_gap_cap": diameter_gap,
            "radius_interval": [lower, upper],
            "limiting_zero_radius_nodes": 0,
        })
        return h, lower, upper

    def exact_diameter(self, h: Fraction, lower: Fraction,
                       upper: Fraction) -> tuple[Fraction, Fraction]:
        self.work["exact_formula_evaluations"] += 1
        beta = (self.h0 * self.h0 - 2 * self.m * self.h0 - self.m * self.m) / (4 * self.h0)
        self.work["diameter_exact_endpoint_comparisons"] += 1
        radius = upper if beta >= 0 else lower
        value = h + beta * radius
        self.plan.update({
            "branch": "diameter_constant_curvature", "beta": beta,
            "fixed_radius": radius, "positive_nodes": 0,
            "grid_intervals": 0, "planned_bisections": 0,
        })
        self.require(value > 0, "invalid_exact_diameter_value",
                     "A nonzero exact envelope must be positive.")
        return value, value

    def one_radius(self, h: Fraction, rho: Fraction,
                   branch: str) -> tuple[Fraction, Fraction]:
        assert self.internal is not None
        self.plan.update({
            "branch": branch, "fixed_radius": rho, "grid_intervals": 0,
        })
        K, _ = self.bounds(h)
        # With rho<=h/m, this spends at most (internal-1)*h/2
        # on the root and no allowance on a radius mesh.
        steps = self.ceil_log2_at_least_one(
            4 * K / (self.m * (self.internal - 1)))
        self.preflight_nodes(1, steps)
        lower, upper, rounding = self.node(h, rho, steps, 0)
        inner = 2 * rho * K / (1 << steps)
        return self.finish_affine(h / 2, lower, upper,
                                  Fraction(0), inner, rounding)

    def radius_grid(self, h: Fraction, a: Fraction,
                    b: Fraction) -> tuple[Fraction, Fraction]:
        assert self.internal is not None
        width = b - a
        self.require(width > 0, "invalid_diameter_grid",
                     "A radius grid requires a positive interval width.")
        K, L = self.bounds(h)
        intervals = self.maximum(1, self.ceil_fraction(
            2 * L * width / ((self.internal - 1) * h)),
            "budget_setup_maximum_comparisons")
        steps = self.ceil_log2_at_least_one(
            8 * K / (self.m * (self.internal - 1)))
        spacing = width / intervals
        self.plan.update({
            "branch": "diameter_restricted_grid",
            "grid_intervals": intervals, "grid_spacing": spacing,
        })
        self.preflight_nodes(intervals + 1, steps)
        lower, upper, rounding = h / 2, None, Fraction(0)
        for index in range(intervals + 1):
            lo, hi, value_width = self.node(h, a + index * spacing, steps, index)
            upper = hi if upper is None else self.maximum(
                upper, hi, "certificate_maximum_comparisons")
            lower = self.maximum(lower, lo, "witness_maximum_comparisons")
            rounding = self.maximum(rounding, value_width,
                                     "rounding_maximum_comparisons")
        assert upper is not None
        mesh = L * spacing / 2
        inner = 2 * b * K / (1 << steps)
        return self.finish_affine(h / 2, lower, upper + mesh,
                                  mesh, inner, rounding)

    def calculate(self) -> tuple[Fraction, Fraction]:
        self.parse()
        assert self.P is not None and self.g is not None
        if self.P == 0 or self.g == 0:
            self.work["exact_formula_evaluations"] += 1
            self.plan.update({"branch": "diameter_zero", "positive_nodes": 0,
                              "grid_intervals": 0, "planned_bisections": 0})
            return Fraction(0), Fraction(0)
        h, lower, upper = self.geometry()
        if self.h1 == 0:
            return self.exact_diameter(h, lower, upper)
        self.work["diameter_collapse_comparisons"] += 1
        if lower == upper:
            return self.one_radius(h, lower, "diameter_collapsed_node")
        allowed, _, _ = self.regime(h)
        if allowed:
            return self.one_radius(h, lower, "diameter_regime_one_node")
        return self.radius_grid(h, lower, upper)

    def run(self) -> DiameterCertificate:
        raw = super().run()
        data = {item.name: getattr(raw, item.name) for item in fields(Certificate)}
        data["envelope"] = self.public_envelope
        return DiameterCertificate(**data)


def certify_joint_diameter(*, g: ExactInput, P: ExactInput, D: ExactInput,
                           model: Model, factor: ExactInput = "2",
                           precision: Precision | None = None,
                           limits: ResourceLimits | None = None) -> DiameterCertificate:
    """Enclose J_D(g,P) for full-gap/norm caps and diameter at most D>0.

    All numerical data use int, Fraction or exact rational strings.
    g>P*D is allowed because inputs are caps. Invalid numerical input,
    unresolved interval signs and exceeded resources produce refusals.
    Python signature errors remain ordinary TypeError.
    """
    return _DiameterEngine(
        "joint", model, P, g, D, factor,
        Precision() if precision is None else precision,
        ResourceLimits() if limits is None else limits).run()


def certify_norm_diameter(P: ExactInput, D: ExactInput, *, model: Model,
                          factor: ExactInput = "2",
                          precision: Precision | None = None,
                          limits: ResourceLimits | None = None) -> DiameterCertificate:
    """Enclose E_D(P); P*D is a derived gap cap, not an observation."""
    return _DiameterEngine(
        "norm", model, P, None, D, factor,
        Precision() if precision is None else precision,
        ResourceLimits() if limits is None else limits).run()
