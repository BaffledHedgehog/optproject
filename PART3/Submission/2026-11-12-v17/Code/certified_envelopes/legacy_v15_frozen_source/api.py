"""Certified scalar envelopes E, B and J for the v15 affine-loss model.

The returned interval encloses the MATHEMATICAL ENVELOPE, not an observed
optimization error.  Only ``envelope_upper`` transfers to an upper bound
on a source error under the assumptions in README_RU.md.

No optimizer, source objective, gradient, LMO, LP or stochastic oracle is
called.  There is no automatic retry, precision increase or resource-cap
increase.  A refusal retains the costs and partial diagnostic trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

from mpmath import __version__ as _mpmath_version
from mpmath.ctx_iv import MPIntervalContext

__version__ = "0.1.0"

ExactInput = str | Fraction | int
EnvelopeName = Literal["E", "B", "J"]


@dataclass(frozen=True)
class Model:
    """Exact model constants: m > 0, H0 >= m, H1 >= 0."""

    m: ExactInput
    H0: ExactInput
    H1: ExactInput


@dataclass(frozen=True)
class Precision:
    """Fixed interval precisions; value_dps defaults to work_dps.

    work_dps is used for bounds and switch signs.  value_dps is used for
    the two final loss evaluations at each retained root bracket.
    Separate values make the evaluation-slack obligation explicit.
    Neither value is changed during a call.
    """

    work_dps: int = 100
    value_dps: int | None = None


@dataclass(frozen=True)
class ResourceLimits:
    """Deterministic per-call caps, not complexity guarantees.

    A plan exceeding a cap is refused before its node loop.  In-loop
    limits also protect partial calls.  The scale cap bounds H1*g/m for
    the working gap; in a norm reduction that gap is P**2/m.
    """

    max_input_bits: int = 4096
    max_precision_digits: int = 1000
    max_scaled_radius_square: ExactInput = "10000"
    max_positive_nodes: int = 10000
    max_bisection_steps: int = 200000
    max_scalar_evaluations: int = 500000
    max_elementary_evaluations: int = 3000000
    max_endpoint_bits: int = 20000


def _json_exact(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_exact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_exact(v) for v in value]
    return value


@dataclass(frozen=True)
class Certificate:
    """A successful envelope enclosure, or a costed refusal.

    ``achieved_factor`` is envelope_upper / envelope_lower, exactly;
    its value for the exact zero envelope is one.  On refusal both
    bounds and achieved_factor are None.  Trace entries on refusal are
    diagnostics, not a completed envelope certificate.
    """

    status: Literal["certified", "refused"]
    envelope: EnvelopeName
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
        """JSON-ready exact strings for every rational certificate value."""
        return _json_exact({
            "status": self.status,
            "envelope": self.envelope,
            "envelope_lower": self.envelope_lower,
            "envelope_upper": self.envelope_upper,
            "achieved_factor": self.achieved_factor,
            "requested_factor": self.requested_factor,
            "internal_factor": self.internal_factor,
            "reason": self.reason,
            "message": self.message,
            "inputs": self.inputs,
            "precision": self.precision,
            "plan": self.plan,
            "work": self.work,
            "elementary_evaluations": self.elementary_evaluations,
            "trace": self.trace,
            "refusal_details": self.refusal_details,
            "bound_semantics": "The interval encloses the mathematical envelope; only envelope_upper bounds source error under the model assumptions.",
            "cost_semantics": "Named started evaluations, resolved/attempted sign queries and explicit maximum selections; not all arithmetic operations or bit cost.",
        })


class _Refusal(Exception):
    def __init__(self, reason: str, message: str, **details: Any):
        super().__init__(message)
        self.reason, self.message, self.details = reason, message, details


_WORK_KEYS = (
    "bound_evaluations", "node_coefficient_preparations",
    "switch_evaluations", "loss_evaluations", "scalar_evaluations_started",
    "bracket_sign_queries", "bracket_sign_decisions",
    "root_sign_queries", "root_sign_decisions", "bisection_updates",
    "sign_order_predicates", "sign_equality_predicates",
    "certificate_maximum_comparisons", "witness_maximum_comparisons",
    "rounding_maximum_comparisons", "derived_lower_maximum_comparisons",
    "budget_setup_maximum_comparisons", "budget_order_comparisons",
    "resource_metadata_maximum_comparisons",
    "budget_ceiling_operations", "rational_to_interval_conversions",
    "interval_endpoint_extractions", "resource_limit_checks",
    "validation_checks", "exact_formula_evaluations",
    "source_objective_calls", "source_gradient_calls", "full_LMO_calls",
    "LP_calls", "optimizer_updates", "stochastic_samples",
)


class _Kernel:
    """A private mpmath interval context; global mp/iv precision is untouched."""

    def __init__(self, engine: _Engine, dps: int):
        self.engine = engine
        self.ctx = MPIntervalContext()
        self.ctx.dps = dps

    def rational(self, value: Fraction | int):
        self.engine.work["rational_to_interval_conversions"] += 1
        q = Fraction(value)
        return self.ctx.mpf(q.numerator) / q.denominator

    def elementary(self, name: str, value):
        e = self.engine
        e.cap(sum(e.elementary.values()) + 1,
              e.limits.max_elementary_evaluations,
              "elementary_evaluations")
        e.elementary[name] += 1
        return getattr(self.ctx, name)(value)

    def pair(self, value):
        ep = self.elementary("exp", value)
        em = self.elementary("exp", -value)
        return (ep + em) / 2, (ep - em) / 2

    def endpoints(self, value) -> tuple[Fraction, Fraction]:
        result: list[Fraction] = []
        for sign, man, exp, bc in value._mpi_:
            self.engine.work["interval_endpoint_extractions"] += 1
            if bc < 0:
                raise _Refusal("nonfinite_interval",
                               "A scalar interval has a nonfinite endpoint.")
            # Check before allocating a potentially enormous power of two.
            positive_exp = self.engine.maximum(int(exp), 0, "resource_metadata_maximum_comparisons")
            negative_exp = self.engine.maximum(-int(exp), 0, "resource_metadata_maximum_comparisons")
            bits = self.engine.maximum(int(man).bit_length() + positive_exp,
                                       1 + negative_exp,
                                       "resource_metadata_maximum_comparisons")
            self.engine.cap(bits, self.engine.limits.max_endpoint_bits,
                            "interval_endpoint_bits")
            numerator = (-1 if sign else 1) * int(man)
            if exp >= 0:
                result.append(Fraction(numerator << int(exp)))
            else:
                result.append(Fraction(numerator, 1 << -int(exp)))
        return result[0], result[1]


class _Engine:
    def __init__(self, envelope: EnvelopeName, model: Model,
                 P: ExactInput | None, g: ExactInput | None,
                 factor: ExactInput, precision: Precision,
                 limits: ResourceLimits):
        self.envelope = envelope
        self.model_raw, self.P_raw, self.g_raw = model, P, g
        self.factor_raw = factor
        self.precision_raw, self.limits = precision, limits
        self.work = {key: 0 for key in _WORK_KEYS}
        self.elementary = {key: 0 for key in ("exp", "sin", "cos", "sqrt")}
        self.inputs: dict[str, Any] = {}
        self.precisions: dict[str, int] = {}
        self.plan: dict[str, Any] = {}
        self.trace: list[dict[str, Any]] = []
        self.factor: Fraction | None = None
        self.internal: Fraction | None = None
        self.kernels: dict[int, _Kernel] = {}

    def cap(self, actual: int | Fraction, limit: int | Fraction,
            name: str) -> None:
        self.work["resource_limit_checks"] += 1
        if actual > limit:
            raise _Refusal("resource_cap", f"The fixed {name} cap was exceeded.",
                           resource=name, required=actual, limit=limit)

    def require(self, condition: bool, reason: str, message: str,
                **details: Any) -> None:
        self.work["validation_checks"] += 1
        if not condition:
            raise _Refusal(reason, message, **details)

    def exact(self, value: Any, name: str) -> Fraction:
        self.require(type(value) in (int, str) or isinstance(value, Fraction),
                     "invalid_input", "Only int, rational string or Fraction inputs are accepted.",
                     parameter=name, received_type=type(value).__name__)
        try:
            q = Fraction(value)
        except (ValueError, ZeroDivisionError, TypeError) as exc:
            raise _Refusal("invalid_input", "The exact rational input is invalid.",
                           parameter=name, error=str(exc)) from exc
        bits = self.maximum(abs(q.numerator).bit_length(), q.denominator.bit_length(),
                            "resource_metadata_maximum_comparisons")
        self.cap(bits,
                 self.limits.max_input_bits, "input_bits")
        return q

    def maximum(self, a: Fraction | int, b: Fraction | int,
                counter: str):
        self.work[counter] += 1
        return a if a >= b else b

    def ceil_fraction(self, value: Fraction) -> int:
        self.work["budget_ceiling_operations"] += 1
        return -(-value.numerator // value.denominator)

    def ceil_log2_at_least_one(self, value: Fraction) -> int:
        self.work["budget_order_comparisons"] += 1
        if value <= 1:
            return 0
        k = value.numerator.bit_length() - value.denominator.bit_length()
        self.work["budget_order_comparisons"] += 1
        if value.numerator > (value.denominator << k):
            k += 1
        return k

    def begin_scalar(self, kind: str) -> None:
        self.cap(self.work["scalar_evaluations_started"] + 1,
                 self.limits.max_scalar_evaluations, "scalar_evaluations")
        self.work["scalar_evaluations_started"] += 1
        self.work[kind] += 1

    def kernel(self, dps: int) -> _Kernel:
        if dps not in self.kernels:
            self.kernels[dps] = _Kernel(self, dps)
        return self.kernels[dps]

    def sign(self, value, kernel: _Kernel,
             scope: Literal["bracket", "root"]) -> tuple[int | None, tuple[Fraction, Fraction]]:
        self.work[f"{scope}_sign_queries"] += 1
        lo, hi = kernel.endpoints(value)
        self.work["sign_order_predicates"] += 1
        if lo > 0:
            sign: int | None = 1
        else:
            self.work["sign_order_predicates"] += 1
            if hi < 0:
                sign = -1
            else:
                self.work["sign_equality_predicates"] += 1
                if lo == 0:
                    self.work["sign_equality_predicates"] += 1
                    sign = 0 if hi == 0 else None
                else:
                    sign = None
        if sign is not None:
            self.work[f"{scope}_sign_decisions"] += 1
        return sign, (lo, hi)

    def parse(self) -> None:
        self.require(isinstance(self.limits, ResourceLimits), "invalid_input",
                     "limits must be ResourceLimits.")
        for key in (
            "max_input_bits", "max_precision_digits", "max_positive_nodes",
            "max_bisection_steps", "max_scalar_evaluations",
            "max_elementary_evaluations", "max_endpoint_bits",
        ):
            value = getattr(self.limits, key)
            self.require(type(value) is int and value >= 0, "invalid_input",
                         "Resource limits must be nonnegative integers.", parameter=key)
        self.require(self.limits.max_input_bits >= 1 and self.limits.max_endpoint_bits >= 1,
                     "invalid_input", "Input and endpoint bit caps must be positive.")
        self.require(isinstance(self.model_raw, Model), "invalid_input", "model must be Model.")
        self.require(isinstance(self.precision_raw, Precision), "invalid_input",
                     "precision must be Precision.")
        dps = self.precision_raw.work_dps
        vdps = dps if self.precision_raw.value_dps is None else self.precision_raw.value_dps
        for name, value in (("work_dps", dps), ("value_dps", vdps)):
            self.require(type(value) is int and value >= 2, "invalid_input",
                         "Interval decimal precision must be an integer of at least two.", parameter=name)
            self.cap(value, self.limits.max_precision_digits, "precision_digits")
        self.precisions = {"work_dps": dps, "value_dps": vdps}
        self.m = self.exact(self.model_raw.m, "m")
        self.h0 = self.exact(self.model_raw.H0, "H0")
        self.h1 = self.exact(self.model_raw.H1, "H1")
        self.require(self.m > 0 and self.h0 >= self.m and self.h1 >= 0,
                     "invalid_input", "Required model conditions are m>0, H0>=m and H1>=0.")
        self.factor = self.exact(self.factor_raw, "factor")
        self.require(self.factor > 1, "invalid_input", "The requested factor must exceed one.")
        self.internal = (1 + self.factor) / 2
        self.P = self.exact(self.P_raw, "P") if self.envelope in ("E", "J") else None
        self.g = self.exact(self.g_raw, "g") if self.envelope in ("B", "J") else None
        self.require((self.P is None or self.P >= 0) and (self.g is None or self.g >= 0),
                     "invalid_input", "Norm and gap caps must be nonnegative.")
        self.scale_cap = self.exact(self.limits.max_scaled_radius_square, "max_scaled_radius_square")
        self.require(self.scale_cap >= 0, "invalid_input", "The scale-square cap must be nonnegative.")
        self.inputs = {"m": self.m, "H0": self.h0, "H1": self.h1}
        if self.P is not None:
            self.inputs["P"] = self.P
        if self.g is not None:
            self.inputs["g"] = self.g

    def bounds(self, gap: Fraction) -> tuple[Fraction, Fraction]:
        """Outward K*,L* bounds, adapted from the audited v15 kernel."""
        self.require(_mpmath_version == "1.3.0", "unsupported_dependency",
                     "The affine kernel is pinned to the reviewed mpmath 1.3.0 implementation.",
                     found_version=_mpmath_version)
        self.cap(self.h1 * gap / self.m, self.scale_cap, "scaled_radius_square")
        self.begin_scalar("bound_evaluations")
        k = self.kernel(self.precisions["work_dps"])
        mi, hi, si, gi = map(k.rational, (self.m, self.h0, self.h1, gap))
        T = k.elementary("sqrt", si * gi / mi)
        ch, sh = k.pair(T)
        A, C = sh / T, (ch - 1) / (T * T)
        forward = hi / si * (ch - 1) + gi * A
        curvature = hi + si * forward
        lipschitz = mi * (A + 2 * ch) + curvature * (C + 2 * A)
        Klo, Khi = k.endpoints(curvature)
        Llo, Lhi = k.endpoints(lipschitz)
        self.require(Khi > 0 and Lhi > 0, "invalid_interval_bound",
                     "Positive upper curvature and Lipschitz bounds were not obtained.")
        self.plan["Kstar_interval"] = [Klo, Khi]
        self.plan["Lstar_interval"] = [Llo, Lhi]
        return Khi, Lhi

    def prepare_node(self, kernel: _Kernel, gap: Fraction, rho: Fraction):
        self.work["node_coefficient_preparations"] += 1
        hi, si, gi, ri = map(kernel.rational, (self.h0, self.h1, gap, rho))
        c = kernel.elementary("sqrt", si * ri)
        a, q = hi / si, kernel.rational(gap - self.m * rho)
        bmax = c if self.h1 * rho <= 1 else kernel.rational(1)
        return c, a, q, bmax, gi

    def node(self, gap: Fraction, rho: Fraction, steps: int, index: int):
        self.require(0 < rho <= gap / self.m, "invalid_node",
                     "An affine node must have 0<rho<=g/m.")
        record: dict[str, Any] = {
            "index": index, "rho": rho, "planned_bisections": steps,
            "completed_bisections": 0, "alpha_bracket": [Fraction(0), Fraction(1)],
            "sign_history": "", "status": "started",
        }
        self.trace.append(record)
        work_kernel = self.kernel(self.precisions["work_dps"])
        work_coefficients = self.prepare_node(work_kernel, gap, rho)

        def evaluate(alpha: Fraction, loss: bool, kernel: _Kernel, coefficients):
            self.begin_scalar("loss_evaluations" if loss else "switch_evaluations")
            c, a, q, bmax, gi = coefficients
            b = kernel.rational(alpha) * bmax
            ch, sh = kernel.pair(c - b)
            cb, sb = kernel.elementary("cos", b), kernel.elementary("sin", b)
            if loss:
                return a * (ch * cb + sh * sb - 1) + (q / c) * (sh * cb + ch * sb)
            return a * c * (sh * cb - ch * sb) + q * (ch * cb - sh * sb) - gi

        lo, up = Fraction(0), Fraction(1)
        for alpha, expected, name in ((lo, 1, "left"), (up, -1, "right")):
            value = evaluate(alpha, False, work_kernel, work_coefficients)
            sign, interval = self.sign(value, work_kernel, "bracket")
            record[f"{name}_residual_interval"] = interval
            if sign is None:
                record["status"] = "unresolved_bracket_sign"
                raise _Refusal("unresolved_interval_sign",
                               "A prescribed bracket-endpoint interval sign is unresolved.",
                               stage="bracket", node=index, alpha=alpha, residual_interval=interval)
            self.require(sign == expected, "invalid_analytic_bracket",
                         "The prescribed analytic bracket did not have its required strict signs.",
                         node=index, alpha=alpha, residual_interval=interval)

        for _ in range(steps):
            self.cap(self.work["bisection_updates"] + 1,
                     self.limits.max_bisection_steps, "bisection_steps")
            mid = (lo + up) / 2
            value = evaluate(mid, False, work_kernel, work_coefficients)
            sign, interval = self.sign(value, work_kernel, "root")
            record["last_residual_interval"] = interval
            record["last_alpha"] = mid
            if sign is None:
                record["status"] = "unresolved_root_sign"
                raise _Refusal("unresolved_interval_sign",
                               "A switch-bisection interval sign is unresolved at the fixed precision.",
                               stage="root", node=index, alpha=mid, residual_interval=interval)
            if sign > 0:
                lo = mid
                record["left_residual_interval"] = interval
                record["sign_history"] += "+"
            elif sign < 0:
                up = mid
                record["right_residual_interval"] = interval
                record["sign_history"] += "-"
            else:
                lo = up = mid
                record["left_residual_interval"] = interval
                record["right_residual_interval"] = interval
                record["sign_history"] += "0"
            self.work["bisection_updates"] += 1
            record["completed_bisections"] += 1
            record["alpha_bracket"] = [lo, up]
            if sign == 0:
                break

        value_kernel = self.kernel(self.precisions["value_dps"])
        value_coefficients = work_coefficients if value_kernel is work_kernel else self.prepare_node(value_kernel, gap, rho)
        low_interval = value_kernel.endpoints(evaluate(up, True, value_kernel, value_coefficients))
        high_interval = value_kernel.endpoints(evaluate(lo, True, value_kernel, value_coefficients))
        lower, upper = low_interval[0], high_interval[1]
        rounding_width = high_interval[1] - high_interval[0]
        self.require(lower <= upper and rounding_width >= 0, "invalid_value_enclosure",
                     "The endpoint loss enclosure is not ordered.", node=index)
        record.update({
            "status": "enclosed", "loss_lower_endpoint_interval": low_interval,
            "loss_upper_endpoint_interval": high_interval,
            "fixed_radius_lower": lower, "fixed_radius_upper": upper,
            "upper_evaluation_rounding_width": rounding_width,
        })
        return lower, upper, rounding_width

    def preflight_nodes(self, positive_nodes: int, steps: int) -> None:
        self.plan["positive_nodes"] = positive_nodes
        self.plan["bisections_per_node"] = steps
        self.plan["planned_bisections"] = positive_nodes * steps
        self.cap(positive_nodes, self.limits.max_positive_nodes, "positive_nodes")
        self.cap(positive_nodes * steps, self.limits.max_bisection_steps, "bisection_steps")
        # Every node has two bracket switches and two endpoint losses.
        self.cap(self.work["scalar_evaluations_started"] + positive_nodes * (steps + 4),
                 self.limits.max_scalar_evaluations, "scalar_evaluations")
        coefficient_sqrts = 1 + (self.precisions["work_dps"] != self.precisions["value_dps"])
        planned_elementary = sum(self.elementary.values()) + positive_nodes * (coefficient_sqrts + 4 * (steps + 4))
        self.cap(planned_elementary, self.limits.max_elementary_evaluations,
                 "elementary_evaluations")

    def finish_affine(self, baseline: Fraction, witness: Fraction,
                      upper: Fraction, mesh: Fraction, inner: Fraction,
                      rounding: Fraction) -> tuple[Fraction, Fraction]:
        assert self.factor is not None and self.internal is not None
        reserve = (self.factor - self.internal) * baseline
        self.plan.update({
            "known_envelope_lower": baseline,
            "sampled_envelope_lower": witness,
            "mesh_allowance": mesh,
            "inner_root_allowance": inner,
            "maximum_upper_evaluation_rounding_width": rounding,
            "reserved_rounding_allowance": reserve,
        })
        self.require(mesh + inner <= (self.internal - 1) * baseline,
                     "invalid_scalar_budget", "The planned mesh/root budget exceeds its allowance.")
        self.require(rounding <= reserve, "insufficient_rounding_slack",
                     "The fixed value precision did not meet the reserved evaluation allowance.",
                     actual_width=rounding, reserved_width=reserve)
        allowance = mesh + inner + rounding
        derived_lower = upper - allowance
        lower = self.maximum(baseline, witness, "derived_lower_maximum_comparisons")
        lower = self.maximum(lower, derived_lower, "derived_lower_maximum_comparisons")
        self.plan["total_additive_allowance"] = allowance
        self.plan["upper_minus_allowance_lower"] = derived_lower
        self.require(0 < lower <= upper, "invalid_envelope_enclosure",
                     "The completed nonzero envelope enclosure is not positive and ordered.")
        self.require(upper <= self.factor * lower, "achieved_factor_exceeded",
                     "The exact returned envelope ratio exceeds the requested factor.")
        return lower, upper

    def affine_norm(self, P: Fraction, branch: str) -> tuple[Fraction, Fraction]:
        assert self.internal is not None
        gap, rho = P * P / self.m, P * P / (self.m * self.m)
        self.plan.update({"branch": branch, "working_gap": gap, "fixed_radius": rho})
        K, _ = self.bounds(gap)
        steps = self.ceil_log2_at_least_one(4 * K / (self.m * (self.internal - 1)))
        self.preflight_nodes(1, steps)
        lower, upper, rounding = self.node(gap, rho, steps, 0)
        inner = 2 * rho * K / (1 << steps)
        return self.finish_affine(gap / 2, lower, upper, Fraction(0), inner, rounding)

    def affine_gap(self, gap: Fraction) -> tuple[Fraction, Fraction]:
        assert self.internal is not None
        self.plan.update({"branch": "affine_gap", "working_gap": gap})
        K, L = self.bounds(gap)
        J = self.maximum(1, self.ceil_fraction(L / (self.m * (self.internal - 1))),
                         "budget_setup_maximum_comparisons")
        steps = self.ceil_log2_at_least_one(4 * K / (self.m * (self.internal - 1)))
        self.plan["grid_intervals"] = J
        self.plan["limiting_zero_radius_nodes"] = 1
        self.preflight_nodes(J, steps)
        lower = upper = gap
        rounding = Fraction(0)
        for j in range(1, J + 1):
            rho = gap * Fraction(j, J) / self.m
            lo, hi, width = self.node(gap, rho, steps, j)
            upper = self.maximum(upper, hi, "certificate_maximum_comparisons")
            lower = self.maximum(lower, lo, "witness_maximum_comparisons")
            rounding = self.maximum(rounding, width, "rounding_maximum_comparisons")
        mesh = L * gap / (2 * self.m * J)
        inner = 2 * (gap / self.m) * K / (1 << steps)
        return self.finish_affine(gap, lower, upper + mesh, mesh, inner, rounding)

    def affine_joint(self, P: Fraction, gap: Fraction) -> tuple[Fraction, Fraction]:
        assert self.internal is not None
        self.plan.update({"branch": "affine_joint_restricted", "working_gap": gap})
        K, L = self.bounds(gap)
        J = self.maximum(1, self.ceil_fraction(2 * L / (self.m * (self.internal - 1))),
                         "budget_setup_maximum_comparisons")
        steps = self.ceil_log2_at_least_one(8 * K / (self.m * (self.internal - 1)))
        a, b = gap * gap / (P * P), gap / self.m
        h = (b - a) / J
        self.plan.update({"grid_intervals": J, "radius_interval": [a, b],
                          "grid_spacing": h, "limiting_zero_radius_nodes": 0})
        self.preflight_nodes(J + 1, steps)
        lower, upper, rounding = gap / 2, None, Fraction(0)
        for j in range(J + 1):
            lo, hi, width = self.node(gap, a + j * h, steps, j)
            if upper is None:
                upper = hi
            else:
                upper = self.maximum(upper, hi, "certificate_maximum_comparisons")
            lower = self.maximum(lower, lo, "witness_maximum_comparisons")
            rounding = self.maximum(rounding, width, "rounding_maximum_comparisons")
        assert upper is not None
        mesh = L * h / 2
        inner = 2 * b * K / (1 << steps)
        return self.finish_affine(gap / 2, lower, upper + mesh, mesh, inner, rounding)

    def exact_envelope(self) -> Fraction:
        self.work["exact_formula_evaluations"] += 1
        beta = (self.h0 * self.h0 - 2 * self.m * self.h0 - self.m * self.m) / (4 * self.h0)
        kappa = 1 + beta / self.m
        self.plan.update({"branch": "constant_curvature", "beta": beta, "kappa": kappa})
        if self.envelope == "E":
            assert self.P is not None
            return kappa * self.P * self.P / self.m
        if self.envelope == "B":
            assert self.g is not None
            return self.g if beta < 0 else kappa * self.g
        assert self.P is not None and self.g is not None
        if self.g >= self.P * self.P / self.m:
            return kappa * self.P * self.P / self.m
        if beta < 0:
            return self.g + beta * self.g * self.g / (self.P * self.P)
        return kappa * self.g

    def calculate(self) -> tuple[Fraction, Fraction]:
        self.parse()
        zero = ((self.envelope == "E" and self.P == 0)
                or (self.envelope == "B" and self.g == 0)
                or (self.envelope == "J" and (self.P == 0 or self.g == 0)))
        if zero:
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
            return self.affine_gap(self.g)
        assert self.P is not None and self.g is not None
        if self.g >= self.P * self.P / self.m:
            return self.affine_norm(self.P, "affine_joint_norm_branch")
        return self.affine_joint(self.P, self.g)

    def run(self) -> Certificate:
        lower = upper = achieved = None
        reason = message = None
        details: dict[str, Any] = {}
        status: Literal["certified", "refused"] = "refused"
        try:
            lower, upper = self.calculate()
            achieved = Fraction(1) if upper == 0 else upper / lower
            status = "certified"
        except _Refusal as exc:
            reason, message, details = exc.reason, exc.message, exc.details
        except Exception as exc:
            reason = "internal_error"
            message = "An unexpected implementation exception prevented certification."
            details = {"exception_type": type(exc).__name__, "exception_message": str(exc)}
        if status == "refused" and self.trace and self.trace[-1]["status"] == "started":
            self.trace[-1]["status"] = "aborted"
        return Certificate(
            status=status, envelope=self.envelope,
            envelope_lower=lower if status == "certified" else None,
            envelope_upper=upper if status == "certified" else None,
            achieved_factor=achieved if status == "certified" else None,
            requested_factor=self.factor, internal_factor=self.internal,
            reason=reason, message=message, inputs=dict(self.inputs),
            precision=dict(self.precisions), plan=dict(self.plan),
            work=dict(self.work), elementary_evaluations=dict(self.elementary),
            trace=tuple(self.trace), refusal_details=dict(details),
        )


def certify_norm(P: ExactInput, *, model: Model, factor: ExactInput = "2",
                 precision: Precision | None = None,
                 limits: ResourceLimits | None = None) -> Certificate:
    """Enclose E(P); envelope_upper bounds loss when ||grad f(x)|| <= P."""
    return _Engine("E", model, P, None, factor,
                   Precision() if precision is None else precision,
                   ResourceLimits() if limits is None else limits).run()


def certify_gap(g: ExactInput, *, model: Model, factor: ExactInput = "2",
                precision: Precision | None = None,
                limits: ResourceLimits | None = None) -> Certificate:
    """Enclose B(g); envelope_upper bounds loss when the full FW gap <= g."""
    return _Engine("B", model, None, g, factor,
                   Precision() if precision is None else precision,
                   ResourceLimits() if limits is None else limits).run()


def certify_joint(P: ExactInput, g: ExactInput, *, model: Model,
                  factor: ExactInput = "2", precision: Precision | None = None,
                  limits: ResourceLimits | None = None) -> Certificate:
    """Enclose J(P,g), requiring both a norm cap P and a full-gap cap g."""
    return _Engine("J", model, P, g, factor,
                   Precision() if precision is None else precision,
                   ResourceLimits() if limits is None else limits).run()
