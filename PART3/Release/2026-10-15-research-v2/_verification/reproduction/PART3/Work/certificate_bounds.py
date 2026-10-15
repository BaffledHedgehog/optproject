"""Analysis utilities, never used as inputs to a benchmark solver.

The gap envelope uses only local suboptimality-dependent curvature. No
strong convexity or unique minimizer is needed. Exact-arithmetic formulas;
floating point evaluations here are numerical diagnostics, not interval proofs.
"""

import math


def gap_envelope(delta, h0, h1, diameter):
    if any(not math.isfinite(v) or v < 0 for v in (delta, h0, h1, diameter)):
        raise ValueError("Need finite nonnegative parameters")
    return max(2*delta, diameter*math.sqrt(2*(h0+h1*delta)*delta))


def certificate_threshold(epsilon, h0, h1, diameter):
    if (not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(diameter)
            or diameter <= 0 or any(not math.isfinite(v) or v < 0 for v in (h0, h1))):
        raise ValueError("Need positive accuracy/diameter and nonnegative curvature")
    if h0 == h1 == 0:
        return epsilon/2
    # Positive root of 2 D^2 (H0+H1*eta)*eta = epsilon^2,
    # rationalized to avoid subtracting nearly equal square roots.
    root = math.hypot(h0, math.sqrt(2*h1)*epsilon/diameter)
    eta = (epsilon/diameter)*(epsilon/diameter)/(h0+root)
    return min(epsilon/2, eta)


def sharp_gap_envelope(delta, h0, h1, diameter):
    """Invert the full quadratic decrease, including its full-step branch.

    Sharp for this decrease inequality, and attained by quadratics when
    H1=0. This does not claim the best possible H1>0 curvature envelope.
    """
    if any(not math.isfinite(v) or v < 0 for v in (delta, h0, h1, diameter)):
        raise ValueError("Need finite nonnegative parameters")
    curvature = diameter*diameter*(h0+h1*delta)
    if 2*delta <= curvature:
        return math.sqrt(2*curvature*delta)
    return delta+curvature/2


def sharp_certificate_threshold(epsilon, h0, h1, diameter):
    """Largest delta whose sharp quadratic-descent envelope is <= epsilon.

    The older conservative threshold is retained above for archived studies.
    Neither threshold is an online solver input.
    """
    if (not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(diameter)
            or diameter <= 0 or any(not math.isfinite(v) or v < 0 for v in (h0, h1))):
        raise ValueError("Need positive accuracy/diameter and nonnegative curvature")
    a, b = diameter*diameter*h0, diameter*diameter*h1
    if b < 2 and epsilon*(2-b) >= 2*a:
        return (epsilon-a/2)/(1+b/2)
    return certificate_threshold(epsilon, h0, h1, diameter)


def _a_coth(a):
    if a < 1e-4:
        return 1+a*a/3-a**4/45
    return a/math.tanh(a)


def _log_sinh_over_x(a):
    if a < 1e-4:
        return a*a/6-a**4/180
    if a < 20:
        return math.log(math.sinh(a)/a)
    return a-math.log(2*a)+math.log1p(-math.exp(-2*a))


def optimal_gap_envelope(delta, h0, h1, diameter):
    """Exact worst-case FW gap for the stated Hessian/value class and diameter.

    Derived from h'' <= D^2*(H0+H1*h), with one-dimensional attaining
    objectives. This is a mathematical sharpness claim, not a priority claim.
    Historical quadratic-envelope helpers remain unchanged above.
    """
    if any(not math.isfinite(v) or v < 0 for v in (delta, h0, h1, diameter)):
        raise ValueError("Need finite nonnegative parameters")
    if delta == 0 or diameter == 0:
        return 0.0
    if h1 == 0:
        return sharp_gap_envelope(delta, h0, h1, diameter)
    a = diameter*math.sqrt(h1)
    if a == 0:
        return sharp_gap_envelope(delta, h0, 0, diameter)
    if h0 > 0:
        # Critical delta = (H0/H1)*(cosh(a)-1), evaluated in logs.
        log_critical = math.log(h0)+2*math.log(diameter)-math.log(2)+2*_log_sinh_over_x(a/2)
        if math.log(delta) <= log_critical:
            return diameter*math.sqrt(delta)*math.sqrt(2*h0+h1*delta)
    return _a_coth(a)*delta+(diameter*diameter*h0)*(math.tanh(a/2)/a)


def optimal_certificate_threshold(epsilon, h0, h1, diameter):
    """Inverse of the exact pointwise gap envelope; offline analysis only."""
    if (not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(diameter)
            or diameter <= 0 or any(not math.isfinite(v) or v < 0 for v in (h0, h1))):
        raise ValueError("Need positive accuracy/diameter and nonnegative curvature")
    if h1 == 0:
        return sharp_certificate_threshold(epsilon, h0, 0, diameter)
    a = diameter*math.sqrt(h1)
    if a == 0:
        return sharp_certificate_threshold(epsilon, h0, 0, diameter)
    if h0 > 0:
        log_critical_gap = math.log(h0)+2*math.log(diameter)+_log_sinh_over_x(a)
        if math.log(epsilon) <= log_critical_gap:
            scaled = epsilon/diameter
            return scaled*(scaled/(h0+math.hypot(h0, math.sqrt(h1)*scaled)))
    offset = (diameter*diameter*h0)*(math.tanh(a/2)/a)
    return (epsilon-offset)/_a_coth(a)


def zero_base_finite_bound(u0, h1, diameter, margin, initial, initial_size):
    """Explicit exact-termination bound for AFW when H0=0 on a polytope.

    The margin is the positive atom separation of the exposed optimal face.
    None of these analysis quantities is passed to an optimizer.
    """
    if (any(not math.isfinite(v) or v <= 0 for v in (u0, diameter, margin, initial))
            or not math.isfinite(h1) or h1 < 0 or initial_size < 1):
        raise ValueError("Need positive U0, diameter, atom margin, M and support size; H1 >= 0")
    b = h1*diameter*diameter
    factor = _a_coth(math.sqrt(b))
    cap = max(initial, 2*h1*u0)
    threshold = min(u0/2, margin/(4*(1+b+factor)))
    local_cap = margin*margin/(8*diameter*diameter*threshold)
    local_lag = math.ceil(math.log2(max(1, cap/local_cap)))
    phases = math.ceil(math.log2(u0/threshold))
    potential = math.ceil(math.log2(max(1, initial*diameter*diameter/u0)))
    good_bound = potential+math.ceil((6+16*b)*phases)+local_lag
    updates = 2*good_bound+initial_size-1
    overhead = math.ceil(math.log2(max(1, 2*h1*u0/initial)))
    return {"threshold": threshold, "gap_factor": factor, "curvature_cap": cap,
            "local_curvature_threshold": local_cap, "local_lag_bound": local_lag,
            "phases": phases, "initial_potential": potential, "good_step_bound": good_bound,
            "update_bound": updates, "function_call_bound": 1+2*updates+overhead}
