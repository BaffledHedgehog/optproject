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
