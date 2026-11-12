"""Independent algebra and stress checks for boundary_math_{en,ru}.tex.

The algebra checks differentiate the examples and integrate the radial
majorant. The numerical checks use exact box LMOs and known minimizers,
including large boundary gradients, negative curvature, and extra normal
directions. These checks supplement the proofs; they do not establish them.
"""

import itertools
import json
import math
import random

import sympy as sp


def verify_symbolic_identities():
    u, v, t = sp.symbols("u v t", real=True)
    m, a, ell = sp.symbols("m a ell", positive=True)
    f = u**2 / 2 + t / sp.pi * (1 - sp.cos(sp.pi * u))
    f += sp.sqrt(1 + v**2) + (sp.pi * t + 1) * v
    grad = [sp.diff(f, z) for z in (u, v)]
    hess = sp.hessian(f, (u, v))
    assert sp.simplify(grad[0] - (u + t * sp.sin(sp.pi * u))) == 0
    assert sp.simplify(grad[1] - (sp.pi * t + 1 + v / sp.sqrt(1 + v**2))) == 0
    assert sp.simplify(hess[0, 0] - (1 + sp.pi * t * sp.cos(sp.pi * u))) == 0
    assert sp.simplify(hess[1, 1] - (1 + v**2) ** sp.Rational(-3, 2)) == 0
    assert hess[0, 1] == 0
    loss = f.subs({u: 1, v: 0}) - f.subs({u: 0, v: 0})
    assert sp.simplify(loss - sp.Rational(1, 2) - 2 * t / sp.pi) == 0
    assert grad[0].subs({u: 1, v: 0}) == 1
    shifted = sum((g - g.subs({u: 0, v: 0})) * z for g, z in zip(grad, (u, v)))
    assert sp.simplify(shifted - u**2 - t * u * sp.sin(sp.pi * u) - v**2 / sp.sqrt(1 + v**2)) == 0

    # A distinct polynomial example checks the same obstruction exactly.
    cubic = m * (u**2 + v**2) / 2 + t * (a * u**2 / 2 - u**3 / 3) + t * a * v / ell
    cg = [sp.diff(cubic, z) for z in (u, v)]
    cs = sum((g - g.subs({u: 0, v: 0})) * z for g, z in zip(cg, (u, v)))
    assert sp.simplify(cs - m * (u**2 + v**2) - t * u**2 * (a - u)) == 0
    assert sp.simplify(cubic.subs({u: a, v: 0}) - m * a**2 / 2 - t * a**3 / 6) == 0

    # Derive, rather than substitute, the tent integral coefficient.
    K, r, p, s = sp.symbols("K r p s", positive=True)
    switch = (K + m) * r / (2 * K)
    integral = sp.integrate(p - m * r + K * s, (s, 0, switch))
    integral += sp.integrate(p + K * (r - s), (s, switch, r))
    coefficient = sp.factor((integral - p * r) / r**2)
    assert sp.simplify(coefficient - (K**2 - 2 * m * K - m**2) / (4 * K)) == 0
    return {"symbolic_identities": "passed", "tent_coefficient": str(coefficient)}


def box_gap(x, gradient, widths):
    # Enumerating vertices independently checks the exact LMO value.
    return max(
        sum(g * (xi - si) for g, xi, si in zip(gradient, x, vertex))
        for vertex in itertools.product(*[(0.0, width) for width in widths])
    )


def certificate(x, gradient, widths, m, L0, L1, curvature_cap=None):
    gap = max(0.0, box_gap(x, gradient, widths))
    norm = math.sqrt(sum(g * g for g in gradient))
    diameter = math.sqrt(sum(w * w for w in widths))
    radius = min(diameter, math.sqrt(gap / m), norm / m)
    curvature = max(m, (L0 + L1 * norm) * math.exp(L1 * radius))
    if curvature_cap is not None:
        curvature = min(curvature, curvature_cap)
    beta = max(0.0, (curvature * curvature - 2 * m * curvature - m * m) / (4 * curvature))
    return gap + beta * radius * radius, gap, norm


def stress_checks():
    rng = random.Random(64380)
    checks = 0
    min_slack = math.inf
    # Polynomial family, including multiple unconstrained normal components.
    for dimension in (2, 3, 7):
        for m, L1, bump in itertools.product((0.25, 2.0), (0.2, 1.0), (0.0, 1.0, 1000.0)):
            widths = [0.7] + [0.4 + 0.1 * j for j in range(dimension - 1)]
            a = widths[0]
            normal = bump * a / L1
            points = [[0.0] * dimension, [a] + [0.0] * (dimension - 1)]
            points += [[rng.random() * w for w in widths] for _ in range(6)]
            for x in points:
                loss = m * sum(z * z for z in x) / 2
                loss += bump * (a * x[0] ** 2 / 2 - x[0] ** 3 / 3)
                loss += normal * sum(x[1:])
                gradient = [m * x[0] + bump * x[0] * (a - x[0])]
                gradient += [m * z + normal for z in x[1:]]
                hessian_norm = max(m, abs(m + bump * (a - 2 * x[0])))
                norm = math.sqrt(sum(z * z for z in gradient))
                assert hessian_norm <= m + L1 * norm + 1e-10
                for cap in (None, m + bump * a):
                    bound, gap, _ = certificate(x, gradient, widths, m, m, L1, cap)
                    assert bound + 1e-8 * max(1.0, loss) >= loss
                    assert m * sum(z * z for z in x) <= gap + 1e-10
                    min_slack = min(min_slack, bound - loss)
                    checks += 1

    # Global smoothness is tested outside Q; secant regularity only inside Q.
    for bump in (0.0, 0.2, 1.0, 10.0, 10000.0):
        for _ in range(60):
            x = [rng.uniform(-10, 10), rng.uniform(-100, 100)]
            gradient = [x[0] + bump * math.sin(math.pi * x[0]),
                        math.pi * bump + 1 + x[1] / math.sqrt(1 + x[1] ** 2)]
            hessian_norm = max(abs(1 + math.pi * bump * math.cos(math.pi * x[0])),
                               (1 + x[1] ** 2) ** -1.5)
            assert hessian_norm <= 1 + math.hypot(*gradient) + 1e-10
            checks += 1

    # Certificate convergence on a fixed instance with a huge gradient floor.
    convergence = []
    bump = 1000.0
    for n in (1, 2, 3, 4, 6, 8):
        z = 10.0 ** -n
        x = [z, 0.0]
        gradient = [z + bump * math.sin(math.pi * z), math.pi * bump + 1]
        loss = z * z / 2 + 2 * bump / math.pi * math.sin(math.pi * z / 2) ** 2
        bound, gap, norm = certificate(x, gradient, [1.0, 1.0], 0.5, 1.0, 1.0)
        assert bound >= loss
        convergence.append({"coordinate": z, "error": loss, "gap": gap,
                            "gradient_norm": norm, "certificate": bound})
    assert convergence[-1]["certificate"] < 1e-5
    assert convergence[-1]["gradient_norm"] > 3000
    return {"numeric_checks": checks, "smallest_certificate_slack": min_slack,
            "fixed_instance_convergence": convergence}


if __name__ == "__main__":
    print(json.dumps({**verify_symbolic_identities(), **stress_checks()}, indent=2))
