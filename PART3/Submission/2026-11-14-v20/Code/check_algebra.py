"""Exact symbolic checks supporting v19; this is not a Lean proof.

The quadratic identities reproduce the scalar algebra used by
LongTails/Proofs/FrankWolfeDescent.lean at commit
5965b785bb0bf47092ac5dcbfa5d8562b48ba70b. No LongTails code is executed.
"""

import sympy as s


def zero(label, expression):
    result = s.simplify(expression)
    assert result == 0, (label, result)
    print(f"PASS: {label}")


g, curvature, trial = s.symbols("g curvature trial", positive=True)
short = g / curvature
zero("quadratic short-step value", short * g - curvature * short**2 / 2 - g**2 / (2 * curvature))
zero("quadratic completion of the square", (short * g - curvature * short**2 / 2) - (trial * g - curvature * trial**2 / 2) - curvature * (trial - short)**2 / 2)

t, b, c, p = s.symbols("t b c p", positive=True)
phi = (s.exp(b * t) - 1 - b * t) / b**2
zero("integrated exponential model second derivative", s.diff(phi, t, 2) - s.exp(b * t))
zero("integrated exponential model at zero", phi.subs(t, 0))
zero("integrated exponential model slope at zero", s.diff(phi, t).subs(t, 0))
zero("zero-exponent limit of the upper model", s.limit(phi, b, 0) - t**2 / 2)
alpha = s.log(1 + b * g / (c * p**2)) / b
model = -g * t + c * p**2 * phi
zero("logarithmic step solves the model derivative", s.diff(model, t).subs(t, alpha))
zero("logarithmic step has the ordinary short-step limit", s.limit(alpha, b, 0) - g / (c * p**2))

u = s.symbols("u", nonnegative=True)
w = (1 + u / 2) * s.log(1 + u) - u
zero("half-decrease residual at zero", w.subs(u, 0))
zero("half-decrease residual derivative at zero", s.diff(w, u).subs(u, 0))
zero("half-decrease residual second derivative is nonnegative", s.diff(w, u, 2) - u / (2 * (1 + u)**2))

M, x = s.symbols("M x", positive=True)
f = M * x + M * x**3 / 3
gap = s.diff(f, x) * x
zero("counterexample Hessian inequality residual", 2 * gap - s.diff(f, x, 2) - 2 * M * x**3)
zero("counterexample nonzero Taylor remainder", f - M * x - M * x**3 / 3)

print("All 13 symbolic identities passed. Sign conclusions use the stated positive-variable assumptions.")
print("No Lean compiler, Lake build, or Comparator audit was run; no formal verification is claimed.")
