"""Finite checks of the new geometry results, including exact rational AFW runs.

The analytic proofs are in article_away.tex and article_zero.tex. This audit
does not establish those universal statements or bibliographic priority.
No pre-existing experiment is modified. The rational optimizer receives only
the atom set, objective, gradient, starting weights and initial estimate.
"""

import argparse
import csv
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil


def ceil_log2(q):
    if q <= 1:
        return 0
    value = max(0, q.numerator.bit_length()-q.denominator.bit_length())
    return value+int(F(2)**value < q)


def ceil_fraction(q):
    return -((-q.numerator)//q.denominator)


def dot(x, y):
    return sum((a*b for a, b in zip(x, y)), F(0))


def point(atoms, weights):
    return tuple(sum((weights[i]*atoms[i][j] for i in weights), F(0))
                 for j in range(len(atoms[0])))


def exact_afw(atoms, weights, objective, gradient, initial, max_updates=80,
              check_update=None):
    weights = dict(weights)
    assert sum(weights.values()) == 1 and all(w > 0 for w in weights.values())
    x = point(atoms, weights)
    fx = objective(x)
    lower = None
    estimate = initial
    trace = []
    calls, rejects, good, drops = 1, 0, 0, 0
    initial_size = len(weights)
    for iteration in range(max_updates+1):
        grad = gradient(x)
        s = min(range(len(atoms)), key=lambda i: (dot(grad, atoms[i]), i))
        gap = dot(grad, tuple(a-b for a, b in zip(x, atoms[s])))
        lower = fx-gap if lower is None else max(lower, fx-gap)
        upper = fx-lower
        assert gap >= upper >= fx >= 0  # the supplied audit objectives have f*=0
        row = {"iteration": iteration, "objective": fx, "gap": gap, "certificate": upper,
               "weights": dict(weights), "x": x, "function_evals": calls,
               "gradient_evals": iteration+1, "lmo_evals": iteration+1}
        trace.append(row)
        if upper == 0:
            assert fx == 0
            return {"trace": trace, "good_steps": good, "drop_steps": drops,
                    "rejections": rejects, "initial_size": initial_size}
        assert iteration < max_updates, "Exact AFW did not terminate within this finite audit cap"
        assert max(v.denominator.bit_length() for v in x) < 100000, "Audit arithmetic-size cap"
        v = max(sorted(weights), key=lambda i: dot(grad, atoms[i]))
        away_gap = dot(grad, tuple(a-b for a, b in zip(atoms[v], x)))
        if gap >= away_gap or len(weights) == 1:
            kind, atom, chosen, cap = "FW", s, gap, F(1)
            direction = tuple(a-b for a, b in zip(atoms[s], x))
        else:
            kind, atom, chosen = "away", v, away_gap
            cap = weights[v]/(1-weights[v])
            direction = tuple(a-b for a, b in zip(x, atoms[v]))
        q2 = dot(direction, direction)
        curvature = estimate/2
        trials = 0
        while True:
            alpha = min(cap, chosen/(curvature*q2))
            trial_x = tuple(a+alpha*b for a, b in zip(x, direction))
            trial_f = objective(trial_x)
            calls += 1
            trials += 1
            if trial_f <= fx-alpha*chosen+curvature*alpha*alpha*q2/2:
                break
            rejects += 1
            curvature *= 2
            assert trials < 100
        is_drop = kind == "away" and alpha == cap
        row.update(kind=kind, atom=atom, chosen_gap=chosen, step=alpha, cap=cap,
                   curvature=curvature, direction_norm_sq=q2, drop=is_drop, trials=trials)
        if check_update is not None:
            check_update(row)
        factor = 1-alpha if kind == "FW" else 1+alpha
        updated = {i: factor*w for i, w in weights.items()}
        updated[atom] = updated.get(atom, F(0))+(alpha if kind == "FW" else -alpha)
        weights = {i: w for i, w in updated.items() if w}
        assert all(w > 0 for w in weights.values()) and sum(weights.values()) == 1
        assert point(atoms, weights) == trial_x
        assert fx-trial_f >= alpha*chosen/2 > 0
        good += not is_drop
        drops += is_drop
        x, fx, estimate = trial_x, trial_f, curvature
        assert iteration+1 <= 2*good+initial_size-1


def audit_problem(name, beta):
    if name == "simplex":
        atoms = tuple(tuple(F(i == j) for j in range(4)) for i in range(4))
        c = (F(0), F(0), F(1), F(2))
        def objective(x):
            t = dot(c, x)
            return t+beta*t**3
        def gradient(x):
            t = dot(c, x)
            return tuple(v*(1+3*beta*t*t) for v in c)
        return atoms, objective, gradient, 30*beta, F(1), {0, 1}
    if name == "box":
        atoms = tuple(itertools.product((F(0), F(1)), (F(-1, 2), F(1, 2))))
        def objective(x):
            t, y = x
            return t+beta*t**3*(1+y*y)
        def gradient(x):
            t, y = x
            return (1+3*beta*t*t*(1+y*y), 2*beta*t**3*y)
        return atoms, objective, gradient, 10*beta, F(1), {0, 1}
    raise ValueError(name)


def rational_finite_audit():
    records = []
    local_updates = 0
    for name, beta, initial in itertools.product(
            ("simplex", "box"), (F(0), F(1, 16), F(1)), (F(1, 4), F(1), F(4), F(64))):
        atoms, objective, gradient, h1, margin, face = audit_problem(name, beta)
        weights = {i: F(1, len(atoms)) for i in range(len(atoms))}
        x = point(atoms, weights)
        grad = gradient(x)
        u0 = dot(grad, x)-min(dot(grad, atom) for atom in atoms)
        d2 = max(dot(tuple(a-b for a, b in zip(v, w)),
                     tuple(a-b for a, b in zip(v, w))) for v in atoms for w in atoms)
        b = h1*d2
        # 2+b >= max(2,sqrt(2b)) >= a*coth(a): a rational conservative
        # factor from the already proved quadratic envelope.
        factor = 2+b
        curvature_cap = max(initial, 2*h1*u0)
        threshold = min(u0/2, margin/(4*(1+b+factor)), margin**2/(8*curvature_cap*d2))
        phases = ceil_log2(u0/threshold)
        potential = ceil_log2(initial*d2/u0)
        good_bound = potential+ceil_fraction((6+16*b)*phases)
        def check(row):
            nonlocal local_updates
            if row["objective"] <= threshold:
                assert row["kind"] == "away" and row["drop"] and row["atom"] not in face
                local_updates += 1
        result = exact_afw(atoms, weights, objective, gradient, initial, check_update=check)
        end = result["trace"][-1]
        n = end["iteration"]
        assert result["good_steps"] <= good_bound
        assert n <= 2*good_bound+len(weights)-1
        assert end["function_evals"] == 1+n+result["rejections"]
        assert end["function_evals"] <= 1+2*n+ceil_log2(2*h1*u0/initial)
        assert set(end["weights"]) <= face
        records.append({"family": name, "beta": str(beta), "initial": str(initial),
                        "iterations": n, "good_steps": result["good_steps"],
                        "drops": result["drop_steps"], "rejections": result["rejections"],
                        "function_evals": end["function_evals"], "gradient_evals": n+1,
                        "lmo_evals": n+1, "exact_objective": str(end["objective"]),
                        "exact_certificate": str(end["certificate"]),
                        "update_bound": 2*good_bound+len(weights)-1,
                        "final_atoms": ",".join(map(str, sorted(end["weights"]))),
                        "threshold": str(threshold)})
    # Plant sources well inside a common local identification neighborhood;
    # every remaining update must drop a bad atom, without a primal-error input.
    planted = 0
    for name, beta in itertools.product(("simplex", "box"), (F(0), F(1, 16), F(1))):
        atoms, objective, gradient, h1, margin, face = audit_problem(name, beta)
        theta = F(1, 100000*(1+int(100*h1)))
        weights = {0: (1-theta)/2, 1: (1-theta)/2, 2: theta/3, 3: 2*theta/3}
        def check_local(row):
            nonlocal local_updates
            assert row["kind"] == "away" and row["drop"] and row["atom"] not in face
            local_updates += 1
        result = exact_afw(atoms, weights, objective, gradient, F(1), check_update=check_local)
        assert result["good_steps"] == 0 and result["drop_steps"] == 2
        assert result["trace"][-1]["certificate"] == 0
        planted += 1
    return {"global_runs": len(records), "planted_local_runs": planted,
            "checked_local_maximal_drops": local_updates,
            "max_updates": max(r["iterations"] for r in records),
            "max_function_evals": max(r["function_evals"] for r in records)}, records


def positive_base_counterexample():
    checks = 0
    for h in (F(1, 1000000), F(1), F(1000)):
        x = F(1, 2)
        curvature = 3*h/2
        for k in range(30):
            gap, q2 = h*x*(x+1), (x+1)**2
            alpha = gap/(curvature*q2)
            next_x = x-alpha*(x+1)
            assert next_x == x/3 > 0 and alpha < 1
            for trial_l, should_pass in ((curvature, True), (curvature/2, False)):
                step = gap/(trial_l*q2)
                trial_x = x-step*(x+1)
                passes = h*trial_x**2/2 <= h*x*x/2-step*gap+trial_l*step*step*q2/2
                assert passes == should_pass
                checks += 1
            x = next_x
    return {"exact_positive_base_trial_checks": checks, "finite_horizon_per_scale": 30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks, records = rational_finite_audit()
    checks.update(positive_base_counterexample())
    if args.output:
        output = args.output.resolve()
        if output.exists() and any(output.iterdir()):
            parser.error("Use a new/empty audit directory")
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(__file__, output/"audit_source.py")
        (output/"checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        with (output/"runs.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        manifest = {"study": "differential_geometry_audit", "date": "2026-10-15",
                    "scope": "finite exact rational tests; not a formal proof, priority search or independent peer review",
                    "arithmetic": "fractions.Fraction; no floating-point optimizer comparisons",
                    "artifact_sha256": {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                                        for name in ("audit_source.py", "checks.json", "runs.csv")}}
        (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
