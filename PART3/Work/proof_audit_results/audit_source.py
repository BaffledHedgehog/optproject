"""Finite exact-rational checks of proof inequalities, not a formal proof.

The curvature-state model overapproximates accepted backtracking histories.
It does not claim these abstract histories arise from actual objectives.
"""

import argparse
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys


def ceil_log2(value):
    """Exact nonnegative ceil(log2(max(1,value))), including dyadic ties."""
    if value <= 1:
        return 0
    result = max(0, value.numerator.bit_length()-value.denominator.bit_length())
    if F(2)**result < value:
        result += 1
    return result


def lag_states(caps, initial, initially_skipped):
    """Maximal lag in a dominating model with arbitrary forward phase jumps.

    If previous/2 exceeds the cap, the trial must pass and halve.
    Otherwise any accepted estimate is <= cap; resetting to the cap
    dominates all such choices for future lag. Same-phase non-lag
    repetitions cannot create lag and are collapsed. No f/g data is used.
    """
    states = 0
    @lru_cache(None)
    def worst(j, previous):
        nonlocal states
        states += 1
        cap = caps[j]
        lag = previous/2 > cap
        accepted = previous/2 if lag else cap
        before, after = ceil_log2(previous/cap), ceil_log2(accepted/cap)
        assert after == (before-1 if lag else 0)
        choices = [0]  # termination after this update
        for next_j in range(j if lag else j+1, len(caps)):
            assert ceil_log2(accepted/caps[next_j]) <= after+next_j-j
            choices.append(worst(next_j, accepted))
        return int(lag)+max(choices)
    maximum = max(worst(j, initial) for j in (range(len(caps)) if initially_skipped else (0,)))
    return maximum, states


def audit_phases():
    configurations = states = inequalities = 0
    ordinary_max, geometric_max = 0, 0
    h_values = (F(0), F(1, 16), F(1), F(16))
    for h0 in h_values:
        for h1 in h_values:
            for d2 in (F(1, 16), F(1), F(16)):
                for u0 in (F(1, 4), F(1), F(8)):
                    for ratio in (F(1, 2), F(1, 4), F(3, 8), F(1, 16), F(255, 256)):
                        epsilon = ratio*u0
                        phases = ceil_log2(u0/epsilon)
                        bs = [u0/F(2)**j for j in range(phases)]
                        assert sum(1/b for b in bs) < 2/epsilon
                        assert sum(bs) < 2*u0
                        inequalities += 2
                        ordinary = [max(2*(h0+h1*b), b/(2*d2)) for b in bs]
                        for b, cap in zip(bs, ordinary):
                            progress = min(b/4, b*b/(8*cap*d2))
                            assert int(b/progress)+1 <= 5+16*h1*d2+16*h0*d2/b
                            denominator = 8*d2*h0+4*d2*h1*b
                            explicit = min(b/4, b*b/denominator) if denominator else b/4
                            assert int(b/explicit)+1 <= 5+4*h1*d2+8*h0*d2/b
                            inequalities += 2
                        for factor in (F(1, 32), F(1, 2), F(1), F(2), F(3), F(4096)):
                            initial = factor*u0/d2
                            p = ceil_log2(initial*d2/u0)
                            maximum, visited = lag_states(tuple(ordinary), initial, False)
                            assert maximum <= p+phases
                            ordinary_max = max(ordinary_max, maximum)
                            configurations += 1
                            states += visited
                            if h0+h1 == 0:
                                continue
                            caps = tuple(2*(h0+h1*b) for b in bs)
                            for left, right in zip(caps, caps[1:]):
                                assert left/2 <= right <= left
                            potential = ceil_log2(initial/(2*(h0+h1*u0)))
                            maximum, visited = lag_states(caps, initial, True)
                            assert maximum <= potential+phases
                            geometric_max = max(geometric_max, maximum)
                            # kappa is arbitrary positive; test independent rational units.
                            for kappa in (F(1, 8), F(1), F(8)):
                                for b, cap in zip(bs, caps):
                                    progress = min(b/4, kappa*b/(2*cap*d2))
                                    assert int(b/progress)+1 <= 5+4*h0*d2/kappa+4*h1*d2*b/kappa
                                    inequalities += 1
                            configurations += 1
                            states += visited
    return {"phase_configurations": configurations, "exact_curvature_states": states,
            "phase_budget_inequalities": inequalities,
            "maximum_ordinary_lag": ordinary_max, "maximum_geometric_lag": geometric_max}


def audit_bridge():
    inequalities = exact_examples = 0
    values = (F(0), F(1, 1024), F(1, 16), F(1, 2), F(1), F(2), F(16))
    for a in (F(0), F(1, 16), F(1), F(16)):
        for b in (F(0), F(1, 16), F(1), F(2), F(16)):
            for delta in values:
                curvature = a+b*delta
                for gap in values:
                    decrease = (gap*gap/(2*curvature) if curvature and gap <= curvature
                                else gap-curvature/2)
                    inverted = (gap*gap <= 2*curvature*delta if 2*delta <= curvature
                                else gap <= delta+curvature/2)
                    assert (decrease <= delta) == inverted
                    inequalities += 1
                # In the full-step inverse branch the threshold is rational.
                for epsilon in values[1:]:
                    if b < 2 and epsilon*(2-b) >= 2*a:
                        eta = (epsilon-a/2)/(1+b/2)
                        c_eta = a+b*eta
                        assert 2*eta >= c_eta
                        assert eta+c_eta/2 == epsilon
                        assert eta >= epsilon/2
                        inequalities += 3
    for h0 in (F(1, 16), F(1), F(16)):
        for diameter in (F(1, 4), F(1), F(4)):
            radius, curvature = diameter/2, h0*diameter**2
            for relative_center in (F(-4), F(-1), F(0), F(1, 2), F(3, 4)):
                center = relative_center*radius
                optimum = max(-radius, center)
                delta = h0*((radius-center)**2-(optimum-center)**2)/2
                gap = h0*(radius-center)*diameter
                assert (gap*gap == 2*curvature*delta if 2*delta <= curvature
                        else gap == delta+curvature/2)
                exact_examples += 1
    return {"rational_bridge_inequalities": inequalities, "attaining_quadratics": exact_examples}


def audit_drops():
    visited = 0
    for initial in (1, 2, 4):
        states = {(initial, 0, 0)}  # size, good steps, FW steps
        for updates in range(13):
            following = set()
            for size, good, fw in states:
                visited += 1
                drops = updates-good
                assert drops <= fw+initial-1
                assert updates <= 2*good+initial-1
                # Maximal possible size growth dominates every real FW update.
                following.add((size+1, good+1, fw+1))
                following.add((size, good+1, fw))
                if size > 1:
                    following.add((size-1, good, fw))
            states = following
    return {"drop_count_states_up_to_12_updates": visited}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = next(p for p in Path(__file__).resolve().parents if p.name == "PART3")
    parser.add_argument("--output", type=Path, default=project/"Work"/"proof_audit_results")
    output = parser.parse_args().output
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new/empty audit directory")
    checks = {**audit_phases(), **audit_bridge(), **audit_drops()}
    output.mkdir(parents=True, exist_ok=True)
    (output/"checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    shutil.copy2(__file__, output/"audit_source.py")
    manifest = {"study": "finite_rational_proof_audit", "date": "2026-10-15",
                "scope": "bounded exact checks of dominating abstract states and inequalities, not a formal proof or independent review",
                "python": sys.version, "platform": platform.platform(), "arithmetic": "fractions.Fraction; no floating-point comparisons",
                "command": [sys.executable, *sys.argv],
                "artifact_sha256": {n: hashlib.sha256((output/n).read_bytes()).hexdigest()
                                    for n in ("checks.json", "audit_source.py")}}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
