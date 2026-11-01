"""Independent finite checks of examples in the 19 October PART3 PDF.

Exact Fractions; no original experiment results or original solver code are
used. This illustrates the exposed-face mechanism, not general performance.
The baseline is intentionally capped; its infinite continuation is supplied
by the analytic recurrence in the manuscript, not proved by a finite run.
"""

import argparse
import csv
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import shutil


C = F(81, 476)
ATOMS = ((F(0), F(-1)), (F(0), F(1)), (F(1), F(-1)), (F(1), F(1)))


def objective(x):
    t, z = x
    return t+C*max(t, F(0))**3*(1+z*z/4)


def gradient(x):
    t, z = x
    if t <= 0:
        return F(1), F(0)
    return 1+3*C*t*t*(1+z*z/4), C*t**3*z/2


def dot(x, y):
    return sum((a*b for a, b in zip(x, y)), F(0))


def run(t0, z0, method, sigma=F(1, 2), cap=64):
    x, estimate = (t0, z0), 9*t0/4
    counts = {'function_calls': 0, 'gradient_calls': 0, 'lmo_calls': 0,
              'endpoint_queries': 0, 'endpoint_cache_hits': 0}
    cache = {}
    def paid_value(point):
        if point in cache:
            return cache[point], True
        value = objective(point)
        counts['function_calls'] += 1
        if point in ATOMS:
            cache[point] = value
        return value, False
    fx, _ = paid_value(x)
    lower = None
    trace = []
    for k in range(cap+1):
        g = gradient(x)
        counts['gradient_calls'] += 1
        s = min(ATOMS, key=lambda v: (dot(g, v), v))
        counts['lmo_calls'] += 1
        d = tuple(a-b for a, b in zip(s, x))
        gap = -dot(g, d)
        lower = fx-gap if lower is None else max(lower, fx-gap)
        upper = fx-lower
        assert 0 <= fx <= upper <= gap
        row = {'iteration': k, 't': x[0], 'z': x[1], 'objective': fx,
               'gap': gap, 'certificate': upper, 'event': 'source',
               'step': '', 'curvature': '', **counts}
        trace.append(row)
        if upper == 0 or k == cap:
            return trace
        scheduled = method == 'dense' or (method == 'dyadic' and ((k+1) & k) == 0)
        if scheduled:
            endpoint_value, hit = paid_value(s)
            counts['endpoint_queries'] += 1
            counts['endpoint_cache_hits'] += hit
            if endpoint_value <= fx-sigma*gap:
                row['event'], row['step'] = 'endpoint_accepted', F(1)
                x, fx = s, endpoint_value
                continue
            row['event'] = 'endpoint_rejected_then_fallback'
        else:
            row['event'] = 'ordinary_fallback'
        curvature = estimate/2
        q2 = dot(d, d)
        for _ in range(100):
            alpha = min(F(1), gap/(curvature*q2))
            trial = tuple(a+alpha*b for a, b in zip(x, d))
            trial_value, _ = paid_value(trial)
            if trial_value <= fx-alpha*gap+curvature*alpha*alpha*q2/2:
                break
            curvature *= 2
        else:
            raise AssertionError('Backtracking cap')
        # Check the paper's exact trajectory, using actual paid f/g/LMO.
        assert alpha == F(1, 2)
        assert curvature == 9*x[0]/8
        assert trial == (x[0]/2, -x[1])
        assert gap == x[0]+F(9, 16)*x[0]**3
        slack = fx-alpha*gap+curvature*alpha*alpha*q2/2-trial_value
        assert slack == x[0]/4+F(189, 15232)*x[0]**3 > 0
        row['step'], row['curvature'] = alpha, curvature
        x, fx, estimate = trial, trial_value, curvature
    raise AssertionError('Unreachable')


def all_checks():
    summaries, traces = [], []
    baselines = {}
    initializations = list(itertools.product((F(1), F(1, 2), F(1, 4)), (F(-1, 3), F(1, 3))))
    sigmas = (F(1, 2), F(9, 10), F(99, 100), 1-F(1, 2**20))
    comparisons = 0
    def retain(t0, z0, method, sigma, trace):
        run_id = f'{t0};{z0};{method};{sigma}'
        end = trace[-1]
        summaries.append({'run_id': run_id, 't0': str(t0), 'z0': str(z0),
                          'method': method, 'sigma': str(sigma),
                          'updates': end['iteration'], 'exact_zero': end['certificate'] == 0,
                          'final_objective': str(end['objective']),
                          'final_certificate': str(end['certificate']),
                          **{name: end[name] for name in ('function_calls', 'gradient_calls', 'lmo_calls',
                                                       'endpoint_queries', 'endpoint_cache_hits')}})
        for row in trace:
            traces.append({'run_id': run_id, **row})
    for t0, z0 in initializations:
        baseline = run(t0, z0, 'ordinary')
        assert baseline[-1]['iteration'] == 64 and baseline[-1]['certificate'] > 0
        assert baseline[-1]['t'] == t0/F(2)**64
        baselines[t0, z0] = baseline
        retain(t0, z0, 'ordinary', '', baseline)
    for (t0, z0), sigma, method in itertools.product(initializations, sigmas, ('dense', 'dyadic')):
        trace = run(t0, z0, method, sigma)
        assert trace[-1]['certificate'] == trace[-1]['objective'] == 0
        ordinary = baselines[t0, z0]
        for row in trace[:-1]:
            baseline = ordinary[row['iteration']]
            for key in ('t', 'z', 'objective', 'gap', 'certificate'):
                assert row[key] == baseline[key]
            assert 0 <= row['function_calls']-baseline['function_calls'] <= len(ATOMS)
            comparisons += 1
        # On this example only the two optimal face endpoints are ever probed.
        assert trace[-1]['endpoint_queries']-trace[-1]['endpoint_cache_hits'] <= 2
        assert trace[-1]['gradient_calls'] == trace[-1]['lmo_calls'] == trace[-1]['iteration']+1
        retain(t0, z0, method, sigma, trace)
    # Independent pointwise counterexample: the ratio limit differs from one.
    pointwise = []
    for a in (F(3, 2), F(2), F(4)):
        for k in (2, 8, 16):
            t = F(1, 2**k)
            delta = t+(a-1)**2*t*t/2
            gap = (2*a-1)*t+(a-1)**2*t*t
            grad = (1+a*(a-1)*t, (a-1)*t)
            x, s = (t, 1-t), (F(0), F(-1))
            assert dot(grad, tuple(u-v for u, v in zip(x, s))) == gap
            pointwise.append({'a': str(a), 't': str(t), 'ratio': float(gap/delta),
                              'limit': float(2*a-1)})
    # Integrated-profile vertex certificate: passes where the frozen profile fails.
    t = F(9, 20)
    g = 1+3*t*t
    gap = t*g
    assert g == F(643, 400) and gap == F(5787, 8000)
    assert g*g > 3*gap*gap and g*g < 6*gap*gap
    checks = {'arithmetic': 'fractions.Fraction; no floating-point optimization decisions',
              'scope': 'Illustrative checks of one exact exposed-face family, not a practical benchmark',
              'baseline_cap': 64, 'baseline_runs': 6, 'endpoint_runs': 48,
              'exactly_terminated_endpoint_runs': 48, 'baseline_runs_nonterminal_at_cap': 6,
              'matched_pre_endpoint_sources': comparisons,
              'maximum_endpoint_updates': max(r['updates'] for r in summaries if r['method'] != 'ordinary'),
              'maximum_paid_endpoint_values': max(r['endpoint_queries']-r['endpoint_cache_hits'] for r in summaries),
              'pointwise_counterexample': pointwise,
              'integrated_vertex_example': {'t': str(t), 'gradient': str(g), 'gap': str(gap),
                                            'integrated_passes': True, 'frozen_profile_passes': False}}
    return checks, summaries, traces


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new or empty output directory')
    checks, summaries, traces = all_checks()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'summary.csv', summaries)
    write_csv(output/'trace.csv', traces)
    (output/'checks.json').write_text(json.dumps(checks, indent=2)+'\n', encoding='utf-8')
    shutil.copy2(__file__, output/'audit_source.py')
    manifest = {'date': '2026-11-01', 'study': 'independent_finite_exposed_face_examples',
                'article': 'new/part3_revised.pdf, Section 5 and Section 6',
                'original_324_runs_reproduced': False,
                'artifact_sha256': {n: hashlib.sha256((output/n).read_bytes()).hexdigest()
                                    for n in ('summary.csv', 'trace.csv', 'checks.json', 'audit_source.py')}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in checks.items() if k != 'pointwise_counterexample'}, indent=2))


if __name__ == '__main__':
    main()
