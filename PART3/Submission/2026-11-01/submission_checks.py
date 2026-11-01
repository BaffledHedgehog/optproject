"""Checks for the publication manuscript and its numerical evidence."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parent
PART3=ROOT.parents[1]
PAPER=ROOT/'Paper'
WORK=PART3/'Work'

def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

def dependencies(paper):
    paper = paper.resolve()
    pending, seen = [paper/'article.tex'], set()
    while pending:
        path = pending.pop().resolve()
        assert path.is_relative_to(paper) and path.is_file(), path
        if path in seen:
            continue
        seen.add(path)
        if path.suffix != '.tex':
            continue
        text = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text(encoding='utf-8-sig'))
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}', text):
            pending.append(paper/(name if Path(name).suffix else name+'.tex'))
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
            path = paper/name
            if not path.suffix:
                path = next(p for p in (path.with_suffix(ext) for ext in ('.pdf', '.png', '.jpg')) if p.exists())
            pending.append(path)
    return sorted(seen)

def inspect_paper():
    import pymupdf
    files = dependencies(PAPER)
    text = '\n'.join(p.read_text(encoding='utf-8-sig') for p in files if p.suffix == '.tex')
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    refs = set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', text))
    bib = set(re.findall(r'\\bibitem\{([^}]+)\}', text))
    cited = {k.strip() for group in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}', text) for k in group.split(',')}
    assert len(labels) == len(set(labels)), 'Duplicate labels'
    assert refs <= set(labels), refs-set(labels)
    assert cited == bib, (cited-bib, bib-cited)
    assert not re.search(r'\b(?:TODO|TBD|FIXME)\b', text)
    log = (PAPER/'build/article.log').read_text(errors='replace')
    for forbidden in ['undefined references', 'multiply-defined labels', 'Overfull', 'undefined on input line', '! LaTeX Error']:
        assert forbidden not in log, forbidden
    doc = pymupdf.open(PAPER/'article.pdf')
    assert 'Yaroslav P. Lyakhov' in doc.metadata.get('author', '')
    assert 'Stepan A. Suvorov' in doc.metadata.get('author', '')
    assert abs(doc[0].rect.width-595.276) < 1
    assert abs(doc[0].rect.height-841.89) < 1
    texts = [p.get_text() for p in doc]
    assert all(len(t.strip()) > 20 for t in texts)
    assert '??' not in '\n'.join(texts)
    fonts = {font[0] for page in doc for font in page.get_fonts(full=True)}
    for ref in fonts:
        if doc.xref_get_key(ref, 'Subtype') == ('name', '/Type3'):
            assert doc.xref_get_key(ref, 'CharProcs')[0] == 'xref'
        else:
            assert doc.extract_font(ref)[3], ('Unembedded font', ref)
    return {'pages': len(doc), 'labels': len(labels), 'citations': len(cited),
            'embedded_fonts': len(fonts), 'pdf_sha256': sha(PAPER/'article.pdf'),
            'dependency_sha256': {p.relative_to(PAPER).as_posix(): sha(p) for p in files}}

def inspect_endpoint():
    base = WORK/'endpoint_recovery_strict_results'
    manifest = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    assert (base/'manifest.sha256').read_text().split()[0] == sha(base/'manifest.json')
    for group in ['result_sha256', 'frozen_source_sha256', 'frozen_data_sha256']:
        for name, expected in manifest[group].items():
            path = (base/name).resolve()
            assert path.is_relative_to(base.resolve())
            assert sha(path) == expected, name
    for name in ['endpoint_fw.py', 'endpoint_benchmark.py']:
        assert sha(WORK/name) == sha(base/'frozen/source'/name)
    with (base/'configurations.csv').open(encoding='utf-8', newline='') as source:
        dimensions = {(r['dataset'], float(r['radius_ratio'])): int(r['dimension'])
                      for r in csv.DictReader(source)}
    sources = transitions = pairs = max_overhead = 0
    with (base/'trace.csv').open(encoding='utf-8', newline='') as source:
        for _, rows in itertools.groupby(csv.DictReader(source), key=lambda r: r['config_id']):
            paths = {}
            for row in rows:
                paths.setdefault(row['method'], []).append(row)
                assert int(row['function_requests']) == int(row['function_evals'])+int(row['cache_hits'])
                assert row['endpoint_accepted'] == 'False'
            ordinary = paths['ordinary']
            for method in ['dense', 'dyadic']:
                probe = paths[method]
                bound = 2*dimensions[probe[0]['dataset'], float(probe[0]['radius_ratio'])]
                count = min(len(ordinary), len(probe))
                for left, right in zip(ordinary[:count], probe[:count]):
                    for key in ['objective', 'fw_gap', 'certificate', 'lower_bound']:
                        assert float(left[key]) == float(right[key]), (method, key)
                    delta = int(right['source_paid_evals'])-int(left['source_paid_evals'])
                    assert 0 <= delta <= bound
                    max_overhead = max(max_overhead, delta)
                for left, right in zip(ordinary[:count-1], probe[:count-1]):
                    for key in ['step', 'accepted_curvature']:
                        assert float(left[key]) == float(right[key]), (method, key)
                sources += count
                transitions += count-1
                pairs += 1
    with (base/'replay_audit.csv').open(encoding='utf-8', newline='') as source:
        audits = list(csv.DictReader(source))
    for row in audits:
        assert row['common_source_prefix_exact'] == 'True'
        assert 0 <= int(row['min_paid_eval_delta_at_common_updates'])
        assert int(row['max_paid_eval_delta_at_common_updates']) <= int(row['vertex_count_bound_2d'])
    assert pairs == len(audits) == 216
    assert sources == manifest['replay_audit']['paired_source_points']
    assert transitions == manifest['replay_audit']['paired_transitions']
    generator = runpy.run_path(str(ROOT/'prepare_submission.py'))
    stats = generator['endpoint_summary']()
    evidence = json.loads((ROOT/'evidence_manifest.json').read_text(encoding='utf-8'))
    assert stats == evidence['endpoint_results']
    for name, expected in evidence['input_sha256'].items():
        assert sha(PART3/name) == expected, name
    for name, expected in evidence['output_sha256'].items():
        assert sha(ROOT/name) == expected, name
    assert sha(ROOT/'prepare_submission.py') == evidence['generator_sha256']
    return {'runs': 324, 'paired_paths': pairs, 'matched_sources': sources,
            'matched_transitions': transitions, 'maximum_paid_overhead': max_overhead,
            'endpoint_accepts': 0, 'summaries': stats}

def inspect_exact_examples():
    base = WORK/'exposed_face_recovery_results'
    manifest = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['artifact_sha256'].items():
        assert sha(base/name) == expected, name
    checks, summaries, traces = runpy.run_path(str(base/'audit_source.py'))['all_checks']()
    assert checks == json.loads((base/'checks.json').read_text(encoding='utf-8'))
    for name, expected in [('summary.csv', summaries), ('trace.csv', traces)]:
        with (base/name).open(encoding='utf-8', newline='') as source:
            actual = list(csv.DictReader(source))
        assert actual == [{k: str(v) for k, v in row.items()} for row in expected], name
    return {'runs': len(summaries), 'trace_rows': len(traces),
            'exact_endpoint_successes': checks['exactly_terminated_endpoint_runs']}

def validate(run_tests=False):
    report={'date':'2026-11-01','paper':inspect_paper(),
            'endpoint':inspect_endpoint(),'exact_examples':inspect_exact_examples(),
            'human_mathematical_review':'pending; prepared for external expert reading'}
    tex='\n'.join(p.read_text(encoding='utf-8-sig') for p in dependencies(PAPER) if p.suffix=='.tex')
    assert 'Observable Certificates and Exact Termination' in tex
    assert not re.search(r'reconstruct|supplied 19 October|historical median|original file.*unavailable',tex,re.I)
    assert 'Generative-tool disclosure' in tex
    if run_tests:
        for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
            os.environ[key]='1'
        sys.path.insert(0,str(WORK))
        suite=unittest.defaultTestLoader.discover(str(WORK),pattern='test_*.py')
        output=io.StringIO(); started=time.perf_counter()
        result=unittest.TextTestRunner(stream=output,verbosity=1).run(suite)
        (ROOT/'unittest_validation.log').write_text(output.getvalue(),encoding='utf-8')
        assert result.wasSuccessful(),output.getvalue()
        report['tests']={'run':result.testsRun,'passed':True,'elapsed_seconds':time.perf_counter()-started}
    (ROOT/'data_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests',action='store_true')
    args=parser.parse_args()
    report=validate(args.tests)
    print(json.dumps({'pages':report['paper']['pages'],'citations':report['paper']['citations'],
                      'endpoint_runs':report['endpoint']['runs'],'exact_runs':report['exact_examples']['runs'],
                      'tests':report.get('tests')},indent=2))
