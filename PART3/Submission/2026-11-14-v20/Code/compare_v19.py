"""Compare reproduced controls with the user's original v19 archive numerically."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def compare(a, b, stats):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key], stats)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y, stats)
    else:
        try:
            x, y = float(a), float(b)
        except (TypeError, ValueError):
            assert a == b, (a, b)
            return
        assert math.isclose(x, y, rel_tol=5e-12, abs_tol=5e-13), (a, b)
        stats['numeric_values'] += 1
        stats['max_absolute_difference'] = max(stats['max_absolute_difference'], abs(x-y))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    report = {'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              'relative_tolerance': 5e-12, 'absolute_tolerance': 5e-13,
              'note': 'Line endings and last-bit floating-point differences do not imply a scientific mismatch.',
              'files': {}}
    with zipfile.ZipFile(args.archive) as archive:
        for name in ('gap_summary.csv', 'gap_summary.json', 'gap_checks.json', 'gap_trace.csv'):
            old = archive.read('Evidence/' + name)
            new = (ROOT / 'Evidence/v19' / name).read_bytes()
            stats = {'byte_identical': old == new, 'numeric_values': 0, 'max_absolute_difference': 0.0}
            parse = json.loads if name.endswith('.json') else lambda s: list(csv.DictReader(io.StringIO(s)))
            compare(parse(old.decode('utf-8-sig')), parse(new.decode('utf-8-sig')), stats)
            stats['numerically_equivalent'] = True
            report['files'][name] = stats
        report['original_code_unchanged'] = all(archive.read('Code/' + name) == (ROOT/'Code'/name).read_bytes()
            for name in ('benchmark_gap.py', 'check_algebra.py'))
    assert report['original_code_unchanged']
    (ROOT/'Evidence/v19/reproduction_check.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
