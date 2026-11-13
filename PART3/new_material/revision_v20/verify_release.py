"""Validate saved evidence, TeX receipts and (optionally) replay all 432 configurations."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'Code'))

def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    for p, digest in read('Evidence/multidim/data_hashes.json').items():
        assert sha(ROOT/'Evidence/multidim/data'/p) == digest, p
    rows = read('Evidence/multidim/summary.json')
    audits = read('Evidence/multidim/audit.json')
    assert len(rows) == len(audits) == 432
    assert len({(r['problem'],r['start'],r['method']) for r in rows}) == 432
    assert len({r['problem'] for r in rows}) == 18
    assert all(a['passed'] for a in audits)
    assert sum(a['points'] for a in audits) == 340433
    assert sum(r['status'] == 'converged' for r in rows) == 423
    assert all(r['method']=='open_loop' for r in rows if r['status']!='converged')
    for folder in (ROOT/'Paper', ROOT/'Paper/ru'):
        receipt = json.loads((folder/'build_receipt.json').read_text())
        assert not receipt['warning_lines'] and not receipt['source_changed_during_build']
        assert len(receipt['passes']) == 3 and all(p['exit']==0 for p in receipt['passes'])
        for name, digest in receipt['source_hashes'].items():
            assert sha(folder/name)==digest, name
    checks = ROOT/'Evidence/checks'; checks.mkdir(exist_ok=True)
    for name, command in [('unit_tests.txt',[sys.executable,'-m','unittest','discover','-s','Code','-p','test_revision.py','-v']),
                          ('algebra_audit.txt',[sys.executable,'Code/check_algebra.py'])]:
        with (checks/name).open('w',encoding='utf-8') as output:
            subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
    replayed = 0
    if args.replay:
        import numpy as np
        from benchmark_multidim import PoissonBall, solve
        protocol = read('Evidence/multidim/protocol.json')
        problems = {}
        for row in rows:
            pid = row['problem']
            if pid not in problems:
                data = read('Evidence/multidim/data/'+pid+'.json')
                problems[pid] = PoissonBall(np.array(data['A']), np.array(data['y']), data['radius'])
            problem = problems[pid]
            start = np.zeros(row['n'])
            if row['start'] != 'zero':
                start[0] = row['radius'] if row['start']=='plus_axis' else -row['radius']
            result, _, _ = solve(problem,start,row['method'],protocol['epsilon'],protocol['max_attempts'],record=False)
            for key, value in result.items():
                if key == 'seconds':
                    continue
                if isinstance(value,float):
                    assert math.isclose(value,row[key],rel_tol=2e-10,abs_tol=2e-12),(pid,row['method'],key,value,row[key])
                else:
                    assert value==row[key],(pid,row['method'],key,value,row[key])
            replayed += 1
            if replayed % 72 == 0:
                print('Replayed',replayed,'/432',flush=True)
    report = {'saved_runs':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),
              'data_hashes_passed':True,'tex_receipts_passed':True,'unit_tests':8,
              'symbolic_identities':13,'replayed_runs':replayed,
              'replay_excludes_elapsed_time':True,'source_sha256':{
                  p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'Code').glob('*.py'))}}
    report_name = 'release_validation.json' if args.replay else 'quick_validation.json'
    (checks/report_name).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':
    main()
