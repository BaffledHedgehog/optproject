"""Build, package and independently compile the publication submission."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import statistics
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
PART3 = ROOT.parents[1]
PROJECT = PART3.parent
WORK = PART3/'Work'
PAPER = ROOT/'Paper'
TITLE = 'Observable Certificates and Exact Termination in Backtracking Frank-Wolfe'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def endpoint_summary():
    records = rows(WORK/'endpoint_recovery_strict_results/summary.csv')
    assert len(records) == 324
    ordinary = {r['config_id']: r for r in records if r['method'] == 'ordinary'}
    assert len(ordinary) == 108
    result = {}
    for method in ('dense', 'dyadic'):
        paired = [r for r in records if r['method'] == method]
        assert len(paired) == 108
        ratios = [float(r['budget_certificate']) /
                  float(ordinary[r['config_id']]['budget_certificate']) for r in paired]
        result[method] = dict(median=statistics.median(ratios),
                              better=sum(v < .99 for v in ratios),
                              worse=sum(v > 1.01 for v in ratios),
                              within=sum(.99 <= v <= 1.01 for v in ratios))
    assert sum(int(r['endpoint_accepts']) for r in records) == 0
    return result


def make_table(stats):
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\caption{Persistent-cache endpoint probes: certificate ratio $U_{\rm probe}/U_{\rm ordinary}$ at 1000 paid objective evaluations, over 108 paired configurations.}',
             r'\label{endpoint:table}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Probe schedule & Median ratio & Better $>1\%$ & Worse $>1\%$ & Within $1\%$ \\', r'\midrule']
    for method in ('dense', 'dyadic'):
        r = stats[method]
        lines.append(f"{method.capitalize()} & {r['median']:.6f} & {r['better']} & {r['worse']} & {r['within']} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (PAPER/'evidence/endpoint_table.tex').write_bytes('\n'.join(lines).encode())


def make_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trace = rows(WORK/'exposed_face_recovery_results/trace.csv')
    def path(identifier):
        return [r for r in trace if r['run_id'] == identifier]
    ordinary = path('1;1/3;ordinary;')
    sigma = str(1-Fraction(1, 2**20))
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.2), constrained_layout=True)
    left = ordinary[:17]
    axes[0].plot([int(r['iteration']) for r in left],
                 [float(Fraction(r['gap'])/Fraction(r['objective'])) for r in left], color='#176B8D')
    axes[0].axhline(1, linestyle=':', color='grey')
    axes[0].set(xlabel='Ordinary FW update', ylabel=r'$G_k/\delta_k$', title='Actual half-step trajectory')
    floor = 1e-7
    for method, label, color in [('ordinary', 'Ordinary FW', '#176B8D'),
                                  ('dense', 'Endpoint every update', '#C06C20'),
                                  ('dyadic', 'Endpoint at dyadic indices', '#57922F')]:
        rs = ordinary[:21] if method == 'ordinary' else path('1;1/3;'+method+';'+sigma)
        x = [int(r['function_calls']) for r in rs]
        y = [max(floor, float(Fraction(r['certificate']))) for r in rs]
        axes[1].semilogy(x, y, color=color, label=label)
        if Fraction(rs[-1]['certificate']) == 0:
            axes[1].scatter(x[-1], floor, color=color, marker='v', zorder=5)
            axes[1].annotate('exact U=0', (x[-1], floor), xytext=(0, 8),
                             textcoords='offset points', ha='center', fontsize=7)
    axes[1].set(xlabel='Paid objective evaluations', ylabel='Observable certificate U',
                title=r'Endpoint variants, $\sigma=1-2^{-20}$')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle('One exact exposed-face example; zeros shown at the plotting floor', fontsize=9)
    fig.savefig(PAPER/'figures/exposed_face_mechanism.pdf',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def generate_evidence():
    stats = endpoint_summary()
    make_table(stats)
    make_figure()
    inputs = [WORK/'endpoint_recovery_strict_results/summary.csv',
              WORK/'exposed_face_recovery_results/trace.csv']
    outputs = [PAPER/'evidence/endpoint_table.tex', PAPER/'figures/exposed_face_mechanism.pdf']
    manifest = {'date': '2026-11-01', 'endpoint_results': stats,
                'input_sha256': {p.relative_to(PART3).as_posix(): digest(p) for p in inputs},
                'output_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in outputs},
                'generator_sha256': digest(Path(__file__))}
    (ROOT/'evidence_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


def write_metadata(report):
    tex = (PAPER/'article.tex').read_text(encoding='utf-8')
    abstract = ' '.join(tex.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0].split())
    from submission_checks import dependencies
    all_tex = '\n'.join(p.read_text(encoding='utf-8-sig') for p in dependencies(PAPER) if p.suffix == '.tex')
    figures = len(re.findall(r'\\begin\{figure\}', all_tex))
    tables = len(re.findall(r'\\begin\{table\}', all_tex))
    metadata = {'title': TITLE, 'authors': ['Yaroslav P. Lyakhov', 'Stepan A. Suvorov'],
                'primary_category': 'math.OC', 'cross_lists': [], 'abstract': abstract,
                'comments': f"{report['paper']['pages']} pages, {figures} figures, {tables} tables",
                'processor': 'pdflatex', 'main_tex': 'article.tex',
                'license': 'To be selected by the submitting author in arXiv',
                'affiliations': 'Not supplied; no institutional affiliation inferred',
                'submission_status': 'Prepared locally; not submitted; expert mathematical reading pending',
                'pdf_sha256': digest(PAPER/'article.pdf')}
    (ROOT/'submission_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    plain = '\n\n'.join([f'Title:\n{TITLE}', 'Authors:\nYaroslav P. Lyakhov; Stepan A. Suvorov',
                           'Primary category:\nmath.OC', 'Abstract:\n'+abstract,
                           'Comments:\n'+metadata['comments'],
                           'Processor: PDFLaTeX\nMain file: article.tex',
                           'Journal reference and DOI: leave blank (no journal publication recorded).',
                           'License: select in your arXiv account. Affiliation and contact details: use your actual details.'])
    (ROOT/'ARXIV_METADATA.txt').write_text(plain+'\n', encoding='utf-8')


def build(engine=None):
    generate_evidence()
    builder = runpy.run_path(str(PAPER/'build_papers.py'))
    compiled = builder['compile_tex'](PAPER, 'article', builder['find_engine'](engine))
    (PAPER/'build/build_manifest.json').write_text(json.dumps({'article': compiled}, indent=2)+'\n', encoding='utf-8')
    from submission_checks import validate
    report = validate()
    shutil.copy2(PAPER/'article.pdf', ROOT/'paper.pdf')
    write_metadata(report)
    return report


def write_zip(path, files):
    if path.exists():
        raise FileExistsError(f'Immutable package exists: {path}; use a new --tag')
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, source in sorted(files.items()):
            assert not name.startswith('/') and '..' not in Path(name).parts and '\\' not in name
            assert source.is_file() and not source.is_symlink()
            item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.create_system = 3
            item.external_attr = 0o100644 << 16
            with source.open('rb') as incoming, archive.open(item, 'w', force_zip64=True) as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024*1024)


def artifact_paths(tag=''):
    assert not tag or re.fullmatch(r'[A-Za-z0-9_-]+', tag)
    suffix = '_'+tag if tag else ''
    return {name: ROOT/(name+suffix+'.zip') for name in ['arxiv_source', 'reproducibility', 'for_gasnikov']}


def package(tag=''):
    from submission_checks import dependencies, validate
    validate()
    artifacts = artifact_paths(tag)
    files = {p.relative_to(PAPER).as_posix(): p for p in dependencies(PAPER)}
    for name in files:
        assert all(re.fullmatch(r'[A-Za-z0-9_+.,=-]+', part) for part in Path(name).parts), name
        assert Path(name).suffix in {'.tex', '.pdf', '.png', '.jpg'}
    assert 'article.pdf' not in files
    write_zip(artifacts['arxiv_source'], files)
    review = {'paper.pdf': ROOT/'paper.pdf', 'COVER_LETTER_RU.txt': ROOT/'COVER_LETTER_RU.txt',
              'REVIEW_GUIDE_RU.txt': ROOT/'REVIEW_GUIDE_RU.txt'}
    write_zip(artifacts['for_gasnikov'], review)

    repro = {}
    denied = {'__pycache__', 'build', 'build_ru', '_verification', 'endpoint_recovery_results'}
    for base in [WORK, PART3/'Data', PART3/'Paper']:
        for current, directories, names in os.walk(base, followlinks=False):
            directories[:] = [d for d in directories if d not in denied]
            for name in names:
                source = Path(current)/name
                if source.is_symlink():
                    raise ValueError(source)
                if base == PART3/'Paper' and name in ['article.pdf', 'article_ru.pdf', 'manuscript.pdf']:
                    continue
                repro[source.relative_to(PROJECT).as_posix()] = source
    prefix = ROOT.relative_to(PROJECT).as_posix()+'/'
    for name, source in files.items():
        repro[prefix+'Paper/'+name] = source
    support = ['prepare_submission.py', 'submission_checks.py', 'requirements.txt', 'evidence_manifest.json',
               'submission_metadata.json', 'ARXIV_METADATA.txt', 'README_RU.txt', 'README_REPRODUCIBILITY.txt',
               'PROVENANCE_AND_NUMERICAL_NOTE.txt', 'editorial_provenance.json', 'data_validation.json']
    for name in support:
        repro[prefix+name] = ROOT/name
    repro[prefix+'Paper/build_papers.py'] = PAPER/'build_papers.py'
    repro[prefix+'paper.pdf'] = ROOT/'paper.pdf'
    for source in (ROOT/'Formal').iterdir():
        if source.is_file():
            repro[prefix+'Formal/'+source.name] = source
    repro['README_REPRODUCIBILITY.txt'] = ROOT/'README_REPRODUCIBILITY.txt'
    write_zip(artifacts['reproducibility'], repro)
    manifest = {'date': '2026-11-01', 'source_entry_point': 'article.tex',
                'source_members': {n: digest(p) for n, p in files.items()},
                'archives': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in artifacts.values()},
                'reproduction_files': len(repro), 'human_review': 'pending'}
    (ROOT/('package_manifest'+('_'+tag if tag else '')+'.json')).write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


def extract(archive, target):
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            dest = (target/member.filename).resolve()
            assert dest.is_relative_to(target.resolve()) and '\\' not in member.filename
            assert (member.external_attr >> 16) & 0o170000 != 0o120000
            dest.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, dest.open('xb') as output:
                shutil.copyfileobj(source, output)


def verify(tag='', engine=None):
    artifacts = artifact_paths(tag)
    temporary = ROOT/'_verification'
    temporary.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='submission_', dir=temporary))
    source = work/'arxiv'
    repro = work/'reproduction'
    extract(artifacts['arxiv_source'], source)
    extract(artifacts['reproducibility'], repro)
    builder = runpy.run_path(str(PAPER/'build_papers.py'))
    compiled = builder['compile_tex'](source, 'article', builder['find_engine'](engine))
    import pymupdf
    def text_hash(path):
        with pymupdf.open(path) as doc:
            return hashlib.sha256('\n'.join(p.get_text() for p in doc).encode()).hexdigest()
    assert text_hash(source/'article.pdf') == text_hash(ROOT/'paper.pdf')
    env = os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[key] = '1'
    relative = ROOT.relative_to(PROJECT).as_posix()
    steps = [
        [sys.executable, relative+'/prepare_submission.py', '--build'],
        [sys.executable, relative+'/submission_checks.py', '--tests'],
        [sys.executable, 'PART3/Work/validate_results.py'],
    ]
    checks = []
    for command in steps:
        result = subprocess.run(command, cwd=repro, env=env, capture_output=True, text=True, errors='replace', timeout=300)
        item = {'command': command, 'returncode': result.returncode,
                'stdout': result.stdout[-4500:], 'stderr': result.stderr[-4500:]}
        checks.append(item)
        if result.returncode:
            (ROOT/'verification_failure.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
            raise RuntimeError(item)
    assert text_hash(repro/relative/'paper.pdf') == text_hash(ROOT/'paper.pdf')
    report = {'status': 'passed', 'arxiv_archive_compiles': True, 'fresh_pdf_text_matches': True,
              'arxiv_compilation': compiled, 'extracted_directory': str(work), 'checks': checks,
              'archive_sha256': {p.name: digest(p) for p in artifacts.values()},
              'human_mathematical_review': 'pending; no approval is inferred from automated checks'}
    (ROOT/('submission_validation'+('_'+tag if tag else '')+'.json')).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--package', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--engine')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    if args.build or not (args.package or args.verify):
        result = build(args.engine)
        print(json.dumps({'built': 'paper.pdf', 'pages': result['paper']['pages'],
                          'endpoint_runs': result['endpoint']['runs'], 'exact_runs': result['exact_examples']['runs']}, indent=2))
    if args.package:
        print(json.dumps(package(args.tag), indent=2))
    if args.verify:
        report = verify(args.tag, args.engine)
        print(json.dumps({'status': report['status'], 'arxiv_archive_compiles': report['arxiv_archive_compiles'],
                          'fresh_pdf_text_matches': report['fresh_pdf_text_matches']}, indent=2))


if __name__ == '__main__':
    main()
