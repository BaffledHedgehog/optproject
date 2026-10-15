"""Plot theoretical envelope comparisons; no optimization outcomes are inferred."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PAPER = Path(__file__).resolve().parent
WORK = PAPER.parent/'Work'
sys.path.insert(0, str(WORK))
from certificate_bounds import optimal_gap_envelope, sharp_gap_envelope


def main():
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9})
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)
    deltas = np.geomspace(1e-5, 1e3, 161)
    exact = [optimal_gap_envelope(float(d), 1, 1, 2) for d in deltas]
    frozen = [sharp_gap_envelope(float(d), 1, 1, 2) for d in deltas]
    axes[0].loglog(deltas, frozen, '--', color='#C67400', label='Frozen quadratic bound')
    axes[0].loglog(deltas, exact, '-', color='#15658C', label='Exact differential bound')
    axes[0].set(xlabel=r'True gap $\delta$', ylabel=r'Upper bound on $G$',
                title=r'$H_0=H_1=1$, $D=2$')
    for delta, old, new in zip(deltas, frozen, exact):
        records.append({'panel': 'positive_base', 'x': delta, 'quadratic': old, 'exact': new})
    bs = np.geomspace(1e-4, 1e4, 161)
    exact = [optimal_gap_envelope(1, 0, float(b), 1) for b in bs]
    frozen = [sharp_gap_envelope(1, 0, float(b), 1) for b in bs]
    axes[1].loglog(bs, frozen, '--', color='#C67400')
    axes[1].loglog(bs, exact, '-', color='#15658C')
    axes[1].set(xlabel=r'$b=H_1D^2$', ylabel=r'Coefficient in $G\leq c\delta$',
                title=r'Zero base curvature, $H_0=0$')
    for b, old, new in zip(bs, frozen, exact):
        records.append({'panel': 'zero_base', 'x': b, 'quadratic': old, 'exact': new})
    for ax in axes:
        ax.grid(True, which='major', alpha=.2)
    axes[0].legend(loc='upper left', fontsize=8)
    evidence = PAPER/'evidence'
    with (evidence/'geometry_grid.csv').open('w', encoding='utf-8', newline='') as source:
        writer = csv.DictWriter(source, list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for extension in ('pdf', 'png'):
        fig.savefig(PAPER/'figures'/('gap_geometry.'+extension), dpi=180)
    plt.close(fig)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = [PAPER/'prepare_geometry_evidence.py', WORK/'certificate_bounds.py',
              WORK/'differential_geometry_results/manifest.json']
    outputs = [evidence/'geometry_grid.csv', PAPER/'figures/gap_geometry.pdf', PAPER/'figures/gap_geometry.png']
    manifest = {'scope': 'Theoretical bound comparisons only; not optimization trajectories or timing results',
                'grid_rows': len(records),
                'input_sha256': {p.relative_to(PAPER.parent).as_posix(): sha(p) for p in inputs},
                'artifact_sha256': {p.relative_to(PAPER).as_posix(): sha(p) for p in outputs}}
    (evidence/'geometry_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Theoretical geometry figure and {len(records)} grid rows generated')


if __name__ == '__main__':
    main()
