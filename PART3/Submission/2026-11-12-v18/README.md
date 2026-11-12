# Certified Frank-Wolfe stopping with a nonzero gradient at the minimum

Revision v18, 12 November 2026. Authors: Yaroslav P. Lyakhov and Stepan A. Suvorov.

- [English paper](paper.pdf), 8 pages.
- [Russian paper](paper_ru.pdf), 8 pages.
- [Russian paragraph-by-paragraph guide](guide_professor.txt).
- [Detailed Russian release instructions](README_RU.txt).
- [Changes from v17](CHANGES_v18.txt).
- [Reproducible code](Code/README.txt) and [release validation](Evidence/release_validation.json).
- [English LaTeX archive](arxiv_source_v18.zip), [Russian LaTeX archive](arxiv_source_ru_v18.zip), and [complete reading packet: restoration instructions](ArchiveParts/README_RU.txt).

The complete reading packet and the full CSV trace are stored as small verified
parts. After downloading the repository, run from this release directory:

```bash
python restore_large_files.py
```

The script restores `reading_packet_v18.zip` and `Evidence/boundary_trace.csv`
byte for byte. It uses only the Python standard library and refuses to overwrite
an existing different file. The original ZIP contents are unchanged.

The main paper now uses the convex compact-domain model with
`||Hessian f(x)|| <= L0 + L1 ||gradient f(x)||`.
The minimum value is absent from the smoothness condition, step rule, and
stopping test. The algorithm does not require the gradient norm to vanish
or the smoothness constants to be supplied.

The paper proves finite certificate-based stopping with complete oracle
counts, carried directional accuracy, invariance of the entire backtracking
execution under a normal linear term, and a quantitative comparison with
the fixed-parameter Algorithm 1 of Vyguzov and Stonyakin (arXiv:2510.16468v5).
The full Frank-Wolfe gap and quadratic backtracking are established methods;
the paper cites their sources.

For `exp(x)+A*y` on `[-1,1] x [0,1]`, all backtracking executions from
`(0.7,0)` take 2 accepted steps, 5 objective evaluations, 3 gradient queries,
and 3 full linear minimizations. The fixed rule takes 11, 14, 66, 610,
and 6045 steps for `A = 0, 1, 10, 100, 1000`, respectively. Both use the same
stopping certificate and tolerance `1e-6`. The separation is proved for the
specified family and fixed step rule. The source paper's adaptive Algorithm 2
is a different method and is not part of this comparison.

Build from this directory with `python build_papers.py`. Run the benchmark
with `python Code/benchmark_boundary.py --output reproduced --plots reproduced/figures`
and the directional checks with `python Code/check_directional.py`.
The PDF build does not run experiments. Numerical calculations use the Python
standard library; plotting needs matplotlib. See the Russian instructions
for TeX dependencies.

[v17](../2026-11-12-v17/README_RU.txt) is preserved as a historical release.
The separate exploratory files in `Research_notes/` are not part of the
current paper or its proofs. This is a research manuscript; no external
submission has been performed.
