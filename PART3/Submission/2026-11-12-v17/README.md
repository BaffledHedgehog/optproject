# Sharp Error Bounds and Adaptive Stopping for Frank-Wolfe beyond Convexity

Yaroslav P. Lyakhov and Stepan A. Suvorov. Revision v17, 12 November 2026.

- [English paper](paper.pdf), 23 pages.
- [Русская статья](paper_ru.pdf), 23 страницы.
- [Краткий разбор](guide_professor.txt).
- [Сборка и воспроизведение](README_RU.txt).
- [Новый скалярный API и результаты](Code/certified_diameter/USAGE_AND_RESULTS.txt).
- [Состав публичного выпуска и отдельного архива](PUBLICATION_SCOPE.txt).

The main paper contains complete central proofs. It replaces repeated radial
arguments with one comparison lemma and adds the sharp diameter envelope.
For `(m,H0,H1,g,P,D)=(1,4,0,1,2,3/5)`, the certificate improves from
`23/16` to `463/400`. The new fixed API panel has ten calls, seven certificates,
three expected refusals and 184 control assertions. No optimizer benchmark
or runtime advantage is inferred.

The [fixed scalar reference](Evidence/joint_reference_result.json) records the
intervals and independent travel-time checks used in the comparison table.

The public release includes the current manuscript sources and figure, both
scalar-certificate packages with their fixed control records, current
proof/build/layout reviews, and the scalar reference above.
Expanded v16 sources and accompanying records are supplied separately as
extended_sources_and_records_v16.zip; that archive is not part of this
repository. The current papers contain complete central proofs.

This package supports rebuilding the current papers and reproducing their
fixed scalar controls. It is not a complete replay archive of earlier
optimization experiments. The new diameter package and its first run are
complete. The earlier November 1 submission directory and other repository
projects are unchanged.
