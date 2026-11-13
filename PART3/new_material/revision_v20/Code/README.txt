Code for revision v20 (14 November 2026)

benchmark_gap.py and check_algebra.py: unmodified sources from the supplied
reading_packet_v19.zip. Their SHA-256 provenance and numerical reproduction
are in Evidence/v19_source_manifest.json and Evidence/v19/reproduction_check.json.

benchmark_multidim.py: Poisson regression on Euclidean balls; eight step rules;
explicit observed-data constants; actual oracle counters; saved full trajectories;
three timing repetitions. Reference intervals are post-run diagnostics only.
AC-FW is our implementation of Algorithms 1-2 of arXiv:2605.15512v2,
not the authors' software. The full ball is the generating dictionary.
Line bisection is a numerical line search with derivative queries, not an
unpriced exact line-minimization oracle.

test_revision.py: eight tests of derivatives, scalar inequality, scaling,
oracle accounting, initial optima, Armijo, bisection and nontrivial LMO endpoints.
report_results.py: figures and bilingual tables generated from saved evidence.
compare_v19.py: numerical comparison with the original user-supplied archive.

See ../README_RU.txt for exact commands, environment and limitations.
Run all commands from the release root. No external service is called.
No inexact oracle or certified floating-point implementation is claimed.
