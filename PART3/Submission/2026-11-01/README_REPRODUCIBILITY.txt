Observable Certificates and Exact Termination in Backtracking Frank-Wolfe
Yaroslav P. Lyakhov and Stepan A. Suvorov — 1 November 2026

Publication entry point:
  PART3/Submission/2026-11-01/Paper/article.tex
  PART3/Submission/2026-11-01/paper.pdf

The older PART3/Paper directory is retained only for baseline source and
evidence provenance. Use the Submission entry point for this manuscript.

From PART3/Submission/2026-11-01:
  python -m pip install -r requirements.txt
  python prepare_submission.py --build
  python submission_checks.py --tests

PDFLaTeX must be on PATH, or supply --engine to prepare_submission.py.
Run with OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1 for numerical
replays. requirements.txt records the environment used in this execution.

The verified endpoint evidence is in:
  PART3/Work/endpoint_recovery_strict_results/
It contains 324 executions, frozen source/data, SHA-256 manifests and the
216-pair common-prefix audit. The non-strict preliminary attempt is not
included. The published table uses only the strict execution.

To execute the protocol again from the extracted repository root:
  python PART3/Work/endpoint_benchmark.py --output NEW_EMPTY_DIRECTORY
For the 54 rational example runs:
  python PART3/Work/exposed_face_recovery.py --output NEW_EMPTY_DIRECTORY
Choose new or empty output directories; frozen evidence is not overwritten.

Preserved baseline CSVs, source snapshots and checks remain under PART3/Work.
Data attribution and raw input hashes are under PART3/Data and the study
manifests. PROVENANCE_AND_NUMERICAL_NOTE.txt in the Submission directory
records the reconstruction history and the unresolved difference from the
older author-supplied PDF; those older numbers are not current measurements.

Formal/ResearchAlgebra.lean under the Submission directory uses Lean
4.35.0-rc2 and core/Std only. It proves two polynomial identities and seven
closed integer checks. It does not formalize the analytic theorems.
The recorded build log, axiom audit and checksums accompany the source.

Automated validation and internal AI-assisted reading are not independent
human mathematical review. No arXiv submission or external endorsement is
asserted by this package. The clean arXiv source archive is supplied as a
separate artifact; this full code/data bundle is for reproducibility.
