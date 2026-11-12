SCALAR CERTIFICATE PACKAGES

Run Python from this directory to import certified_envelopes (v16) or
certified_diameter (v17). Both packages require Python 3.10+ and mpmath==1.3.0.
The v17 package privately reuses byte-identical reviewed v16 primitives;
it does not import or edit the sibling v16 package.

Each package documents exact input semantics, resource limits, structured
refusals and its immutable first-run record. The upper endpoint bounds source
error only under the stated mathematical assumptions. An envelope lower
endpoint is not a lower bound on an observed error.

Read each USAGE_AND_RESULTS.txt for input and result semantics. The release-root
README_RU.txt gives the exact snapshot-copy commands for a new v17 run; the v16
usage file gives its own reproduction procedure. Existing run_001 files must
not be overwritten. This release did not rerun old optimization experiments
or old scalar protocols.
