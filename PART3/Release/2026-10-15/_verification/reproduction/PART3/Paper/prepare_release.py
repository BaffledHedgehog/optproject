"""Build and verify printable papers, a minimal TeX archive and a full reproduction package."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from build_papers import compile_tex, find_engine
from validate_article import dependencies, inspect_pdf, validate


PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, cwd, environment, timeout=300):
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=timeout)
    output = result.stdout+result.stderr
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{output[-7000:]}")
    return {"command": [str(s) for s in command], "exit_code": result.returncode,
            "output": output, "output_sha256": hashlib.sha256(output.encode()).hexdigest()}


def archive(path, files):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for source, name in sorted(files, key=lambda p: p[1]):
            target.write(source, name)


def extract(path, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as source:
        for name in source.namelist():
            assert (destination/name).resolve().is_relative_to(destination.resolve())
        source.extractall(destination)


def csv_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def rendered_hash(path):
    import pymupdf
    doc = pymupdf.open(path)
    return hashlib.sha256(doc[0].get_pixmap().samples).hexdigest()


def reproduction_files():
    selected = set()
    for name in ("README.md", "info.txt", "requirements.txt", "requirements-paper.txt"):
        selected.add(PROJECT/name)
    for directory in (PROJECT/"Data", PROJECT/"Translations"):
        selected.update(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    work = PROJECT/"Work"
    selected.update(work.glob("*.py"))
    selected.update(work.glob("*.md"))
    selected.add(work/"publication_literature.json")
    for study in work.iterdir():
        if study.is_dir() and (study/"manifest.json").is_file():
            selected.update(p for p in study.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for pattern in ("*.tex", "*.py", "*.pdf", "*.txt"):
        selected.update(PAPER.glob(pattern))
    for directory in (PAPER/"figures", PAPER/"evidence"):
        selected.update(p for p in directory.rglob("*") if p.is_file())
    return [(p, "PART3/"+str(p.relative_to(PROJECT)).replace("\\", "/")) for p in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine")
    parser.add_argument("--output", type=Path, default=PROJECT/"Release"/"2026-10-15")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new/empty release directory; existing releases are preserved")
    output.mkdir(parents=True, exist_ok=True)
    engine = find_engine(args.engine)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    verification = {"scope": "Local isolated build and reproduction; no external arXiv upload or independent peer review",
                    "date": "2026-10-15", "checks": {}}
    print("Building final article and extended report", flush=True)
    for name in ("article", "manuscript"):
        verification["checks"][name+"_build"] = compile_tex(PAPER, name, engine)
    verification["article"] = validate(PAPER)
    verification["technical_report"] = inspect_pdf(PAPER/"manuscript.pdf")
    shutil.copy2(PAPER/"article.pdf", output/"article.pdf")
    shutil.copy2(PAPER/"manuscript.pdf", output/"technical_report.pdf")
    shutil.copy2(PAPER/"PRINT_README_RU.txt", output/"README_RU.txt")
    shutil.copy2(PROJECT/"Work"/"publication_literature.json", output/"literature_comparison.json")

    closure = dependencies(PAPER)
    archive(output/"arxiv_source.zip", [(p, str(p.relative_to(PAPER)).replace("\\", "/")) for p in closure])
    staging = output/"_verification"
    arxiv = staging/"arxiv"
    extract(output/"arxiv_source.zip", arxiv)
    print(f"Compiling isolated TeX extraction ({len(closure)} input files)", flush=True)
    compile_tex(arxiv, "article", engine)
    isolated = validate(arxiv, evidence=False)
    assert isolated["pdf"]["text_sha256"] == verification["article"]["pdf"]["text_sha256"]
    verification["isolated_source"] = isolated

    print("Packing code, raw data, source snapshots and all recorded studies", flush=True)
    files = reproduction_files()
    archive(output/"reproducibility.zip", files)
    repro = staging/"reproduction"
    extract(output/"reproducibility.zip", repro)
    print(f"Checking extracted reproduction package ({len(files)} files)", flush=True)
    verification["checks"]["extracted_tests"] = run(
        [sys.executable, "-m", "unittest", "discover", "-s", "PART3/Work", "-p", "test_*.py"], repro, environment)
    assert "Ran 58 tests" in verification["checks"]["extracted_tests"]["output"]
    verification["checks"]["extracted_archives"] = run(
        [sys.executable, "PART3/Work/validate_results.py"], repro, environment)

    print("Reproducing all 108 AC-FW runs from the extracted archived source", flush=True)
    verification["checks"]["extracted_comparator_replay"] = run(
        [sys.executable, "PART3/Work/count_comparator_results/source/count_comparator.py",
         "--output", "PART3/Work/reproduced_count_comparator"], repro, environment)
    old = csv_rows(repro/"PART3/Work/count_comparator_results/summary.csv")
    new = csv_rows(repro/"PART3/Work/reproduced_count_comparator/summary.csv")
    def key(row):
        return row["dataset"], float(row["radius_ratio"]), row["start"]
    indexed = {key(r): r for r in new}
    fields = ("objective", "certificate", "iteration", "function_evals", "gradient_evals", "lmo_evals", "accepted_steps", "rejected_steps")
    assert len(old) == len(new) == len(indexed) == 108
    for row in old:
        for field in fields:
            assert row[field] == indexed[key(row)][field], (key(row), field)
    verification["replayed_method_records"] = {"records": len(new), "exactly_matching_fields": fields}
    verification["checks"]["replayed_archive_validation"] = run(
        [sys.executable, "PART3/Work/validate_results.py", "PART3/Work/reproduced_count_comparator"], repro, environment)

    print("Regenerating article evidence and rebuilding both extracted PDFs", flush=True)
    verification["checks"]["extracted_evidence"] = run(
        [sys.executable, "PART3/Paper/prepare_article_evidence.py"], repro, environment)
    extracted_paper = repro/"PART3/Paper"
    for original in (PAPER/"evidence").glob("*"):
        if original.suffix in (".tex", ".csv"):
            assert original.read_bytes() == (extracted_paper/"evidence"/original.name).read_bytes()
    assert rendered_hash(PAPER/"figures/count_comparators.pdf") == rendered_hash(extracted_paper/"figures/count_comparators.pdf")
    verification["evidence_roundtrip"] = "All five tables and the 80-row claim ledger match byte-for-byte; regenerated figure rendering matches"
    for name in ("article", "manuscript"):
        compile_tex(extracted_paper, name, engine)
    extracted_article = validate(extracted_paper)
    assert extracted_article["pdf"]["text_sha256"] == verification["article"]["pdf"]["text_sha256"]
    extracted_report = inspect_pdf(extracted_paper/"manuscript.pdf")
    assert extracted_report["text_sha256"] == verification["technical_report"]["text_sha256"]
    verification["extracted_papers"] = {"article": extracted_article["pdf"], "technical_report": extracted_report}

    article_tex = (PAPER/"article.tex").read_text(encoding="utf-8")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", article_tex, re.S).group(1)
    metadata = ["Title: Certificate Bounds for Backtracking Frank-Wolfe under Function-Dependent Curvature",
                "Authors: Yaroslav P. Lyakhov; Stepan Suvorov", "Proposed primary category: math.OC",
                "Processor: PDFLaTeX; main file article.tex", "", "Abstract (TeX math retained):",
                re.sub(r"\s+", " ", abstract).strip(), "", "No external submission has been made.",
                "Affiliations and contact email were not supplied. The authors select the distribution license at submission.", ""]
    (output/"submission_metadata.txt").write_text("\n".join(metadata), encoding="utf-8")
    (output/"verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    artifacts = [p for p in output.iterdir() if p.is_file() and p.name != "manifest.json"]
    manifest = {"date": "2026-10-15", "authors": ["Yaroslav P. Lyakhov", "Stepan Suvorov"],
                "artifact_sha256": {p.name: digest(p) for p in artifacts},
                "artifact_bytes": {p.name: p.stat().st_size for p in artifacts},
                "reproduction_files": len(files), "arxiv_source_files": len(closure),
                "external_submission": False, "independent_peer_review": False,
                "builder_sha256": digest(__file__)}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Release verified: {output}", flush=True)
    print(f"Article {verification['article']['pdf']['pages']} pages; technical report {verification['technical_report']['pages']} pages; 108 exact replays")


if __name__ == "__main__":
    main()
