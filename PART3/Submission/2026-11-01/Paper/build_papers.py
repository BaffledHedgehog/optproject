"""Build the focused article and expanded report with a standard PDFLaTeX."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent


def find_engine(explicit=None):
    candidate = explicit or os.environ.get("PDFLATEX") or shutil.which("pdflatex")
    if candidate is None:
        raise RuntimeError("Set PDFLATEX or pass --engine with the path to pdflatex")
    return str(candidate)


def compile_tex(directory, name, engine):
    directory = Path(directory).resolve()
    (directory/"build").mkdir(exist_ok=True)
    command = [engine, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
               "-no-shell-escape", "-recorder", "-output-directory=build", name+".tex"]
    last = ""
    for attempt in range(4):
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True, errors="replace", timeout=120)
        last = result.stdout+result.stderr
        if result.returncode:
            raise RuntimeError(last[-7000:])
        log = (directory/"build"/(name+".log")).read_text(errors="replace")
        if attempt >= 1 and not any(s in log for s in ("Rerun to get cross-references right", "Rerun to get outlines right")):
            break
    forbidden = ("undefined references", "multiply-defined labels", "Overfull", "! LaTeX Error", "undefined on input line")
    if any(s in log for s in forbidden):
        raise RuntimeError("Unresolved typesetting issue:\n"+"\n".join(line for line in log.splitlines()
                           if any(s in line for s in forbidden)))
    source = directory/"build"/(name+".pdf")
    shutil.copy2(source, directory/(name+".pdf"))
    return {"pdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "passes": attempt+1,
            "engine": engine, "command": command, "log_sha256": hashlib.sha256(
                (directory/"build"/(name+".log")).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine")
    parser.add_argument("--only", choices=("article", "manuscript"))
    args = parser.parse_args()
    engine = find_engine(args.engine)
    reports = {}
    for name in ((args.only,) if args.only else ("article", "manuscript")):
        reports[name] = compile_tex(ROOT, name, engine)
        print(f"Built {name}.pdf in {reports[name]['passes']} passes; clean references and boxes")
    (ROOT/"build"/"build_manifest.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
