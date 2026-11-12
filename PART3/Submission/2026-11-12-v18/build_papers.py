"""Build the English and Russian PDFs without running scientific controls."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def build(language, args):
    folder = ROOT / ("manuscript_" + language)
    if not folder.exists():
        folder = ROOT / "Paper" / ("ru" if language == "ru" else "")
    env = os.environ.copy()
    engine = "pdflatex" if language == "en" else "xelatex"
    command = [engine]
    if args.format_root:
        config = Path(args.format_root).resolve() / ("manuscript_" + language) / "_texbuild"
        env["TEXMFCNF"] = str(config) + ":/etc/texmf/web2c:/usr/share/texlive/texmf-dist/web2c"
        env["TEXFONTMAPS"] = str(config) + ":"
        fmt = "pdflatex.fmt" if language == "en" else "xelatex_ru.fmt"
        command.append("-fmt=" + str(config / fmt))
    job = "paper" if language == "en" else "paper_ru"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.tex"))}
    records = []
    for number in range(1, args.passes + 1):
        call = command + ["-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-jobname=" + job]
        if language == "ru" and number < args.passes:
            call.append("-no-pdf")
        call.append("article.tex")
        start = time.monotonic()
        log = folder / ("compile_" + str(number) + ".log")
        with log.open("w") as output:
            run = subprocess.run(call, cwd=folder, env=env, stdout=output, stderr=subprocess.STDOUT)
        records.append({"pass": number, "exit": run.returncode, "seconds": time.monotonic() - start})
        print(language, "pass", number, "exit", run.returncode, flush=True)
        if run.returncode:
            print(log.read_text(errors="replace")[-4000:], flush=True)
            break
    logpath = folder / (job + ".log")
    content = logpath.read_text(errors="replace") if logpath.exists() else ""
    warnings = [line for line in content.splitlines() if any(w in line for w in
                ("Overfull", "Missing character", "undefined", "multiply defined", "Rerun to get", "Warning:"))]
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.tex"))}
    receipt = {"release": "v18", "language": language, "passes": records,
               "warning_lines": warnings, "source_changed_during_build": before != after,
               "source_hashes": after, "scientific_calls": 0}
    (folder / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for line in content.splitlines():
        if "Output written" in line:
            print(language + ": " + line, flush=True)
    print(language, "warnings", warnings, flush=True)
    return max(records[-1]["exit"], int(before != after))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=("en", "ru"), default=["en", "ru"])
    parser.add_argument("--passes", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--format-root", help="Optional existing private TeX format root")
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.languages)) as pool:
        codes = list(pool.map(lambda lang: build(lang, args), args.languages))
    sys.exit(max(codes))
