"""Build and check the Russian copy without modifying the English publication."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

PAPER = Path(__file__).resolve().parent


def main():
    engine = shutil.which("xelatex")
    if engine is None:
        raise RuntimeError("XeLaTeX is required (with fontspec, babel Russian and Times New Roman)")
    subprocess.run([sys.executable, str(PAPER / "ru" / "make_figures.py")], check=True)
    build = PAPER / "build_ru"
    build.mkdir(exist_ok=True)
    command = [engine, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
               "-no-shell-escape", "-recorder", "-output-directory=build_ru", "article_ru.tex"]
    for attempt in range(4):
        result = subprocess.run(command, cwd=PAPER, capture_output=True, text=True,
                                errors="replace", timeout=120)
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-7000:])
        log = (build / "article_ru.log").read_text(errors="replace")
        if attempt >= 1 and not any(s in log for s in
                ("Rerun to get cross-references right", "Rerun to get outlines right")):
            break
    forbidden = ("undefined references", "multiply-defined labels", "Overfull",
                 "! LaTeX Error", "undefined on input line", "Missing character")
    if any(s in log for s in forbidden):
        raise RuntimeError("Unresolved typesetting problem: " + "\n".join(
            line for line in log.splitlines() if any(s in line for s in forbidden)))
    from validate_article import dependencies
    files = dependencies(PAPER, "article_ru.tex")
    text = "\n".join(p.read_text(encoding="utf-8") for p in files if p.suffix == ".tex")
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    assert len(labels) == len(set(labels))
    assert set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text)) <= set(labels)
    cited = {k.strip() for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", text)
             for k in group.split(",")}
    assert cited == set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    # Numbered mathematical statements must match the English source exactly,
    # apart from whitespace, terminal punctuation and translated text conditions.
    def equations(source):
        values = []
        for match in re.finditer(r"\\begin\{(equation|align)\}([\s\S]*?)\\end\{\1\}", source):
            value = re.sub(r"\\text\{[^}]*\}", "TEXT", match.group(2))
            values.append(re.sub(r"\s+", "", value).rstrip(".,"))
        return values
    checked = 0
    for name in ("article_intro", "article_fw", "article_away", "article_zero",
                 "article_experiments", "article_appendix"):
        original = (PAPER / (name + ".tex")).read_text(encoding="utf-8")
        translated = (PAPER / "ru" / (name + ".tex")).read_text(encoding="utf-8")
        assert equations(original) == equations(translated), name
        checked += len(equations(original))
    for path in (PAPER / "ru" / "evidence").glob("*.tex"):
        original = (PAPER / "evidence" / path.name).read_text(encoding="utf-8")
        translated = path.read_text(encoding="utf-8")
        numeric_rows = lambda t: [line.strip() for line in t.splitlines()
                                  if " & " in line and re.search(r"&\s*\d", line)]
        assert numeric_rows(original) == numeric_rows(translated), path.name
    target = PAPER / "article_ru.pdf"
    shutil.copy2(build / "article_ru.pdf", target)
    helpers = [Path(__file__).resolve(), PAPER / "ru" / "make_figures.py"]
    report = {"engine": engine, "passes": attempt + 1, "numbered_math_environments": checked,
              "labels": len(labels), "citations": len(cited), "translated_tables": 5,
              "pdf_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "source_sha256": {str(p.relative_to(PAPER)).replace("\\", "/"):
                                hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "build_helpers_sha256": {str(p.relative_to(PAPER)).replace("\\", "/"):
                                       hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}}
    (PAPER / "ru" / "translation_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"Russian PDF built: {checked} numbered math environments, {len(labels)} labels, "
          f"{len(cited)} citations; all five tables retain their original numeric rows")


if __name__ == "__main__":
    main()
