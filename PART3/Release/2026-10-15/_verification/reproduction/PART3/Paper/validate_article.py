"""Check the article's dependency closure, numerical evidence and PDF portability."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependencies(paper, main="article.tex"):
    paper = Path(paper).resolve()
    pending, visited = [paper/main], set()
    while pending:
        path = pending.pop().resolve()
        assert path.is_relative_to(paper) and path.is_file(), f"Missing/external dependency {path}"
        if path in visited:
            continue
        visited.add(path)
        if path.suffix != ".tex":
            continue
        text = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text(encoding="utf-8"))
        for name in re.findall(r"\\(?:input|include)\{([^}]+)\}", text):
            pending.append(paper/(name if Path(name).suffix else name+".tex"))
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            candidate = paper/name
            if not candidate.suffix:
                candidate = next((candidate.with_suffix(s) for s in (".pdf", ".png", ".jpg")
                                  if candidate.with_suffix(s).exists()), candidate)
            pending.append(candidate)
    return sorted(visited)


def inspect_pdf(path):
    import pymupdf
    doc = pymupdf.open(path)
    assert len(doc) > 0
    assert "Yaroslav P. Lyakhov" in doc.metadata.get("author", "")
    assert "Stepan Suvorov" in doc.metadata.get("author", "")
    texts = [page.get_text() for page in doc]
    assert all(len(t.strip()) > 20 for t in texts), "Empty/unreadable PDF page"
    assert not any(marker in "\n".join(texts) for marker in ("??", "Author names and affiliations to be confirmed"))
    font_refs = {f[0] for page in doc for f in page.get_fonts(full=True)}
    type3_fonts = 0
    for ref in font_refs:
        if doc.xref_get_key(ref, "Subtype") == ("name", "/Type3"):
            # Type 3 fonts embed PDF glyph procedures, not an extractable font file.
            kind, procedures = doc.xref_get_key(ref, "CharProcs")
            assert kind == "xref", f"Missing embedded glyph procedures for font {ref}"
            procedures_ref = int(procedures.split()[0])
            glyphs = doc.xref_get_keys(procedures_ref)
            assert glyphs, f"Empty embedded font {ref}"
            for glyph in glyphs:
                kind, stream = doc.xref_get_key(procedures_ref, glyph)
                assert kind == "xref", (ref, glyph, "Missing glyph stream")
                stream_ref = int(stream.split()[0])
                assert doc.xref_is_stream(stream_ref) and doc.xref_stream(stream_ref), (ref, glyph)
            type3_fonts += 1
        else:
            assert doc.extract_font(ref)[3], f"Unembedded font {ref}"
    assert doc.embfile_count() == 0
    return {"pages": len(doc), "embedded_fonts": len(font_refs), "type3_procedure_fonts": type3_fonts,
            "page_width_pt": doc[0].rect.width,
            "page_height_pt": doc[0].rect.height, "text_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
            "pdf_sha256": sha(path)}


def validate(paper=ROOT, evidence=True):
    paper = Path(paper).resolve()
    files = dependencies(paper)
    tex = "\n".join(p.read_text(encoding="utf-8") for p in files if p.suffix == ".tex")
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    assert len(labels) == len(set(labels)), "Duplicate labels"
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", tex))
    assert refs <= set(labels), refs-set(labels)
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    cited = {key.strip() for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", tex) for key in group.split(",")}
    assert cited == bib, ("Missing/unused bibliography entries", cited-bib, bib-cited)
    assert not re.search(r"\b(?:TODO|TBD|FIXME|XXX)\b|Working manuscript|to be confirmed", tex)
    log_path = paper/"build"/"article.log"
    log = log_path.read_text(errors="replace")
    assert not any(s in log for s in ("undefined references", "multiply-defined labels", "Overfull", "undefined on input line"))
    report = {"dependencies": {str(p.relative_to(paper)).replace("\\", "/"): sha(p) for p in files},
              "labels": len(labels), "citations": len(cited), "pdf": inspect_pdf(paper/"article.pdf")}
    assert abs(report["pdf"]["page_width_pt"]-595.276) < 1 and abs(report["pdf"]["page_height_pt"]-841.89) < 1, "Article should use A4"
    if evidence:
        work = paper.parent/"Work"
        m = json.loads((paper/"evidence"/"manifest.json").read_text(encoding="utf-8"))
        assert sha(paper/"prepare_article_evidence.py") == m["generator_sha256"]
        for n, digest in m["artifact_sha256"].items():
            assert sha(paper/"evidence"/n) == digest
        for n, digest in m["figure_sha256"].items():
            assert sha(paper/n) == digest
        for n, digest in m["source_manifest_sha256"].items():
            assert sha(work/n/"manifest.json") == digest
        for n, digest in m["source_csv_sha256"].items():
            assert sha(work/n) == digest
        def read(study, name):
            with (work/study/name).open(encoding="utf-8") as f:
                return list(csv.DictReader(f))
        old, additional = read("multidata_results", "summary.csv"), read("count_comparator_results", "summary.csv")
        joined = old+additional
        identities = {(r["dataset"], r["radius_ratio"], r["start"], r["method"]) for r in joined}
        assert len(joined) == len(identities) == 756
        cases = {(r["dataset"], r["radius_ratio"], r["start"]) for r in joined}
        assert len(cases) == 108
        budget = read("multidata_results", "budget.csv")+read("count_comparator_results", "budget.csv")
        primary = [r for r in budget if r["oracle"] == "function_evals" and r["budget"] == "1000"]
        assert len(primary) == 756 and all(r["budget_reached_or_certified"] == "True" for r in primary)
        assert sum(r["certified"] == "True" for r in primary if r["method"] == "AC_FW") == 13
        monitors = read("certificate_results", "summary.csv")
        assert len(monitors) == 2160
        assert sum(r["certified"] == "True" for r in monitors if r["profile"] == "scalar") == 129
        assert sum(r["certified"] == "True" for r in monitors if r["profile"] == "known_mu") == 226
        with (paper/"evidence"/"claim_ledger.csv").open(encoding="utf-8") as f:
            ledger = list(csv.DictReader(f))
        assert len(ledger) >= 80
        report["evidence"] = {"primary_method_records": len(joined), "distinct_cases": len(cases),
                              "uncensored_1000f_records": len(primary), "claim_records": len(ledger)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=ROOT)
    parser.add_argument("--source-only", action="store_true", help="Skip CSV provenance checks in a minimal extracted source package")
    args = parser.parse_args()
    report = validate(args.paper, evidence=not args.source_only)
    path = args.paper/"build"/"article_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Article valid: {report['pdf']['pages']} A4 pages, {report['pdf']['embedded_fonts']} embedded fonts, "
          f"{report['citations']} resolved references; {len(report['dependencies'])} local source dependencies")


if __name__ == "__main__":
    main()
