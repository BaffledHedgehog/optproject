"""Package the existing v18 papers, sources and evidence without rerunning experiments."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
TEX_NAMES = (
    "article.tex", "intro.tex", "model.tex", "stopping.tex",
    "normal_gradient.tex", "experiments.tex", "references.tex",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_zip(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (2026, 11, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(entries)
        for name, data in entries.items():
            assert archive.read(name) == data


def source_archive(language):
    source = ROOT / "Paper" / ("ru" if language == "ru" else "")
    entries = {name: (source / name).read_bytes() for name in TEX_NAMES}
    if language == "ru":
        entries["experiments.tex"] = entries["experiments.tex"].replace(b"../figures/", b"figures/")
    figure = "figures/boundary_normal_benchmark.pdf"
    entries[figure] = (ROOT / "Paper" / figure).read_bytes()
    engine = "xelatex" if language == "ru" else "pdflatex"
    note = (
        "v18 LaTeX source archive, 12 November 2026.\n"
        f"Compile article.tex with {engine} three times.\n"
        "The existing figure is included; no experiment is run by LaTeX.\n"
    )
    if language == "ru":
        note += (
            "Needs fontspec, polyglossia, Liberation Serif/Sans and DejaVu Sans Mono.\n"
            "The figure path was normalized from ../figures/ to figures/ for this archive.\n"
        )
    entries["BUILD.txt"] = note.encode()
    filename = "arxiv_source_ru_v18.zip" if language == "ru" else "arxiv_source_v18.zip"
    write_zip(ROOT / filename, entries)
    return filename


def main():
    for source, target in (("Paper/paper.pdf", "paper.pdf"), ("Paper/ru/paper_ru.pdf", "paper_ru.pdf")):
        shutil.copyfile(ROOT / source, ROOT / target)
    sources = [source_archive(language) for language in ("en", "ru")]
    validation_path = ROOT / "Evidence/release_validation.json"
    validation = json.loads(validation_path.read_text())
    validation["archive_checks"] = {
        "source_archives": sources,
        "crc_and_content_match": True,
        "tex_inputs_and_figure_included": True,
        "ru_figure_path_normalized": True,
        "scientific_runs_during_packaging": 0,
    }
    validation["sha256"] = {
        name: sha256(ROOT / name) for name in
        ("paper.pdf", "paper_ru.pdf", "guide_professor.txt", *sources,
         "Code/benchmark_boundary.py", "Code/check_directional.py")
    }
    validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n")
    names = [
        "paper.pdf", "paper_ru.pdf", "guide_professor.txt", "README.md", "README_RU.txt",
        "CHANGES_v18.txt", "ARXIV_METADATA.txt", "build_papers.py", "package_release.py", *sources,
    ]
    for folder, patterns in (("Code", ("*.py", "*.txt")), ("Evidence", ("*.json", "*.csv", "*.txt"))):
        for pattern in patterns:
            names.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / folder).glob(pattern)))
    for language in ("en", "ru"):
        prefix = "Paper/ru/" if language == "ru" else "Paper/"
        names.extend(prefix + name for name in TEX_NAMES)
        names.append(prefix + "build_receipt.json")
    names.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / "Paper/figures").glob("*")) if p.is_file())
    names = sorted(set(names))
    entries = {name: (ROOT / name).read_bytes() for name in names}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()}
    entries["SHA256SUMS.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    write_zip(ROOT / "reading_packet_v18.zip", entries)
    for name in ("paper.pdf", "paper_ru.pdf", *sources, "reading_packet_v18.zip"):
        path = ROOT / name
        print(f"{name}: {path.stat().st_size} bytes; SHA256 {sha256(path)}")


if __name__ == "__main__":
    main()
