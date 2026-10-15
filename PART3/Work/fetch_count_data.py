"""Fetch two named, unmodified files from official CC BY 4.0 UCI archives."""

import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = next(p for p in Path(__file__).resolve().parents if p.name.upper() == "PART3")/"Data"
DATASETS = {
    "seoul": {"title": "Seoul Bike Sharing Demand", "year": 2020,
              "doi": "10.24432/C5F62R", "creator": "UCI dataset citation (no named creator listed)",
              "url": "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip",
              "page": "https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand",
              "file": "SeoulBikeData.csv"},
    "metro": {"title": "Metro Interstate Traffic Volume", "year": 2019,
              "doi": "10.24432/C5X60B", "creator": "John Hogue",
              "url": "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip",
              "page": "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
              "file": "Metro_Interstate_Traffic_Volume.csv.gz"},
}


def main():
    for name, metadata in DATASETS.items():
        with urlopen(metadata["url"], timeout=30) as response:
            payload = response.read()
        with ZipFile(io.BytesIO(payload)) as archive:
            content = archive.read(metadata["file"])
        directory = ROOT/name
        directory.mkdir(parents=True, exist_ok=True)
        target = directory/metadata["file"]
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError(f"Existing data differs from official download: {target}")
        target.write_bytes(content)
        source = {**metadata, "license": "CC BY 4.0",
                  "license_url": "https://creativecommons.org/licenses/by/4.0/",
                  "archive_sha256": hashlib.sha256(payload).hexdigest(),
                  "file_sha256": hashlib.sha256(content).hexdigest(),
                  "changes": "Named file retained byte-for-byte; preprocessing is in count_problems.py."}
        (directory/"source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
        print(f"Saved {name}: {len(content)} bytes; sha256={source['file_sha256']}")


if __name__ == "__main__":
    main()
