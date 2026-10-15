"""Download only the two named daily-data files from the official UCI archive."""

import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1] / "Data" / "bike_sharing"
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def main():
    with urlopen(URL, timeout=30) as response:
        payload = response.read()
    ROOT.mkdir(parents=True, exist_ok=True)
    files = {}
    with ZipFile(io.BytesIO(payload)) as archive:
        for name in ("day.csv", "Readme.txt"):
            content = archive.read(name)
            target = ROOT / name
            if target.exists() and target.read_bytes() != content:
                raise RuntimeError(f"Existing data differs from official download: {target}")
            target.write_bytes(content)
            files[name] = hashlib.sha256(content).hexdigest()
    metadata = {
        "title": "Bike Sharing", "creator": "Hadi Fanaee-T", "year": 2013,
        "doi": "10.24432/C5W894", "download_url": URL,
        "source_page": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "archive_sha256": hashlib.sha256(payload).hexdigest(), "files_sha256": files,
        "changes": "Only day.csv and original Readme.txt retained; raw contents unchanged.",
    }
    (ROOT / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved verified daily data and attribution to {ROOT}")


if __name__ == "__main__":
    main()
