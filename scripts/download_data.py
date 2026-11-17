"""Download the exact 60 original Parquet files from the pinned 3W revision."""

import argparse, hashlib, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--destination", type=Path, default=ROOT / "raw")
    a = p.parse_args()
    m = json.loads((ROOT / "data/manifest.json").read_text())
    for f in m["files"]:
        target = a.destination / f["path"]
        if target.exists():
            data = target.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/petrobras/3W/{m['commit']}/{f['path']}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        assert blob_sha(data) == f["sha"], f["path"]
        if "size" in f:
            assert len(data) == f["size"], f["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f["path"])


if __name__ == "__main__":
    main()
