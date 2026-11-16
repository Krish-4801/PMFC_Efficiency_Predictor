"""Download and verify the FPV-DB dataset used by the drone weight component.

Usage:
    python data/download_data.py            # download if missing, verify hashes
    python data/download_data.py --force    # re-download

The canonical dataset is regenerated daily upstream, so hashes pin the exact
snapshot used for the reported results (accessed 2026-11-16). If upstream has
moved, the script reports a hash mismatch instead of silently changing results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
BASE = "https://raw.githubusercontent.com/fpvdb/fpv-db-data/main/"

FILES = {
    "quads.csv":    "dc5de7d66b22fdf5892b71285a96b06727eea887d62099d607478ece95234fe7",
    "motors.csv":   "99e0828a5eda528eee04fa6f1f3d8186f5cbe761329ee19a37104e6ac4591c0a",
    "batteries.csv": "6daf6fd6d294c1834af3807aedc66d993399adf593917631c40fe709dfe9a97c",
    "props.csv":    "cfaa9484219da46a6738fd40dedfb545f31c9f058aead633b31bb65d05105002",
    "stacks.csv":   "0bab72142ead17cdd58b843f8d840e4a0277c56079236a8f03398d48abc6803e",
    "cameras.csv":  "92bcbc1cbf8b9d5f84e76c6596d5f2b224081c59235c15f04724fd91b637a03a",
    "vtx.csv":      "e2d1d5f267ddefc5f856bad7cf173a749bedbf3d03a96e61d092679202096fd6",
    "README.md":    None,
    "LICENSE":      None,
}

PROVENANCE = {
    "name": "FPV-DB: open dataset of FPV drone component specifications",
    "doi": "10.5281/zenodo.23613812",
    "url": "https://github.com/fpvdb/fpv-db-data",
    "license": "CC BY 4.0",
    "attribution": "FPV-DB (fpv-db.com)",
    "snapshot_date": "2026-11-16",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main(force: bool = False) -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    problems = []
    for name, expected in FILES.items():
        dest = RAW / name
        if dest.exists() and not force:
            status = "cached"
        else:
            print(f"downloading {name} ...")
            try:
                urllib.request.urlretrieve(BASE + name, dest)
            except Exception as exc:  # pragma: no cover - network dependent
                problems.append(f"{name}: download failed ({exc})")
                continue
            status = "downloaded"
        if expected:
            got = sha256(dest)
            if got != expected:
                problems.append(
                    f"{name}: hash mismatch (expected {expected[:12]}..., got {got[:12]}...). "
                    "Upstream may have been regenerated; results are pinned to the snapshot."
                )
            else:
                print(f"  {name:16s} {status:10s} sha256 OK")
        else:
            print(f"  {name:16s} {status}")

    (RAW / "PROVENANCE.json").write_text(json.dumps(PROVENANCE, indent=2))
    if problems:
        print("\nWARNING:")
        for p in problems:
            print(" -", p)
        return 1
    print("\nDataset verified. Snapshot:", PROVENANCE["snapshot_date"])
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    sys.exit(main(ap.parse_args().force))
