"""Freeze / verify the evidence used by the manuscript (run 7, seeds 5000+; see common.py).

    python tools/evidence.py freeze   # write evidence/MANIFEST.json (SHA-256 of every result file)
    python tools/evidence.py verify   # fail if any file changed or is missing
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from common import HERE, RUN as RUN4, RUN_NAME
MANIFEST = HERE / "evidence" / "MANIFEST.json"
PATTERNS = ["*/results/*/results.parquet", "*/results/*/client_reports.parquet",
            "*/results/*/config.yaml", "*/results/*/run_meta.json"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files() -> list[Path]:
    return sorted(p for pat in PATTERNS for p in RUN4.glob(pat))


def freeze() -> None:
    entries = {str(p.relative_to(RUN4)).replace("\\", "/"): sha256(p) for p in files()}
    MANIFEST.write_text(json.dumps({"root": f"powersnr/kaggle_runs/{RUN_NAME}", "files": entries}, indent=1))
    print(f"froze {len(entries)} files")


def verify() -> None:
    man = json.loads(MANIFEST.read_text())["files"]
    bad = [k for k, h in man.items() if not (RUN4 / k).exists() or sha256(RUN4 / k) != h]
    if bad:
        sys.exit(f"EVIDENCE CHANGED OR MISSING: {bad}")
    print(f"evidence OK ({len(man)} files)")


if __name__ == "__main__":
    {"freeze": freeze, "verify": verify}[sys.argv[1]]()
