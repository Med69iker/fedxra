"""Download the five credit-scoring datasets into data/raw/<name>/.

Usage (from the powersnr/ folder):
    python scripts/download_data.py                # all datasets
    python scripts/download_data.py gmsc taiwan    # a subset

Sources
-------
gmsc        Kaggle competition "GiveMeSomeCredit"          -> data/raw/gmsc/cs-training.csv
taiwan      UCI id 350 (Default of Credit Card Clients)    -> data/raw/taiwan/taiwan.csv (+ original .xls)
german      UCI id 144 (Statlog German Credit)             -> data/raw/german/german.data
homecredit  Kaggle competition "home-credit-default-risk"  -> data/raw/homecredit/application_train.csv
lendingclub Kaggle dataset "wordsforthewise/lending-club"  -> data/raw/lendingclub/accepted_2007_to_2018Q4.csv.gz

Kaggle credentials are read by the kaggle package from ~/.kaggle/kaggle.json (never printed).
Kaggle competitions require that the user accepted the competition rules on kaggle.com;
a 403 is reported as a failure and NOT worked around.

UCI sets are downloaded directly from the UCI archive (static/public/<id>/...zip); see
download_uci for why ucimlrepo is not used.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

KAGGLE_COMP = {
    "gmsc": ("GiveMeSomeCredit", "cs-training.csv"),
    "homecredit": ("home-credit-default-risk", "application_train.csv"),
}
KAGGLE_DS = {
    "lendingclub": ("wordsforthewise/lending-club", "accepted_2007_to_2018Q4.csv.gz"),
}
UCI = {"taiwan": 350, "german": 144}  # UCI ids (documentation)


def _kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _unzip_member(zip_path: Path, member: str, out_dir: Path) -> Path:
    """Extract exactly one named member (no path traversal: we write by basename)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        match = [n for n in names if Path(n).name == member]
        if not match:
            raise FileNotFoundError(f"{member} not in {zip_path.name}: {names}")
        target = out_dir / member
        with zf.open(match[0]) as src, open(target, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    zip_path.unlink()
    return target


def _finalize(out_dir: Path, fname: str) -> Path:
    """Kaggle may deliver `fname` or `fname.zip`; normalise to `fname`."""
    plain = out_dir / fname
    zipped = out_dir / (fname + ".zip")
    if zipped.exists():
        return _unzip_member(zipped, fname, out_dir)
    if plain.exists():
        # a .gz file sometimes arrives zipped under the same name; detect by magic bytes
        with open(plain, "rb") as fh:
            magic = fh.read(4)
        if magic[:2] == b"PK":
            tmp = plain.with_suffix(plain.suffix + ".zip")
            plain.rename(tmp)
            return _unzip_member(tmp, fname, out_dir)
        return plain
    raise FileNotFoundError(f"download produced neither {plain} nor {zipped}")


def download_kaggle_competition(name: str) -> Path:
    comp, fname = KAGGLE_COMP[name]
    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / fname).exists():
        return out / fname
    api = _kaggle_api()
    api.competition_download_file(comp, fname, path=str(out), quiet=False)
    return _finalize(out, fname)


def download_kaggle_dataset(name: str) -> Path:
    ds, fname = KAGGLE_DS[name]
    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / fname).exists():
        return out / fname
    api = _kaggle_api()
    api.dataset_download_file(ds, fname, path=str(out), quiet=False)
    return _finalize(out, fname)


UCI_URLS = {
    "taiwan": ("https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
               "default of credit card clients.xls"),
    "german": ("https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip",
               "german.data"),
}


def download_uci(name: str) -> Path:
    """Direct download of the UCI archive (ucimlrepo's fetch_ucirepo fails for these two
    legacy datasets with DatasetNotFoundError because UCI hosts no processed CSV for them).

    taiwan: the .xls (header on 2nd row) is converted to taiwan.csv (requires xlrd), keeping the
            original column names (ID, LIMIT_BAL, SEX, ..., AGE, PAY_0, ..., default payment next month).
    german: german.data (space separated, 20 attributes + class 1=good/2=bad) is kept as is.
    """
    import urllib.request

    url, member = UCI_URLS[name]
    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    final = out / ("taiwan.csv" if name == "taiwan" else "german.data")
    if final.exists():
        return final
    zpath = out / f"{name}.zip"
    import ssl

    import certifi

    # Use certifi's CA bundle: the system OpenSSL store of some Python builds reports
    # the UCI chain as expired. Certificate verification stays ON.
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, timeout=120, context=ctx) as resp, open(zpath, "wb") as fh:
        fh.write(resp.read())
    extracted = _unzip_member(zpath, member, out)
    if name == "taiwan":
        df = pd.read_excel(extracted, header=1, engine="xlrd")
        df.to_csv(final, index=False)
    return final


def main(names: list[str]) -> int:
    status = {}
    for name in names:
        print(f"== {name}", flush=True)
        try:
            if name in KAGGLE_COMP:
                p = download_kaggle_competition(name)
            elif name in KAGGLE_DS:
                p = download_kaggle_dataset(name)
            elif name in UCI:
                p = download_uci(name)
            else:
                raise ValueError(f"unknown dataset {name}")
            status[name] = f"OK {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} MB)"
        except Exception as e:  # report and continue with the other datasets (not silent)
            msg = str(e).splitlines()[0][:300] if str(e) else ""
            status[name] = f"FAILED {type(e).__name__}: {msg}"
        print(f"   {status[name]}", flush=True)
    print("\nSummary")
    for k, v in status.items():
        print(f"  {k:12s} {v}")
    return 0 if all(v.startswith("OK") for v in status.values()) else 1


if __name__ == "__main__":
    sel = sys.argv[1:] or ["gmsc", "taiwan", "german", "homecredit", "lendingclub"]
    sys.exit(main(sel))
