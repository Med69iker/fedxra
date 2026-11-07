"""Build the Kaggle version of the round-9 research lab (development seeds of runs 2 and 3 only).

1. kaggle/lab_bundle/  -> private dataset "<user>/powersnr-lab": lab.zip with the powersnr package, the research
                          lab scripts and the development client reports.
2. kaggle/lab_kernel_<x>/ -> private notebooks "<user>/powersnr-lab-<x>", each running research/lab2.py on a
                          share of the report files (balanced by the number of bank reports).

Usage (from powersnr/):
    python kaggle/build_lab.py
    kaggle datasets create -p kaggle/lab_bundle          # first time; later: datasets version -m "..."
    kaggle kernels push -p kaggle/lab_kernel_a           # ... b, c, d, e
"""
from __future__ import annotations

import glob
import json
import shutil
import zipfile
from pathlib import Path

from build_kaggle import _cell, kaggle_user

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CANDS = ["r9_base", "r9_unif_trim0.3_dd1", "r9_pow_trim0.3_dd1", "r10_pow_it", "r10_fit_ts", "r10_cv"]
EXPS = ["E4", "E9", "E6", "E5", "E10", "E7", "E3", "E11", "E2"]
# notebook -> list of (experiment, file index within that experiment's sorted report files); about 3000 bank reports each
SPLIT = {
    "a": [("E5", 0), ("E6", 0)],
    "b": [("E5", 1), ("E6", 1)],
    "c": [("E9", 0), ("E3", 0)],
    "d": [("E9", 1), ("E3", 1), ("E2", 0)],
    "e": [("E11", 0), ("E11", 1), ("E10", 0), ("E10", 1), ("E7", 0)],
    "f": [("E4", 0), ("E4", 1)],
}


def report_files(exp: str) -> list[str]:
    fs = [f.replace("\\", "/") for f in glob.glob(str(ROOT / "kaggle_runs" / "run[23]" / "**" / f"*_{exp}" /
                                                       "client_reports.parquet"), recursive=True)]
    fs = [f for f in fs if not any(x in f.split("/kaggle_runs/")[1] for x in ("/powersnr/", "/export/"))]
    return sorted(set(fs))


def build_bundle(user: str) -> dict:
    out = HERE / "lab_bundle"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rel = {}
    with zipfile.ZipFile(out / "lab.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in (ROOT / "powersnr").rglob("*.py"):
            z.write(f, f"lab/{f.relative_to(ROOT).as_posix()}")
        for name in ["lab.py", "lab2.py", "candidates.py", "summarize2.py"]:
            z.write(ROOT / "research" / name, f"lab/research/{name}")
        for exp in EXPS:
            rel[exp] = []
            for f in report_files(exp):
                run = f.split("/kaggle_runs/")[1].split("/")[0]
                arc = f"reports/{run}_{Path(f).parent.name}/client_reports.parquet"
                z.write(f, f"lab/{arc}")
                rel[exp].append(arc)
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "powersnr-lab", "id": f"{user}/powersnr-lab", "licenses": [{"name": "CC-BY-SA-4.0"}]}, indent=2))
    return rel


def build_kernel(user: str, x: str, units: list[tuple[str, list[str]]]) -> None:
    slug = f"powersnr-lab-{x}"
    out = HERE / f"lab_kernel_{x}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cells = [
        _cell(f"# Power-SNR round-9/10 lab ({x})\nDevelopment seeds only (runs 2-3). Output: /kaggle/working/lab_{x}.zip",
              "markdown"),
        _cell(f"UNITS = {units!r}\nCANDS = {CANDS!r}\nJOBS = 4"),
        _cell("""
import glob, os, shutil, subprocess, sys, zipfile
work = '/kaggle/working/lab'
hit = glob.glob('/kaggle/input/**/research/lab2.py', recursive=True)
if hit:
    shutil.copytree(os.path.dirname(os.path.dirname(hit[0])), work, dirs_exist_ok=True)
else:
    z = glob.glob('/kaggle/input/**/lab.zip', recursive=True)[0]
    with zipfile.ZipFile(z) as zf:
        zf.extractall('/kaggle/working')
assert os.path.exists(f'{work}/research/lab2.py'), os.listdir(work)
os.chdir(work)
print(os.listdir(work), len(glob.glob('reports/*/client_reports.parquet')))
"""),
        _cell("""
outs = []
for exp, files in UNITS:
    tag = f"{exp}_" + "_".join(os.path.basename(os.path.dirname(f)).split("_")[0] + os.path.basename(os.path.dirname(f)).split("_")[1][-6:] for f in files)
    out = f'/kaggle/working/res/breadth10_{tag}'
    cmd = [sys.executable, 'research/lab2.py', '--reports', *files, '--out', out, '--jobs', str(JOBS), '--cands', *CANDS]
    print('>>>', exp, files, flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, 'PYTHONUTF8': '1'})
    outs.append(out)
shutil.make_archive(f'/kaggle/working/lab_{X}', 'zip', '/kaggle/working/res')
print(outs)
""".replace("{X}", x)),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(
        {"id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
         "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": False,
         "dataset_sources": [f"{user}/powersnr-lab"], "competition_sources": [], "kernel_sources": []}, indent=2))


def main():
    user = kaggle_user()
    rel = build_bundle(user)
    for exp, fs in rel.items():
        print(exp, fs)
    for x, parts in SPLIT.items():
        units = {}
        for exp, i in parts:
            units.setdefault(exp, []).append(rel[exp][i])
        build_kernel(user, x, list(units.items()))
        print(x, list(units.items()))
    print("bundle", round((HERE / "lab_bundle" / "lab.zip").stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
