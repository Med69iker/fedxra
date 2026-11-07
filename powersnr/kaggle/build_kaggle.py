"""Build the two things Kaggle needs:

1. kaggle/bundle/   -> a private Kaggle *dataset* "<user>/powersnr-code" holding the code,
                       the configs and the small UCI files (taiwan, german).
2. kaggle/kernel/   -> a private Kaggle *notebook* "<user>/powersnr-experiments" that mounts
                       the code bundle plus the competition/dataset inputs and runs the configs.

Usage (from the powersnr/ folder):
    python kaggle/build_kaggle.py                         # build only (nothing is uploaded)
    python kaggle/build_kaggle.py --experiments E1 E2 E8  # choose what the notebook runs

Upload (after checking the generated folders):
    kaggle datasets create -p kaggle/bundle --dir-mode zip     # first time
    kaggle datasets version -p kaggle/bundle -m "update" --dir-mode zip   # later
    kaggle kernels push -p kaggle/kernel
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

# Kaggle sources mounted under /kaggle/input/.
COMPETITIONS = ["GiveMeSomeCredit", "home-credit-default-risk"]
DATASETS = ["wordsforthewise/lending-club"]
SHIPPED_RAW = ["taiwan", "german"]   # small UCI files travel inside the code bundle


def kaggle_user() -> str:
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    return json.loads(cfg.read_text())["username"]


def build_bundle(user: str) -> Path:
    out = HERE / "bundle"
    if out.exists():
        shutil.rmtree(out)  # generated folder only
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for d in ["powersnr", "analysis", "configs", "tests"]:
        shutil.copytree(ROOT / d, out / d, ignore=ignore)
    shutil.copy(ROOT / "requirements.txt", out / "requirements.txt")
    for name in SHIPPED_RAW:
        src = ROOT / "data" / "raw" / name
        if not src.exists():
            raise FileNotFoundError(f"{src} missing: run scripts/download_data.py first")
        shutil.copytree(src, out / "data" / "raw" / name, ignore=ignore)
    meta = {"title": "powersnr-code", "id": f"{user}/powersnr-code",
            "licenses": [{"name": "CC-BY-SA-4.0"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def _cell(src: str, kind: str = "code") -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build_kernel(user: str, experiments: list[str], jobs: int, seeds: int | None, slug: str = "powersnr-experiments") -> Path:
    out = HERE / ("kernel" if slug == "powersnr-experiments" else f"kernel_{slug}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cells = [
        _cell("# Power-SNR experiments\nRuns the selected configs from the `powersnr-code` dataset and saves "
              "results, figures and LaTeX tables to `/kaggle/working/output.zip`.", "markdown"),
        _cell(f"""
EXPERIMENTS = {experiments!r}   # config prefixes in configs/, e.g. "E1" -> configs/E1_*.yaml
JOBS = {jobs}                    # Kaggle CPU sessions have 4 cores
SEEDS = {seeds!r}                # None = use the number of seeds in each config
"""),
        _cell("""
import glob, os, shutil, subprocess, sys
src = glob.glob('/kaggle/input/**/powersnr-code', recursive=True) or glob.glob('/kaggle/input/powersnr-code*')
src = src[0]
work = '/kaggle/working/powersnr'
if os.path.exists(work):
    shutil.rmtree(work)
shutil.copytree(src, work)
# The bundle is uploaded with --dir-mode zip; Kaggle may or may not extract the folder archives.
import zipfile
for z in glob.glob(os.path.join(work, '*.zip')):
    name = os.path.splitext(os.path.basename(z))[0]
    with zipfile.ZipFile(z) as zf:
        zf.extractall(os.path.join(work, name))
    os.remove(z)
for name in ['powersnr', 'analysis', 'configs', 'tests', 'data']:
    inner = os.path.join(work, name, name)
    if os.path.isdir(inner) and os.listdir(os.path.join(work, name)) == [name]:
        tmp = os.path.join(work, name + '__tmp')
        shutil.move(inner, tmp)
        os.rmdir(os.path.join(work, name))
        shutil.move(tmp, os.path.join(work, name))
assert os.path.exists(os.path.join(work, 'powersnr', 'runner.py')), os.listdir(work)
os.chdir(work)
sys.path.insert(0, work)
print('code from', src)
for f in sorted(glob.glob('/kaggle/input/**/*.csv*', recursive=True))[:20]:
    print(f, round(os.path.getsize(f) / 1e6, 1), 'MB')
"""),
        _cell("""
import numpy, pandas, sklearn, shap, xgboost, lightgbm
print({m.__name__: m.__version__ for m in [numpy, pandas, sklearn, shap, xgboost, lightgbm]})
subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'] if os.path.exists('tests') else ['true'])
"""),
        _cell("""
runs = []
for exp in EXPERIMENTS:
    for cfg in sorted(glob.glob(f'configs/{exp}_*.yaml')) or sorted(glob.glob(f'configs/{exp}.yaml')):
        cmd = [sys.executable, '-m', 'powersnr.runner', '--config', cfg, '--jobs', str(JOBS)]
        if SEEDS is not None:
            cmd += ['--seeds', str(SEEDS)]
        print('>>>', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        runs.append(sorted(glob.glob('results/*'))[-1])
print(runs)
"""),
        _cell("""
for run in runs:
    # assets are a convenience; a plotting failure must not lose the results
    subprocess.run([sys.executable, '-m', 'analysis.make_paper_assets', '--run', run,
                    '--out', os.path.join('paper_assets', os.path.basename(run))], check=False)
# Archive only results + assets (not the code copy).
os.makedirs('/kaggle/working/export', exist_ok=True)
for d in ['results', 'paper_assets']:
    if os.path.exists(d):
        shutil.copytree(d, f'/kaggle/working/export/{d}', dirs_exist_ok=True)
shutil.make_archive('/kaggle/working/output', 'zip', '/kaggle/working/export')
print('saved /kaggle/working/output.zip')
"""),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                        "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / "powersnr-experiments.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{user}/{slug}", "title": slug,
            "code_file": "powersnr-experiments.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_internet": False,
            "dataset_sources": [f"{user}/powersnr-code", *DATASETS],
            "competition_sources": COMPETITIONS, "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["E1", "E2", "E8"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--slug", default="powersnr-experiments", help="Kaggle notebook slug (one per parallel run)")
    ap.add_argument("--kernel-only", action="store_true")
    a = ap.parse_args()
    user = kaggle_user()
    if not a.kernel_only:
        print("bundle:", build_bundle(user))
    print("kernel:", build_kernel(user, a.experiments, a.jobs, a.seeds, a.slug))


if __name__ == "__main__":
    main()
