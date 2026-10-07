"""Generate the Google Colab notebook from the repository (single source of truth).

The notebook trains every registered run of stages B, C and D (configs/experiments.yaml), runs the analyses of
stage E (robustness, counterfactual probes, CUDA efficiency) and the XLS-R comparator.  It is resumable: results
are copied to Google Drive after each run and runs already on Drive are skipped.

By default the notebook takes the code from the public repository at a pinned commit (PUBLIC_COMMIT), recomputes
the code version (SHA-256 over src/bscan, configs and tests) and stops if it differs from the version of this
checkout; the metadata table is checked against its SHA-256.  With --embedded the source files and the metadata
table are embedded in the notebook instead (compressed, with a SHA-256 manifest), for a code version that is not
published yet.  Both notebooks run the same code and use the same Drive folder.

Usage:
    python .scripts/build_colab_notebook.py              (writes code/BSCAN_full_pipeline.ipynb)
    python .scripts/build_colab_notebook.py --embedded   (writes code/BSCAN_full_pipeline_embedded.ipynb)
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMBED = sorted([p for p in (ROOT / "src" / "bscan").rglob("*.py")] + [p for p in (ROOT / "configs").glob("*.yaml")]
               + [ROOT / "tests" / "__init__.py", ROOT / "tests" / "smoke_test.py"])
# public repository and the commit the notebook checks out: its src/bscan, configs, tests and metadata table must give
# the current code version (the notebook checks it).  Update after publishing a new code version; never rewrite the
# public history, or this commit disappears.
PUBLIC_REPO = "https://github.com/hamidhosen42/BSCAN-Dual-Branch-Spectro-Cepstral-Residual-Attention-Network-for-Bengali-Deepfake-Audio-Detection"
PUBLIC_COMMIT = "98fe3aa2dbb9411c671fb8a91006229490bbcb29"
ARCHIVES = {
    "banglafake": {"url": "https://huggingface.co/datasets/sifat1221/banglaFake/resolve/main/final_data.zip",
                   "file": "data/banglafake/final_data.zip", "extract_to": "data/banglafake/extracted",
                   "sha256": "3034353ef96a9f8c8a8118e65682b31b36de14c4f9159a2ee0e5cd5a9dfaa8b0", "bytes": 5642775456},
    # Mendeley version 4 (DOI 10.17632/4ftmwt86vr.4); its 4,500 audio files are byte-identical to version 1
    "mendeley": {"url": "https://data.mendeley.com/public-api/zip/4ftmwt86vr/download/4",
                 "file": "data/mendeley/4ftmwt86vr-4.zip", "extract_to": "data/mendeley/extracted",
                 "sha256": "aefdfbf6351849c186717767cd1c901a91e7033ccc1bc6af65161995bb86f4d3", "bytes": 519835011},
}
# one notebook for the whole study (written to code/): CUDA smoke test, every registered training run
# (stages B, C, D: main models, representation study, trained ablation, baselines, seeds), the analyses
# (stage E: robustness, counterfactual probes, CUDA latency) and the XLS-R comparator.  Resumable via Google Drive.
NOTEBOOKS = {
    "full_pipeline": (["B", "C", "D", "E"], True,
                      "Complete pipeline: smoke test, all training runs incl. the trained ablation, and all analyses"),
}


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


def b64gz(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data, mtime=0)).decode()


def payload():
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in EMBED}
    manifest = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    ver = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    return files, manifest, ver


def code_step(ver: str, meta_sha: str, embedded: tuple | None) -> tuple[str, str, dict]:
    """Step 5: the code and the metadata table, either embedded (files, manifest, meta_b64) or checked out from
    PUBLIC_REPO at PUBLIC_COMMIT.  Both write the same files to WORK and define `meta`."""
    if embedded:
        files, manifest, meta_b64 = embedded
        enc = {k: b64gz(v) for k, v in files.items()}
        return "5. Code (embedded)", "The source files and the metadata table, compressed; each is checked against its SHA-256.", code(f"""
import base64, gzip
FILES = {json.dumps(enc)}
MANIFEST = {json.dumps(manifest)}
for rel, blob in FILES.items():
    data = gzip.decompress(base64.b64decode(blob))
    assert hashlib.sha256(data).hexdigest() == MANIFEST[rel], rel
    p = os.path.join(WORK, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)
open(f"{{WORK}}/CODE_VERSION.txt", "w").write(CODE_VERSION)
META = gzip.decompress(base64.b64decode("{meta_b64}"))
assert hashlib.sha256(META).hexdigest() == "{meta_sha}"
os.makedirs(f"{{WORK}}/metadata", exist_ok=True)
open(f"{{WORK}}/metadata/unified_metadata.csv", "wb").write(META)
import pandas as pd
meta = pd.read_csv(f"{{WORK}}/metadata/unified_metadata.csv", dtype={{"utterance_id": str}}, low_memory=False)
print(f"code {{CODE_VERSION}}: {{len(FILES)}} files verified; metadata: {{len(meta)}} recordings")
""")
    repo = PUBLIC_REPO.removeprefix("https://")
    return "5. Code from the public repository", (
        f"Checks out commit `{PUBLIC_COMMIT[:7]}` of [{repo}]({PUBLIC_REPO}), recomputes the code version "
        f"(SHA-256 over `src/bscan`, `configs` and `tests`) and stops unless it is `{ver}`. "
        "The metadata table is checked against its SHA-256."), code(f"""
REPO = "{PUBLIC_REPO}"
COMMIT = "{PUBLIC_COMMIT}"
META_SHA256 = "{meta_sha}"
CLONE = "/content/bscan_repo"
def git(*args):
    r = subprocess.run(["git", "-C", CLONE, *args], capture_output=True, text=True)
    assert r.returncode == 0, f"git {{' '.join(args)}}: {{r.stderr}}"
    return r.stdout.strip()
os.makedirs(CLONE, exist_ok=True)
if not os.path.isdir(f"{{CLONE}}/.git"): git("init", "-q")
if subprocess.run(["git", "-C", CLONE, "cat-file", "-e", COMMIT + "^{{commit}}"], capture_output=True).returncode:
    git("fetch", "-q", "--depth", "1", REPO, COMMIT)
git("checkout", "-q", "-f", COMMIT)
# code version: SHA-256 over the files of src/bscan, configs and tests (same rule as .scripts/build_colab_notebook.py)
code_files = sorted(glob.glob("src/bscan/**/*.py", root_dir=CLONE, recursive=True) + glob.glob("configs/*.yaml", root_dir=CLONE)
                    + ["tests/__init__.py", "tests/smoke_test.py"])
MANIFEST = {{f: hashlib.sha256(open(f"{{CLONE}}/{{f}}", "rb").read()).hexdigest() for f in code_files}}
ver = hashlib.sha256(json.dumps(MANIFEST, sort_keys=True).encode()).hexdigest()[:12]
assert ver == CODE_VERSION, f"commit {{COMMIT[:7]}} holds code version {{ver}}, not {{CODE_VERSION}}"
assert hashlib.sha256(open(f"{{CLONE}}/metadata/unified_metadata.csv", "rb").read()).hexdigest() == META_SHA256
for rel in code_files + ["metadata/unified_metadata.csv"]:  # only the verified files go to the working directory
    os.makedirs(os.path.dirname(f"{{WORK}}/{{rel}}"), exist_ok=True)
    shutil.copy(f"{{CLONE}}/{{rel}}", f"{{WORK}}/{{rel}}")
open(f"{{WORK}}/CODE_VERSION.txt", "w").write(CODE_VERSION)
import pandas as pd
meta = pd.read_csv(f"{{WORK}}/metadata/unified_metadata.csv", dtype={{"utterance_id": str}}, low_memory=False)
print(f"code {{CODE_VERSION}} from commit {{COMMIT[:7]}}: {{len(MANIFEST)}} files verified; metadata: {{len(meta)}} recordings")
""")


def build(key: str, ver: str, meta_sha: str, embedded: tuple | None = None) -> dict:
    stages, ssl, title = NOTEBOOKS[key]
    origin = ("The source files and the metadata table are embedded in step 5 and checked against their SHA-256 before use."
              if embedded else
              f"Step 5 takes the code from the public repository at commit `{PUBLIC_COMMIT[:7]}` and stops unless it "
              "is this code version (a SHA-256 hash of `src/bscan`, `configs` and `tests`).")
    intro = md(f"""
# BSCAN — {title}

**Code version `{ver}`.** {origin} Generated by `.scripts/build_colab_notebook.py`.

**How to run:** Runtime → Change runtime type → **GPU** (T4 or better) → Runtime → **Run all**. Allow Google Drive access when asked.
Results are saved to `MyDrive/BSCAN_{ver}/` after every run. If Colab disconnects, simply **Run all again**: finished runs are skipped.
Each code version uses its own Drive folder, so results of different code versions never mix.
At the end the notebook downloads `BSCAN_results_{ver}.zip` (a copy stays on Drive); import it into the project with `python .scripts/build_all.py --import <zip>`.

Stages {', '.join(stages) if stages else '—'} of `configs/experiments.yaml`{' and the XLS-R comparator' if ssl else ''}.
The data are downloaded from their original hosts and every file is checked against its SHA-256. The CUDA smoke test must pass
before anything is trained. Do not edit the code cells: the results would then no longer belong to this code version.
""")
    steps = [
        ("1. Settings", "Nothing needs to be changed here.", code(f"""
USE_DRIVE = True
DRIVE_DIR = "/content/drive/MyDrive/BSCAN_{ver}"   # one folder per code version: no mixing
STAGES = {stages!r}          # B, C, D: training stages from configs/experiments.yaml; E: analyses
RUN_OPTIONAL_SSL = {ssl!r}
CODE_VERSION = {ver!r}
WORK = "/content/bscan"
""")),
        ("2. GPU check", "Stops if no CUDA GPU is attached.", code("""
import subprocess, sys, os, json, time, shutil, hashlib, glob
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "nvidia-smi not found")
import torch
assert torch.cuda.is_available(), "No CUDA GPU: Runtime -> Change runtime type -> GPU, then Run all."
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0))
TIMING = {}
def tick(k): TIMING[k] = time.time()
def tock(k): TIMING[k] = round(time.time() - TIMING[k], 1); print(f"[time] {k}: {TIMING[k]} s")
""")),
        ("3. Google Drive", f"Every finished run is copied to `MyDrive/BSCAN_{ver}/`.", code("""
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(f"{DRIVE_DIR}/runs", exist_ok=True)
""")),
        ("4. Python packages", "librosa and soundfile are pinned; PyTorch is the one Colab provides.", code("""
tick("pip")
pkgs = ["librosa==1.0.0", "soundfile==0.14.0", "pyyaml", "tabulate", "ultralytics-thop"]
if RUN_OPTIONAL_SSL: pkgs.append("transformers")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
tock("pip")
import importlib
for m in ["numpy", "scipy", "sklearn", "librosa", "soundfile", "pandas", "yaml"]:
    print(m, getattr(importlib.import_module(m), "__version__", "?"))
""")),
        code_step(ver, meta_sha, embedded),
        ("6. Data", "BanglaFake (Hugging Face) and the Mendeley corpus (version 4) are downloaded from their original hosts; "
                    "every audio file is checked against the SHA-256 in the metadata table.", code(f"""
ARCHIVES = {json.dumps(ARCHIVES, indent=1)}
def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()
tick("download")
for name, a in ARCHIVES.items():
    dst = os.path.join(WORK, a["file"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not (os.path.exists(dst) and os.path.getsize(dst) == a["bytes"]):
        subprocess.run(["curl", "-L", "--fail", "--retry", "10", "--retry-delay", "10", "-C", "-", "-s", "-S", "-o", dst, a["url"]], check=True)
    got = sha256_file(dst)
    if got != a["sha256"]:  # not fatal: every audio file is verified individually below
        print(f"WARNING {{name}}: archive SHA-256 {{got[:16]}}... differs from the recorded {{a['sha256'][:16]}}...")
    out = os.path.join(WORK, a["extract_to"])
    if not os.path.exists(out + "/.done"):
        os.makedirs(out, exist_ok=True); subprocess.run(["unzip", "-q", "-o", dst, "-d", out], check=True)
        open(out + "/.done", "w").write("ok")
tock("download")
from concurrent.futures import ThreadPoolExecutor
root = f"{{WORK}}/data"
def check(row):
    p = os.path.join(root, row[0])
    return row[0] if (not os.path.exists(p) or sha256_file(p) != row[1]) else None
with ThreadPoolExecutor(8) as ex:
    bad = [r for r in ex.map(check, meta[["audio_path", "sha256"]].itertuples(index=False)) if r]
print("audio files verified:", len(meta) - len(bad), "/", len(meta)); assert not bad, bad[:10]
""")),
        ("7. Stage A: CUDA smoke test", "Must pass before anything is trained.", code("""
# Stage A: smoke test on CUDA (must pass)
tick("smoke")
os.chdir(WORK)
env = dict(os.environ, PYTHONPATH=f"{WORK}/src", BSCAN_CODE_VERSION=CODE_VERSION)  # recorded in every environment.json
r = subprocess.run([sys.executable, "-m", "tests.smoke_test", "--device", "cuda", "--out", f"{WORK}/results/smoke_cuda",
                    "--metadata", f"{WORK}/metadata/unified_metadata.csv", "--audio_root", f"{WORK}/data"],
                   env=env, capture_output=True, text=True)
print("\\n".join(l for l in r.stdout.splitlines() if l.startswith("[") or "SMOKE" in l))
tock("smoke")
assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
""")),
        ("8. Stages B–D: training and evaluation", "Every registered neural run: main models, representation study, "
         "trained ablation, baselines and extra seeds. Runs already on Drive are skipped.", code("""
# Stages B / C / D: train + evaluate every registered neural run of the selected stages (resumable)
sys.path.insert(0, f"{WORK}/src")
from bscan.registry import load_registry, expected_runs
REG = load_registry()
common = [f"data.audio_root={WORK}/data", f"data.cache_dir={WORK}/data/cache",
          f"data.metadata_csv={WORK}/metadata/unified_metadata.csv", f"out_dir={WORK}/results/runs", "data.num_workers=2"]
os.makedirs(f"{WORK}/results", exist_ok=True)
todo = [r for r in expected_runs(REG, stages=tuple(s for s in STAGES if s in "BCD"))
        if REG["models"][r.model]["kind"] == "neural"]
print(f"{len(todo)} runs in stages {STAGES}")
def on_drive(name): return USE_DRIVE and os.path.exists(f"{DRIVE_DIR}/runs/{name}/metrics.json")
for run in todo:
    if on_drive(run.name):
        print("skip (on Drive):", run.name); continue
    ov = common + [f"train.seed={run.seed}", f"data.protocol={run.protocol}"]
    for step in ("train", "evaluate"):
        tick(f"{run.name}:{step}")
        log = f"{WORK}/results/{run.name}.{step}.log"
        with open(log, "w") as fh:
            p = subprocess.run([sys.executable, "-m", f"bscan.{step}", "--config", f"configs/{run.model}.yaml", *ov],
                               env=env, stdout=fh, stderr=subprocess.STDOUT)
        tock(f"{run.name}:{step}")
        print(open(log).read()[-1500:])
        assert p.returncode == 0, f"{run.name} {step} failed"
    if USE_DRIVE:
        shutil.copytree(f"{WORK}/results/runs/{run.name}", f"{DRIVE_DIR}/runs/{run.name}", dirs_exist_ok=True)
        for f in glob.glob(f"{WORK}/results/{run.name}.*.log"): shutil.copy(f, f"{DRIVE_DIR}/runs/{run.name}/")
""")),
        ("9. Stage E: analyses", "Robustness to 14 perturbations, counterfactual probes (inference only) and CUDA latency.", code("""
# Stage E: robustness, counterfactual probes (inference only) and CUDA efficiency
if "E" in STAGES:
    for kind in ("robustness", "probes"):
        a = REG["analyses"][kind]
        for rn in a["runs"]:
            src = f"{DRIVE_DIR}/runs/{rn}" if USE_DRIVE else f"{WORK}/results/runs/{rn}"
            if not os.path.exists(f"{src}/best.pt"):
                print(f"skip {kind} for {rn}: trained run not found (run its stage first)"); continue
            if os.path.exists(f"{src}/{kind}_summary.csv"):
                print(f"skip {kind} for {rn}: done"); continue
            tick(f"{rn}:{kind}")
            p = subprocess.run([sys.executable, "-m", "bscan.robustness", "--run", src, "--mode", kind,
                                "--splits", *a["splits"], "--max_files", str(a["max_files"]), "--device", "cuda",
                                "--metadata", f"{WORK}/metadata/unified_metadata.csv", "--audio_root", f"{WORK}/data"],
                               env=env, capture_output=True, text=True)
            tock(f"{rn}:{kind}")
            print(p.stdout[-2500:]); assert p.returncode == 0, p.stderr[-3000:]
    tick("efficiency")
    p = subprocess.run([sys.executable, "-m", "bscan.efficiency", "--device", "cuda", "--rtf_files", "100",
                        "--metadata", f"{WORK}/metadata/unified_metadata.csv", "--audio_root", f"{WORK}/data",
                        "--out", f"{WORK}/results/efficiency"], env=env, capture_output=True, text=True)
    tock("efficiency"); print(p.stdout[-3000:]); assert p.returncode == 0, p.stderr[-3000:]
    if USE_DRIVE: shutil.copytree(f"{WORK}/results/efficiency", f"{DRIVE_DIR}/efficiency", dirs_exist_ok=True)
""")),
        ("10. XLS-R comparator", "Frozen XLS-R 300M features with logistic regression (P1, seed 42).", code("""
# XLS-R comparator (registered in stage C): frozen XLS-R 300M + logistic regression
if RUN_OPTIONAL_SSL and not (USE_DRIVE and os.path.exists(f"{DRIVE_DIR}/runs/ssl_xlsr_probe__P1__seed42/metrics.json")):
    tick("ssl")
    p = subprocess.run([sys.executable, "-m", "bscan.ssl_probe", "--protocol", "P1", "--seed", "42",
                        "--metadata", f"{WORK}/metadata/unified_metadata.csv", "--audio_root", f"{WORK}/data",
                        "--cache", f"{WORK}/data/cache/ssl", "--out_dir", f"{WORK}/results/runs"], env=env, capture_output=True, text=True)
    tock("ssl"); print(p.stdout[-3000:]); assert p.returncode == 0, p.stderr[-3000:]
    if USE_DRIVE:
        shutil.copytree(f"{WORK}/results/runs/ssl_xlsr_probe__P1__seed42", f"{DRIVE_DIR}/runs/ssl_xlsr_probe__P1__seed42", dirs_exist_ok=True)
""")),
        ("11. Progress", "Which registered runs are complete (on Drive or in this session).", code("""
# Progress over the whole registry (Drive)
rows = []
for r in expected_runs(REG, include_optional=True):
    if REG["models"][r.model]["kind"] == "local": continue
    rows.append({"run": r.name, "stage": r.stage, "done": on_drive(r.name) or os.path.exists(f"{WORK}/results/runs/{r.name}/metrics.json")})
prog = pd.DataFrame(rows); print(prog.groupby("stage").done.agg(["sum", "count"]))
print(prog[~prog.done].to_string(index=False) if (~prog.done).any() else "all registered Colab runs are complete")
""")),
        ("12. Results package", f"Writes `BSCAN_results_{ver}.zip`, keeps a copy on Drive and downloads it.", code("""
# Package everything for return
out = f"/content/BSCAN_results_{CODE_VERSION}"
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
src_runs = f"{DRIVE_DIR}/runs" if USE_DRIVE else f"{WORK}/results/runs"
shutil.copytree(src_runs, f"{out}/runs", dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.npy"))
for extra in ("smoke_cuda", "efficiency"):
    for base in (f"{WORK}/results", DRIVE_DIR):
        if os.path.exists(f"{base}/{extra}"): shutil.copytree(f"{base}/{extra}", f"{out}/{extra}", dirs_exist_ok=True)
json.dump({"code_version": CODE_VERSION, "stages": STAGES, "timing_sec": TIMING, "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "cuda": torch.version.cuda}, open(f"{out}/colab_session.json", "w"), indent=2)
zp = shutil.make_archive(out, "zip", out)
if USE_DRIVE: shutil.copy(zp, DRIVE_DIR)
print("results:", zp, round(os.path.getsize(zp) / 2**20, 1), "MB")
from google.colab import files; files.download(zp)
""")),
    ]
    cells = [intro]
    for head, note, cell in steps:
        cells += [md(f"## {head}\n\n{note}"), cell]
    for i, c in enumerate(cells):  # stable cell ids (required from nbformat 4.5 on)
        c["id"] = f"cell-{i:02d}"
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the Colab notebook in code/.")
    ap.add_argument("--embedded", action="store_true",
                    help="embed the source files and the metadata table instead of taking them from the public "
                         "repository (for a code version that is not published yet)")
    args = ap.parse_args()
    files, manifest, ver = payload()
    mb = (ROOT / "metadata" / "unified_metadata.csv").read_bytes()
    meta_sha = hashlib.sha256(mb).hexdigest()
    out = ROOT / "code"
    out.mkdir(parents=True, exist_ok=True)
    for key in NOTEBOOKS:
        nb = build(key, ver, meta_sha, (files, manifest, b64gz(mb)) if args.embedded else None)
        p = out / f"BSCAN_{key}{'_embedded' if args.embedded else ''}.ipynb"
        p.write_text(json.dumps(nb, indent=1))
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 2**20:.2f} MB)")
    if args.embedded:
        print(f"code version {ver}; {len(files)} embedded files")
    else:
        print(f"code version {ver}; code from {PUBLIC_REPO} at commit {PUBLIC_COMMIT[:7]} "
              "(the notebook stops if that commit holds another code version)")


if __name__ == "__main__":
    sys.exit(main())
