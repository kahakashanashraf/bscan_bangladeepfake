"""Build the Kaggle release of the benchmark (Phase 2 / 15) and its report.

No audio is redistributed; both corpora are fetched from their original hosts by a download/verify script:
  * BanglaFake (no specific licence named: none on Hugging Face or GitHub, the paper says only "open license")
  * Mendeley Bangla Audio Dataset, version 4 (listed as CC BY 4.0, with a dataset usage agreement that does not
    allow redistribution without the provider's written consent)
The release holds metadata, SHA-256 checksums, splits, descriptors, audits and scores.  Splits are defined by metadata
only.  Per-recording scores come only from real runs in results/runs/ (never from results/runs_PLACEHOLDER/), so
re-running this script after the GPU runs are imported adds their scores.  Every number in the README is computed
from the files.

Usage:
    python .scripts/build_kaggle_release.py [--out release/BSCAN_Bengali_Audio_Deepfake_Benchmark]
Upload (Kaggle CLI >= 1.8; the release folder holds dataset-metadata.json):
    kaggle datasets version -p <out> --dir-mode zip -d -m "<notes>"
-d deletes the older versions, which contained the Mendeley audio; generate the new version's DOI only after that
upload, and never pass -d again on a later upload, or it deletes the version the DOI points to.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAME = "BSCAN_Bengali_Audio_Deepfake_Benchmark"
TITLE = "Bengali Deepfake Forensics Dataset (BSCAN)"
SUBTITLE = "Bengali deepfake audio: leakage-aware splits, checksums, descriptors, scores"
KAGGLE_ID = "hosen42/bengali-deepfake-forensics-dataset"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_ID}"
# Kaggle DOIs are per dataset version: after an upload, generate a DOI for that version on Kaggle and set it here
# (None leaves the DOI out of README.md and CITATION.cff). 10.34740/KAGGLE/DSV/20278870 = version 7 (2026-10-03), the
# version with the scores of all runs that the article cites
DOI = "10.34740/KAGGLE/DSV/20278870"
CODE_URL = "https://github.com/hamidhosen42/BSCAN-Dual-Branch-Spectro-Cepstral-Residual-Attention-Network-for-Bengali-Deepfake-Audio-Detection"
ARTICLE = "BSCAN: Dual-Branch Spectro-Cepstral Residual Attention Network for Bengali Deepfake Audio Detection"
AUTHORS = ["Kahakashan Ashraf", "Md. Hamid Hosen", "Mahfuzulhoq Chowdhury"]
BF_URL = "https://huggingface.co/datasets/sifat1221/banglaFake/resolve/main/final_data.zip"
BF_SHA = "3034353ef96a9f8c8a8118e65682b31b36de14c4f9159a2ee0e5cd5a9dfaa8b0"
BF_BYTES = 5642775456
# Mendeley version 4 (DOI 10.17632/4ftmwt86vr.4); its audio files are byte-identical to version 1
MEN_URL = "https://data.mendeley.com/public-api/zip/4ftmwt86vr/download/4"
MEN_SHA = "aefdfbf6351849c186717767cd1c901a91e7033ccc1bc6af65161995bb86f4d3"
MEN_BYTES = 519835011
MEN_TERMS = ("listed under CC BY 4.0; its versions 3 and 4 add a dataset usage agreement that, among other terms, "
             "allows academic and research use only and forbids redistribution without the provider's written consent "
             "and any re-identification of speakers")
BF_TERMS = ("names no specific licence (its Hugging Face and GitHub repositories carry none; its paper says only "
            "\"open license\")")
MEN_TITLE = "Bangla Audio Dataset: Original and DeepFake Voices for AI-Based Voice Analysis and Detection"
# audit outputs that back the dataset statements of the article (copied; recording paths mapped to release_path)
AUDITS = ["results/phase1/duplicate_summary.json", "results/phase1/near_duplicate_pairs.csv",
          "results/phase1/univariate_separability.csv", "results/phase3/split_overlap.csv",
          "results/phase3/split_counts_P1.csv", "results/phase3/split_counts_P2.csv",
          "results/phase3/filename_leakage.csv", "results/phase3/metadata_shortcut_diagnostics.csv",
          "results/phase3/model_visible_shortcut_diagnostics.csv"]
# nothing private may leave the machine: local paths and access tokens (hf_ratio is a descriptor name, hence the
# long-token form for Hugging Face tokens)
LEAK = re.compile("|".join([re.escape(str(Path.home())), r"/Users/", r"/home/", r"/content/drive", r"KGAT_",
                            r"gh[pousr]_[A-Za-z0-9]{20,}", r"hf_[A-Za-z0-9]{30,}"]))
SPLIT_LABEL = {"train": "train", "validation": "validation", "internal_test": "internal test"}


def release_path(row) -> str:
    # path after scripts/download_external_datasets.py extracts each source archive under external/<corpus>/
    corpus = "mendeley" if row.dataset_id == "MEN" else "banglafake"
    return f"external/{corpus}/" + row.audio_path.split("extracted/", 1)[1]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank (Mann-Whitney) AUC with average ranks for ties; nan unless both classes are present."""
    y = np.asarray(y)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if not n1 or not n0:
        return float("nan")
    r = pd.Series(np.asarray(s, dtype=float)).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def real_runs():
    for mf in sorted((ROOT / "results" / "runs").glob("*/metrics.json")):
        m = json.loads(mf.read_text())
        if not m.get("PLACEHOLDER"):
            yield mf.parent, m


DOWNLOAD_SCRIPT = f'''"""Download both source corpora from their original hosts and verify every file (SHA-256).

Neither corpus is redistributed with this release: BanglaFake {BF_TERMS}, and the Mendeley corpus is
{MEN_TERMS}. By running this script you obtain the audio from the providers and accept their terms.

Each archive is extracted to <external>/<corpus>/, so every release_path (external/<corpus>/...) resolves under
<external>. The default <external> is ./external in the current folder. On Kaggle use --external /kaggle/temp/external:
the dataset folder is read-only, and /kaggle/working is saved as notebook output, which would redistribute the audio.
Files that fail their checksum are extracted again from the archive.

    python scripts/download_external_datasets.py [--external external] [--corpus banglafake mendeley]
"""
import argparse, hashlib, subprocess, zipfile
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1]
SOURCES = {{
    "banglafake": dict(url="{BF_URL}", sha256="{BF_SHA}", bytes={BF_BYTES}, archive="banglafake_final_data.zip"),
    "mendeley": dict(url="{MEN_URL}", sha256="{MEN_SHA}", bytes={MEN_BYTES}, archive="mendeley_4ftmwt86vr-4.zip"),
}}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def fetch(src, dst):
    if dst.exists() and dst.stat().st_size == src["bytes"]: return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "--fail", "--retry", "10", "-C", "-", "-o", str(dst), src["url"]], check=True)
    if sha(dst) != src["sha256"]:
        print(f"warning: {{dst.name}} differs from the archive used in the article; every file is checked below")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external", type=Path, default=Path("external"))
    ap.add_argument("--corpus", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    a = ap.parse_args()
    ext = a.external.resolve()
    meta = pd.read_csv(R / "metadata" / "unified_metadata.csv", low_memory=False)
    failed = 0
    for corpus in a.corpus:
        rows = meta[meta.release_path.str.startswith(f"external/{{corpus}}/")]
        files = [ext / p.split("/", 1)[1] for p in rows.release_path]
        z = ext / SOURCES[corpus]["archive"]
        if not all(f.exists() for f in files):
            fetch(SOURCES[corpus], z)
            zipfile.ZipFile(z).extractall(ext / corpus)
        bad = [f for f, s in zip(files, rows.sha256) if not f.exists() or sha(f) != s]
        if bad:  # damaged or partial files: extract those members again from the archive and re-check
            fetch(SOURCES[corpus], z)
            with zipfile.ZipFile(z) as zf:
                for f in bad:
                    zf.extract(f.relative_to(ext / corpus).as_posix(), ext / corpus)
            sums = dict(zip(files, rows.sha256))
            bad = [f for f in bad if not f.exists() or sha(f) != sums[f]]
        bad = [str(f) for f in bad]
        print(f"{{corpus}}: {{len(rows) - len(bad)}}/{{len(rows)}} files verified under {{ext / corpus}}")
        failed += len(bad)
        if bad:
            print("  e.g.", bad[:3])
    if failed:
        raise SystemExit(f"{{failed}} files missing or changed")
'''

VERIFY_SCRIPT = '''"""Verify the recordings of the selected corpora against metadata/unified_metadata.csv (SHA-256).

No audio ships with this release; both corpora are looked up under <external> after
scripts/download_external_datasets.py has run (default ./external). Exits non-zero if any selected
recording is absent or does not match its checksum.

    python scripts/verify_dataset.py [--external external] [--corpus banglafake mendeley]
"""
import argparse, hashlib
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--external", type=Path, default=Path("external"))
ap.add_argument("--corpus", nargs="+", choices=["banglafake", "mendeley"], default=["banglafake", "mendeley"])
a = ap.parse_args()
ext = a.external.resolve()
meta = pd.read_csv(R / "metadata" / "unified_metadata.csv", low_memory=False)
meta = meta[meta.release_path.str.split("/").str[1].isin(a.corpus)]
files = [ext / p.split("/", 1)[1] for p in meta.release_path]
present = [f.exists() for f in files]
bad = [str(f) for f, s, ok in zip(files, meta.sha256, present) if ok and sha(f) != s]
print(f"files listed: {len(meta)} | present: {sum(present)} | checksum mismatches: {len(bad)}")
absent = meta.loc[[not ok for ok in present], "dataset_id"].value_counts().to_dict()
if absent:
    print("absent (run scripts/download_external_datasets.py first):", absent)
if bad or absent:
    raise SystemExit(f"{len(bad)} checksum mismatches, {sum(absent.values())} absent; e.g. {bad[:5]}")
'''

COLUMNS = {
    "metadata/unified_metadata.csv": [
        ("release_path", "where the recording lies after scripts/download_external_datasets.py (`external/<corpus>/...`)"),
        ("dataset_id", "BF-SUST, BF-MOZ (Common Voice subset), BF-NEWS or MEN (Mendeley version 4)"),
        ("label", "0 = bona fide, 1 = spoof"),
        ("speaker_id", "speaker as stated by the source layout; BF-VITS-voice for every BanglaFake spoof; unknown if not stated"),
        ("set_id", "Mendeley set (S1-S15: five speakers reading the same 30 sentences) or Common Voice speaker set; unknown otherwise"),
        ("gender", "as stated by the source; unknown otherwise"),
        ("utterance_key", "links a bona fide recording to its spoof rendition of the same utterance"),
        ("text_group", "hash of the normalised sentence text; P1 splits are grouped by it"),
        ("duration_sec", "duration in seconds"), ("sample_rate", "sample rate in Hz"),
        ("sha256", "SHA-256 of the file bytes"),
        ("split_P1", "split under protocol P1"), ("split_P2", "split under protocol P2"),
        ("exclude_from_metrics", "True for BF-NEWS (bona fide label unverified; excluded from every metric in the article)"),
        ("licence", "terms of the source audio (no audio is redistributed in this release)")],
    "metadata/recording_descriptors.csv": [
        ("release_path, sha256", "join keys to metadata/unified_metadata.csv"),
        ("duration_sec, sample_rate", "as in the metadata"),
        ("snr_proxy_db", "energy-percentile SNR proxy (dB)"),
        ("leading_silence_sec, trailing_silence_sec", "silence before the first and after the last active frame"),
        ("activity_ratio", "fraction of active (non-silent) frames"),
        ("peak_abs, rms_dbfs, lufs", "peak amplitude, RMS level (dBFS) and integrated loudness (LUFS)"),
        ("clip_ratio", "fraction of full-scale samples"),
        ("dc_offset", "mean sample value"), ("band_edge_hz, rolloff95_hz", "spectral band edge and 95 % roll-off (Hz)"),
        ("zcr", "zero-crossing rate"), ("exact_zero_ratio", "fraction of samples that are exactly zero")],
}


def table(head: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" * len(head)
    sep = ["---:" if a == "r" else "---" for a in align]
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in [head, sep] + rows) + "\n"


def f3(x: float) -> str:
    return "–" if x is None or np.isnan(x) else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "release" / NAME))
    out = Path(ap.parse_args().out).resolve()
    if out.exists():
        shutil.rmtree(out)
    for d in ("metadata", "splits", "scripts", "audits", "scores"):
        (out / d).mkdir(parents=True)

    meta = pd.read_csv(ROOT / "metadata/unified_metadata.csv", dtype={"utterance_id": str}, low_memory=False)
    meta["release_path"] = [release_path(r) for r in meta.itertuples()]
    meta["exclude_from_metrics"] = meta.dataset_id.eq("BF-NEWS")
    meta["licence"] = meta.dataset_id.map(lambda d: "Mendeley Data: CC BY 4.0 with usage agreement; not redistributed"
                                          if d == "MEN" else "no specific licence named; not redistributed")
    keep = [c for c, _ in COLUMNS["metadata/unified_metadata.csv"]]
    meta[keep].to_csv(out / "metadata/unified_metadata.csv", index=False)
    for proto in ("P1", "P2"):
        for sp, g in meta.groupby(f"split_{proto}"):
            if sp != "unused":
                (out / "splits" / f"{proto}_{sp}.txt").write_text("\n".join(g.release_path) + "\n")
    to_rel = dict(zip(meta.audio_path, meta.release_path))

    inv = pd.read_csv(ROOT / "metadata" / "dataset_inventory.csv", dtype={"utterance_id": str}, low_memory=False)
    inv = inv.merge(meta[["sha256", "release_path", "licence"]], on="sha256", how="left", validate="one_to_one")
    inv["license"] = inv.pop("licence")  # same terms as unified_metadata (the inventory's own column predates them)
    inv.drop(columns=["audio_path"]).to_csv(out / "dataset_inventory.csv", index=False)

    desc = pd.read_csv(ROOT / "metadata/recording_descriptors.csv", low_memory=False)
    desc = meta[["audio_path", "release_path", "sha256"]].merge(desc, on="audio_path", how="left", validate="one_to_one")
    assert desc.duration_sec.notna().all(), "recording descriptors missing for some recordings"
    desc.drop(columns="audio_path").to_csv(out / "metadata/recording_descriptors.csv", index=False)
    has_zero = "exact_zero_ratio" in desc.columns

    # audits: recording paths mapped to release_path
    for a in AUDITS:
        src, dst = ROOT / a, out / "audits" / Path(a).name
        if src.suffix == ".csv":
            t = pd.read_csv(src, low_memory=False)
            for c in ("path_a", "path_b", "audio_path"):
                if c in t.columns:
                    t[c] = t[c].map(to_rel)
                    assert t[c].notna().all(), f"{a}: {c} has recordings outside the metadata"
            t.to_csv(dst, index=False)
        else:
            shutil.copyfile(src, dst)
    dup = json.loads((ROOT / "results/phase1/duplicate_summary.json").read_text())
    ov = pd.read_csv(ROOT / "results/phase3/split_overlap.csv")
    sep = pd.read_csv(ROOT / "results/phase1/univariate_separability.csv").set_index(["dataset_id", "descriptor"]).separability

    # per-recording scores of every real run
    score_rows = []
    for run, m in real_runs():
        dst = out / "scores" / run.name
        dst.mkdir()
        for p in sorted(run.glob("predictions_*.csv")):
            sp = p.stem.removeprefix("predictions_")
            f = pd.read_csv(p, low_memory=False)
            rel = f.audio_path.map(to_rel)
            assert rel.notna().all(), f"{p}: recordings outside the metadata"
            f.insert(0, "release_path", rel)
            f.drop(columns="audio_path").to_csv(dst / p.name, index=False)
            a, ref = auc(f.label.to_numpy(), f.score.to_numpy()), m["splits"].get(sp, {}).get("auc", float("nan"))
            assert np.isnan(a) == np.isnan(ref) and (np.isnan(a) or abs(a - ref) < 1e-6), f"{p}: AUC {a} vs metrics {ref}"
            model, protocol, _ = run.name.split("__")  # neural runs' metrics.json has no model/protocol fields
            score_rows.append({"run": run.name, "protocol": m.get("protocol", protocol), "model": m.get("model", model),
                               "split": sp, "n": len(f), "auc": a})
        for fn in ("metrics.json", "environment.json"):
            if (run / fn).exists():
                shutil.copyfile(run / fn, dst / fn)
    scores = pd.DataFrame(score_rows)

    for s in ("build_inventory.py", "duplicate_check.py", "leakage_check.py"):
        # the release keeps them in a visible scripts/ folder, so point their usage lines there
        (out / "scripts" / s).write_text((ROOT / ".scripts" / s).read_text().replace(".scripts/", "scripts/"))
    (out / "scripts/download_external_datasets.py").write_text(DOWNLOAD_SCRIPT)
    (out / "scripts/verify_dataset.py").write_text(VERIFY_SCRIPT)

    # ---- numbers for the README, all from the files written above ----
    counts = meta.groupby(["dataset_id", "label"]).size().unstack(fill_value=0)
    hours = meta.groupby("dataset_id").duration_sec.sum() / 3600
    used = meta[~meta.exclude_from_metrics]
    spk = {d: meta[(meta.dataset_id == d) & (meta.label == 0) & (meta.speaker_id != "unknown")].speaker_id.nunique()
           for d in counts.index}
    # every BF-SUST bona fide file carries the same placeholder speaker ID; the SUST TTS corpus was recorded by
    # one male voice talent (Ahmad et al. 2021, Acoust Sci Technol 42:326-332, Sec. 3.3)
    spk["BF-SUST"] = 1
    sr = {(d, l): "/".join(f"{int(v) / 1000:g}" for v in sorted(g.sample_rate.unique()))
          for (d, l), g in meta.groupby(["dataset_id", "label"])}
    corpora = [("BF-SUST", "BanglaFake, SUST TTS subset", "none named"),
               ("BF-MOZ", "BanglaFake, Common Voice subset", "none named"),
               ("BF-NEWS", "BanglaFake, news subset (label unverified)", "none named"),
               ("MEN", "Mendeley Bangla Audio Dataset, version 4", "CC BY 4.0 + usage agreement")]
    corpus_rows = [[d, name, f"{counts.loc[d].get(0, 0):,}", f"{counts.loc[d].get(1, 0):,}",
                    f"{hours[d]:.2f}", spk[d] or "–", f"{sr[(d, 0)]} / {sr.get((d, 1), '–')}", lic]
                   for d, name, lic in corpora]
    split_rows = []
    for proto in ("P1", "P2"):
        for sp, g in meta.groupby(f"split_{proto}", sort=False):
            c = g.label.value_counts()
            split_rows.append([proto, SPLIT_LABEL.get(sp, sp.replace("external_", "external: ")),
                               ", ".join(sorted(g.dataset_id.unique())), f"{c.get(0, 0):,}", f"{c.get(1, 0):,}",
                               f"`splits/{proto}_{sp}.txt`"])
    order = {"train": 0, "validation": 1, "internal test": 2}
    split_rows.sort(key=lambda r: (r[0], order.get(r[1], 3), r[1]))
    shared_text = ov[ov.shared_text_group > 0]
    ts = desc.assign(label=meta.label.values, dataset_id=meta.dataset_id.values)
    ts = ts[ts.dataset_id == "BF-SUST"].groupby("label").trailing_silence_sec.median()
    cues = [("BF-SUST", "trailing_silence_sec", "trailing silence"), ("BF-SUST", "exact_zero_ratio", "fraction of exactly-zero samples"),
            ("BF-SUST", "dc_offset", "DC offset"), ("BF-MOZ", "sample_rate", "sample rate"), ("MEN", "sample_rate", "sample rate")]
    cue_rows = [[d, lab, f3(sep.get((d, c), float("nan")))] for d, c, lab in cues if (d, c) in sep.index]
    if len(scores):
        piv = scores.pivot(index="run", columns="split", values="auc")
        model = scores.groupby("run").model.first().str.replace("content-agnostic", "coarse")
        cols = [c for c in ["validation", "internal_test", "external_BF-SUST", "external_BF-MOZ", "external_MEN"] if c in piv.columns]
        score_tab = table(["run", "model"] + cols, [[r, model[r]] + [f3(piv.loc[r, c]) for c in cols] for r in piv.index],
                          "ll" + "r" * len(cols))
    else:
        score_tab = "_No real run is available yet._\n"
    n_runs = scores.run.nunique() if len(scores) else 0
    scores_note = (f"Scores of all {n_runs} runs of the article are included: the neural models (trained on Google Colab), "
                   "the XLS-R reference and the non-neural baselines." if n_runs else "")
    zero_note = ("" if has_zero else "\nThe per-recording `exact_zero_ratio` descriptor (fraction of exactly-zero samples) is added in a "
                 "later version; its single-descriptor separability is already in `audits/univariate_separability.csv`.\n")
    today = datetime.date.today().isoformat()
    cite_doi = f" (DOI [{DOI}](https://doi.org/{DOI}))" if DOI else ""
    bib_doi = f"  doi       = {{{DOI}}},\n" if DOI else ""

    def doc(fn: str, cols_: list) -> str:
        rows = [[f"`{c}`", t] for c, t in cols_ if has_zero or c != "exact_zero_ratio"]
        return f"\n**`{fn}`**\n\n" + table(["column", "meaning"], rows)

    readme = f"""# {TITLE}

Companion data release of the article *{ARTICLE}* by {', '.join(AUTHORS)} (manuscript in preparation).
Code: {CODE_URL}

It brings the two public Bengali deepfake-audio corpora used in the article into **one audited table**:
{len(meta):,} recordings ({meta.duration_sec.sum() / 3600:.2f} h), each with SHA-256 checksums of the file and of
the decoded samples, recording descriptors, the two leakage-aware evaluation protocols (P1, P2) as exact file lists,
the duplicate and overlap audits, and per-recording scores of the reference detectors. **No audio is redistributed**:
BanglaFake {BF_TERMS}, and the Mendeley Data release carries a usage agreement that does not allow
redistribution without the provider's written consent. One command fetches both corpora from their official hosts and checks every file against the listed
checksums.

## Corpora

{table(["dataset_id", "source", "bona fide", "spoof", "hours", "bona fide speakers", "sample rate (kHz) bona fide / spoof", "terms of the audio"], corpus_rows, "llrrrrll")}
* **BanglaFake** (Fahad, Asif & Sikder, arXiv:2505.10885; https://huggingface.co/datasets/sifat1221/banglaFake): every spoof
  comes from one single-speaker VITS model. The archive holds {counts.drop('MEN')[0].sum():,} bona fide and
  {counts.drop('MEN')[1].sum():,} spoof recordings, which differs from the 12,260 real and 13,260 deepfake utterances
  stated in the BanglaFake paper.
* **{MEN_TITLE}** (Mendeley Data), version 4
  (Dipto, Ayan & Faria, 2024, DOI 10.17632/4ftmwt86vr.4; audio byte-identical to version 1): 15 sets of five speakers
  reading the same 30 sentences; the generator is not documented. It is {MEN_TERMS}; obtain it from Mendeley Data
  and follow those terms.
* **BF-NEWS**: labelled bona fide only by its folder name; the label cannot be verified, so these
  {counts.loc['BF-NEWS'].sum():,} recordings are flagged `exclude_from_metrics` and are excluded from every metric
  in the article. Without them: {len(used):,} recordings, {int((used.label == 0).sum()):,} bona fide and
  {int((used.label == 1).sum()):,} spoof.

## Files

| path | content |
|---|---|
| `metadata/unified_metadata.csv` | one row per recording ({len(meta):,}); the table every other file joins to |
| `metadata/recording_descriptors.csv` | per-recording signal descriptors |
| `dataset_inventory.csv` | full inventory: container facts, file and decoded-PCM SHA-256, provenance notes |
| `splits/P1_*.txt`, `splits/P2_*.txt` | exact file lists (`release_path`) of every split |
| `audits/` | duplicate, overlap and shortcut audits (see below) |
| `scores/<run>/predictions_<split>.csv` | per-recording scores (`score` = spoof logit, `prob_spoof`) and the run's `metrics.json` |
| `scripts/` | download/verify scripts and the inventory, duplicate and split code used in the article |

## Protocols

No file appears in more than one split of a protocol. **P1** trains on BF-SUST, split 70/15/15 by sentence
(`text_group`, seed 42), so a bona fide recording and its spoof rendition never cross splits; the other corpora are
external tests. **P2** trains on Mendeley, split 9/3/3 by set, so train, validation and test are speaker- and
sentence-disjoint; BanglaFake is the external test.

{table(["protocol", "split", "corpora", "bona fide", "spoof", "file list"], split_rows, "lllrrl")}
## Known shortcuts (read before training)

Class labels are confounded with recording properties in every corpus. Median trailing silence in BF-SUST is
{ts.get(0, float('nan')):.2f} s for bona fide and {ts.get(1, float('nan')):.2f} s for spoof recordings. Single-descriptor separability
(max(AUC, 1 − AUC); 0.5 = uninformative, `audits/univariate_separability.csv`):

{table(["corpus", "descriptor", "separability"], cue_rows, "llr")}
A detector that never hears speech content can therefore score highly in-corpus; see `scores/stats_hgb_*`
(gradient boosting on 11 coarse window descriptors) and the controlled preprocessing described in the article.

## Audits

* Exact duplicates among all {dup['n_files_checked']:,} recordings: {dup['exact_file_sha256']['groups']} groups by file
  hash and {dup['exact_pcm_sha256']['groups']} by decoded samples.
* Near duplicates: {dup['near_duplicate_candidates']:,} candidate pairs (log-Mel fingerprint cosine ≥
  {dup['thresholds']['fingerprint_cosine_candidate']} among the {dup['thresholds']['k'] - 1} nearest neighbours; k =
  {dup['thresholds']['k']} counts the recording itself), {dup['near_duplicate_confirmed_pairs']} confirmed by waveform
  cross-correlation ≥ {dup['thresholds']['xcorr_confirm']} (`audits/near_duplicate_pairs.csv`).
* Split overlap (`audits/split_overlap.csv`): no two splits share a file, decoded stream or utterance. Shared sentence
  groups occur only between {'; '.join(f"`{r.split_a}` and `{r.split_b}` of {r.protocol.replace('split_', '')} ({r.shared_text_group})" for r in shared_text.itertuples())}:
  a sentence read in both BF-SUST and the Common Voice subset by different speakers, not a duplicate recording.

## Per-recording scores

AUC recomputed from the released score files (bona fide vs spoof):

{score_tab}
{scores_note}
The statistics baseline here is evaluated per recording; the article's shortcut table reports window-descriptor
diagnostics (`audits/model_visible_shortcut_diagnostics.csv`), so its AUCs can differ in the third decimal.

## Columns
{doc("metadata/unified_metadata.csv", COLUMNS["metadata/unified_metadata.csv"])}{doc("metadata/recording_descriptors.csv", COLUMNS["metadata/recording_descriptors.csv"])}{zero_note}
## Get and verify the audio

```
python scripts/download_external_datasets.py           # fetches both corpora into ./external and checks all {len(meta):,} files
python scripts/verify_dataset.py --external external   # re-checks them later (non-zero exit if any is absent or changed)
```
The download is about {(BF_BYTES + MEN_BYTES) / 1e9:.1f} GB (BanglaFake {BF_BYTES / 1e9:.1f} GB, Mendeley {MEN_BYTES / 1e9:.1f} GB);
`--corpus mendeley` or `--corpus banglafake` fetches one corpus only. On Kaggle, pass `--external /kaggle/temp/external`
to both scripts: the dataset folder is read-only, and `/kaggle/working` is saved as notebook output, so audio placed
there would be redistributed. Do not save or re-upload the downloaded audio. Source archives: BanglaFake `final_data.zip` SHA-256 `{BF_SHA}`; Mendeley version 4
zip SHA-256 `{MEN_SHA}`. To rebuild the inventory, audits and splits from the raw archives, extract both into one folder
and run `scripts/build_inventory.py --root <folder> --out work/inventory`, then `scripts/duplicate_check.py` and
`scripts/leakage_check.py` with `--out` (and `--splits_dir`, `--metadata_dir`) pointing to a new folder.
Requirements: Python 3.10+, pandas, numpy, soundfile, librosa, scikit-learn, curl.

## Licence and citation

Audio is not included: obtain BanglaFake from Hugging Face and the Mendeley corpus from Mendeley Data under their
terms. Splits, audits, scores and documentation: CC BY 4.0; scripts: MIT. Per-recording data derived from the
Mendeley corpus are provided for research use under its usage agreement, and no ownership of them is claimed. Please cite this dataset{cite_doi},
the article (citation will be updated on publication) and both source datasets; see `CITATION.cff`.

```
@misc{{bscan_dataset,
  title     = {{{TITLE}}},
  author    = {{{' and '.join(AUTHORS)}}},
  publisher = {{Kaggle}},
  year      = {{{today[:4]}}},
{bib_doi}  url       = {{{KAGGLE_URL}}}
}}
```
Release built {today}.
"""
    # column documentation must match the files
    for fn, cols_ in COLUMNS.items():
        written = pd.read_csv(out / fn, nrows=0).columns.tolist()
        documented = [c.strip() for names, _ in cols_ for c in names.split(",") if has_zero or c.strip() != "exact_zero_ratio"]
        assert written == documented, f"{fn}: columns {written} != documented {documented}"
    (out / "README.md").write_text(readme)
    assert 6 <= len(TITLE) <= 50 and 20 <= len(SUBTITLE) <= 80, "Kaggle title/subtitle length limits"
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": TITLE, "id": KAGGLE_ID, "subtitle": SUBTITLE, "description": readme, "isPrivate": False,
        "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": ["bengali", "tabular", "deepfake", "anti-spoofing", "speech"]}, indent=2, ensure_ascii=False))
    (out / "LICENSE.md").write_text(
        "# Licences\n\n* No audio is included in this release.\n"
        f"* {MEN_TITLE} (Mendeley Data), by Md Akteruzzaman Dipto, Nafis Sadique Ayan and Sultana Razia Faria (2024), "
        f"DOI 10.17632/4ftmwt86vr.4. It is {MEN_TERMS}. Obtain it from Mendeley Data and follow those terms.\n"
        "* BanglaFake, by Istiaq Ahmed Fahad, Kamruzzaman Asif and Sifat Sikder (arXiv:2505.10885): it "
        f"{BF_TERMS}. Obtain it from {BF_URL}.\n"
        "* Compiled by the authors of this release. Splits, audits, scores and documentation: CC BY 4.0; scripts: MIT. "
        "Per-recording data derived from the Mendeley corpus (metadata, checksums, descriptors) are provided for research "
        "use under its usage agreement, and no ownership of them is claimed.\n")
    (out / "CITATION.cff").write_text(
        "cff-version: 1.2.0\nmessage: If you use this release, please cite it, the article and the source datasets.\n"
        f"title: \"{TITLE}\"\ntype: dataset\nauthors:\n" +
        "".join(f"  - name: \"{a}\"\n" for a in AUTHORS) +
        f"date-released: {today}\n" + (f"doi: {DOI}\n" if DOI else "") +
        f"url: \"{KAGGLE_URL}\"\nrepository-code: \"{CODE_URL}\"\nlicense: CC-BY-4.0\n"
        "references:\n"
        f"  - type: unpublished\n    title: \"{ARTICLE}\"\n    notes: \"Manuscript in preparation\"\n    authors:\n" +
        "".join(f"      - name: \"{a}\"\n" for a in AUTHORS) +
        "  - type: data\n    title: \"Bangla Audio Dataset: Original and DeepFake Voices "
        "for AI-Based Voice Analysis and Detection\"\n    doi: 10.17632/4ftmwt86vr.4\n    authors:\n      - name: Md Akteruzzaman Dipto\n"
        "      - name: Nafis Sadique Ayan\n      - name: Sultana Razia Faria\n  - type: article\n    title: \"BanglaFake: Constructing "
        "and Evaluating a Specialized Bengali Deepfake Audio Dataset\"\n    url: https://arxiv.org/abs/2505.10885\n    authors:\n"
        "      - name: Istiaq Ahmed Fahad\n      - name: Kamruzzaman Asif\n      - name: Sifat Sikder\n")
    (out / "dataset_card.md").write_text(
        f"# Dataset card: {TITLE}\n\n**Motivation.** Leakage-aware, reproducible evaluation of Bengali audio deepfake "
        f"detection, including cross-corpus tests.\n\n**Composition.** {len(meta):,} recordings "
        f"({meta.duration_sec.sum() / 3600:.2f} h): BanglaFake SUST ({counts.loc['BF-SUST', 0]:,} bona fide / "
        f"{counts.loc['BF-SUST', 1]:,} spoof), Common Voice ({counts.loc['BF-MOZ', 0]:,} / {counts.loc['BF-MOZ', 1]:,}), "
        f"news ({counts.loc['BF-NEWS', 0]:,} bona fide, label unverified), Mendeley version 4 ({counts.loc['MEN', 0]:,} / "
        f"{counts.loc['MEN', 1]:,}; {meta[meta.dataset_id == 'MEN'].speaker_id.nunique()} speakers). All WAV PCM-16 mono.\n\n"
        "**Collection.** Created by the original dataset authors; this release adds metadata, checksums, splits, "
        "descriptors, audits and scores only.\n\n**Known biases and shortcuts.** The class is confounded with "
        "recording properties in every corpus (trailing silence, DC offset and zero padding in BanglaFake-SUST; sample "
        "rate, leading silence and loudness in the Common Voice subset; sample rate, clipping and noise level in "
        "Mendeley). Detectors trained on these data should be evaluated with the provided controls.\n\n**Uses.** "
        "Research on audio deepfake detection. Not for identifying or profiling speakers.\n\n**Personal data.** Voice "
        "recordings are biometric personal data. No recordings are redistributed here; users obtain them from the "
        "providers under the providers' terms.\n\n**Maintenance.** Report issues in the Discussion tab of "
        f"{KAGGLE_URL} or at {CODE_URL}.\n")

    audio = [str(p.relative_to(out)) for p in out.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3")]
    leaks = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and LEAK.search(p.read_text(errors="ignore"))]
    junk = [str(p.relative_to(out)) for p in out.rglob("*") if p.name in (".DS_Store", "__pycache__", "__MACOSX")]
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    rep = {"release": str(out.relative_to(ROOT)) if ROOT in out.parents else out.name, "kaggle": KAGGLE_URL,
           "doi": DOI, "files_listed": int(len(meta)), "audio_files_in_release": len(audio), "release_bytes": size,
           "hours_total": round(float(meta.duration_sec.sum()) / 3600, 3),
           "by_dataset_label": {f"{d}|{l}": int(n) for (d, l), n in meta.groupby(["dataset_id", "label"]).size().items()},
           "exact_zero_ratio": has_zero, "score_runs": sorted(scores.run.unique().tolist()) if len(scores) else [],
           "private_strings": leaks, "junk_files": junk}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/kaggle_release_summary.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    if audio or leaks or junk:
        sys.exit("release NOT ready: audio files, private strings or junk files (see above)")


if __name__ == "__main__":
    main()
