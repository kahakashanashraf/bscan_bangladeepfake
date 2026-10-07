"""Summarise the Phase-1 inventory into machine-readable tables and the release inventory CSV.

Outputs (in --out):
    counts.csv                 files and hours per dataset x label
    duration_stats.csv         duration distribution per dataset x label
    format_stats.csv           sample rate / channels / codec per dataset x label
    signal_stats.csv           median (IQR) signal descriptors per dataset x label
    univariate_separability.csv  per-subset AUC of each descriptor used alone as a real/fake score
    id_pairing.csv             utterance-id overlap between real and fake within each subset
    mendeley_speakers.csv      files per Mendeley speaker / set / gender / label
    errors.csv                 unreadable files
and metadata/dataset_inventory.csv (release-facing columns).

Usage:
    python .scripts/summarize_inventory.py --inventory data/inventory --out results/phase1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DESCRIPTORS = [
    "duration_sec", "sample_rate", "file_size_bytes", "peak_abs", "rms_dbfs", "lufs", "exact_zero_ratio",
    "clip_ratio", "dc_offset", "leading_silence_sec", "trailing_silence_sec", "activity_ratio",
    "snr_proxy_db", "rolloff95_hz", "rolloff99_hz", "band_edge_hz", "zcr",
]

RELEASE_COLUMNS = [
    "dataset_id", "dataset_name", "source", "original_split", "new_split", "audio_path", "label", "label_name",
    "speaker_id", "set_id", "gender", "utterance_id", "generator", "language", "duration_sec", "sample_rate",
    "channels", "codec", "container", "frames", "file_size_bytes", "source_dataset", "sha256", "pcm_sha256",
    "license", "notes",
]

NOTES = {
    "BF-SUST": "SUST TTS corpus speaker (bona fide) / BanglaFake VITS voice (spoof)",
    "BF-MOZ": "Mozilla Common Voice speakers (bona fide) / BanglaFake VITS voice reading Common Voice text (spoof)",
    "BF-NEWS": "bona fide only; no spoof counterpart in the release",
    "MEN": "15 sets x 5 speakers; each set reads the same 30 sentences; spoof generator undocumented",
}


def q(s: pd.Series, p: float) -> float:
    s = s.dropna()
    return float(np.percentile(s, p)) if len(s) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="data/inventory")
    ap.add_argument("--out", default="results/phase1")
    ap.add_argument("--release_csv", default="metadata/dataset_inventory.csv")
    args = ap.parse_args()
    inv, out = Path(args.inventory), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inv / "inventory_full.csv", dtype={"utterance_id": str}, low_memory=False)
    err = df[df["error"].fillna("") != ""]
    err.to_csv(out / "errors.csv", index=False)
    df = df[df["error"].fillna("") == ""].copy()
    df["label_name"] = df["label"].map({0: "bonafide", 1: "spoof"})
    keys = ["dataset_id", "label_name"]

    counts = df.groupby(keys).agg(files=("audio_path", "size"), hours=("duration_sec", lambda s: s.sum() / 3600)).round(3)
    counts.to_csv(out / "counts.csv")

    dur = df.groupby(keys)["duration_sec"].agg(
        mean="mean", sd="std", median="median", min="min", max="max",
        p05=lambda s: q(s, 5), p95=lambda s: q(s, 95),
        frac_le_6s=lambda s: float((s <= 6.0).mean()),
    ).round(3)
    dur.to_csv(out / "duration_stats.csv")

    fmt = (
        df.groupby(keys + ["sample_rate", "channels", "container", "codec"]).size().rename("files").reset_index()
    )
    fmt.to_csv(out / "format_stats.csv", index=False)

    sig_rows = []
    for (d, l), g in df.groupby(keys):
        row = {"dataset_id": d, "label_name": l}
        for c in DESCRIPTORS:
            if c in g:
                row[f"{c}_median"] = round(q(g[c], 50), 4)
                row[f"{c}_iqr"] = f"{q(g[c], 25):.4g}-{q(g[c], 75):.4g}"
        sig_rows.append(row)
    pd.DataFrame(sig_rows).to_csv(out / "signal_stats.csv", index=False)

    # univariate separability: descriptor used alone as score for 'spoof' within each subset
    sep_rows = []
    for d, g in df.groupby("dataset_id"):
        if g["label"].nunique() < 2:
            continue
        for c in DESCRIPTORS:
            v = g[c]
            ok = v.notna()
            if ok.sum() < 10 or v[ok].nunique() < 2:
                auc = 0.5 if ok.sum() >= 10 else float("nan")
            else:
                auc = roc_auc_score(g.loc[ok, "label"], v[ok])
            sep_rows.append({
                "dataset_id": d, "descriptor": c, "auc_spoof_high": round(float(auc), 4),
                "separability": round(float(max(auc, 1 - auc)), 4) if auc == auc else float("nan"),
                "direction": "higher=>spoof" if auc >= 0.5 else "lower=>spoof",
                "n": int(ok.sum()),
            })
    pd.DataFrame(sep_rows).sort_values(["dataset_id", "separability"], ascending=[True, False]).to_csv(
        out / "univariate_separability.csv", index=False
    )

    pair_rows = []
    for d, g in df.groupby("dataset_id"):
        if d == "MEN":
            key = g["speaker_id"] + "/" + g["utterance_id"].astype(str)
        else:
            key = g["utterance_id"].astype(str)
        real, fake = set(key[g["label"] == 0]), set(key[g["label"] == 1])
        def num_range(ids):
            nums = pd.to_numeric(pd.Series(sorted(ids)).str.extract(r"(\d+)$")[0], errors="coerce").dropna()
            return (int(nums.min()), int(nums.max())) if len(nums) else (None, None)
        pair_rows.append({
            "dataset_id": d, "real_ids": len(real), "fake_ids": len(fake), "ids_in_both": len(real & fake),
            "real_numeric_range": num_range({k.split('/')[-1] for k in real}),
            "fake_numeric_range": num_range({k.split('/')[-1] for k in fake}),
        })
    pd.DataFrame(pair_rows).to_csv(out / "id_pairing.csv", index=False)

    men = df[df["dataset_id"] == "MEN"]
    if len(men):
        men.groupby(["set_id", "speaker_id", "gender", "label_name"]).size().rename("files").reset_index().to_csv(
            out / "mendeley_speakers.csv", index=False
        )

    rel = df.copy()
    rel["original_split"] = "none (no official split)"
    rel["new_split"] = "TBD (Phase 3)"
    rel["source_dataset"] = rel["dataset_id"].map({
        "BF-SUST": "BanglaFake/SUST", "BF-MOZ": "BanglaFake/Mozilla-CommonVoice",
        "BF-NEWS": "BanglaFake/News", "MEN": "Mendeley/4ftmwt86vr.4",
    })
    rel["notes"] = rel["dataset_id"].map(NOTES)
    for c in RELEASE_COLUMNS:
        if c not in rel:
            rel[c] = "unknown"
    rel[RELEASE_COLUMNS].to_csv(args.release_csv, index=False)

    summary = {
        "files_ok": int(len(df)), "files_error": int(len(err)),
        "total_hours": round(float(df["duration_sec"].sum() / 3600), 3),
        "by_dataset_label": {f"{a}|{b}": int(n) for (a, b), n in df.groupby(keys).size().items()},
    }
    (out / "phase1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
