"""Phase 3: group-aware splits, overlap checks, and metadata-only shortcut diagnostics.

1. Builds the leakage-aware splits (written to splits/ and metadata/):
   P1 (BanglaFake -> others): BF-SUST split 70/15/15 into train / validation / internal_test,
      grouped by sentence text, so that each bona fide recording and its VITS counterpart (and every
      repeated sentence) stay inside one split.  BF-MOZ, BF-NEWS and MEN are held out entirely as
      external tests.
   P2 (Mendeley -> others): MEN split by *set* (5 speakers reading the same 30 sentences) into
      9 / 3 / 3 sets, so train, validation and test are speaker- and sentence-disjoint.
   Official source data are never modified; splits are defined through metadata only.
2. Checks every pair of splits for shared file hashes, shared decoded audio, confirmed
   near-duplicate recordings, shared speakers and shared utterance ids.
3. Trains simple metadata-only classifiers (no audio content) on each protocol's training split
   and evaluates them on every test split.  High AUC here means the label is predictable from
   recording/format properties alone, i.e. a shortcut is available to any detector.

Usage:
    python .scripts/leakage_check.py --inventory data/inventory --phase1 results/phase1 --out results/phase3
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
WINDOW_SEC = 6.0  # original BSCAN window

FEATURE_GROUPS = {
    "duration": ["duration_sec"],
    "pad_fraction_6s": ["pad_fraction_6s"],
    "sample_rate": ["sample_rate"],
    "bitrate": ["bitrate_kbps"],
    "loudness": ["peak_abs", "rms_dbfs", "lufs"],
    "silence": ["leading_silence_sec", "trailing_silence_sec", "activity_ratio", "exact_zero_ratio"],
    "bandwidth": ["rolloff95_hz", "rolloff99_hz", "band_edge_hz"],
    "snr_proxy": ["snr_proxy_db"],
    "all_metadata": [
        "duration_sec", "pad_fraction_6s", "sample_rate", "bitrate_kbps", "peak_abs", "rms_dbfs", "lufs",
        "leading_silence_sec", "trailing_silence_sec", "activity_ratio", "exact_zero_ratio", "rolloff95_hz",
        "rolloff99_hz", "band_edge_hz", "snr_proxy_db", "zcr", "dc_offset", "clip_ratio",
    ],
}


def eer_and_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def assign_groups(groups: np.ndarray, fractions: tuple[float, ...], names: tuple[str, ...], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    cuts = np.cumsum(np.array(fractions) / sum(fractions) * len(uniq)).round().astype(int)
    out, start = {}, 0
    for name, end in zip(names, cuts):
        for g in uniq[start:end]:
            out[g] = name
        start = end
    return out


def build_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split_P1"] = "unused"
    sust = df["dataset_id"] == "BF-SUST"
    # group = sentence text: keeps each bona fide file with its VITS counterpart (same integer id,
    # different zero-padding) and keeps the 37 duplicated sentences inside one split
    amap = assign_groups(df.loc[sust, "text_group"].astype(str).to_numpy(), (0.70, 0.15, 0.15),
                         ("train", "validation", "internal_test"), SEED)
    df.loc[sust, "split_P1"] = df.loc[sust, "text_group"].astype(str).map(amap)
    df.loc[df["dataset_id"] == "BF-MOZ", "split_P1"] = "external_BF-MOZ"
    df.loc[df["dataset_id"] == "BF-NEWS", "split_P1"] = "external_BF-NEWS"
    df.loc[df["dataset_id"] == "MEN", "split_P1"] = "external_MEN"

    df["split_P2"] = "unused"
    men = df["dataset_id"] == "MEN"
    smap = assign_groups(df.loc[men, "set_id"].astype(str).to_numpy(), (9, 3, 3),
                         ("train", "validation", "internal_test"), SEED)
    df.loc[men, "split_P2"] = df.loc[men, "set_id"].astype(str).map(smap)
    for d in ("BF-SUST", "BF-MOZ", "BF-NEWS"):
        df.loc[df["dataset_id"] == d, "split_P2"] = f"external_{d}"
    return df


def overlap_report(df: pd.DataFrame, split_col: str, near: pd.DataFrame) -> list[dict]:
    rows = []
    splits = [s for s in df[split_col].unique() if s != "unused"]
    path2split = dict(zip(df["audio_path"], df[split_col]))
    for a, b in combinations(sorted(splits), 2):
        A, B = df[df[split_col] == a], df[df[split_col] == b]
        row = {"protocol": split_col, "split_a": a, "split_b": b}
        for key in ("sha256", "pcm_sha256", "utterance_key", "text_group"):
            row[f"shared_{key}"] = len(set(A[key].astype(str)) & set(B[key].astype(str)))
        spk_a = set(A.loc[A["speaker_id"] != "unknown", "speaker_id"])
        spk_b = set(B.loc[B["speaker_id"] != "unknown", "speaker_id"])
        row["shared_known_speakers"] = sorted(spk_a & spk_b)
        if len(near):
            conf = near[near["confirmed_same_recording"]]
            sa, sb = conf["path_a"].map(path2split), conf["path_b"].map(path2split)
            row["confirmed_near_duplicate_pairs"] = int((((sa == a) & (sb == b)) | ((sa == b) & (sb == a))).sum())
        rows.append(row)
    return rows


def run_diagnostics(df: pd.DataFrame, split_col: str) -> list[dict]:
    rows = []
    train = df[df[split_col] == "train"]
    tests = [s for s in df[split_col].unique() if s not in ("train", "unused")]
    for gname, cols in FEATURE_GROUPS.items():
        Xtr = train[cols].astype(float)
        keep = Xtr.notna().all(axis=1)
        if keep.sum() < 50 or train.loc[keep, "label"].nunique() < 2:
            continue
        if Xtr[keep].nunique().max() <= 1:
            rows.append({"protocol": split_col, "features": gname, "note": "constant in training split (uninformative)"})
            continue
        model = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)) if len(cols) <= 4
                 else HistGradientBoostingClassifier(random_state=SEED))
        model.fit(Xtr[keep], train.loc[keep, "label"])
        s_tr = model.predict_proba(Xtr[keep])[:, 1]
        _, thr = eer_and_threshold(train.loc[keep, "label"].to_numpy(), s_tr)
        for t in tests:
            te = df[df[split_col] == t]
            Xte = te[cols].astype(float)
            ok = Xte.notna().all(axis=1)
            y = te.loc[ok, "label"].to_numpy()
            s = model.predict_proba(Xte[ok])[:, 1]
            r = {"protocol": split_col, "features": gname, "model": type(model[-1] if hasattr(model, "steps") else model).__name__,
                 "test_split": t, "n": int(ok.sum()), "n_spoof": int(y.sum()),
                 "acc_at_train_eer_thr": round(float(np.mean((s >= thr).astype(int) == y)), 4)}
            if len(np.unique(y)) == 2:
                auc = roc_auc_score(y, s)
                eer, _ = eer_and_threshold(y, s)
                r.update(auc=round(float(auc), 4), eer=round(eer, 4))
            else:
                r.update(auc=float("nan"), eer=float("nan"),
                         false_positive_rate=round(float(np.mean(s >= thr)), 4) if y.sum() == 0 else float("nan"))
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="data/inventory")
    ap.add_argument("--phase1", default="results/phase1")
    ap.add_argument("--out", default="results/phase3")
    ap.add_argument("--splits_dir", default="splits")
    ap.add_argument("--metadata_dir", default="metadata")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.inventory) / "inventory_full.csv", dtype={"utterance_id": str}, low_memory=False)
    df = df[df["error"].fillna("") == ""].reset_index(drop=True)
    df["pad_fraction_6s"] = np.clip(1.0 - df["duration_sec"] / WINDOW_SEC, 0.0, 1.0)
    df["bitrate_kbps"] = df["file_size_bytes"] * 8 / df["duration_sec"] / 1000.0

    near_path = Path(args.phase1) / "near_duplicate_pairs.csv"
    near = pd.read_csv(near_path) if near_path.exists() and near_path.stat().st_size > 5 else pd.DataFrame()

    df = build_splits(df)

    # machine-readable split definitions (paths only; audio is never copied).
    # Sentence texts (BanglaFake transcripts, licence unknown) are replaced by a hash AFTER the
    # split assignment: clustering only needs equality, and the assignment above is unchanged.
    import hashlib
    df["text_group"] = df["text_group"].astype(str).map(lambda t: "TG:" + hashlib.sha1(t.encode()).hexdigest()[:16])
    sd, md = Path(args.splits_dir), Path(args.metadata_dir)
    sd.mkdir(exist_ok=True)
    md.mkdir(exist_ok=True)
    cols = ["audio_path", "dataset_id", "label", "speaker_id", "set_id", "gender", "utterance_id", "utterance_key",
            "text_group", "duration_sec", "sample_rate", "sha256", "split_P1", "split_P2"]
    df[cols].to_csv(md / "unified_metadata.csv", index=False)
    for proto in ("P1", "P2"):
        for s, g in df.groupby(f"split_{proto}"):
            if s == "unused":
                continue
            g[cols].to_csv(md / f"{proto}_{s}.csv", index=False)
            (sd / f"{proto}_{s}.txt").write_text("\n".join(g["audio_path"]) + "\n")

    split_counts = (df.groupby(["split_P1", "dataset_id", "label"]).size().rename("files").reset_index())
    split_counts.to_csv(out / "split_counts_P1.csv", index=False)
    df.groupby(["split_P2", "dataset_id", "label"]).size().rename("files").reset_index().to_csv(
        out / "split_counts_P2.csv", index=False)

    ov = overlap_report(df, "split_P1", near) + overlap_report(df, "split_P2", near)
    pd.DataFrame(ov).to_csv(out / "split_overlap.csv", index=False)

    diag = pd.DataFrame(run_diagnostics(df, "split_P1") + run_diagnostics(df, "split_P2"))
    diag.to_csv(out / "metadata_shortcut_diagnostics.csv", index=False)

    # filename leakage: numeric part of the utterance id used alone as a score, per subset
    fn_rows = []
    for d, g in df.groupby("dataset_id"):
        if g["label"].nunique() < 2:
            continue
        num = pd.to_numeric(g["utterance_id"].astype(str).str.extract(r"(\d+)$")[0], errors="coerce")
        ok = num.notna()
        if ok.sum() > 10 and num[ok].nunique() > 1:
            auc = roc_auc_score(g.loc[ok, "label"], num[ok])
            fn_rows.append({"dataset_id": d, "auc_id_number": round(float(auc), 4),
                            "separability": round(float(max(auc, 1 - auc)), 4), "n": int(ok.sum())})
    pd.DataFrame(fn_rows).to_csv(out / "filename_leakage.csv", index=False)

    summary = {
        "splits_P1": {k: int(v) for k, v in df["split_P1"].value_counts().items()},
        "splits_P2": {k: int(v) for k, v in df["split_P2"].value_counts().items()},
        "seed": SEED,
    }
    (out / "phase3_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
