"""Exact and near-duplicate audit over the inventory (Phase 1 / Phase 3).

Exact duplicates:  identical file bytes (sha256) or identical decoded PCM (pcm_sha256).
Near duplicates:   candidate pairs from nearest neighbours of the log-mel fingerprint
                   (cosine), confirmed by the peak normalised cross-correlation of the
                   16 kHz waveforms (same recording, possibly re-encoded or trimmed).

Outputs (in --out):
    exact_duplicate_groups.csv   one row per file that shares a hash with another file
    near_duplicate_pairs.csv     candidate pairs with fingerprint cosine and confirmed xcorr
    duplicate_summary.json

Usage:
    python .scripts/duplicate_check.py --inventory data/inventory --root data --out results/phase1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

XCORR_CONFIRM = 0.95   # peak normalised cross-correlation to call two clips the same recording
COS_CANDIDATE = 0.995  # fingerprint cosine similarity to shortlist a pair for waveform checking
K_NEIGHBOURS = 6


def exact_groups(df: pd.DataFrame, key: str) -> pd.DataFrame:
    g = df[df[key].notna() & (df[key] != "")].groupby(key)
    dup = g.filter(lambda x: len(x) > 1).copy()
    if dup.empty:
        return dup
    dup["group_size"] = dup.groupby(key)[key].transform("size")
    dup["hash_type"] = key
    return dup[[key, "hash_type", "group_size", "audio_path", "dataset_id", "label", "speaker_id", "duration_sec"]]


def load16k(path: Path) -> np.ndarray:
    import librosa
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-9)


def peak_xcorr(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.signal import fftconvolve

    if len(a) < len(b):
        a, b = b, a
    c = fftconvolve(a, b[::-1], mode="full")
    return float(np.max(np.abs(c)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="data/inventory")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results/phase1")
    args = ap.parse_args()
    inv, root, out = Path(args.inventory), Path(args.root).resolve(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inv / "inventory_full.csv", dtype={"utterance_id": str}, low_memory=False)
    df = df[df["error"].fillna("") == ""].reset_index(drop=True)

    ex = pd.concat([exact_groups(df, "sha256"), exact_groups(df, "pcm_sha256")], ignore_index=True)
    ex.to_csv(out / "exact_duplicate_groups.csv", index=False)

    # ---------------- near duplicates ----------------
    from sklearn.neighbors import NearestNeighbors

    fp = np.load(inv / "fingerprints.npy")
    rows = df["fingerprint_row"].astype(int).to_numpy()
    X = fp[rows]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    cand = set()
    for i in range(len(X)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            if 1.0 - d >= COS_CANDIDATE and i != j:
                cand.add((min(i, j), max(i, j), float(1.0 - d)))
    pairs = []
    cache: dict[int, np.ndarray] = {}
    for i, j, cos in sorted(cand):
        for k in (i, j):
            if k not in cache:
                cache[k] = load16k(root / df.at[k, "audio_path"])
        xc = peak_xcorr(cache[i], cache[j])
        pairs.append(
            dict(
                path_a=df.at[i, "audio_path"], path_b=df.at[j, "audio_path"],
                dataset_a=df.at[i, "dataset_id"], dataset_b=df.at[j, "dataset_id"],
                label_a=int(df.at[i, "label"]), label_b=int(df.at[j, "label"]),
                speaker_a=df.at[i, "speaker_id"], speaker_b=df.at[j, "speaker_id"],
                dur_a=float(df.at[i, "duration_sec"]), dur_b=float(df.at[j, "duration_sec"]),
                fingerprint_cosine=round(cos, 6), peak_xcorr=round(xc, 4),
                confirmed_same_recording=bool(xc >= XCORR_CONFIRM),
            )
        )
        if len(cache) > 4000:
            cache.clear()
    near = pd.DataFrame(pairs)
    near.to_csv(out / "near_duplicate_pairs.csv", index=False)

    def count_groups(key: str) -> dict:
        sub = ex[ex["hash_type"] == key] if not ex.empty else ex
        if sub.empty:
            return {"files_in_groups": 0, "groups": 0, "cross_label_groups": 0, "cross_dataset_groups": 0}
        g = sub.groupby(key)
        return {
            "files_in_groups": int(len(sub)),
            "groups": int(g.ngroups),
            "cross_label_groups": int((g["label"].nunique() > 1).sum()),
            "cross_dataset_groups": int((g["dataset_id"].nunique() > 1).sum()),
        }

    conf = near[near["confirmed_same_recording"]] if not near.empty else near
    summary = {
        "n_files_checked": int(len(df)),
        "exact_file_sha256": count_groups("sha256"),
        "exact_pcm_sha256": count_groups("pcm_sha256"),
        "near_duplicate_candidates": int(len(near)),
        "near_duplicate_confirmed_pairs": int(len(conf)),
        "near_confirmed_cross_label": int((conf["label_a"] != conf["label_b"]).sum()) if len(conf) else 0,
        "near_confirmed_cross_dataset": int((conf["dataset_a"] != conf["dataset_b"]).sum()) if len(conf) else 0,
        "thresholds": {"fingerprint_cosine_candidate": COS_CANDIDATE, "xcorr_confirm": XCORR_CONFIRM, "k": K_NEIGHBOURS},
    }
    (out / "duplicate_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
