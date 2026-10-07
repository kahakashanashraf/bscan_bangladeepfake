"""Phase 3b: shortcut cues that survive preprocessing (what the detector can actually see).

Every recording is passed through the same preprocessing as the detector (16 kHz polyphase
resampling, segmentation scheme, per-window peak normalisation, padding) and simple, content-
agnostic descriptors are computed per window with the detector's STFT framing (n_fft 1024,
hop 512, centred).  Logistic-regression and gradient-boosting classifiers trained on these
descriptors (P1: BF-SUST train; P2: MEN train) are evaluated on every other split, per window
and per recording (mean of window scores).  This is done for both segmentation schemes so the
effect of the duration-controlled scheme on shortcut availability can be measured.

Descriptors (per window):
  pad_fraction        fraction of the window that is padding
  floor_frame_ratio   frames whose energy is <= peak-frame - 80 dB (Mel floor after power_to_db(ref=max))
  lead_floor_frac     leading run of such floor frames / frames
  trail_floor_frac    trailing run of floor frames / frames
  quiet_frame_ratio   frames <= peak-frame - 40 dB
  noise_floor_db      10th-percentile frame energy relative to the loudest frame
  dc                  mean sample value
  clip_ratio          fraction of samples with |x| >= 0.999 after peak normalisation
  lf_ratio            energy below 100 Hz / total
  hf_ratio            energy above 4 kHz / total
  centroid_hz         mean spectral centroid

Usage:
    python .scripts/model_visible_shortcuts.py --metadata metadata/unified_metadata.csv --root data --out results/phase3
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bscan.audio import load_audio, segment  # noqa: E402

N_FFT, HOP, SR = 1024, 512, 16000
GROUPS = {
    "padding": ["pad_fraction", "floor_frame_ratio", "lead_floor_frac", "trail_floor_frac"],
    "silence_level": ["quiet_frame_ratio", "noise_floor_db", "dc", "clip_ratio"],
    "spectral_balance": ["lf_ratio", "hf_ratio", "centroid_hz"],
}
GROUPS["all"] = sum(GROUPS.values(), [])
SEED = 42


def window_descriptors(x: np.ndarray) -> dict:
    pad = np.pad(x, (N_FFT // 2, N_FFT // 2))
    n = 1 + (len(pad) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = pad[idx] * np.hanning(N_FFT)[None, :]
    P = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    e = P.sum(axis=1)
    edb = 10 * np.log10(e + 1e-20)
    floor = edb <= edb.max() - 80.0
    lead = int(np.argmax(~floor)) if (~floor).any() else n
    trail = int(np.argmax(~floor[::-1])) if (~floor).any() else n
    f = np.fft.rfftfreq(N_FFT, 1 / SR)
    tot = P.sum() + 1e-20
    cen = (P * f[None, :]).sum(axis=1) / (e + 1e-20)
    return {
        "floor_frame_ratio": float(floor.mean()),
        "lead_floor_frac": lead / n,
        "trail_floor_frac": trail / n,
        "quiet_frame_ratio": float((edb <= edb.max() - 40.0).mean()),
        "noise_floor_db": float(np.percentile(edb, 10) - edb.max()),
        "dc": float(np.mean(x)),
        "clip_ratio": float(np.mean(np.abs(x) >= 0.999)),
        "lf_ratio": float(P[:, f < 100].sum() / tot),
        "hf_ratio": float(P[:, f > 4000].sum() / tot),
        "centroid_hz": float(np.mean(cen[~floor])) if (~floor).any() else 0.0,
    }


def process(args: tuple[str, str]) -> list[dict]:
    import hashlib

    rel, root = args
    x, _ = load_audio(Path(root) / rel)
    seed = int(hashlib.sha1(rel.encode()).hexdigest()[:8], 16)  # deterministic per file
    rows = []
    for scheme in ("orig6s", "trim_repeat", "controlled"):
        for s in segment(x, scheme=scheme, seed=seed):
            d = window_descriptors(s.audio)
            d.update(audio_path=rel, scheme=scheme, window=s.index, pad_fraction=s.pad_fraction)
            rows.append(d)
    return rows


def evaluate(win: pd.DataFrame, meta: pd.DataFrame, proto: str) -> list[dict]:
    col = f"split_{proto}"
    w = win.merge(meta[["audio_path", "label", col]], on="audio_path")
    out = []
    for scheme, ws in w.groupby("scheme"):
        tr = ws[ws[col] == "train"]
        for gname, cols in GROUPS.items():
            for mname, model in (
                ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
                ("hgb", HistGradientBoostingClassifier(random_state=SEED)),
            ):
                model.fit(tr[cols], tr["label"])
                for split in sorted(s for s in ws[col].unique() if s not in ("train", "unused")):
                    te = ws[ws[col] == split]
                    s = model.predict_proba(te[cols])[:, 1]
                    rec = {"protocol": proto, "scheme": scheme, "features": gname, "model": mname, "test_split": split,
                           "n_windows": int(len(te)), "n_files": int(te["audio_path"].nunique())}
                    if te["label"].nunique() == 2:
                        rec["auc_window"] = round(float(roc_auc_score(te["label"], s)), 4)
                        f = te.assign(s=s).groupby("audio_path").agg(s=("s", "mean"), y=("label", "first"))
                        rec["auc_file"] = round(float(roc_auc_score(f["y"], f["s"])), 4)
                    else:
                        rec["mean_p_spoof"] = round(float(np.mean(s)), 4)
                    out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="metadata/unified_metadata.csv")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results/phase3")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.metadata, dtype={"utterance_id": str}, low_memory=False)
    root = str(Path(args.root).resolve())
    with Pool(args.workers) as pool:
        rows = [r for rs in pool.imap_unordered(process, [(p, root) for p in meta["audio_path"]], chunksize=32) for r in rs]
    # fixed row order: HistGradientBoosting's internal early-stopping split depends on row order
    win = pd.DataFrame(rows).sort_values(["scheme", "audio_path", "window"]).reset_index(drop=True)
    win.to_csv(out / "window_descriptors.csv.gz", index=False)
    res = pd.DataFrame(evaluate(win, meta, "P1") + evaluate(win, meta, "P2"))
    res.to_csv(out / "model_visible_shortcut_diagnostics.csv", index=False)
    summ = win.merge(meta[["audio_path", "dataset_id", "label"]], on="audio_path").groupby(
        ["scheme", "dataset_id", "label"])[GROUPS["all"]].median().round(4)
    summ.to_csv(out / "window_descriptor_medians.csv")
    print(json.dumps({"windows": int(len(win)), "files": int(win["audio_path"].nunique())}, indent=2))


if __name__ == "__main__":
    main()
