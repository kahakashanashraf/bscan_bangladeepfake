"""Shared helpers for the generalisation experiments (outside src/bscan, so the paper's code version is unchanged).

Data: the corpora are extracted under data/ exactly as in the main pipeline (data/<metadata audio_path>); the splits
are the paper's split lists (splits/P1_*.txt, P2_*.txt).  Every recording goes through the paper's controlled
preprocessing (16 kHz mono, peak normalisation, silence trimming, 6 s windows with 10% overlap, repeat-padding,
DC removal, -60 dBFS dither), so results are comparable with the paper's tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bscan.audio import load_audio, segment  # noqa: E402
from bscan.metrics import aggregate_by_file, cluster_bootstrap_ci, eer, select_threshold  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "experiments" / "generalization" / "work"   # features, models and scores (gitignored)
SR = 16000
PROTOCOL_SPLITS = {
    "P1": ["train", "validation", "internal_test", "external_BF-MOZ", "external_MEN"],
    "P2": ["train", "validation", "internal_test", "external_BF-SUST", "external_BF-MOZ"],
}


def metadata() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "metadata" / "unified_metadata.csv")
    return m.set_index("audio_path")


def split_files(protocol: str, split: str) -> list[str]:
    return [l.strip() for l in (ROOT / "splits" / f"{protocol}_{split}.txt").read_text().splitlines() if l.strip()]


def windows_of(audio_path: str | Path, seed: int = 0) -> list[np.ndarray]:
    """The 6 s windows the paper's models see for one recording (controlled preprocessing)."""
    p = Path(audio_path)
    x, _ = load_audio(p if p.is_absolute() else DATA / p)
    return [s.audio for s in segment(x, scheme="controlled", seed=seed)]


def recording_metrics(rec_ids, scores, labels, groups=None, thr: float | None = None) -> dict:
    """AUC, EER and (when groups are given) a cluster-bootstrap CI of the AUC, from per-window or per-file scores."""
    from sklearn.metrics import roc_auc_score
    ids, s, y = aggregate_by_file(rec_ids, scores, labels)
    out = {"n": len(ids), "auc": float(roc_auc_score(y, s)), "eer": float(eer(y, s)[0])}
    if groups is not None:
        g = pd.Series(groups, index=rec_ids).groupby(level=0).first().reindex(ids).values
        lo, hi, _ = cluster_bootstrap_ci(lambda r: roc_auc_score(y[r], s[r]) if len(set(y[r])) == 2 else float("nan"),
                                         g, n_boot=1000)
        out.update(auc_lo=lo, auc_hi=hi)
    if thr is not None:
        pred = (s >= thr).astype(int)
        out.update(acc=float((pred == y).mean()), spec=float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
                   sens=float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)))
    return out
