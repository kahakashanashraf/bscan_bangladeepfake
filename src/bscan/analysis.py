"""Systematic error analysis and calibration summaries (Phases 11 and 15).

Error analysis is systematic, not anecdotal: every recording of a split is assigned to TP / TN /
FP / FN at the run's validation-derived threshold, and error rates are tabulated over
pre-registered groupings:
    corpus, speaker (known speakers), gender (Mendeley), Mendeley set,
    duration tertiles, SNR-proxy tertiles, trailing-silence bins, clipping (any / none),
    confidence (|score - threshold|) quintiles.
For each grouping, the false-negative rate is computed on spoofs and the false-positive rate on
bona fide recordings, with counts.  A second table compares descriptor distributions of false
negatives against true positives (medians, Mann-Whitney U p-values).

Usage:
    python -m bscan.analysis --runs bscan_controlled__P1__seed42 lcnn_lfcc__P1__seed42
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .registry import ROOT, resolve

DESC = ["duration_sec", "snr_proxy_db", "trailing_silence_sec", "leading_silence_sec", "activity_ratio",
        "clip_ratio", "dc_offset", "lufs", "band_edge_hz"]


def _load(name: str, split: str):
    st, path = resolve(name)
    if path is None or not (path / f"predictions_{split}.csv").exists():
        return st, None, None
    m = json.loads((path / "metrics.json").read_text())
    return st, pd.read_csv(path / f"predictions_{split}.csv", low_memory=False), m["threshold_logit"]


def error_tables(name: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    st, f, thr = _load(name, split)
    if f is None:
        return pd.DataFrame(), pd.DataFrame(), st
    desc = pd.read_csv(ROOT / "metadata" / "recording_descriptors.csv")
    meta = pd.read_csv(ROOT / "metadata" / "unified_metadata.csv", usecols=["audio_path", "gender", "set_id"],
                       low_memory=False)
    f = f.merge(desc, on="audio_path", how="left").merge(meta, on="audio_path", how="left")
    pred = (f["score"] >= thr).astype(int)
    f["outcome"] = np.select([(f.label == 1) & (pred == 1), (f.label == 0) & (pred == 0), (f.label == 0) & (pred == 1)],
                             ["TP", "TN", "FP"], "FN")
    f["confidence"] = (f["score"] - thr).abs()
    groupings = {
        "corpus": f["dataset_id"],
        "speaker": f["speaker_id"].where(f["speaker_id"] != "unknown"),
        "gender": f["gender"].where(f["gender"] != "unknown"),
        "set": f["set_id"].where(f["set_id"] != "unknown"),
        "duration_tertile": pd.qcut(f["duration_sec"], 3, labels=["short", "medium", "long"]),
        "snr_proxy_tertile": pd.qcut(f["snr_proxy_db"], 3, labels=["low", "mid", "high"], duplicates="drop"),
        "trailing_silence": pd.cut(f["trailing_silence_sec"], [-1, 0.05, 0.3, 100], labels=["<0.05s", "0.05-0.3s", ">0.3s"]),
        "clipping": np.where(f["clip_ratio"] > 0, "clipped", "unclipped"),
        "confidence_quintile": pd.qcut(f["confidence"], 5, labels=["q1 (least)", "q2", "q3", "q4", "q5 (most)"],
                                       duplicates="drop"),
    }
    rows = []
    for gname, g in groupings.items():
        for level, sub in f.groupby(g, observed=True):
            spoof, bona = sub[sub.label == 1], sub[sub.label == 0]
            rows.append({"run": name, "split": split, "grouping": gname, "level": str(level), "n": len(sub),
                         "n_spoof": len(spoof), "fn": int((spoof.outcome == "FN").sum()),
                         "fnr": float((spoof.outcome == "FN").mean()) if len(spoof) else np.nan,
                         "n_bonafide": len(bona), "fp": int((bona.outcome == "FP").sum()),
                         "fpr": float((bona.outcome == "FP").mean()) if len(bona) else np.nan})
    grp = pd.DataFrame(rows)
    comp = []
    fn, tp = f[f.outcome == "FN"], f[f.outcome == "TP"]
    for c in DESC:
        a, b = fn[c].dropna(), tp[c].dropna()
        p = float(mannwhitneyu(a, b).pvalue) if len(a) > 2 and len(b) > 2 else np.nan
        comp.append({"run": name, "split": split, "descriptor": c, "n_fn": len(a), "n_tp": len(b),
                     "fn_median": float(a.median()) if len(a) else np.nan,
                     "tp_median": float(b.median()) if len(b) else np.nan, "mann_whitney_p": p})
    return grp, pd.DataFrame(comp), st


def calibration_table(names: list[str], splits: list[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        st, path = resolve(n)
        if path is None:
            continue
        m = json.loads((path / "metrics.json").read_text())
        for sp in splits:
            v = m["splits"].get(sp)
            if v and "ece" in v:
                rows.append({"run": n, "split": sp, "status": st, "ece": v["ece"], "brier": v["brier"],
                             "auc": v["auc"], "accuracy": v["accuracy"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["bscan_controlled__P1__seed42", "lfcc_only__P1__seed42",
                                                  "lcnn_lfcc__P1__seed42", "bscan_orig6s__P1__seed42"])
    ap.add_argument("--splits", nargs="+", default=["internal_test", "external_BF-MOZ", "external_MEN"])
    args = ap.parse_args()
    out = ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    g_all, c_all, statuses = [], [], {}
    for n in args.runs:
        for sp in args.splits:
            g, c, st = error_tables(n, sp)
            statuses[n] = st
            if len(g):
                g["status"], c["status"] = st, st
                g_all.append(g)
                c_all.append(c)
    if g_all:
        pd.concat(g_all).to_csv(out / "error_groups.csv", index=False)
        pd.concat(c_all).to_csv(out / "error_fn_vs_tp.csv", index=False)
    calibration_table(args.runs, args.splits).to_csv(out / "calibration.csv", index=False)
    print(f"wrote {out}; run statuses: {statuses}")


if __name__ == "__main__":
    main()
