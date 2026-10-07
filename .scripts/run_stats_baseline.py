"""Recording-statistics baseline (coarse recording statistics), evaluated exactly like the neural models.

A histogram gradient-boosting classifier is trained on the 11 coarse window descriptors
of `.scripts/model_visible_shortcuts.py` (padding, silence/level, spectral balance; no speech
content) from the protocol's TRAIN split.  Window scores (log-odds) are mean-pooled per recording;
the threshold is the validation EER point; metrics and cluster-bootstrap CIs come from
`bscan.evaluate`, so the output files have the same format as every neural run:

    results/runs/stats_hgb_{scheme}__{protocol}__seed{seed}/metrics.json, predictions_{split}.csv

Usage:
    python .scripts/run_stats_baseline.py --schemes orig6s controlled --protocols P1 P2 --seed 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bscan.evaluate import file_table, split_metrics  # noqa: E402
from bscan.metrics import select_threshold  # noqa: E402
from bscan.utils import environment_info, save_json  # noqa: E402

DESCRIPTORS = ["pad_fraction", "floor_frame_ratio", "lead_floor_frac", "trail_floor_frac", "quiet_frame_ratio",
               "noise_floor_db", "dc", "clip_ratio", "lf_ratio", "hf_ratio", "centroid_hz"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=str(ROOT / "results/phase3/window_descriptors.csv.gz"))
    ap.add_argument("--metadata", default=str(ROOT / "metadata/unified_metadata.csv"))
    ap.add_argument("--schemes", nargs="+", default=["orig6s", "controlled"])
    ap.add_argument("--protocols", nargs="+", default=["P1", "P2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", default=str(ROOT / "results/runs"))
    args = ap.parse_args()

    win = pd.read_csv(args.windows)
    meta = pd.read_csv(args.metadata, dtype={"utterance_id": str}, low_memory=False)
    keep = ["audio_path", "label", "dataset_id", "speaker_id", "text_group", "split_P1", "split_P2"]
    win = win.merge(meta[keep], on="audio_path", how="left")
    assert win["label"].notna().all()

    for scheme in args.schemes:
        ws = win[win["scheme"] == scheme].sort_values(["audio_path", "window"]).reset_index(drop=True)
        for proto in args.protocols:
            col = f"split_{proto}"
            rd = Path(args.out_dir) / f"stats_hgb_{scheme}__{proto}__seed{args.seed}"
            rd.mkdir(parents=True, exist_ok=True)
            tr = ws[ws[col] == "train"]
            t0 = time.perf_counter()
            clf = HistGradientBoostingClassifier(random_state=args.seed).fit(tr[DESCRIPTORS], tr["label"])
            fit_sec = time.perf_counter() - t0
            tables = {}
            for sp in sorted(s for s in ws[col].unique() if s not in ("train", "unused")):
                part = ws[ws[col] == sp].reset_index(drop=True)
                logits = clf.decision_function(part[DESCRIPTORS])  # log-odds, comparable to network logits
                part[["audio_path", "window", "pad_fraction", "label", "dataset_id"]].assign(logit=logits).to_csv(
                    rd / f"windows_{sp}.csv.gz", index=False)
                tables[sp] = file_table(part, logits)
                tables[sp].to_csv(rd / f"predictions_{sp}.csv", index=False)
            thr = select_threshold(tables["validation"]["label"].to_numpy(), tables["validation"]["score"].to_numpy(), "eer")
            res = {"model": "HistGradientBoosting on 11 content-agnostic window descriptors", "scheme": scheme,
                   "protocol": proto, "seed": args.seed, "descriptors": DESCRIPTORS, "threshold_rule": "eer",
                   "threshold_logit": thr, "fit_sec": fit_sec, "train_windows": int(len(tr)), "splits": {}}
            for sp, f in tables.items():
                m = split_metrics(f, thr, args.n_boot, args.seed)
                rel = m.pop("_reliability", None)
                if rel:
                    pd.DataFrame(rel).to_csv(rd / f"reliability_{sp}.csv", index=False)
                res["splits"][sp] = m
                if m.get("n_bonafide") and m.get("n_spoof"):
                    print(f"{scheme:10s} {proto} {sp:17s} AUC {m['auc']:.4f} [{m['auc_ci95'][0]:.3f},{m['auc_ci95'][1]:.3f}] "
                          f"EER {m['eer']:.4f} acc {m['accuracy']:.4f} sens {m['recall_spoof_sensitivity']:.3f} "
                          f"spec {m['specificity_bonafide_recall']:.3f}")
                else:
                    print(f"{scheme:10s} {proto} {sp:17s} single-class: frac predicted spoof {m['frac_pred_spoof']:.4f}")
            save_json(res, rd / "metrics.json")
            save_json(environment_info(), rd / "environment.json")


if __name__ == "__main__":
    main()
