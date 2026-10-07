"""Evaluate a trained run on every split of its protocol.

Threshold discipline: the operating threshold is chosen ONCE on the validation split
(cfg.eval.threshold_rule, per-recording scores) and applied unchanged to every other split.
Primary unit: the recording (mean of window logits).  Window-level metrics are also saved.

Confidence intervals: cluster bootstrap over independent units -
    BanglaFake splits: sentence groups (a bona fide file and its VITS rendition stay together)
    Mendeley splits:   speakers
Single-class splits (BF-NEWS, flagged) report the fraction of recordings classified as spoof.

Usage:
    python -m bscan.evaluate --config configs/bscan_controlled.yaml [train.seed=123 ...]

Writes to cfg.run_dir(): predictions_{split}.csv (per recording), windows_{split}.csv.gz,
metrics.json, reliability_{split}.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentCfg, load_config, parse_overrides
from .data import WindowDataset, build_feature_cache
from .metrics import aggregate_by_file, binary_metrics, brier, cluster_bootstrap_ci, ece, eer, select_threshold
from .models import build_model, model_features
from .train import make_loader, predict_logits
from .utils import pick_device, save_json


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cluster_ids(files: pd.DataFrame) -> np.ndarray:
    return np.where(files["dataset_id"].eq("MEN"), "SPK:" + files["speaker_id"].astype(str),
                    "SEN:" + files["text_group"].astype(str))


def file_table(index: pd.DataFrame, logits: np.ndarray) -> pd.DataFrame:
    w = index.assign(logit=logits)
    f = w.groupby("audio_path").agg(label=("label", "first"), dataset_id=("dataset_id", "first"),
                                     speaker_id=("speaker_id", "first"), text_group=("text_group", "first"),
                                     score=("logit", "mean"), n_windows=("logit", "size"),
                                     max_pad_fraction=("pad_fraction", "max")).reset_index()
    f["prob_spoof"] = _sigmoid(f["score"].to_numpy())
    return f


def split_metrics(f: pd.DataFrame, thr: float, n_boot: int, seed: int) -> dict:
    y, s = f["label"].to_numpy().astype(int), f["score"].to_numpy()
    groups = cluster_ids(f)
    m = binary_metrics(y, s, thr)
    if len(np.unique(y)) == 2:
        for name, fn in (("auc", lambda i: m_auc(y[i], s[i])), ("eer", lambda i: eer(y[i], s[i])[0]),
                         ("accuracy", lambda i: float(np.mean((s[i] >= thr) == y[i]))),
                         ("f1_macro", lambda i: binary_metrics(y[i], s[i], thr)["f1_macro"])):
            lo, hi, _ = cluster_bootstrap_ci(fn, groups, n_boot=n_boot, seed=seed)
            m[f"{name}_ci95"] = [lo, hi]
        p = _sigmoid(s)
        m["brier"] = brier(y, p)
        m["ece"], rel = ece(y, p)
        m["_reliability"] = rel
    else:
        frac = lambda i: float(np.mean(s[i] >= thr))  # noqa: E731
        m["frac_pred_spoof"] = frac(np.arange(len(s)))
        lo, hi, _ = cluster_bootstrap_ci(frac, groups, n_boot=n_boot, seed=seed)
        m["frac_pred_spoof_ci95"] = [lo, hi]
    m["n_clusters"] = int(len(np.unique(groups)))
    return m


def m_auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def run_evaluation(cfg: ExperimentCfg, log=print) -> dict:
    import torch

    rd = cfg.run_dir()
    feats = model_features(cfg.model)
    stats = json.loads((rd / "norm_stats.json").read_text())

    # build every feature cache BEFORE initialising the GPU (worker processes are forked on Linux)
    splits = ["validation"] + [s for s in cfg.data.eval_splits if s != "validation"]
    dirs = {}
    for sp in splits:
        try:
            dirs[sp] = build_feature_cache(cfg.data, sp, feats, log=log)
        except ValueError:
            log(f"  skip {sp}: not defined for protocol {cfg.data.protocol}")

    device = pick_device(cfg.train.device)
    amp = cfg.train.amp and device.type == "cuda"
    ck = torch.load(rd / "best.pt", map_location=device, weights_only=False)
    model = build_model(cfg.model).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    tables = {}
    for sp, d in dirs.items():
        ds = WindowDataset(d, feats, stats)
        logits = predict_logits(model, make_loader(ds, 64, False, cfg.data.num_workers, 0, device), device, amp)
        win = ds.index.assign(logit=logits)
        win.to_csv(rd / f"windows_{sp}.csv.gz", index=False)
        tables[sp] = file_table(ds.index, logits)
        tables[sp].to_csv(rd / f"predictions_{sp}.csv", index=False)

    val = tables["validation"]
    thr = select_threshold(val["label"].to_numpy(), val["score"].to_numpy(), cfg.eval.threshold_rule)
    results = {"threshold_rule": cfg.eval.threshold_rule, "threshold_logit": thr, "best_epoch": ck["epoch"],
               "splits": {}}
    for sp, f in tables.items():
        m = split_metrics(f, thr, cfg.eval.n_boot, cfg.train.seed)
        rel = m.pop("_reliability", None)
        if rel:
            pd.DataFrame(rel).to_csv(rd / f"reliability_{sp}.csv", index=False)
        results["splits"][sp] = m
        if "auc" in m and m["n_bonafide"] and m["n_spoof"]:
            log(f"  {sp:18s} n={m['n']:5d} AUC {m['auc']:.4f} EER {m['eer']:.4f} acc {m['accuracy']:.4f} "
                f"F1m {m['f1_macro']:.4f} sens {m['recall_spoof_sensitivity']:.3f} spec {m['specificity_bonafide_recall']:.3f}")
        else:
            log(f"  {sp:18s} n={m['n']:5d} single-class: frac predicted spoof {m.get('frac_pred_spoof', float('nan')):.4f}")
    save_json(results, rd / "metrics.json")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    run_evaluation(load_config(args.config, parse_overrides(args.overrides)))


if __name__ == "__main__":
    main()
