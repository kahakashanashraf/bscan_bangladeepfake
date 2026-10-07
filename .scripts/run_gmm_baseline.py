"""LFCC-GMM baseline (the classical ASVspoof countermeasure baseline), same protocol as BSCAN.

Front-end: the first 20 static LFCCs of the BSCAN LFCC tensor plus their deltas and delta-deltas
(60 dimensions per frame), computed with the same preprocessing scheme and caches as the neural
models.  Two diagonal-covariance GMMs (bona fide, spoof) are trained with EM on a seeded random
subset of TRAIN frames.  Window score = mean per-frame log-likelihood ratio (spoof - bona fide);
recording score = mean over windows; threshold = validation EER point; metrics and cluster-
bootstrap CIs come from bscan.evaluate, so the outputs match every other run:

    results/runs/lfcc_gmm_{scheme}__{protocol}__seed{seed}/metrics.json, predictions_{split}.csv

Usage:
    python .scripts/run_gmm_baseline.py --protocols P1 P2 --scheme controlled --components 256
"""

from __future__ import annotations

import os

# EM with multithreaded BLAS is not bit-reproducible (the reduction order varies between runs; verified on
# Apple Accelerate: three runs, three different mixtures). Single-threaded BLAS makes the baseline
# deterministic. These variables must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bscan.config import load_config, parse_overrides  # noqa: E402
from bscan.data import build_feature_cache  # noqa: E402
from bscan.evaluate import file_table, split_metrics  # noqa: E402
from bscan.metrics import select_threshold  # noqa: E402
from bscan.utils import environment_info, save_json  # noqa: E402

N_STATIC = 20


def frames(arr: np.ndarray) -> np.ndarray:
    """[n, 3, 40, T] float16 -> [n, T, 60] float32 (20 static + 20 delta + 20 delta-delta)."""
    x = np.asarray(arr[:, :, :N_STATIC, :], dtype=np.float32)  # [n, 3, 20, T]
    return x.reshape(x.shape[0], 3 * N_STATIC, x.shape[-1]).transpose(0, 2, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["P1", "P2"])
    ap.add_argument("--scheme", default="controlled")
    ap.add_argument("--components", type=int, default=256)
    ap.add_argument("--frames_per_class", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", default=str(ROOT / "results/runs"))
    args = ap.parse_args()

    for proto in args.protocols:
        cfg = load_config(ROOT / "configs/lfcc_only.yaml",
                          parse_overrides([f"data.protocol={proto}", f"data.scheme={args.scheme}"]))
        rd = Path(args.out_dir) / f"lfcc_gmm_{args.scheme}__{proto}__seed{args.seed}"
        rd.mkdir(parents=True, exist_ok=True)
        tr_dir = build_feature_cache(cfg.data, "train", ("lfcc",))
        idx = pd.read_csv(tr_dir / "index.csv")
        X = frames(np.load(tr_dir / "lfcc.npy", mmap_mode="r")[: len(idx)])
        y = idx["label"].to_numpy()
        rng = np.random.default_rng(args.seed)
        gmms, t_fit = {}, 0.0
        for lab in (0, 1):
            F = X[y == lab].reshape(-1, 3 * N_STATIC)
            F = F[rng.choice(len(F), size=min(args.frames_per_class, len(F)), replace=False)]
            t0 = time.perf_counter()
            gmms[lab] = GaussianMixture(n_components=args.components, covariance_type="diag", max_iter=100,
                                        reg_covar=1e-4, random_state=args.seed).fit(F)
            t_fit += time.perf_counter() - t0
            print(f"{proto} GMM class {lab}: {len(F):,} frames, converged={gmms[lab].converged_}, "
                  f"iterations={gmms[lab].n_iter_}")
        tables = {}
        for sp in [s for s in cfg.data.eval_splits]:
            try:
                d = build_feature_cache(cfg.data, sp, ("lfcc",))
            except ValueError:
                continue
            ix = pd.read_csv(d / "index.csv", low_memory=False)
            A = frames(np.load(d / "lfcc.npy", mmap_mode="r")[: len(ix)])
            n, T, D = A.shape
            flat = A.reshape(-1, D)
            llr = (gmms[1].score_samples(flat) - gmms[0].score_samples(flat)).reshape(n, T).mean(axis=1)
            ix.assign(logit=llr).to_csv(rd / f"windows_{sp}.csv.gz", index=False)
            tables[sp] = file_table(ix, llr)
            tables[sp].to_csv(rd / f"predictions_{sp}.csv", index=False)
        thr = select_threshold(tables["validation"]["label"].to_numpy(), tables["validation"]["score"].to_numpy(), "eer")
        res = {"model": f"LFCC-GMM ({args.components} diagonal components per class; 20 static LFCC + deltas)",
               "scheme": args.scheme, "protocol": proto, "seed": args.seed, "threshold_rule": "eer",
               "threshold_logit": thr, "fit_sec": t_fit, "frames_per_class": args.frames_per_class, "splits": {}}
        for sp, f in tables.items():
            m = split_metrics(f, thr, args.n_boot, args.seed)
            rel = m.pop("_reliability", None)
            if rel:
                pd.DataFrame(rel).to_csv(rd / f"reliability_{sp}.csv", index=False)
            res["splits"][sp] = m
            if m.get("n_bonafide") and m.get("n_spoof"):
                print(f"  {sp:17s} AUC {m['auc']:.4f} [{m['auc_ci95'][0]:.3f},{m['auc_ci95'][1]:.3f}] EER {m['eer']:.4f} "
                      f"acc {m['accuracy']:.4f} sens {m['recall_spoof_sensitivity']:.3f} spec {m['specificity_bonafide_recall']:.3f}")
            else:
                print(f"  {sp:17s} single-class: frac predicted spoof {m['frac_pred_spoof']:.4f}")
        save_json(res, rd / "metrics.json")
        save_json(environment_info(), rd / "environment.json")


if __name__ == "__main__":
    main()
