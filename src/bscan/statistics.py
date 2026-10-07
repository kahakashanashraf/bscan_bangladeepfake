"""Multi-seed aggregation, paired comparisons and the cross-corpus matrix (Phase 13).

Every output row carries `status`: real (all inputs are real runs), placeholder (at least one
input is a synthetic placeholder) or missing.  Paired tests compare two runs on the SAME
recordings of the same split:
    DeLong test            difference of correlated ROC AUCs (recording-level scores)
    exact McNemar test     decisions at each run's own validation-derived threshold
    paired cluster bootstrap  CI of the AUC and EER differences, resampling sentence groups
                              (BanglaFake) or speakers (Mendeley)
Usage:
    python -m bscan.statistics            (writes results/stats/*.csv)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import cluster_ids
from .metrics import cluster_bootstrap_ci, delong_test, eer, mcnemar_exact
from .registry import ROOT, expected_runs, load_registry, resolve

METRICS = ["auc", "eer", "accuracy", "f1_macro", "recall_spoof_sensitivity", "specificity_bonafide_recall", "ece",
           "brier"]
SPLITS = {"P1": ["internal_test", "external_BF-MOZ", "external_MEN"],
          "P2": ["internal_test", "external_BF-SUST", "external_BF-MOZ"]}
# corpus seen at test time, per protocol split (for the cross-corpus matrix)
CORPUS = {("P1", "internal_test"): "BF-SUST", ("P1", "external_BF-MOZ"): "BF-MOZ", ("P1", "external_MEN"): "MEN",
          ("P2", "internal_test"): "MEN", ("P2", "external_BF-SUST"): "BF-SUST", ("P2", "external_BF-MOZ"): "BF-MOZ"}
TRAIN_CORPUS = {"P1": "BF-SUST", "P2": "MEN"}


def run_metrics(name: str) -> tuple[str, dict | None]:
    st, path = resolve(name)
    return st, (json.loads((path / "metrics.json").read_text()) if path else None)


def predictions(name: str, split: str) -> tuple[str, pd.DataFrame | None]:
    st, path = resolve(name)
    if path is None or not (path / f"predictions_{split}.csv").exists():
        return "missing", None
    return st, pd.read_csv(path / f"predictions_{split}.csv", low_memory=False)


def combine_status(sts: list[str]) -> str:
    if not sts or "missing" in sts:
        return "missing"
    return "placeholder" if "placeholder" in sts else "real"


def multiseed_summary(reg: dict | None = None) -> pd.DataFrame:
    reg = reg or load_registry()
    runs = expected_runs(reg)
    rows = []
    for (model, proto), grp in pd.DataFrame([r.__dict__ for r in runs]).groupby(["model", "protocol"]):
        seeds = sorted(grp["seed"].unique())
        for split in SPLITS[proto]:
            vals = {m: [] for m in METRICS}
            sts = []
            for s in seeds:
                st, m = run_metrics(f"{model}__{proto}__seed{s}")
                sts.append(st)
                if m and split in m["splits"]:
                    for k in METRICS:
                        v = m["splits"][split].get(k)
                        if v is not None and v == v:
                            vals[k].append(float(v))
            row = {"model": model, "protocol": proto, "split": split, "corpus": CORPUS[(proto, split)],
                   "n_seeds_expected": len(seeds), "status": combine_status(sts)}
            for k, v in vals.items():
                row[f"{k}_n"] = len(v)
                row[f"{k}_mean"] = float(np.mean(v)) if v else np.nan
                row[f"{k}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
            # CI from the seed-42 run (cluster bootstrap), for single-seed reporting
            st42, m42 = run_metrics(f"{model}__{proto}__seed42")
            if m42 and split in m42["splits"]:
                for k in ("auc", "eer", "accuracy", "f1_macro"):
                    ci = m42["splits"][split].get(f"{k}_ci95")
                    if ci:
                        row[f"{k}_seed42_ci_low"], row[f"{k}_seed42_ci_high"] = ci
            rows.append(row)
    return pd.DataFrame(rows)


def paired(name_a: str, name_b: str, split: str, n_boot: int = 2000, seed: int = 0) -> dict:
    st_a, pa = predictions(name_a, split)
    st_b, pb = predictions(name_b, split)
    out = {"run_a": name_a, "run_b": name_b, "split": split, "status": combine_status([st_a, st_b])}
    if pa is None or pb is None:
        return out
    ma, mb = run_metrics(name_a)[1], run_metrics(name_b)[1]
    j = pa.merge(pb[["audio_path", "score"]], on="audio_path", suffixes=("_a", "_b"))
    y = j["label"].to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        return out
    sa, sb = j["score_a"].to_numpy(), j["score_b"].to_numpy()
    d = delong_test(y, sa, sb)
    ca = (sa >= ma["threshold_logit"]).astype(int) == y
    cb = (sb >= mb["threshold_logit"]).astype(int) == y
    mc = mcnemar_exact(ca, cb)
    groups = cluster_ids(j)
    from sklearn.metrics import roc_auc_score

    def d_auc(i):
        yy = y[i]
        return float(roc_auc_score(yy, sa[i]) - roc_auc_score(yy, sb[i])) if len(np.unique(yy)) == 2 else float("nan")

    def d_eer(i):
        yy = y[i]
        return float(eer(yy, sa[i])[0] - eer(yy, sb[i])[0]) if len(np.unique(yy)) == 2 else float("nan")

    lo, hi, _ = cluster_bootstrap_ci(d_auc, groups, n_boot=n_boot, seed=seed)
    elo, ehi, _ = cluster_bootstrap_ci(d_eer, groups, n_boot=n_boot, seed=seed)
    out.update({"n": int(len(j)), "auc_a": d["auc_a"], "auc_b": d["auc_b"], "delta_auc": d["auc_a"] - d["auc_b"],
                "delta_auc_ci_low": lo, "delta_auc_ci_high": hi, "delong_z": d["z"], "delong_p": d["p_value"],
                "eer_a": eer(y, sa)[0], "eer_b": eer(y, sb)[0], "delta_eer_ci_low": elo, "delta_eer_ci_high": ehi,
                "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
                "mcnemar_a_right_b_wrong": mc["a_right_b_wrong"], "mcnemar_a_wrong_b_right": mc["a_wrong_b_right"],
                "mcnemar_p": mc["p_value"]})
    return out


def paired_comparisons(reg: dict | None = None, n_boot: int = 2000) -> pd.DataFrame:
    reg = reg or load_registry()
    registered = {r.name for r in expected_runs(reg)}
    rows = []
    for c in reg["comparisons"]:
        for proto in ("P1", "P2"):
            a, b = f"{c['a']}__{proto}__seed42", f"{c['b']}__{proto}__seed42"
            if a not in registered or b not in registered:
                continue  # the pair is not part of the pre-specified design for this protocol
            for split in SPLITS[proto]:
                r = paired(a, b, split, n_boot=n_boot)
                r["protocol"] = proto
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df) and "delong_p" in df:
        # Holm correction within each protocol x split family of comparisons
        df["delong_p_holm"] = np.nan
        for _, idx in df.groupby(["protocol", "split"]).groups.items():
            p = df.loc[idx, "delong_p"].to_numpy(dtype=float)
            ok = ~np.isnan(p)
            if ok.any():
                order = np.argsort(p[ok])
                adj = np.empty(ok.sum())
                m = ok.sum()
                running = 0.0
                for rank, k in enumerate(order):
                    running = max(running, min(1.0, (m - rank) * p[ok][k]))
                    adj[k] = running
                vals = np.full(len(p), np.nan)
                vals[ok] = adj
                df.loc[idx, "delong_p_holm"] = vals
    return df


def cross_corpus_matrix(summary: pd.DataFrame, models: list[str], metric: str = "auc") -> pd.DataFrame:
    rows = []
    for model in models:
        for proto in ("P1", "P2"):
            sub = summary[(summary.model == model) & (summary.protocol == proto)]
            if sub.empty:
                continue
            row = {"model": model, "train_corpus": TRAIN_CORPUS[proto], "status": combine_status(list(sub.status))}
            for _, r in sub.iterrows():
                row[f"test_{r['corpus']}"] = r[f"{metric}_mean"]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    out = ROOT / "results" / "stats"
    out.mkdir(parents=True, exist_ok=True)
    s = multiseed_summary()
    s.to_csv(out / "multiseed_summary.csv", index=False)
    p = paired_comparisons()
    p.to_csv(out / "paired_comparisons.csv", index=False)
    models = ["bscan_controlled", "mel_only", "lfcc_only", "lcnn_lfcc", "lfcc_gmm_controlled", "stats_hgb_controlled"]
    for metric in ("auc", "eer"):
        cross_corpus_matrix(s, models, metric).to_csv(out / f"cross_corpus_{metric}.csv", index=False)
    print(f"wrote {out}: {len(s)} summary rows, {len(p)} paired comparisons; statuses: "
          f"{s['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
