"""Evaluation metrics, validation-only threshold selection, calibration and statistical tests.

Conventions: label 1 = spoof (positive class), score = P(spoof) or logit (higher = more spoof-like).
Sensitivity = spoof recall (true-positive rate); specificity = bona fide recall.

Statistical helpers:
    cluster_bootstrap_ci  percentile CI that resamples *groups* (files, speakers) rather than rows,
                          so repeated chunks of one recording are not treated as independent.
    mcnemar_exact         paired comparison of two classifiers' errors on the same items.
    delong_test           paired comparison of two correlated ROC AUCs (DeLong et al., 1988;
                          fast algorithm of Sun & Xu, 2014).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


# ----------------------------------------------------------------------------- EER / threshold
def eer(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the score threshold at which it occurs (linear interpolation on the ROC)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1.0 - tpr
    d = fnr - fpr
    i = int(np.where(d <= 0)[0][0])  # first point where FNR <= FPR
    if i == 0:
        return float((fpr[0] + fnr[0]) / 2), float(thr[0])
    # interpolate between i-1 (d>0) and i (d<=0)
    w = d[i - 1] / (d[i - 1] - d[i])
    e = fpr[i - 1] + w * (fpr[i] - fpr[i - 1])
    t_hi, t_lo = thr[i - 1], thr[i]
    t = t_lo if not np.isfinite(t_hi) else t_hi + w * (t_lo - t_hi)
    return float(e), float(t)


def select_threshold(y_val: np.ndarray, s_val: np.ndarray, rule: str = "eer") -> float:
    """Pick the operating threshold on VALIDATION data only.

    rule='eer'           threshold at the validation EER point
    rule='balanced_acc'  threshold maximising validation balanced accuracy
    """
    y_val, s_val = np.asarray(y_val), np.asarray(s_val, dtype=float)
    if rule == "eer":
        return eer(y_val, s_val)[1]
    if rule == "balanced_acc":
        cand = np.unique(s_val)
        best_t, best = cand[0], -1.0
        for t in cand:
            p = (s_val >= t).astype(int)
            ba = 0.5 * (np.mean(p[y_val == 1] == 1) + np.mean(p[y_val == 0] == 0))
            if ba > best:
                best, best_t = ba, t
        return float(best_t)
    raise ValueError(rule)


# ----------------------------------------------------------------------------- point metrics
def binary_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    p = (s >= threshold).astype(int)
    tp = int(np.sum((p == 1) & (y == 1)))
    tn = int(np.sum((p == 0) & (y == 0)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    n = len(y)

    def div(a, b):
        return float(a / b) if b else float("nan")

    prec_s, rec_s = div(tp, tp + fp), div(tp, tp + fn)
    prec_b, rec_b = div(tn, tn + fn), div(tn, tn + fp)
    # F1 from counts: 0 (not undefined) when a class is present but never predicted correctly
    f1_s, f1_b = div(2 * tp, 2 * tp + fp + fn), div(2 * tn, 2 * tn + fn + fp)
    both = len(np.unique(y)) == 2
    e, _ = eer(y, s) if both else (float("nan"), float("nan"))
    return {
        "n": n, "n_spoof": int(y.sum()), "n_bonafide": int(n - y.sum()),
        "threshold": float(threshold),
        "accuracy": div(tp + tn, n),
        "balanced_accuracy": 0.5 * (rec_s + rec_b) if both else float("nan"),
        "precision_spoof": prec_s, "recall_spoof_sensitivity": rec_s,
        "specificity_bonafide_recall": rec_b,
        "f1_spoof": f1_s, "f1_macro": float(np.mean([f1_s, f1_b])) if (both and f1_s == f1_s and f1_b == f1_b) else float("nan"),
        "auc": float(roc_auc_score(y, s)) if both else float("nan"),
        "eer": e,
        "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ----------------------------------------------------------------------------- calibration
def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    """Expected calibration error (equal-width bins) and the reliability-diagram table."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total, table = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            conf, acc = float(p[m].mean()), float(y[m].mean())
            total += m.mean() * abs(acc - conf)
            table.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                          "mean_pred": conf, "frac_spoof": acc})
    return float(total), table


# ----------------------------------------------------------------------------- resampling CIs
def cluster_bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    groups: Sequence,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Percentile CI for stat_fn(row_indices), resampling whole groups with replacement.

    stat_fn receives an index array into the original rows (duplicates allowed) and returns a
    scalar; resamples for which it returns NaN (e.g. one class missing) are discarded.
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([members[g] for g in pick])
        v = stat_fn(rows)
        if v == v:
            vals.append(v)
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), vals
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), vals


# ----------------------------------------------------------------------------- paired tests
def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test on paired per-item correctness vectors."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b))  # A right, B wrong
    n10 = int(np.sum(~a & b))  # A wrong, B right
    n = n01 + n10
    p = float(stats.binomtest(min(n01, n10), n, 0.5).pvalue) if n > 0 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": p}


def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_test(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray) -> dict:
    """Two-sided DeLong test for the difference of two correlated AUCs on the same items."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")  # positives first
    y = y[order]
    preds = np.vstack([np.asarray(s_a, float)[order], np.asarray(s_b, float)[order]])
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "z": float("nan"), "p_value": float("nan")}
    k = preds.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float("nan"), "p_value": float("nan")}
    z = diff / np.sqrt(var)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z)))}


def aggregate_by_file(file_ids: Sequence, scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool chunk scores (logits) to one score per original recording."""
    file_ids = np.asarray(file_ids)
    uniq, inv = np.unique(file_ids, return_inverse=True)
    s = np.bincount(inv, weights=np.asarray(scores, float)) / np.bincount(inv)
    lab = np.zeros(len(uniq), dtype=int)
    lab[inv] = np.asarray(labels).astype(int)
    return uniq, s, lab
