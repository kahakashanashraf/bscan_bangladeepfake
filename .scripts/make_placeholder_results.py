"""Create SYNTHETIC PLACEHOLDER results for every registered run that has no real result yet.

Purpose: let the complete pipeline (statistics, tables, figures, reports, manuscript) be built and
checked end to end before the Colab runs finish.  Placeholders are NOT results:
  * they live only in results/runs_PLACEHOLDER/ (never in results/runs/)
  * every metrics.json has "PLACEHOLDER": true; every CSV has a PLACEHOLDER column
  * scores are random draws (seeded) on the real file lists of each split
  * every table cell / macro / figure built from them is visibly marked as a placeholder,
    and .scripts/final_qc.py refuses to pass while any placeholder is in use.
A real run in results/runs/ always takes precedence over a placeholder of the same name.

Usage:
    python .scripts/make_placeholder_results.py          (creates only what is missing)
    python .scripts/make_placeholder_results.py --clean  (deletes all placeholders)
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bscan.config import load_config, parse_overrides  # noqa: E402
from bscan.evaluate import file_table, split_metrics  # noqa: E402
from bscan.metrics import select_threshold  # noqa: E402
from bscan.registry import PLACEHOLDER_DIR, expected_runs, load_registry, resolve  # noqa: E402
from bscan.robustness import PROBES, ROBUSTNESS  # noqa: E402
from bscan.utils import save_json  # noqa: E402

SPLITS = {"P1": ["validation", "internal_test", "external_BF-MOZ", "external_MEN", "external_BF-NEWS"],
          "P2": ["validation", "internal_test", "external_BF-SUST", "external_BF-MOZ", "external_BF-NEWS"]}
BANNER = "SYNTHETIC PLACEHOLDER - NOT A RESULT"


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16))


def config_name(model: str) -> str:
    return model if (ROOT / "configs" / f"{model}.yaml").exists() else "bscan_controlled"


def make_run(model: str, proto: str, seed: int, meta: pd.DataFrame) -> Path:
    name = f"{model}__{proto}__seed{seed}"
    rd = PLACEHOLDER_DIR / name
    rd.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / f"{config_name(model)}.yaml",
                      parse_overrides([f"data.protocol={proto}", f"train.seed={seed}"]))
    save_json({**cfg.to_dict(), "PLACEHOLDER": True, "note": BANNER}, rd / "config.json")
    col = f"split_{proto}"
    tables = {}
    for sp in SPLITS[proto]:
        f = meta[meta[col] == sp]
        if f.empty:
            continue
        r = _rng(name, sp)
        sep = r.uniform(0.3, 2.5)  # arbitrary separation: placeholder only
        score = r.standard_normal(len(f)) + sep * f["label"].to_numpy()
        idx = f[["audio_path", "label", "dataset_id", "speaker_id", "text_group"]].assign(pad_fraction=0.0)
        tables[sp] = file_table(idx, score)
        tables[sp]["PLACEHOLDER"] = True
        tables[sp].to_csv(rd / f"predictions_{sp}.csv", index=False)
    thr = select_threshold(tables["validation"]["label"].to_numpy(), tables["validation"]["score"].to_numpy(), "eer")
    res = {"PLACEHOLDER": True, "note": BANNER, "threshold_rule": "eer", "threshold_logit": thr, "best_epoch": None,
           "splits": {}}
    for sp, f in tables.items():
        m = split_metrics(f, thr, n_boot=200, seed=seed)
        m.pop("_reliability", None)
        res["splits"][sp] = m
    save_json(res, rd / "metrics.json")
    r = _rng(name, "log")
    ep = np.arange(1, 16)
    pd.DataFrame({"epoch": ep, "train_loss": 0.7 * np.exp(-ep / 4) + r.uniform(0, .02, 15),
                  "val_loss": 0.7 * np.exp(-ep / 5) + r.uniform(0, .05, 15), "PLACEHOLDER": True}).to_csv(
        rd / "train_log.csv", index=False)
    return rd


def make_analysis(kind: str, run_name: str, splits: list[str]) -> None:
    st, path = resolve(run_name)
    if st == "real" and (path / f"{kind}_summary.csv").exists():
        return
    rd = PLACEHOLDER_DIR / run_name
    rd.mkdir(parents=True, exist_ok=True)
    conds = ROBUSTNESS if kind == "robustness" else PROBES
    rows = []
    for sp in splits:
        base = _rng(run_name, sp, kind).uniform(0.6, 0.99)
        for c in conds:
            r = _rng(run_name, sp, kind, c)
            auc = base if c == "clean" else max(0.4, base - r.uniform(0.0, 0.25))
            rows.append({"split": sp, "condition": c, "n": 600, "auc": auc, "eer": 1 - auc, "accuracy": auc - 0.05,
                         "sensitivity": auc - r.uniform(0, .1), "specificity": auc - r.uniform(0, .1),
                         "mean_score_shift_bonafide": 0.0 if c == "clean" else r.normal(0, 1),
                         "mean_score_shift_spoof": 0.0 if c == "clean" else r.normal(0, 1),
                         "decision_flip_rate": 0.0 if c == "clean" else r.uniform(0, .4),
                         "share_input_changed": 0.0 if c == "clean" else r.uniform(0.5, 1.0),
                         "decision_flip_rate_changed": float("nan") if c == "clean" else r.uniform(0, .5),
                         "share_clipped_bonafide": r.uniform(0, 1) if c.startswith("clip_") else float("nan"),
                         "share_clipped_spoof": r.uniform(0, 1) if c.startswith("clip_") else float("nan"),
                         "PLACEHOLDER": True})
    pd.DataFrame(rows).to_csv(rd / f"{kind}_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    if args.clean:
        shutil.rmtree(PLACEHOLDER_DIR, ignore_errors=True)
        print("removed", PLACEHOLDER_DIR)
        return
    reg = load_registry()
    meta = pd.read_csv(ROOT / "metadata" / "unified_metadata.csv", dtype={"utterance_id": str}, low_memory=False)
    made = []
    for run in expected_runs(reg, include_optional=True):
        if resolve(run.name)[0] == "missing":
            make_run(run.model, run.protocol, run.seed, meta)
            made.append(run.name)
    for kind in ("robustness", "probes"):
        a = reg["analyses"][kind]
        for rn in a["runs"]:
            make_analysis(kind, rn, a["splits"])
    eff = ROOT / "results" / "efficiency" / "efficiency_cuda_PLACEHOLDER.csv"
    if not any((ROOT / "results" / "efficiency").glob("efficiency_cuda_threads*.csv")):
        pd.DataFrame([{"config": c, "device": "cuda", "b1_mean_ms": np.nan, "PLACEHOLDER": True}
                      for c in ("bscan_controlled", "mel_only", "lfcc_only", "dual_plain", "lcnn_lfcc")]).to_csv(eff, index=False)
    (PLACEHOLDER_DIR / "README.txt").write_text(BANNER + "\nCreated by .scripts/make_placeholder_results.py; "
                                                "replace by running the Colab notebooks.\n")
    print(f"placeholder runs created: {len(made)}")


if __name__ == "__main__":
    main()
