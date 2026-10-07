"""Smoke test on a tiny stratified subset (Phase 5).  Must pass before any expensive training.

Checks: audio loading and resampling; the three segmentation schemes; feature dimensions;
batch collation; forward / loss / backward / optimizer step for every model family; NaN/Inf
detection; 2-epoch training through the real training entry point; checkpoint save/load
(bit-identical logits after reload); validation-only threshold selection and metrics through the
real evaluation entry point; and CPU determinism (two runs with the same seed must match).

Usage:
    python -m tests.smoke_test [--device auto|cpu|mps|cuda] [--files 16] [--out results/smoke]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bscan.audio import load_audio, segment  # noqa: E402
from bscan.config import load_config, parse_overrides  # noqa: E402
from bscan.data import WindowDataset, build_feature_cache, load_split, norm_stats  # noqa: E402
from bscan.evaluate import run_evaluation  # noqa: E402
from bscan.features import FEATURE_SHAPES, extract  # noqa: E402
from bscan.models import build_model, count_parameters  # noqa: E402
from bscan.train import make_loader, predict_logits, run_training  # noqa: E402
from bscan.utils import environment_info, pick_device, save_json, seed_everything  # noqa: E402

RESULTS: list[dict] = []


def check(name: str):
    def deco(fn):
        def run(*a, **k):
            t0 = time.perf_counter()
            try:
                detail = fn(*a, **k)
                RESULTS.append({"check": name, "status": "PASS", "sec": round(time.perf_counter() - t0, 2),
                                "detail": detail})
                print(f"[PASS] {name} ({time.perf_counter() - t0:.1f}s) {detail if detail else ''}")
                return detail
            except Exception as exc:  # noqa: BLE001
                RESULTS.append({"check": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                                "trace": traceback.format_exc()})
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
                return None
        return run
    return deco


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--files", type=int, default=16, help="files per split (half bona fide, half spoof)")
    ap.add_argument("--out", default=str(ROOT / "results" / "smoke"))
    ap.add_argument("--metadata", default=str(ROOT / "metadata" / "unified_metadata.csv"))
    ap.add_argument("--audio_root", default=str(ROOT / "data"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    device = pick_device(args.device)
    common = [f"data.metadata_csv={args.metadata}", f"data.audio_root={args.audio_root}",
              f"data.cache_dir={out / 'cache'}", f"data.max_files_per_split={args.files}", "data.num_workers=0",
              "train.epochs=2", "train.batch_size=4", f"train.device={device.type}", f"out_dir={out / 'runs'}",
              "eval.n_boot=50"]

    def cfg_for(name: str, extra: list[str] | None = None):
        return load_config(ROOT / "configs" / f"{name}.yaml", parse_overrides(common + (extra or [])))

    # 1. audio loading ------------------------------------------------------------------------
    @check("audio loading + resampling (one file per corpus)")
    def t_audio():
        c = cfg_for("bscan_controlled")
        info = {}
        for sp in ("train", "external_BF-MOZ", "external_MEN"):
            f = load_split(c.data, sp).iloc[0]
            x, sr = load_audio(Path(args.audio_root) / f["audio_path"])
            assert np.isfinite(x).all() and x.dtype == np.float32 and len(x) > 0
            exp = round(f["duration_sec"] * 16000)
            assert abs(len(x) - exp) <= 2, (len(x), exp)
            info[sp] = {"native_sr": sr, "samples_16k": len(x)}
        return info
    t_audio()

    # 2. segmentation -------------------------------------------------------------------------
    @check("segmentation schemes (orig6s / trim_repeat / controlled)")
    def t_seg():
        c = cfg_for("bscan_controlled")
        f = load_split(c.data, "train").iloc[0]
        x, _ = load_audio(Path(args.audio_root) / f["audio_path"])
        res = {}
        for sch in ("orig6s", "trim_repeat", "controlled"):
            segs = segment(x, sch, seed=1)
            assert segs and all(len(s.audio) == 96000 and np.isfinite(s.audio).all() for s in segs)
            assert all(0.0 <= s.pad_fraction <= 1.0 for s in segs)
            again = segment(x, sch, seed=1)
            assert all(np.array_equal(a.audio, b.audio) for a, b in zip(segs, again)), "non-deterministic"
            res[sch] = len(segs)
        return res
    t_seg()

    # 3. features -----------------------------------------------------------------------------
    @check("feature dimensions and finiteness (mel 3x128x188, lfcc/mfcc 3x40x188)")
    def t_feat():
        y = np.random.default_rng(0).standard_normal(96000).astype(np.float32) * 0.1
        F = extract(y, ("mel", "lfcc", "mfcc"))
        for k, v in F.items():
            assert v.shape == FEATURE_SHAPES[k] + (188,), (k, v.shape)
            assert np.isfinite(v).all(), k
        return {k: list(v.shape) for k, v in F.items()}
    t_feat()

    # 4. cache + collation --------------------------------------------------------------------
    state = {}

    @check("feature cache + normalisation (train only) + batch collation")
    def t_cache():
        c = cfg_for("bscan_controlled", ["model.branches=[mel, lfcc, mfcc]"])
        tr = build_feature_cache(c.data, "train", ("mel", "lfcc", "mfcc"), workers=2)
        st = norm_stats(tr, ("mel", "lfcc", "mfcc"))
        ds = WindowDataset(tr, ("mel", "lfcc", "mfcc"), st)
        # the cached arrays are over-allocated; the statistics must use only the windows actually written
        assert all(s["n_windows"] == min(5000, len(ds)) for s in st.values()), (st["mel"]["n_windows"], len(ds))
        dl = make_loader(ds, 4, True, 0, 0, device)
        feats, y, _ = next(iter(dl))
        shapes = {k: list(v.shape) for k, v in feats.items()}
        assert shapes["mel"] == [4, 3, 128, 188] and shapes["lfcc"] == [4, 3, 40, 188]
        for k, v in feats.items():
            assert bool(v.isfinite().all()), k
        state.update(ds=ds, stats=st)
        return {"windows": len(ds), "batch": shapes, "labels": y.tolist()}
    t_cache()

    # 5. every model family: forward / loss / backward / step ------------------------------------
    @check("forward/loss/backward/step for all model variants (NaN/Inf guarded)")
    def t_models():
        import torch

        ds = state["ds"]
        dl = make_loader(ds, 4, True, 0, 0, device)
        feats, y, _ = next(iter(dl))
        feats = {k: v.to(device) for k, v in feats.items()}
        y = y.to(device)
        out = {}
        for name in ("bscan_controlled", "mel_only", "lfcc_only", "mfcc_only", "mel_mfcc", "dual_no_se",
                     "dual_no_tattn", "dual_plain", "lcnn_lfcc"):
            c = cfg_for(name)
            seed_everything(0)
            m = build_model(c.model).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            logits = m(feats)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            grads_ok = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
            opt.step()
            assert logits.shape == (4,) and bool(torch.isfinite(logits).all()) and grads_ok and torch.isfinite(loss)
            out[name] = {"params": count_parameters(m), "loss": round(loss.item(), 4)}
        assert out["bscan_controlled"]["params"] == 162819, "BSCAN parameter count changed"
        return out
    t_models()

    # 6. two-epoch training through the real entry point + reload ------------------------------
    @check(f"2-epoch training + checkpoint reload on {device.type}")
    def t_train():
        import torch

        c = cfg_for("bscan_controlled")
        rd = run_training(c, log=lambda s: None)
        log = json.loads((rd / "train_summary.json").read_text())
        ck = torch.load(rd / "best.pt", map_location=device, weights_only=False)
        m = build_model(c.model).to(device)
        m.load_state_dict(ck["model_state_dict"])
        va = WindowDataset(build_feature_cache(c.data, "validation", ("mel", "lfcc")), ("mel", "lfcc"),
                           json.loads((rd / "norm_stats.json").read_text()))
        l1 = predict_logits(m, make_loader(va, 8, False, 0, 0, device), device, False)
        l2 = predict_logits(m, make_loader(va, 8, False, 0, 0, device), device, False)
        assert np.array_equal(l1, l2), "inference not repeatable"
        assert np.isfinite(l1).all()
        state["run_dir"] = rd
        return {"epochs_run": log["epochs_run"], "best_epoch": log["best_epoch"], "params": log["params"],
                "train_windows": log["train_windows"]}
    t_train()

    # 7. evaluation entry point: validation-only threshold, metrics on all splits ----------------
    @check("evaluation: validation-derived threshold applied to all splits; metrics finite")
    def t_eval():
        c = cfg_for("bscan_controlled")
        res = run_evaluation(c, log=lambda s: None)
        sp = res["splits"]
        assert "validation" in sp and "internal_test" in sp and "external_MEN" in sp
        for name, m in sp.items():
            if "auc" in m and m["n_bonafide"] and m["n_spoof"]:
                assert 0.0 <= m["auc"] <= 1.0 and 0.0 <= m["eer"] <= 1.0
        assert np.isfinite(res["threshold_logit"])
        return {k: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in m.items()
                    if kk in ("n", "auc", "eer", "accuracy", "frac_pred_spoof")} for k, m in sp.items()}
    t_eval()

    # 8. determinism on CPU --------------------------------------------------------------------
    @check("CPU determinism: identical training losses for identical seeds")
    def t_det():
        import pandas as pd

        logs = []
        for rep in (1, 2):
            c = load_config(ROOT / "configs" / "bscan_controlled.yaml",
                            parse_overrides(common + ["train.device=cpu", f"out_dir={out / f'det{rep}'}"]))
            rd = run_training(c, log=lambda s: None)
            logs.append(pd.read_csv(rd / "train_log.csv")[["train_loss", "val_loss"]].to_numpy())
        assert np.array_equal(logs[0], logs[1]), f"runs differ: {logs}"
        return {"train_loss_epoch1": float(logs[0][0, 0])}
    t_det()

    report = {"device": str(device), "environment": environment_info(), "files_per_split": args.files,
              "checks": RESULTS, "all_passed": all(r["status"] == "PASS" for r in RESULTS)}
    save_json(report, out / "smoke_report.json")
    print(f"\nSMOKE TEST {'PASSED' if report['all_passed'] else 'FAILED'} "
          f"({sum(r['status'] == 'PASS' for r in RESULTS)}/{len(RESULTS)} checks) -> {out / 'smoke_report.json'}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
