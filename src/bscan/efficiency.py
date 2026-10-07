"""Computational-efficiency measurements (Phase 14).

Measured, not estimated:
  * parameters (exact) and serialized FP32 state_dict size
  * multiply-accumulate operations for one 6 s window (thop profiler; documented tool)
  * model-only latency per batch after warm-up (mean, std, median, p95), batch 1 and 16
  * feature-extraction time per 6 s window (CPU, single process)
  * end-to-end real-time factor on real recordings: load -> resample -> segment -> features ->
    model, divided by audio duration (RTF < 1 means faster than real time on that hardware)
  * peak GPU memory (CUDA only)
Latency does not depend on trained weights, so randomly initialised models are timed.

Usage:
    python -m bscan.efficiency --configs bscan_controlled mel_only lcnn_lfcc --device cpu --threads 1
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import load_audio, segment
from .config import load_config
from .features import FEATURE_SHAPES, extract
from .models import build_model, count_parameters, model_features
from .utils import environment_info, pick_device, save_json, stable_seed

ROOT = Path(__file__).resolve().parents[2]


def _sync(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _dummy(feats, batch, device):
    import torch

    return {k: torch.randn(batch, *FEATURE_SHAPES[k], 188, device=device) for k in feats}


def time_model(model, feats, device, batch, warmup, iters) -> dict:
    import torch

    x = _dummy(feats, batch, device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        _sync(device)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            _sync(device)
            ts.append((time.perf_counter() - t0) * 1000)
    ts = np.asarray(ts)
    return {"batch": batch, "iters": iters, "mean_ms": float(ts.mean()), "std_ms": float(ts.std(ddof=1)),
            "median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
            "per_window_ms": float(ts.mean() / batch)}


def macs(model, feats) -> float | None:
    try:
        import copy

        import torch
        from thop import profile

        m = copy.deepcopy(model).cpu().eval()
        x = {k: torch.randn(1, *FEATURE_SHAPES[k], 188) for k in feats}
        total, _ = profile(m, inputs=(x,), verbose=False)
        return float(total)
    except Exception:  # noqa: BLE001
        return None


def e2e_rtf(model, cfg, stats, device, files: list[Path], seed: int = 0) -> dict:
    import torch

    feats = model_features(cfg.model)
    t_load = t_feat = t_model = 0.0
    audio_sec = 0.0
    model.eval()
    for p in files:
        t0 = time.perf_counter()
        x, _ = load_audio(p)
        segs = segment(x, scheme=cfg.data.scheme, window_sec=cfg.data.window_sec, overlap=cfg.data.overlap,
                       seed=stable_seed(str(p)))
        t1 = time.perf_counter()
        batch = {k: [] for k in feats}
        for s in segs:
            f = extract(s.audio, feats)
            for k in feats:
                m = np.asarray(stats[k]["mean"], np.float32)[:, None, None] if stats else 0.0
                sd = (np.asarray(stats[k]["std"], np.float32)[:, None, None] + 1e-8) if stats else 1.0
                batch[k].append((f[k] - m) / sd)
        t2 = time.perf_counter()
        with torch.inference_mode():
            model({k: torch.from_numpy(np.stack(v)).to(device) for k, v in batch.items()})
            _sync(device)
        t3 = time.perf_counter()
        t_load += t1 - t0
        t_feat += t2 - t1
        t_model += t3 - t2
        audio_sec += len(x) / 16000
    total = t_load + t_feat + t_model
    return {"files": len(files), "audio_sec": audio_sec, "load_resample_segment_sec": t_load,
            "feature_sec": t_feat, "model_sec": t_model, "total_sec": total, "rtf": total / audio_sec,
            "rtf_model_only": t_model / audio_sec}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["bscan_controlled", "mel_only", "lfcc_only", "dual_plain",
                                                       "lcnn_lfcc"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = library default)")
    ap.add_argument("--rtf_files", type=int, default=100)
    ap.add_argument("--metadata", default=str(ROOT / "metadata/unified_metadata.csv"))
    ap.add_argument("--audio_root", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results/efficiency"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.metadata, dtype={"utterance_id": str}, low_memory=False)
    pool = meta[meta["split_P1"] == "internal_test"].sort_values("audio_path")
    files = [Path(args.audio_root) / p for p in pool.sample(n=min(args.rtf_files, len(pool)), random_state=0)["audio_path"]]

    rows = []
    for name in args.configs:
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        feats = model_features(cfg.model)
        torch.manual_seed(0)
        model = build_model(cfg.model).to(device)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        rec = {"config": name, "device": str(device), "threads": torch.get_num_threads(),
               "params": count_parameters(model), "state_dict_mb": buf.getbuffer().nbytes / 2**20,
               "macs_per_window": macs(model, feats)}
        warm, iters = (20, 100) if device.type != "cpu" else (5, 30)
        for b in (1, 16):
            t = time_model(model, feats, device, b, warm, iters)
            rec.update({f"b{b}_{k}": v for k, v in t.items() if k != "batch"})
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            time_model(model, feats, device, 16, 2, 5)
            rec["cuda_peak_mem_mb_b16"] = torch.cuda.max_memory_allocated() / 2**20
        # feature extraction alone (single process, CPU), per 6 s window
        y = np.random.default_rng(0).standard_normal(96000).astype(np.float32) * 0.1
        extract(y, feats)
        t0 = time.perf_counter()
        for _ in range(50):
            extract(y, feats)
        rec["feature_ms_per_window_cpu"] = (time.perf_counter() - t0) / 50 * 1000
        rec.update({f"e2e_{k}": v for k, v in e2e_rtf(model, cfg, None, device, files).items()})
        rows.append(rec)
        print(f"{name:18s} params {rec['params']:,}  MACs {rec['macs_per_window'] / 1e6 if rec['macs_per_window'] else float('nan'):8.1f} M  "
              f"b1 {rec['b1_mean_ms']:.2f}±{rec['b1_std_ms']:.2f} ms (p95 {rec['b1_p95_ms']:.2f})  "
              f"b16/window {rec['b16_per_window_ms']:.2f} ms  feat {rec['feature_ms_per_window_cpu']:.2f} ms  "
              f"RTF {rec['e2e_rtf']:.4f}")
    df = pd.DataFrame(rows)
    tag = f"{device.type}_threads{torch.get_num_threads()}"
    df.to_csv(out / f"efficiency_{tag}.csv", index=False)
    save_json({"environment": environment_info(), "device": str(device), "threads": torch.get_num_threads(),
               "rtf_files": len(files), "note": "latency with random weights; e2e RTF on real internal_test recordings"},
              out / f"efficiency_{tag}_meta.json")


if __name__ == "__main__":
    main()
