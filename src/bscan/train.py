"""Train one model on the training split of a protocol; select the checkpoint on validation loss.

Usage:
    python -m bscan.train --config configs/bscan_controlled.yaml [train.seed=123 data.scheme=orig6s ...]

Writes to cfg.run_dir():
    config.json, environment.json, norm_stats.json, train_log.csv, best.pt, train_summary.json

Only the training split influences the weights and the normalisation statistics; only the
validation split influences checkpoint selection, learning-rate scheduling and early stopping.
No test split is read here.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .config import ExperimentCfg, load_config, parse_overrides
from .data import WindowDataset, build_feature_cache, norm_stats
from .metrics import aggregate_by_file, eer
from .models import build_model, count_parameters, model_features
from .utils import environment_info, pick_device, save_json, seed_everything


def make_loader(ds, batch_size, shuffle, num_workers, seed, device):
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)

    def _init(worker_id):
        np.random.seed(seed + worker_id)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=(device.type == "cuda"), generator=g, worker_init_fn=_init,
                      persistent_workers=num_workers > 0)


def predict_logits(model, loader, device, amp: bool) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.inference_mode():
        for feats, _, _ in loader:
            feats = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
            with torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(feats)
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def run_training(cfg: ExperimentCfg, log=print) -> "Path":
    import torch
    import torch.nn as nn

    seed_everything(cfg.train.seed, cfg.train.deterministic)
    device = pick_device(cfg.train.device)
    amp = cfg.train.amp and device.type == "cuda"
    rd = cfg.run_dir()
    rd.mkdir(parents=True, exist_ok=True)
    save_json(cfg.to_dict(), rd / "config.json")
    save_json({**environment_info(), "device": str(device), "amp": amp}, rd / "environment.json")

    feats = model_features(cfg.model)
    t0 = time.perf_counter()
    tr_dir = build_feature_cache(cfg.data, "train", feats, log=log)
    va_dir = build_feature_cache(cfg.data, "validation", feats, log=log)
    t_cache = time.perf_counter() - t0
    stats = norm_stats(tr_dir, feats, seed=cfg.train.seed)
    save_json(stats, rd / "norm_stats.json")

    tr_ds, va_ds = WindowDataset(tr_dir, feats, stats), WindowDataset(va_dir, feats, stats)
    tr_dl = make_loader(tr_ds, cfg.train.batch_size, True, cfg.data.num_workers, cfg.train.seed, device)
    va_dl = make_loader(va_ds, 64, False, cfg.data.num_workers, cfg.train.seed, device)

    model = build_model(cfg.model).to(device)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.train.plateau_factor,
                                                       patience=cfg.train.plateau_patience)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log(f"{cfg.name} | {cfg.data.protocol} | seed {cfg.train.seed} | device {device} | params {n_params:,} | "
        f"train windows {len(tr_ds):,} | val windows {len(va_ds):,}")

    best, best_epoch, bad, history = float("inf"), -1, 0, []
    va_idx = va_ds.index
    for epoch in range(1, cfg.train.epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        tot, n, nonfinite = 0.0, 0, 0
        for feats_b, y, _ in tr_dl:
            feats_b = {k: v.to(device, non_blocking=True) for k, v in feats_b.items()}
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = crit(model(feats_b), y)
            if not torch.isfinite(loss):
                nonfinite += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(y)
            n += len(y)
        tr_loss = tot / max(n, 1)

        logits = predict_logits(model, va_dl, device, amp)
        yv = va_ds.labels
        va_loss = float(crit(torch.from_numpy(logits), torch.from_numpy(yv)))
        _, fs, fy = aggregate_by_file(va_idx["audio_path"].to_numpy(), logits, yv)
        va_eer_file = eer(fy, fs)[0]
        va_eer_win = eer(yv, logits)[0]
        sched.step(va_loss)
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_eer_file": va_eer_file,
               "val_eer_window": va_eer_win, "lr": opt.param_groups[0]["lr"], "nonfinite_batches": nonfinite,
               "epoch_sec": time.perf_counter() - t_ep}
        history.append(rec)
        log(f"  epoch {epoch:2d} train_loss {tr_loss:.4f} val_loss {va_loss:.4f} "
            f"val_EER(file) {va_eer_file:.4f} lr {rec['lr']:.1e} ({rec['epoch_sec']:.0f}s)")
        if va_loss < best - 1e-6:
            best, best_epoch, bad = va_loss, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": va_loss,
                        "config": cfg.to_dict()}, rd / "best.pt")
        else:
            bad += 1
            if bad >= cfg.train.patience:
                log(f"  early stopping at epoch {epoch} (best epoch {best_epoch})")
                break
    pd.DataFrame(history).to_csv(rd / "train_log.csv", index=False)
    save_json({"best_epoch": best_epoch, "best_val_loss": best, "epochs_run": len(history), "params": n_params,
               "train_windows": len(tr_ds), "val_windows": len(va_ds), "feature_cache_sec": t_cache,
               "train_sec": float(sum(h["epoch_sec"] for h in history))}, rd / "train_summary.json")
    return rd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.seed=123")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.overrides))
    run_training(cfg)


if __name__ == "__main__":
    main()
