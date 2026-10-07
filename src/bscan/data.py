"""Split manifests, feature caching and the PyTorch dataset.

Each split is expanded into windows by the configured segmentation scheme.  Features for every
window are computed once (CPU, librosa) and stored as float16 memory-mapped arrays:

    {cache_dir}/{cache_key}/{split}/index.csv          one row per window
    {cache_dir}/{cache_key}/{split}/{feature}.npy      [n_windows, 3, F, T] float16

cache_key hashes the preprocessing settings, the feature definitions version and the exact file
list, so a cache can never be reused for a different configuration.  Normalisation statistics
(per feature and per channel) are computed from the TRAIN split only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import load_audio, segment
from .config import DataCfg
from .features import FEATURE_SHAPES, extract
from .utils import stable_seed

FEATURE_VERSION = "features-v1"  # bump if features.py changes


# ----------------------------------------------------------------------------- manifests
def load_split(cfg: DataCfg, split: str) -> pd.DataFrame:
    """Rows of the unified metadata belonging to one split of the configured protocol."""
    meta = pd.read_csv(cfg.metadata_csv, dtype={"utterance_id": str}, low_memory=False)
    col = f"split_{cfg.protocol}"
    df = meta[meta[col] == split].sort_values("audio_path").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"split {split!r} is empty for protocol {cfg.protocol}")
    if cfg.max_files_per_split:
        # stratified, seeded subset (smoke tests)
        parts = [g.sample(n=min(len(g), cfg.max_files_per_split // 2 or 1), random_state=0)
                 for _, g in df.groupby("label")]
        df = pd.concat(parts).sort_values("audio_path").reset_index(drop=True)
    return df


def cache_key(cfg: DataCfg, features: tuple[str, ...], files: list[str]) -> str:
    h = hashlib.sha1()
    h.update(json.dumps({"scheme": cfg.scheme, "window": cfg.window_sec, "overlap": cfg.overlap,
                         "features": sorted(features), "version": FEATURE_VERSION}, sort_keys=True).encode())
    for f in files:
        h.update(f.encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- extraction
def _windows_for_file(args):
    rel, root, scheme, window_sec, overlap, features = args
    x, orig_sr = load_audio(Path(root) / rel)
    segs = segment(x, scheme=scheme, window_sec=window_sec, overlap=overlap, seed=stable_seed(rel))
    out = []
    for s in segs:
        feats = extract(s.audio, features)
        out.append(({k: v.astype(np.float16) for k, v in feats.items()},
                    {"audio_path": rel, "window": s.index, "start_sec": s.start_sec, "pad_fraction": s.pad_fraction}))
    return out


def build_feature_cache(cfg: DataCfg, split: str, features: tuple[str, ...], workers: int | None = None,
                        log=print) -> Path:
    files_df = load_split(cfg, split)
    files = files_df["audio_path"].tolist()
    key = cache_key(cfg, features, files)
    d = Path(cfg.cache_dir) / key / split
    done = d / "DONE"
    if done.exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, os.cpu_count() or 1)
    args = [(f, cfg.audio_root, cfg.scheme, cfg.window_sec, cfg.overlap, tuple(features)) for f in files]

    # upper bound on the number of windows (from durations) so features stream straight to disk
    hop_sec = cfg.window_sec * (1.0 - cfg.overlap)
    dur = files_df["duration_sec"].to_numpy()
    ub = int(np.sum(2 + np.ceil(np.maximum(0.0, dur - cfg.window_sec) / hop_sec)))
    n_frames = 1 + int(round(cfg.window_sec * 16000)) // 512
    arrs = {k: np.lib.format.open_memmap(d / f"{k}.npy", mode="w+", dtype=np.float16,
                                         shape=(ub,) + FEATURE_SHAPES[k] + (n_frames,)) for k in features}
    rows, j = [], 0
    # fork is fast on Linux (Colab); on macOS fork after torch/Objective-C initialisation is unsafe
    ctx = get_context("fork") if sys.platform.startswith("linux") else get_context("spawn")
    with ctx.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_windows_for_file, args, chunksize=16)):
            for feats, meta in res:
                if j >= ub:
                    raise RuntimeError("window upper bound exceeded; check durations in metadata")
                for k in features:
                    arrs[k][j] = feats[k]
                meta["row"] = j
                rows.append(meta)
                j += 1
            if (i + 1) % 2000 == 0:
                log(f"  [{split}] features for {i + 1}/{len(files)} files")
    for a in arrs.values():
        a.flush()
    del arrs
    index = pd.DataFrame(rows).merge(
        files_df[["audio_path", "label", "dataset_id", "speaker_id", "set_id", "text_group"]], on="audio_path", how="left")
    index.to_csv(d / "index.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"split": split, "files": len(files), "windows": len(index),
                                             "features": list(features), "scheme": cfg.scheme, "key": key}, indent=2))
    done.write_text("ok")
    return d


def norm_stats(train_dir: Path, features: tuple[str, ...], max_windows: int = 5000, seed: int = 0) -> dict:
    """Per-feature, per-channel mean/std from (a seeded subset of) the training windows.

    The cached arrays are allocated with an upper bound on the window count, so only the first
    meta.json["windows"] rows hold features; the unused zero rows after them are never sampled.
    """
    n = int(json.loads((Path(train_dir) / "meta.json").read_text())["windows"])
    stats = {}
    for k in features:
        arr = np.load(train_dir / f"{k}.npy", mmap_mode="r")
        if n > len(arr):
            raise RuntimeError(f"{train_dir}: meta.json lists {n} windows but {k}.npy holds {len(arr)}")
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=min(max_windows, n), replace=False))
        x = np.asarray(arr[idx], dtype=np.float64)  # [n, 3, F, T]
        mean = x.mean(axis=(0, 2, 3))
        std = x.std(axis=(0, 2, 3))
        stats[k] = {"mean": mean.tolist(), "std": std.tolist(), "n_windows": int(len(idx))}
    return stats


# ----------------------------------------------------------------------------- torch dataset
class WindowDataset:
    """Returns ({feature: tensor[3,F,T]}, label, row) for each cached window."""

    def __init__(self, split_dir: Path, features: tuple[str, ...], stats: dict):
        import torch  # noqa: F401  (import here so the module works without torch for caching)

        self.dir = Path(split_dir)
        self.features = tuple(features)
        self.index = pd.read_csv(self.dir / "index.csv", dtype={"utterance_id": str}, low_memory=False)
        self.labels = self.index["label"].to_numpy().astype(np.float32)
        self._arr = {}
        self.mean = {k: np.asarray(stats[k]["mean"], np.float32)[:, None, None] for k in self.features}
        self.std = {k: np.asarray(stats[k]["std"], np.float32)[:, None, None] + 1e-8 for k in self.features}

    def __len__(self) -> int:
        return len(self.index)

    def _array(self, k):
        if k not in self._arr:  # opened lazily so each DataLoader worker has its own handle
            self._arr[k] = np.load(self.dir / f"{k}.npy", mmap_mode="r")
        return self._arr[k]

    def __getitem__(self, i):
        import torch

        feats = {k: torch.from_numpy(((self._array(k)[i].astype(np.float32) - self.mean[k]) / self.std[k]))
                 for k in self.features}
        return feats, torch.tensor(self.labels[i]), i
