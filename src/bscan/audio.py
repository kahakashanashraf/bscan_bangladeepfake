"""Audio loading, resampling and segmentation.

Two segmentation schemes are provided:

"orig6s"       Reproduces the original notebook (code/banglafkpart.ipynb, cell 14): resample to
               16 kHz, peak-normalise, 6 s windows with 10 % overlap (hop 5.4 s); recordings of <= 6 s are zero-padded at
               the end; the tail that does not fill a whole window is discarded; each window is
               peak-normalised again.  Zero padding makes the recording duration directly visible
               to the model (a potential label shortcut when durations differ by class).

"trim_repeat"  Duration-controlled variant: leading/trailing silence is trimmed (energy threshold
               relative to the recording peak), recordings shorter than the window are filled by
               repeating the audio instead of zeros, and the final window is aligned to the end of
               the recording so no speech is discarded.  Each window is peak-normalised.

Segments carry the fraction of the window that is padding so that the leakage audit can relate
model scores to padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


@dataclass
class Segment:
    audio: np.ndarray       # float32 [window_samples]
    index: int              # window index within the recording
    start_sec: float        # start time in the (trimmed) recording
    pad_fraction: float     # fraction of the window that is padding (zeros or repeated audio)


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Read any soundfile-supported file, mix to mono, resample (polyphase) to target_sr."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, int(sr) // g).astype(np.float32)
    return x, int(sr)


def peak_normalize(x: np.ndarray) -> np.ndarray:
    p = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x / p).astype(np.float32) if p > 0 else x.astype(np.float32)


def trim_silence(x: np.ndarray, sr: int = TARGET_SR, top_db: float = 40.0) -> np.ndarray:
    """Remove leading/trailing frames quieter than (peak - top_db) dB (25 ms frames, 10 ms hop)."""
    win, hop = int(0.025 * sr), int(0.010 * sr)
    if len(x) <= win:
        return x
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    e = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    active = np.where(e > e.max() - top_db)[0]
    if len(active) == 0:
        return x
    return x[active[0] * hop: min(len(x), active[-1] * hop + win)]


def condition(x: np.ndarray, dc_remove: bool = False, dither_dbfs: float | None = None,
              seed: int = 0) -> np.ndarray:
    """Optional signal conditioning applied to each peak-normalised window.

    dc_remove    subtract the window mean (removes the DC offset left by some vocoders)
    dither_dbfs  add white Gaussian noise with this RMS level (dB re full scale); masks the
                 difference between digital silence and recorded room noise in pauses.
                 The noise is seeded per window so preprocessing is deterministic.
    """
    y = x.astype(np.float32, copy=True)
    if dc_remove:
        y = y - np.float32(y.mean())
    if dither_dbfs is not None:
        rng = np.random.default_rng(seed)
        y = y + (10 ** (dither_dbfs / 20.0)) * rng.standard_normal(len(y)).astype(np.float32)
    return y


def segment(x: np.ndarray, scheme: str = "orig6s", window_sec: float = 6.0, overlap: float = 0.10,
            sr: int = TARGET_SR, dc_remove: bool = False, dither_dbfs: float | None = None,
            seed: int = 0) -> list[Segment]:
    """Split one recording into windows; see module docstring for the schemes.

    "controlled" = "trim_repeat" followed by DC removal and -60 dBFS seeded dither
    (the duration-, DC- and noise-floor-controlled preprocessing evaluated in Phase 3).
    """
    if scheme == "controlled":
        segs = segment(x, "trim_repeat", window_sec, overlap, sr)
        return [Segment(condition(s.audio, True, -60.0 if dither_dbfs is None else dither_dbfs, seed + s.index),
                        s.index, s.start_sec, s.pad_fraction) for s in segs]
    if dc_remove or dither_dbfs is not None:
        return [Segment(condition(s.audio, dc_remove, dither_dbfs, seed + s.index), s.index, s.start_sec,
                        s.pad_fraction) for s in segment(x, scheme, window_sec, overlap, sr)]
    win = int(round(window_sec * sr))
    hop = int(win * (1.0 - overlap))
    if scheme == "orig6s":
        x = peak_normalize(x)
        if len(x) <= win:
            buf = np.zeros(win, dtype=np.float32)
            buf[: len(x)] = x
            return [Segment(peak_normalize(buf), 0, 0.0, 1.0 - len(x) / win)]
        return [Segment(peak_normalize(x[s: s + win]), i, s / sr, 0.0)
                for i, s in enumerate(range(0, len(x) - win + 1, hop))]
    if scheme == "trim_repeat":
        x = trim_silence(peak_normalize(x), sr)
        if len(x) == 0:
            x = np.zeros(1, dtype=np.float32)
        if len(x) <= win:
            reps = int(np.ceil(win / len(x)))
            return [Segment(peak_normalize(np.tile(x, reps)[:win]), 0, 0.0, 1.0 - len(x) / win)]
        starts = list(range(0, len(x) - win + 1, hop))
        if starts[-1] + win < len(x):
            starts.append(len(x) - win)  # final window aligned to the end: no speech discarded
        return [Segment(peak_normalize(x[s: s + win]), i, s / sr, 0.0) for i, s in enumerate(starts)]
    raise ValueError(f"unknown segmentation scheme: {scheme}")
