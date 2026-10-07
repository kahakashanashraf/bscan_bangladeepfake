"""Time-frequency front-ends (numpy/librosa, CPU).

"mel" and "lfcc" reproduce the original notebook (code/banglafkpart.ipynb, cell 23):
    STFT: n_fft 1024, win 1024 (Hann), hop 512, centred frames -> T = 188 for a 6 s clip at 16 kHz
    mel : 128-band power mel spectrogram, power_to_db(ref=max, top_db=80), + delta + delta-delta
    lfcc: 40 triangular filters equally spaced on the linear-frequency axis (peak 1, unnormalised),
          natural log(E + 1e-9), DCT-II (orthonormal) keeping all 40 coefficients, + deltas
"mfcc" is added for the representation study and is built to differ from "lfcc" only in the
    frequency warping of the filterbank: 40 mel filters (librosa default Slaney filters),
    natural log(E + 1e-9), DCT-II keeping all 40 coefficients, + deltas.
All outputs are float32 arrays shaped [3, F, T] (static, delta, delta-delta).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 128
N_CEPS = 40
LOG_EPS = 1e-9


@lru_cache(maxsize=8)
def linear_filterbank(sr: int = SR, n_fft: int = N_FFT, n_filters: int = N_CEPS) -> np.ndarray:
    """Triangular filters with centres equally spaced over FFT bins (as in the original code)."""
    n_freqs = n_fft // 2 + 1
    edges = np.linspace(0, n_freqs - 1, n_filters + 2, dtype=int)
    fb = np.zeros((n_filters, n_freqs), dtype=np.float32)
    for i in range(n_filters):
        a, c, b = edges[i], edges[i + 1], edges[i + 2]
        if c > a:
            fb[i, a:c] = np.linspace(0, 1, c - a, endpoint=False, dtype=np.float32)
        if b > c:
            fb[i, c:b] = np.linspace(1, 0, b - c, endpoint=False, dtype=np.float32)
    return fb


@lru_cache(maxsize=8)
def mel_filterbank(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_CEPS) -> np.ndarray:
    import librosa

    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels).astype(np.float32)


def _stack_deltas(x: np.ndarray) -> np.ndarray:
    import librosa

    return np.stack([x, librosa.feature.delta(x), librosa.feature.delta(x, order=2)], axis=0).astype(np.float32)


def power_spectrogram(y: np.ndarray) -> np.ndarray:
    import librosa

    return np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2  # float32, as in the original


def mel_feature(y: np.ndarray, sr: int = SR, S: np.ndarray | None = None) -> np.ndarray:
    import librosa

    if S is None:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    else:
        mel = librosa.feature.melspectrogram(S=S, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return _stack_deltas(librosa.power_to_db(mel, ref=np.max))


def _cepstra(S: np.ndarray, fb: np.ndarray) -> np.ndarray:
    import librosa

    log_e = np.log(fb @ S + LOG_EPS).astype(np.float32)
    return librosa.feature.mfcc(S=log_e, n_mfcc=fb.shape[0])


def lfcc_feature(y: np.ndarray, sr: int = SR, S: np.ndarray | None = None) -> np.ndarray:
    S = power_spectrogram(y) if S is None else S
    return _stack_deltas(_cepstra(S, linear_filterbank(sr, N_FFT, N_CEPS)))


def mfcc_feature(y: np.ndarray, sr: int = SR, S: np.ndarray | None = None) -> np.ndarray:
    S = power_spectrogram(y) if S is None else S
    return _stack_deltas(_cepstra(S, mel_filterbank(sr, N_FFT, N_CEPS)))


EXTRACTORS = {"mel": mel_feature, "lfcc": lfcc_feature, "mfcc": mfcc_feature}
FEATURE_SHAPES = {"mel": (3, N_MELS), "lfcc": (3, N_CEPS), "mfcc": (3, N_CEPS)}


def extract(y: np.ndarray, names: tuple[str, ...], sr: int = SR) -> dict[str, np.ndarray]:
    """Compute the requested features from one 16 kHz segment, sharing the STFT where possible."""
    S = power_spectrogram(y) if ("lfcc" in names or "mfcc" in names or "mel" in names) else None
    return {n: EXTRACTORS[n](y, sr, S) for n in names}
