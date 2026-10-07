"""Build a per-file inventory of every audio file in the source datasets (Phase 1).

For each file this records provenance (dataset, subset, label, speaker/utterance ids
where derivable from the source layout), container facts (sample rate, channels,
codec subtype, frames, duration, size), content hashes (file SHA-256 and decoded-PCM
SHA-256), and signal descriptors used later by the leakage audit (level, clipping,
exact-zero samples, leading/trailing silence, activity ratio, spectral roll-off,
an energy-percentile SNR proxy) plus a compact log-mel fingerprint for
near-duplicate search.

Nothing is inferred that the source layout does not state: unknown fields are
written as "unknown".

Usage:
    python .scripts/build_inventory.py --root data --out data/inventory
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

FINGERPRINT_SR_FMAX = 8000.0  # common band for all corpora (lowest native Nyquist is 8 kHz)
N_FP_MELS = 40

# --------------------------------------------------------------------------------------
# Provenance rules (derived only from the documented folder layout of each source)
# --------------------------------------------------------------------------------------
BANGLAFAKE_SUBSETS = {
    "deepfake_data_sust": "BF-SUST",
    "deepfake_data_mozilla": "BF-MOZ",
    "deepfake_data_news": "BF-NEWS",
}
BANGLAFAKE_LABEL_DIRS = {"real_wav": 0, "deepfake_wav": 1}
MENDELEY_LABEL_DIRS = {"real": 0, "fake": 1}
MENDELEY_SPEAKER_RE = re.compile(r"^S(\d+)([MF])(\d+)$", re.IGNORECASE)
MOZILLA_NAME_RE = re.compile(r"^common_voice_(s\d+)_\d+$")

LICENSES = {
    "BF": "unknown (no licence on HF card or GitHub; paper says 'open license')",
    "MEN": "CC BY 4.0",
}


def classify(path: Path, root: Path) -> dict | None:
    """Map a file path to dataset/subset/label/speaker using only folder names."""
    parts = [p for p in path.relative_to(root).parts]
    lower = [p.lower() for p in parts]

    subset = next((BANGLAFAKE_SUBSETS[p] for p in lower if p in BANGLAFAKE_SUBSETS), None)
    if subset is not None:
        label = next((BANGLAFAKE_LABEL_DIRS[p] for p in lower if p in BANGLAFAKE_LABEL_DIRS), None)
        if label is None:
            return None
        stem = path.stem
        moz = MOZILLA_NAME_RE.match(stem)
        if label == 1:
            speaker = "BF-VITS-voice"  # single male VITS voice (BanglaFake paper)
            generator = "VITS TTS (BanglaFake; trained on SUST TTS corpus)"
        elif subset == "BF-SUST":
            speaker = "BF-SUST-speaker"  # the one SUST TTS voice talent (Ahmad et al. 2021, Sec. 3.3)
            generator = "none (bona fide)"
        elif subset == "BF-MOZ" and moz:
            speaker = f"BF-MOZ-{moz.group(1)}"  # Common Voice speaker tag embedded in the file name
            generator = "none (bona fide)"
        else:
            speaker = "unknown"
            generator = "none (bona fide)"
        # utterance key: identifies the text/utterance shared by a bona fide file and its spoof
        # counterpart (SUST ids are zero-padded for real files but not for fakes: 01001 vs 1001)
        utt_key = f"{subset}:{int(stem)}" if (subset in ("BF-SUST", "BF-NEWS") and stem.isdigit()) else f"{subset}:{stem}"
        return dict(
            dataset_id=subset,
            dataset_name="BanglaFake",
            source="huggingface.co/datasets/sifat1221/banglaFake (final_data.zip)",
            label=label,
            speaker_id=speaker,
            set_id=f"BF-MOZ-{moz.group(1)}" if (subset == "BF-MOZ" and moz) else "unknown",
            gender="male" if label == 1 else "unknown",
            generator=generator,
            license=LICENSES["BF"],
            utterance_key=utt_key,
        )

    if "dataset" in lower and any(p in MENDELEY_LABEL_DIRS for p in lower):
        label = next(MENDELEY_LABEL_DIRS[p] for p in lower if p in MENDELEY_LABEL_DIRS)
        spk = next((parts[i] for i, p in enumerate(parts) if MENDELEY_SPEAKER_RE.match(p)), None)
        m = MENDELEY_SPEAKER_RE.match(spk) if spk else None
        set_id = f"S{int(m.group(1))}" if m else "unknown"
        return dict(
            dataset_id="MEN",
            dataset_name="Mendeley Bangla Audio Dataset (v4)",
            source="doi:10.17632/4ftmwt86vr.4",  # audio byte-identical to version 1 (.1)
            label=label,
            speaker_id=f"MEN-{spk.upper()}" if spk else "unknown",
            set_id=set_id,
            gender=({"M": "male", "F": "female"}[m.group(2).upper()] if m else "unknown"),
            generator="unknown (not documented)" if label == 1 else "none (bona fide)",
            license=LICENSES["MEN"],
            # sentence index within a set (each set's 5 speakers read the same 30 sentences)
            utterance_key=f"MEN:{set_id}:{path.stem}",
        )
    return None


# --------------------------------------------------------------------------------------
# Signal descriptors
# --------------------------------------------------------------------------------------
def frame_energy_db(x: np.ndarray, sr: int, win_s: float = 0.025, hop_s: float = 0.010) -> np.ndarray:
    win = max(1, int(round(win_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    if len(x) < win:
        return np.array([10 * np.log10(np.mean(x ** 2) + 1e-12)])
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    e = np.mean(x[idx] ** 2, axis=1)
    return 10 * np.log10(e + 1e-12)


def spectral_rolloff(x: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Return (rolloff95_hz, rolloff99_hz, band_edge_hz) from the long-term power spectrum.

    band_edge_hz = highest frequency whose smoothed PSD is within 60 dB of the PSD peak,
    a proxy for the effective bandwidth (e.g. codec or vocoder low-pass).
    """
    from scipy.signal import welch

    nper = min(2048, len(x))
    if nper < 64:
        return float("nan"), float("nan"), float("nan")
    f, p = welch(x, fs=sr, nperseg=nper)
    c = np.cumsum(p)
    if c[-1] <= 0:
        return float("nan"), float("nan"), float("nan")
    r95 = float(f[np.searchsorted(c, 0.95 * c[-1])])
    r99 = float(f[min(len(f) - 1, np.searchsorted(c, 0.99 * c[-1]))])
    pdb = 10 * np.log10(p + 1e-20)
    k = max(1, len(pdb) // 128)
    smooth = np.convolve(pdb, np.ones(k) / k, mode="same")
    above = np.where(smooth > smooth.max() - 60.0)[0]
    edge = float(f[above[-1]]) if len(above) else float("nan")
    return r95, r99, edge


def fingerprint(x: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    fmax = min(FINGERPRINT_SR_FMAX, sr / 2)
    m = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=256, n_mels=N_FP_MELS, fmax=fmax)
    lm = np.log(m + 1e-10)
    return np.concatenate([lm.mean(axis=1), lm.std(axis=1)]).astype(np.float32)


def describe(args: tuple[str, str]) -> dict:
    path_str, root_str = args
    path, root = Path(path_str), Path(root_str)
    rec = classify(path, root)
    if rec is None:
        return {"audio_path": str(path.relative_to(root)), "error": "unclassified"}
    rec = dict(rec)
    rec["audio_path"] = str(path.relative_to(root))
    rec["utterance_id"] = path.stem
    rec["language"] = "bn"
    try:
        raw = path.read_bytes()
        rec["file_size_bytes"] = len(raw)
        rec["sha256"] = hashlib.sha256(raw).hexdigest()
        info = sf.info(path_str)
        rec.update(
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            frames=int(info.frames),
            duration_sec=float(info.frames / info.samplerate) if info.samplerate else float("nan"),
            container=info.format,
            codec=info.subtype,
        )
        pcm, sr = sf.read(path_str, dtype="int16", always_2d=True)
        rec["pcm_sha256"] = hashlib.sha256(np.ascontiguousarray(pcm).tobytes()).hexdigest()
        x = pcm.astype(np.float32).mean(axis=1) / 32768.0

        n = len(x)
        rec["peak_abs"] = float(np.max(np.abs(x))) if n else 0.0
        rec["clip_ratio"] = float(np.mean(np.abs(pcm) >= 32767)) if n else 0.0
        rec["exact_zero_ratio"] = float(np.mean(pcm == 0)) if n else 0.0
        rec["dc_offset"] = float(np.mean(x)) if n else 0.0
        rec["rms_dbfs"] = float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)) if n else float("nan")
        try:
            import pyloudnorm as pyln

            rec["lufs"] = float(pyln.Meter(sr).integrated_loudness(x)) if n >= int(0.5 * sr) else float("nan")
        except Exception:
            rec["lufs"] = float("nan")

        e = frame_energy_db(x, sr)
        thr = e.max() - 40.0
        active = np.where(e > thr)[0]
        hop_s = 0.010
        if len(active):
            rec["leading_silence_sec"] = float(active[0] * hop_s)
            rec["trailing_silence_sec"] = float(max(0.0, rec["duration_sec"] - (active[-1] * hop_s + 0.025)))
            rec["activity_ratio"] = float(len(active) / len(e))
        else:
            rec["leading_silence_sec"] = rec["trailing_silence_sec"] = float("nan")
            rec["activity_ratio"] = 0.0
        # energy-percentile SNR proxy (not a calibrated SNR; documented as a proxy)
        rec["snr_proxy_db"] = float(np.percentile(e, 95) - np.percentile(e, 10))
        r95, r99, edge = spectral_rolloff(x, sr)
        rec.update(rolloff95_hz=r95, rolloff99_hz=r99, band_edge_hz=edge)
        zc = np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))) if n > 1 else 0.0
        rec["zcr"] = float(zc)
        rec["_fingerprint"] = fingerprint(x, sr).tolist()
        rec["error"] = ""
    except Exception as exc:  # corrupt / unreadable file
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip()


def attach_transcripts(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Join the BanglaFake metadata transcripts onto utterance_key; Mendeley has none."""
    texts: dict[str, str] = {}
    for meta in root.rglob("metadata.*"):
        sub = next((BANGLAFAKE_SUBSETS[p.lower()] for p in meta.parts if p.lower() in BANGLAFAKE_SUBSETS), None)
        if sub is None:
            continue
        raw = meta.read_text(encoding="utf-8").splitlines()
        if raw and raw[0].lower().startswith("filename"):
            sep = "|" if "|" in raw[0] else ","
            raw = raw[1:]
        else:
            sep = "|"
        for line in raw:
            if sep not in line:
                continue
            key, text = line.split(sep, 1)
            key = key.strip()
            key = str(int(key)) if key.isdigit() else key
            texts[f"{sub}:{key}"] = _norm_text(text)
    df["transcript"] = df["utterance_key"].map(texts).fillna("unknown") if "utterance_key" in df else "unknown"
    # group key for leakage-safe splitting: identical sentence text => same group; unknown text => utterance key
    df["text_group"] = np.where(df["transcript"] != "unknown", "TXT:" + df["transcript"], df.get("utterance_key", ""))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="folder containing the extracted datasets")
    ap.add_argument("--out", default="data/inventory")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wavs = sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3"})
    wavs = [w for w in wavs if "/__MACOSX/" not in w and not Path(w).name.startswith("._")]
    print(f"audio files found: {len(wavs)}", file=sys.stderr)

    with Pool(args.workers) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(describe, [(w, str(root)) for w in wavs], chunksize=64)):
            recs.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  processed {i + 1}/{len(wavs)}", file=sys.stderr)

    df = pd.DataFrame(recs).sort_values("audio_path").reset_index(drop=True)
    df = attach_transcripts(df, root)
    fps = df.pop("_fingerprint") if "_fingerprint" in df else None
    if fps is not None:
        ok = fps.notna()
        np.save(out / "fingerprints.npy", np.stack([np.asarray(v, dtype=np.float32) for v in fps[ok]]))
        df.loc[ok, "fingerprint_row"] = np.arange(ok.sum())
    df.to_csv(out / "inventory_full.csv", index=False)
    meta = {
        "n_files": int(len(df)),
        "n_errors": int((df["error"].astype(str) != "").sum()),
        "root": str(root),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "soundfile": sf.__version__,
    }
    (out / "inventory_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
