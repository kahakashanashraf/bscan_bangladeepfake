"""Test-time perturbations (robustness) and counterfactual shortcut probes; inference only.

Every perturbation is applied to the 16 kHz recording BEFORE the model's own preprocessing
(segmentation scheme, peak normalisation, features), i.e. it simulates a changed input signal.
No model is retrained.  All randomness is seeded per recording, so results are reproducible.
For every recording and condition the windows that reach the model are compared with the clean
ones, so conditions that the preprocessing undoes (no change of the model input) are reported as
such and kept out of summary ranges; decision flips are also reported over changed recordings.

Robustness conditions (deployment-like):
    noise_snr{20,10,5}      white Gaussian noise at the given SNR w.r.t. the recording power
    mp3_{64,32}k, aac_32k   lossy codecs through ffmpeg (encode + decode)
    telephone               8 kHz resampling round trip + 300-3400 Hz band-pass
    clip_+6dB               +6 dB gain with hard clipping at full scale.  Every model peak-normalises
                            its input, so a pure level change is a no-op by construction and only the
                            clipping remains; the share of recordings that clip is reported per class
    speed_{0.95,1.05}       resampling speed change (tempo and pitch change together)
    pitch_{+1,-1}st         pitch shift by one semitone, duration preserved
    reverb_rt{0.3,0.6}      synthetic exponentially decaying room impulse response

Counterfactual shortcut probes (each targets one cue found in Phase 3; the controlled preprocessing
removes several of these cues by construction, which the change shares make visible):
    append_silence_0.44     append 0.44 s of digital silence (the VITS trailing-silence cue)
    trim_trailing           remove trailing silence (below -40 dB re peak)
    add_dc_neg              add a DC offset of -0.001 relative to the peak (the vocoder DC cue)
    remove_dc               subtract the mean (removes any DC cue)
    dither_-60dBFS          add -60 dBFS white noise (masks digital silence)

Usage (on a returned run directory):
    python -m bscan.robustness --run results/runs/bscan_controlled__P1__seed42 \
        --splits internal_test external_BF-MOZ external_MEN --max_files 600 --device auto
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import TARGET_SR, load_audio, segment
from .config import ExperimentCfg, DataCfg, ModelCfg, TrainCfg, EvalCfg
from .features import extract
from .metrics import binary_metrics
from .models import build_model, model_features
from .utils import pick_device, save_json, stable_seed

# ----------------------------------------------------------------------------- perturbations
def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def codec_roundtrip(x: np.ndarray, codec: str, bitrate: str) -> np.ndarray:
    import soundfile as sf

    ext = {"libmp3lame": "mp3", "aac": "m4a"}[codec]
    with tempfile.TemporaryDirectory() as td:
        src, enc, dec = Path(td) / "in.wav", Path(td) / f"enc.{ext}", Path(td) / "out.wav"
        sf.write(src, x, TARGET_SR, subtype="PCM_16")
        subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), "-c:a", codec, "-b:a", bitrate,
                        str(enc)], check=True)
        subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", str(enc), "-ar", str(TARGET_SR), "-ac", "1",
                        str(dec)], check=True)
        y, _ = sf.read(dec, dtype="float32")
    return y


def trim_trailing_silence(x: np.ndarray, sr: int = TARGET_SR, top_db: float = 40.0) -> np.ndarray:
    """Cut everything after the last frame louder than (peak frame - top_db); the start is untouched."""
    win, hop = int(0.025 * sr), int(0.010 * sr)
    if len(x) <= win:
        return x
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    e = 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    active = np.where(e > e.max() - top_db)[0]
    if not len(active) or active[-1] == n - 1:  # ends in sound: nothing to trim (keep the unanalysed tail)
        return x
    return x[: active[-1] * hop + win]


def synthetic_rir(rt60: float, rng: np.random.Generator, sr: int = TARGET_SR) -> np.ndarray:
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)  # -60 dB at t = rt60
    h[0] = 1.0
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def perturb(x: np.ndarray, name: str, seed: int) -> np.ndarray:
    from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt

    rng = np.random.default_rng(seed)
    x = x.astype(np.float32)
    if name == "clean":
        return x
    if name.startswith("noise_snr"):
        snr = float(name.replace("noise_snr", ""))
        p = np.mean(x ** 2) + 1e-12
        return (x + rng.standard_normal(len(x)).astype(np.float32) * np.sqrt(p / 10 ** (snr / 10))).astype(np.float32)
    if name == "mp3_64k":
        return codec_roundtrip(x, "libmp3lame", "64k")
    if name == "mp3_32k":
        return codec_roundtrip(x, "libmp3lame", "32k")
    if name == "aac_32k":
        return codec_roundtrip(x, "aac", "32k")
    if name == "telephone":
        y = resample_poly(resample_poly(x, 1, 2), 2, 1)
        sos = butter(4, [300, 3400], btype="bandpass", fs=TARGET_SR, output="sos")
        return sosfiltfilt(sos, y).astype(np.float32)
    if name.startswith("clip_"):
        # clip(x * g, -1, 1) = g * clip(x, -1/g, 1/g): identical after the models' peak normalisation, and
        # bit-exact (input unchanged) when nothing clips
        return np.clip(x, -clip_level(name), clip_level(name)).astype(np.float32)
    if name.startswith("speed_"):
        f = float(name.replace("speed_", ""))
        up, down = (20, int(round(20 * f))) if f >= 1 else (int(round(20 / f)), 20)
        return resample_poly(x, up, down).astype(np.float32)
    if name.startswith("pitch_"):
        import librosa

        st = float(name.replace("pitch_", "").replace("st", ""))
        return librosa.effects.pitch_shift(x, sr=TARGET_SR, n_steps=st).astype(np.float32)
    if name.startswith("reverb_rt"):
        rt = float(name.replace("reverb_rt", ""))
        y = fftconvolve(x, synthetic_rir(rt, rng))[: len(x)]
        return (y / (np.max(np.abs(y)) + 1e-9) * np.max(np.abs(x))).astype(np.float32)
    # ---- counterfactual probes
    if name == "append_silence_0.44":
        return np.concatenate([x, np.zeros(int(0.44 * TARGET_SR), np.float32)])
    if name == "trim_trailing":
        return trim_trailing_silence(x)
    if name == "add_dc_neg":
        return (x - 0.001 * float(np.max(np.abs(x)))).astype(np.float32)
    if name == "remove_dc":
        return (x - np.float32(x.mean())).astype(np.float32)
    if name == "dither_-60dBFS":
        return (x + 1e-3 * rng.standard_normal(len(x)).astype(np.float32)).astype(np.float32)
    raise ValueError(f"unknown perturbation {name}")


ROBUSTNESS = ["clean", "noise_snr20", "noise_snr10", "noise_snr5", "mp3_64k", "mp3_32k", "aac_32k", "telephone",
              "clip_+6dB", "speed_0.95", "speed_1.05", "pitch_+1st", "pitch_-1st", "reverb_rt0.3", "reverb_rt0.6"]
def clip_level(name: str) -> float:
    """A recording clips under clip_<g>dB if its 16 kHz peak exceeds this level."""
    return 10 ** (-float(name.replace("clip_", "").replace("dB", "")) / 20)
PROBES = ["clean", "append_silence_0.44", "trim_trailing", "add_dc_neg", "remove_dc", "dither_-60dBFS"]


# ----------------------------------------------------------------------------- scoring
def load_run(run_dir: Path, device):
    import torch

    cfgd = json.loads((run_dir / "config.json").read_text())
    cfg = ExperimentCfg(name=cfgd["name"], out_dir=cfgd["out_dir"], data=DataCfg(**cfgd["data"]),
                        model=ModelCfg(**cfgd["model"]), train=TrainCfg(**cfgd["train"]), eval=EvalCfg(**cfgd["eval"]))
    stats = json.loads((run_dir / "norm_stats.json").read_text())
    model = build_model(cfg.model).to(device)
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    if not (run_dir / "metrics.json").exists():
        raise FileNotFoundError(f"{run_dir}: metrics.json missing; evaluate the run first (its threshold is needed)")
    thr = json.loads((run_dir / "metrics.json").read_text())["threshold_logit"]
    return cfg, stats, model, thr


def model_windows(x: np.ndarray, cfg: ExperimentCfg, seed: int, scheme: str | None = None) -> list:
    """The windows that reach the model for one 16 kHz recording, using the run's preprocessing."""
    return segment(x, scheme=scheme or cfg.data.scheme, window_sec=cfg.data.window_sec, overlap=cfg.data.overlap,
                   seed=seed)


def input_difference(a: list, b: list) -> float:
    """Largest absolute sample difference between two window lists (inf if the windowing differs)."""
    if len(a) != len(b):
        return float("inf")
    return max((float(np.max(np.abs(s.audio - t.audio))) for s, t in zip(a, b)), default=0.0)


def score_recording(x: np.ndarray, cfg: ExperimentCfg, stats: dict, model, device, seed: int,
                    scheme: str | None = None, segs: list | None = None) -> float:
    """Mean window logit for one (possibly perturbed) 16 kHz recording, using the run's preprocessing."""
    import torch

    feats_names = model_features(cfg.model)
    segs = segs if segs is not None else model_windows(x, cfg, seed, scheme)
    batch = {k: [] for k in feats_names}
    for s in segs:
        f = extract(s.audio, feats_names)
        for k in feats_names:
            m = np.asarray(stats[k]["mean"], np.float32)[:, None, None]
            sd = np.asarray(stats[k]["std"], np.float32)[:, None, None] + 1e-8
            batch[k].append((f[k].astype(np.float16).astype(np.float32) - m) / sd)
    with torch.inference_mode():
        t = {k: torch.from_numpy(np.stack(v)).to(device) for k, v in batch.items()}
        return float(model(t).float().mean().cpu())


def run_conditions(run_dir: Path, splits: list[str], conditions: list[str], max_files: int, device_pref: str,
                   metadata_csv: str | None = None, audio_root: str | None = None, seed: int = 0,
                   out_name: str = "robustness") -> pd.DataFrame:
    if not conditions or conditions[0] != "clean":
        raise ValueError("the first condition must be 'clean' (every other condition is compared with it)")
    device = pick_device(device_pref)
    cfg, stats, model, thr = load_run(run_dir, device)
    meta = pd.read_csv(metadata_csv or cfg.data.metadata_csv, dtype={"utterance_id": str}, low_memory=False)
    root = Path(audio_root or cfg.data.audio_root)
    col = f"split_{cfg.data.protocol}"
    rows = []
    for sp in splits:
        files = meta[meta[col] == sp]
        if files.empty:
            continue
        # fixed, stratified subset (same files for every condition and every model)
        if max_files and len(files) > max_files:
            files = pd.concat([g.sample(n=min(len(g), max_files // files["label"].nunique()), random_state=seed)
                               for _, g in files.groupby("label")])
        for _, r in files.sort_values("audio_path").iterrows():
            x, _ = load_audio(root / r["audio_path"])
            base = stable_seed(r["audio_path"])
            clean_segs = None
            for c in conditions:
                y = perturb(x, c, seed=stable_seed(r["audio_path"], c))
                segs = model_windows(y, cfg, base)
                if c == "clean":
                    clean_segs = segs
                diff = 0.0 if c == "clean" else input_difference(segs, clean_segs)
                rows.append({"split": sp, "audio_path": r["audio_path"], "label": int(r["label"]),
                             "dataset_id": r["dataset_id"], "condition": c,
                             "score": score_recording(y, cfg, stats, model, device, seed=base, segs=segs),
                             "input_changed": bool(diff > 0), "input_max_abs_diff": diff,
                             "clipped": bool(c.startswith("clip_") and np.max(np.abs(x)) > clip_level(c))})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"{out_name}_scores.csv.gz", index=False)

    summ = []
    for (sp, c), g in df.groupby(["split", "condition"]):
        m = binary_metrics(g["label"].to_numpy(), g["score"].to_numpy(), thr)
        gi = g.set_index("audio_path")
        clean = df[(df.split == sp) & (df.condition == "clean")].set_index("audio_path")["score"].reindex(gi.index)
        d = gi["score"] - clean
        flips = (gi["score"] >= thr) != (clean >= thr)
        changed = gi["input_changed"].astype(bool)

        def by_label(s, lab):
            return float(s[gi["label"] == lab].mean()) if (gi["label"] == lab).any() else float("nan")

        summ.append({"split": sp, "condition": c, "n": m["n"], "auc": m["auc"], "eer": m["eer"], "accuracy": m["accuracy"],
                     "sensitivity": m["recall_spoof_sensitivity"], "specificity": m["specificity_bonafide_recall"],
                     "mean_score_shift_bonafide": by_label(d, 0), "mean_score_shift_spoof": by_label(d, 1),
                     "decision_flip_rate": float(flips.mean()),
                     "share_input_changed": float(changed.mean()),
                     "decision_flip_rate_changed": float(flips[changed].mean()) if changed.any() else float("nan"),
                     "share_clipped_bonafide": by_label(gi["clipped"].astype(float), 0) if c.startswith("clip_") else float("nan"),
                     "share_clipped_spoof": by_label(gi["clipped"].astype(float), 1) if c.startswith("clip_") else float("nan")})
    s = pd.DataFrame(summ)
    s.to_csv(run_dir / f"{out_name}_summary.csv", index=False)
    save_json({"run": str(run_dir), "threshold_logit": thr, "conditions": conditions, "splits": splits,
               "max_files": max_files, "device": str(device), "code_version": os.environ.get("BSCAN_CODE_VERSION")},
              run_dir / f"{out_name}_meta.json")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", nargs="+", default=["internal_test", "external_BF-MOZ", "external_MEN"])
    ap.add_argument("--mode", choices=["robustness", "probes"], default="robustness")
    ap.add_argument("--max_files", type=int, default=600)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--audio_root", default=None)
    args = ap.parse_args()
    conds = ROBUSTNESS if args.mode == "robustness" else PROBES
    s = run_conditions(Path(args.run), args.splits, conds, args.max_files, args.device, args.metadata,
                       args.audio_root, out_name=args.mode)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
