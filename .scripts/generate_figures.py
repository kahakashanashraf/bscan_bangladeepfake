"""Generate every figure from result files (no numbers typed by hand).

Figures are written as vector PDFs to PLOS/figures/ (the figure files of PLOS/BSCAN_PLOS_ONE.tex), named after
the number the figure receives in the manuscript (Fig1 ... Fig9, S1_Fig; .scripts/final_qc.py checks that the
order of the figure floats matches).  Every figure whose inputs are all real is also exported as a PLOS
submission file, PLOS/submission/<name>.tif (RGB, 300 dpi, LZW); placeholder figures are never exported, and the
TIFFs of the previous run are deleted first.
If any input of a figure is a synthetic placeholder, the figure is overprinted with a red
"SYNTHETIC PLACEHOLDER - NOT A RESULT" banner, the same text is written into the PDF metadata (Keywords)
so that .scripts/final_qc.py can detect it in the file itself, and the figure is listed as such in
results/figures/figure_manifest.csv.  Every figure has exactly one manifest row per run: a figure whose
inputs do not exist yet gets a 'missing' row, and figure files of the previous run are deleted first, so a
stale file can never stand in for a figure that was not regenerated.

Usage:
    python .scripts/generate_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bscan.registry import load_registry, resolve  # noqa: E402

FIG = ROOT / "PLOS" / "figures"
SUBMISSION = ROOT / "PLOS" / "submission"  # upload-ready TIFFs of the real figures
OUT = ROOT / "results" / "figures"
MANIFEST: list[dict] = []
PLACEHOLDER_MARK = "SYNTHETIC PLACEHOLDER - NOT A RESULT"
# PLOS figure guidelines: Arial (DejaVu Sans only as a per-glyph fallback), embedded as TrueType, not Type 3
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
# colour-blind-safe palette (Okabe-Ito)
C = {"bscan_controlled": "#0072B2", "lcnn_lfcc": "#D55E00", "lfcc_gmm_controlled": "#009E73",
     "stats_hgb_controlled": "#7F7F7F", "mel_only": "#E69F00", "lfcc_only": "#CC79A7", "mfcc_only": "#56B4E9",
     "mel_mfcc": "#F0E442", "bscan_orig6s": "#000000", "stats_hgb_orig6s": "#BBBBBB",
     "dual_plain": "#D9D9D9", "dual_no_se": "#A6A6A6", "dual_no_tattn": "#595959"}
# readable names of test-time conditions (robustness perturbations and counterfactual probes)
COND = {"noise_snr20": "White noise, 20 dB SNR", "noise_snr10": "White noise, 10 dB SNR",
        "noise_snr5": "White noise, 5 dB SNR", "mp3_64k": "MP3, 64 kbit/s", "mp3_32k": "MP3, 32 kbit/s",
        "aac_32k": "AAC, 32 kbit/s", "telephone": "Telephone channel", "clip_+6dB": "Clipping (+6 dB gain)",
        "speed_0.95": "Speed \u00d70.95", "speed_1.05": "Speed \u00d71.05", "pitch_+1st": "Pitch +1 semitone",
        "pitch_-1st": "Pitch \u22121 semitone", "reverb_rt0.3": "Reverberation, RT60 0.3 s",
        "reverb_rt0.6": "Reverberation, RT60 0.6 s",
        "append_silence_0.44": "append 0.44 s silence", "trim_trailing": "trim trailing silence",
        "add_dc_neg": "add DC offset", "remove_dc": "remove DC offset", "dither_-60dBFS": "add \u221260 dBFS noise"}
LABEL = {k: v["label"] for k, v in load_registry()["models"].items()}
ROBUSTNESS_CODES = ["noise_snr20", "noise_snr10", "noise_snr5", "mp3_64k", "mp3_32k", "aac_32k", "telephone", "clip_+6dB",
                    "speed_0.95", "speed_1.05", "pitch_+1st", "pitch_-1st", "reverb_rt0.3", "reverb_rt0.6"]
# the names of the tables (.scripts/generate_tables.py TABLE_LABEL); the registry labels stay as registered
LABEL.update({"stats_hgb_controlled": "Recording statistics", "stats_hgb_orig6s": "Recording statistics (orig. prep.)",
              "bscan_orig6s": "BSCAN (orig. prep.)", "dual_plain": "Dual, no SE, no attention",
              "dual_no_tattn": "Dual, no attention", "ssl_xlsr_probe": "XLS-R (frozen) + logistic regression"})
SPLITNAME = {("P1", "internal_test"): "BF-SUST (internal test)", ("P1", "external_BF-MOZ"): "BF-MOZ",
             ("P1", "external_MEN"): "MEN (cross-dataset)", ("P2", "internal_test"): "MEN (internal test)",
             ("P2", "external_BF-SUST"): "BF-SUST", ("P2", "external_BF-MOZ"): "BF-MOZ"}


def rel(path) -> str:
    """Source path relative to the repository root (the manifest never records local absolute paths)."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def save(fig, name: str, caption: str, status: str, sources: list[str]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"Title": f"{name}: {caption}"}
    if status != "real":
        fig.text(0.5, 0.5, PLACEHOLDER_MARK, color="red", alpha=0.35, fontsize=22,
                 ha="center", va="center", rotation=25, weight="bold", transform=fig.transFigure, zorder=1000)
        meta["Keywords"] = PLACEHOLDER_MARK
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata=meta)  # vector figure of the manuscript
    if status == "real":  # PLOS upload file: TIFF, RGB (no alpha), 300 dpi, LZW compression
        SUBMISSION.mkdir(parents=True, exist_ok=True)
        fig.savefig(SUBMISSION / f"{name}.tif", dpi=300, bbox_inches="tight", facecolor="white",
                    pil_kwargs={"compression": "tiff_lzw"})
        from PIL import Image
        with Image.open(SUBMISSION / f"{name}.tif") as im:
            rgb = im.convert("RGB")
        rgb.save(SUBMISSION / f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))
    plt.close(fig)
    MANIFEST.append({"figure": name, "status": status, "caption": caption, "sources": ";".join(rel(x) for x in sources)})


def skip(name: str, caption: str, reason: str) -> None:
    """Manifest row for a figure that could not be drawn (its inputs do not exist yet); no file is written."""
    MANIFEST.append({"figure": name, "status": "missing", "caption": caption, "sources": reason})


def label_panels(axes, x: float = -0.02, y: float = 1.02) -> None:
    """Bold panel letters (A, B, C, ...) referenced by the figure captions."""
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.text(x, y, "ABCDEFGH"[i], transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")


def combine(sts: list[str]) -> str:
    return "missing" if "missing" in sts else ("placeholder" if "placeholder" in sts else "real")


# ----------------------------------------------------------------------------- diagrams (not data)
def box(ax, x, y, w, h, text, fc="#EAF2FB", ec="#0072B2", fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#333333",
                                 shrinkA=0, shrinkB=0, zorder=5))


MANUAL = ROOT / "PLOS" / "figures_manual"  # hand-drawn figures that replace a generated diagram (e.g. Fig1.tif)


def use_manual(name: str, caption: str, sources: list[str]) -> bool:
    """Use a hand-drawn PLOS/figures_manual/<name>.tif|.tiff|.png instead of drawing the figure.

    The raster file becomes the PLOS upload file PLOS/submission/<name>.tif (RGB, LZW, its own resolution) and,
    unless PLOS/figures_manual/<name>.pdf (a vector export of the same drawing) exists, also the manuscript
    figure PLOS/figures/<name>.pdf.  PLOS limits are checked and reported: 300-600 dpi, width 2.63-7.5 in,
    height at most 8.75 in.  Returns False when there is no manual file, so the diagram is drawn as before.
    """
    raster = next((MANUAL / f"{name}.{e}" for e in ("tif", "tiff", "png") if (MANUAL / f"{name}.{e}").exists()), None)
    vector = MANUAL / f"{name}.pdf"
    if raster is None:
        if vector.exists():
            raise RuntimeError(f"{rel(vector)} found: also export the drawing as {name}.tif or {name}.png "
                               "(300-600 dpi) for the PLOS upload")
        return False
    from PIL import Image
    with Image.open(raster) as im:
        dpi = float(round((im.info.get("dpi") or (0, 0))[0] or 0))  # PNG stores 299.9994 for 300
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            flat = Image.new("RGB", im.size, "white")
            flat.paste(im, mask=im.getchannel("A"))
        else:
            flat = im.convert("RGB")
    res = dpi if dpi >= 1 else 300.0  # no resolution stored in the file: read it as 300 dpi
    w_in, h_in = flat.width / res, flat.height / res
    problems = [p for p, bad in (
        (f"resolution {dpi:g} dpi (PLOS: 300-600; none stored, read as 300)" if dpi < 1 else
         f"resolution {dpi:g} dpi (PLOS: 300-600)", not 300 <= res <= 600),
        (f"width {w_in:.2f} in (PLOS: 2.63-7.5 in)", not 2.63 <= w_in <= 7.5),
        (f"height {h_in:.2f} in (PLOS: at most 8.75 in)", h_in > 8.75)) if bad]
    FIG.mkdir(parents=True, exist_ok=True)
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    flat.save(SUBMISSION / f"{name}.tif", compression="tiff_lzw", dpi=(res, res))
    if vector.exists():
        (FIG / f"{name}.pdf").write_bytes(vector.read_bytes())
    else:
        flat.save(FIG / f"{name}.pdf", "PDF", resolution=res)
    for p in problems:
        print(f"[figure warning] {name} (hand-drawn): {p}")
    MANIFEST.append({"figure": name, "status": "real", "caption": f"{caption} (hand-drawn: {raster.name})",
                     "sources": ";".join([rel(raster)] + [rel(x) for x in sources])})
    return True


FIG1_EXAMPLE = "banglafake/extracted/final_data/deepfake_data_mozilla/real_wav/common_voice_s1_103.wav"


def _fig1_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Waveform (16 kHz, every 60th sample) and the static Mel (dB) and LFCC maps of the first window of one bona fide
    BF-MOZ recording (Mozilla Common Voice, CC0) after the controlled preprocessing, for Fig 1.  Recomputed with the
    pipeline's own code when the corpus is in data/, otherwise read from results/figures/fig1_example_window.npz
    (made once from the pinned BanglaFake archive, SHA-256 checked), so the figure needs no audio."""
    cache = OUT / "fig1_example_window.npz"
    audio = ROOT / "data" / FIG1_EXAMPLE
    if audio.exists():
        from bscan.audio import load_audio, segment
        from bscan.features import lfcc_feature, mel_feature
        x, _ = load_audio(audio)
        w0 = segment(x, scheme="controlled", seed=0)[0].audio
        prev = dict(np.load(cache)) if cache.exists() else {}
        np.savez_compressed(cache, mel=mel_feature(w0)[0].astype(np.float16),
                            lfcc=lfcc_feature(w0)[0].astype(np.float16), wave=x[::60].astype(np.float16),
                            **{k: v for k, v in prev.items() if k == "info"})
    d = np.load(cache)
    return d["mel"].astype(float), d["lfcc"].astype(float), d["wave"].astype(float)


def fig_architecture():
    """Pipeline and architecture diagram in the layout chosen by the authors: (A) recording, controlled preprocessing,
    the Mel and LFCC branches, fusion and the recording score; (B) the SE-ResBlock and its SE module; (C) temporal
    attention pooling.  Drawn in inch coordinates on a 7.3 in wide canvas (7.5 in with the save padding, the PLOS
    maximum) with Arial at 8 pt or more, no title inside the image.  The waveform and feature maps are real: one bona
    fide BF-MOZ recording and its first window after the controlled preprocessing (_fig1_example).  A hand-drawn
    PLOS/figures_manual/Fig1.tif (or .png) replaces it (see use_manual)."""
    if use_manual("Fig1", "Architecture and preprocessing", ["src/bscan/models/bscan.py", "src/bscan/audio.py"]):
        return
    # mathtext in Arial, so that the symbols of panel C (w_t, h_t, f) use the same font as the rest
    with plt.rc_context({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold"}):
        _draw_architecture()


def _draw_architecture() -> None:
    """Body of fig_architecture, drawn inside its rc_context."""
    from matplotlib.patches import Circle

    FS, FT, FL = 8, 9, 12  # body, title and panel-letter font sizes (pt); PLOS: 8-12 pt
    INK = "#333333"
    X, AR, DL, PR = "×", "→", "Δ", "′"  # times, arrow, Delta, prime
    PAL = {"grey": ("#EEF1F5", "#5B6B7F"), "peach": ("#FDF0E0", "#C8955A"), "lav": ("#F1ECFB", "#8E6CCB"),
           "lavbox": ("#F8F6FD", "#8E6CCB"), "blue": ("#D8E8F7", "#3C7FB9"), "pink": ("#F9DADD", "#D46A73"),
           "green": ("#DDF0D8", "#5AA35A"), "yellow": ("#FCEBC4", "#D9A12B"), "bn": ("#E4E6EA", "#8A9099"),
           "fuse": ("#F2EAFC", "#9B6FD6"), "fusebox": ("#FBF9FE", "#9B6FD6"), "score": ("#EAF6E6", "#5AA35A"),
           "scorebox": ("#F7FBF5", "#5AA35A"), "beige": ("#F6F2EA", "#9A8F7A")}
    W, H = 7.3, 8.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def rbox(x, y, w, h, style, r=0.05, z=1):
        fc, ec = PAL[style]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=0.8,
                                    zorder=z))

    def frame(x, y, w, h, fc, ec, z=0, lw=0.9, dash=(4, 2), r=0.06):  # dashed container
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                    ls=(0, dash), zorder=z))

    def tbox(x, y, w, h, text, style, z=2):
        rbox(x, y, w, h, style, z=z)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS, linespacing=1.15, zorder=z + 1)

    def arr(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK,
                                     shrinkA=0, shrinkB=0, zorder=6))

    def line(xs, ys):
        ax.plot(xs, ys, color=INK, lw=0.9, solid_joinstyle="miter", zorder=5)

    def title(x, y, bold, rest="", fs=FT):
        tb = ax.text(x, y, bold, ha="left", va="top", fontsize=fs, weight="bold", zorder=7)
        if rest:
            fig.canvas.draw()
            bb = tb.get_window_extent().transformed(ax.transData.inverted())
            ax.text(bb.x1 + 0.05, y, rest, ha="left", va="top", fontsize=fs, zorder=7)

    def letter(y, s):
        ax.text(0.0, y, s, ha="left", va="top", fontsize=FL, weight="bold")

    def circle(x, y, sym=None):
        ax.add_patch(Circle((x, y), 0.08 if sym is None else 0.09, fc="#F9D2D5" if sym is None else "white",
                            ec="#D46A73" if sym is None else INK, lw=0.8 if sym is None else 0.9, zorder=3))
        if sym:
            ax.text(x, y, sym, ha="center", va="center", fontsize=FS + 1, zorder=4)

    mel, lfcc, wave = _fig1_example()

    # ---- panel A, row 1: recording, conversion, controlled preprocessing
    letter(H, "A")
    r1t, r1h = H - 0.08, 1.2
    r1b = r1t - r1h
    rbox(0.25, r1b, 1.3, r1h, "grey")
    ax.text(0.9, r1t - 0.07, "1. Recording", ha="center", va="top", fontsize=FT, weight="bold")
    ax.plot(np.linspace(0.38, 1.42, wave.size), r1b + 0.66 + 0.24 * wave / np.abs(wave).max(), color="#1F4E79",
            lw=0.35, zorder=3)
    ax.text(0.9, r1b + 0.22, "(variable length\nand sample rate)", ha="center", va="center", fontsize=FS,
            linespacing=1.15)
    yc1 = r1b + r1h / 2
    arr(1.55, yc1, 1.7, yc1)
    tbox(1.7, yc1 - 0.31, 1.12, 0.62, "Convert to mono\nand resample\nto 16 kHz", "peach")
    arr(2.82, yc1, 2.97, yc1)
    rbox(2.97, r1b, 4.28, r1h, "lav")
    ax.text(3.05, r1t - 0.07, "2. Controlled preprocessing", ha="left", va="top", fontsize=FT, weight="bold")
    sb_t, sb_b = r1t - 0.32, r1b + 0.07
    for x, w, head, items in ((3.07, 2.12, "Per recording", ["Peak-normalise", "Trim silence",
                                                             "6 s windows (10% overlap)", "Repeat-pad if shorter"]),
                              (5.27, 1.9, "Per window", ["Peak-normalise", "Remove DC offset",
                                                         "Add −60 dBFS dither"])):
        rbox(x, sb_b, w, sb_t - sb_b, "lavbox", r=0.04, z=2)
        ax.text(x + 0.08, sb_t - 0.05, head, ha="left", va="top", fontsize=FS, weight="bold", zorder=3)
        ax.text(x + 0.1, sb_t - 0.21, "\n".join("•  " + s for s in items), ha="left", va="top", fontsize=FS,
                linespacing=1.2, zorder=3)
    y_bus = r1b - 0.14
    line([3.3, 3.3, 0.12], [r1b, y_bus, y_bus])

    # ---- panel A, the two branches (same layers, separate weights)
    bh, gap = 1.75, 0.1
    mt = y_bus - 0.12
    rows = [("mel", mt, "blue", "3. Mel-spectrogram branch", "(128 Mel bands, dB)", mel,
             ["Mel-spectrogram", f"3 {X} 128 {X} 188", f"(Mel + {DL} + {DL}{DL})"], 128, "Mel embedding (64)"),
            ("lfcc", mt - bh - gap, "pink", "3. LFCC branch", "(40 linear filters, log, DCT)", lfcc,
             ["LFCC", f"3 {X} 40 {X} 188", f"(LFCC + {DL} + {DL}{DL})"], 40, "LFCC embedding (64)")]
    bar_mid = {}
    line([0.12, 0.12], [y_bus, rows[1][1] - 0.78])
    for key, t, style, tb, tr, img, labels, F, emb in rows:
        b = t - bh
        frame(0.25, b, 5.75, bh, *(("#EDF5FC", "#4A90C8") if key == "mel" else ("#FDEFF0", "#D9666F")))
        title(0.33, t - 0.06, tb, tr)
        yc = t - 0.78
        arr(0.12, yc, 0.33, yc)
        ext = (0.33, 1.23, yc - 0.25, yc + 0.25)
        if key == "mel":  # dB relative to the window maximum (power_to_db, ref=max, top_db=80)
            ax.imshow(img, origin="lower", aspect="auto", extent=ext, cmap="viridis", vmin=-80, vmax=0, zorder=2,
                      interpolation="nearest")
        else:  # all 40 coefficients; colour range from c1-c39 (c0, the log energy, saturates)
            lo, hi = np.percentile(img[1:], [1, 99])
            ax.imshow(img, origin="lower", aspect="auto", extent=ext, cmap="turbo", vmin=lo, vmax=hi, zorder=2,
                      interpolation="nearest")
        ax.add_patch(plt.Rectangle((0.33, yc - 0.25), 0.9, 0.5, fill=False, ec="#555555", lw=0.6, zorder=3))
        ax.text(0.78, yc - 0.31, labels[0], ha="center", va="top", fontsize=FS, weight="bold")
        ax.text(0.78, yc - 0.445, "\n".join(labels[1:]), ha="center", va="top", fontsize=FS, linespacing=1.15)
        arr(1.23, yc, 1.31, yc)
        tbox(1.31, yc - 0.25, 0.62, 0.5, "z-score\n(training\nstatistics)", "yellow")
        arr(1.93, yc, 2.0, yc)
        ex0, ex1 = 2.0, 5.66
        frame(ex0, b + 0.1, ex1 - ex0, t - 0.32 - (b + 0.1), "none", "#8C96A3", z=1, lw=0.8, dash=(3, 2), r=0.05)
        title(ex0 + 0.06, t - 0.36, "4. Feature encoder", "(same layers, separate weights)", fs=FS)
        Fh = F // 2
        blocks = [(0.74, f"SE-ResBlock\n3 {AR} 32", "blue", f"32{X}{F}{X}188"),
                  (0.56, f"Max-pool\n2 {X} 2", "pink", f"32{X}{Fh}{X}94"),
                  (0.74, f"SE-ResBlock\n32 {AR} 64", "blue", f"64{X}{Fh}{X}94"),
                  (0.6, "Average\npool over\nfrequency", "green", f"64{X}94"),
                  (0.6, "Temporal\nattention\npooling", "yellow", "64")]
        x = ex0 + 0.05
        for i, (w, txt, st, shape) in enumerate(blocks):
            tbox(x, yc - 0.25, w, 0.5, txt, st)
            ax.text(x + w / 2, yc - 0.37, shape, ha="center", va="top", fontsize=FS)
            if i < len(blocks) - 1:
                arr(x + w, yc, x + w + 0.08, yc)
            x += w + 0.08
        arr(x - 0.08, yc, 5.74, yc)
        ax.add_patch(FancyBboxPatch((5.74, yc - 0.6), 0.18, 1.2, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc="#9CC7EE" if key == "mel" else "#F4A9B0", ec=PAL[style][1], lw=0.8, zorder=2))
        ax.text(5.83, yc, emb, rotation=90, ha="center", va="center", fontsize=FS, zorder=3)
        bar_mid[key] = yc

    # ---- panel A, fusion and classifier, window logit, recording score
    cx0, cw = 6.1, 1.15
    ctop, fh = mt, 1.9
    rbox(cx0, ctop - fh, cw, fh, "fuse")
    ax.text(cx0 + cw / 2, ctop - 0.06, "5. Fusion and\nclassifier", ha="center", va="top", fontsize=FT,
            weight="bold", linespacing=1.1)
    y = ctop - 0.42
    spans = []
    for i, (txt, h) in enumerate([("Concatenate\n64 + 64 = 128", 0.36), ("FC 128", 0.2), ("ReLU", 0.2),
                                  ("Dropout 0.3", 0.2), ("FC 1", 0.2)]):
        tbox(cx0 + 0.07, y - h, cw - 0.14, h, txt, "fusebox")
        spans.append((y, y - h))
        if i < 4:
            arr(cx0 + cw / 2, y - h, cx0 + cw / 2, y - h - 0.065)
        y -= h + 0.065
    cy = spans[0][0] - 0.18
    for key in ("mel", "lfcc"):
        arr(5.92, bar_mid[key], cx0 + 0.07, cy + (0.06 if key == "mel" else -0.06))
    xl, ylog = cx0 + 0.92, ctop - fh - 0.2
    arr(xl, spans[-1][1], xl, ylog + 0.08)
    circle(xl, ylog)
    ax.text(xl - 0.13, ylog, "Window logit", ha="right", va="center", fontsize=FS)
    st = ylog - 0.2
    sh = st - (rows[1][1] - bh)
    rbox(cx0, st - sh, cw, sh, "score")
    ax.text(cx0 + 0.06, st - 0.05, "6. Recording\nscore", ha="left", va="top", fontsize=FT, weight="bold",
            linespacing=1.1)
    arr(xl, ylog - 0.08, xl, st - 0.42)
    tbox(cx0 + 0.07, st - 0.74, cw - 0.14, 0.32, "Mean of\nwindow logits", "scorebox")
    xr = cx0 + 0.18
    arr(xr, st - 0.74, xr, st - 0.92)
    circle(xr, st - 1.0)
    ax.text(xr + 0.13, st - 1.0, "Recording\nscore (positive\n= synthetic)", ha="left", va="center", fontsize=FS,
            linespacing=1.1)

    # ---- panel B: SE-ResBlock and SE module
    pBt, pBh = rows[1][1] - bh - 0.2, 1.85
    letter(pBt + 0.02, "B")
    frame(0.25, pBt - pBh, 7.0, pBh, "#F1F7FC", "#4A90C8")
    title(0.35, pBt - 0.06, "SE-ResBlock", f"(C {AR} C{PR} channels)")
    yb = pBt - 0.5
    ax.text(0.72, yb, f"Input\nC {X} F {X} T", ha="center", va="center", fontsize=FS, linespacing=1.2)
    x = 1.2
    arr(1.08, yb, x, yb)
    for w, txt, sty in [(0.52, f"Conv\n3 {X} 3", "blue"), (0.36, "BN", "bn"), (0.42, "ReLU", "pink"),
                        (0.52, f"Conv\n3 {X} 3", "blue"), (0.36, "BN", "bn"), (0.36, "SE", "yellow")]:
        tbox(x, yb - 0.2, w, 0.4, txt, sty)
        arr(x + w, yb, x + w + 0.09, yb)
        x += w + 0.09
    xp = x + 0.09
    circle(xp, yb, "+")
    arr(xp + 0.09, yb, xp + 0.2, yb)
    tbox(xp + 0.2, yb - 0.2, 0.42, 0.4, "ReLU", "pink")
    arr(xp + 0.62, yb, xp + 0.74, yb)
    ax.text(xp + 0.78, yb, f"Output\nC{PR} {X} F {X} T", ha="left", va="center", fontsize=FS, linespacing=1.2)
    ys = yb - 0.4
    line([1.13, 1.13, 1.55], [yb, ys, ys])
    tbox(1.55, ys - 0.14, 2.3, 0.28, f"Shortcut: identity, or 1 {X} 1 Conv + BN", "bn")
    line([3.85, xp], [ys, ys])
    arr(xp, ys, xp, yb - 0.09)
    se_t, se_b = pBt - 1.1, pBt - pBh + 0.08
    frame(0.4, se_b, 6.7, se_t - se_b, "#FFF9EB", "#D9A12B", z=1, lw=0.8, dash=(3, 2), r=0.05)
    title(0.5, se_t - 0.05, "Squeeze-and-excitation (SE)", f"bottleneck max(C{PR}/16, 4) = 4 units", fs=FS)
    yse = (se_t + se_b) / 2 - 0.08
    x = 0.55
    sechain = [(0.95, f"Global average\npool (C{PR})", "lavbox"), (0.55, f"FC\nC{PR} {AR} 4", "lavbox"),
               (0.42, "ReLU", "pink"), (0.55, f"FC\n4 {AR} C{PR}", "lavbox"), (0.6, "Sigmoid", "yellow"),
               (0.7, "Scale\nchannels", "beige")]
    for i, (w, txt, sty) in enumerate(sechain):
        tbox(x, yse - 0.19, w, 0.38, txt, sty, z=3)
        if i < len(sechain) - 1:
            arr(x + w, yse, x + w + 0.1, yse)
        x += w + 0.1

    # ---- panel C: temporal attention pooling
    pCt, pCh = pBt - pBh - 0.14, 0.95
    letter(pCt + 0.02, "C")
    frame(0.25, pCt - pCh, 7.0, pCh, "#F1F9EF", "#5AA35A")
    title(0.35, pCt - 0.06, "Temporal attention pooling", f"(input: 64 {X} T{PR})")
    yc_ = pCt - 0.52
    ax.text(0.62, yc_, f"Input\n64 {X} T{PR}", ha="center", va="center", fontsize=FS, linespacing=1.2)
    x = 1.1
    arr(0.92, yc_, x, yc_)
    cchain = [(0.85, f"1 {X} 1 Conv1d\n64 {AR} 64", "blue"), (0.42, "ReLU", "pink"),
              (0.85, f"1 {X} 1 Conv1d\n64 {AR} 1", "blue"), (0.72, "Softmax\nover time", "yellow")]
    for i, (w, txt, sty) in enumerate(cchain):
        tbox(x, yc_ - 0.2, w, 0.4, txt, sty)
        if i < len(cchain) - 1:
            arr(x + w, yc_, x + w + 0.1, yc_)
        x += w + 0.1
    x -= 0.1
    xm = x + 1.25
    arr(x, yc_, xm - 0.09, yc_)
    ax.text((x + xm) / 2, yc_ + 0.07, "weights " + r"$w_t$" + f", 1 {X} T{PR}", ha="center", va="bottom",
            fontsize=FS)
    circle(xm, yc_, X)
    ysk = yc_ - 0.33
    line([1.0, 1.0, xm], [yc_, ysk, ysk])
    arr(xm, ysk, xm, yc_ - 0.09)
    ax.text(xm - 0.08, ysk + 0.05, "features " + r"$\mathbf{h}_t$" + f", 64 {X} T{PR}", ha="right", va="bottom",
            fontsize=FS)
    arr(xm + 0.09, yc_, xm + 0.33, yc_)
    ax.text(xm + 0.38, yc_, "Output " + r"$\mathbf{f} = \Sigma_t\, w_t\, \mathbf{h}_t$", ha="left", va="center",
            fontsize=FS)
    ax.text(xm + 0.38, yc_ - 0.17, "(64 values)", ha="left", va="center", fontsize=FS)
    save(fig, "Fig1", "Architecture and preprocessing", "real",
         ["src/bscan/models/bscan.py", "src/bscan/audio.py", "src/bscan/features.py",
          "results/figures/fig1_example_window.npz"])


def fig_corpus_cues():
    d = pd.read_csv(ROOT / "metadata/recording_descriptors.csv")
    m = pd.read_csv(ROOT / "metadata/unified_metadata.csv", usecols=["audio_path", "dataset_id", "label"], low_memory=False)
    d = d.merge(m, on="audio_path")
    w = pd.read_csv(ROOT / "results/phase3/window_descriptors.csv.gz", usecols=["scheme", "audio_path", "dc"])
    w = w[w.scheme == "trim_repeat"].groupby("audio_path").dc.mean().rename("dc_win").reset_index()
    d = d.merge(w, on="audio_path", how="left")
    groups = [("BF-SUST", 0), ("BF-SUST", 1), ("BF-MOZ", 0), ("BF-MOZ", 1), ("BF-NEWS", 0), ("MEN", 0), ("MEN", 1)]
    # BF-NEWS is labelled bona fide by folder name only (unverified): grey
    names = [f"{g}\n{'unverified' if g == 'BF-NEWS' else ('synthetic' if l else 'bona fide')}" for g, l in groups]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.4))
    for ax, col, lab in ((axes[0], "duration_sec", "Duration (s)"), (axes[1], "trailing_silence_sec", "Trailing silence (s)"),
                         (axes[2], "dc_win", "DC offset (full scale)")):
        data = [d[(d.dataset_id == g) & (d.label == l)][col].dropna().to_numpy() for g, l in groups]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6, medianprops={"color": "black", "lw": 1})
        for patch, (g, l) in zip(bp["boxes"], groups):
            patch.set_facecolor("#999999" if g == "BF-NEWS" else ("#D55E00" if l else "#0072B2"))
            patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(names, rotation=90)
        ax.set_ylabel(lab)
    axes[0].set_ylim(0, 16)
    fig.tight_layout()
    label_panels(axes)
    save(fig, "Fig2", "Recording-level cues by corpus and class", "real",
         ["metadata/recording_descriptors.csv", "results/phase3/window_descriptors.csv.gz"])


def fig_shortcuts():
    r = pd.read_csv(ROOT / "results/phase3/model_visible_shortcut_diagnostics.csv")
    r = r[(r.model == "hgb") & (r.features == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6), sharey=True)
    schemes = [("orig6s", "Original", "#000000"), ("trim_repeat", "Trim + repeat-pad", "#E69F00"),
               ("controlled", "Controlled", "#0072B2")]
    for ax, proto, splits in ((axes[0], "P1", ["internal_test", "external_BF-MOZ", "external_MEN"]),
                              (axes[1], "P2", ["internal_test", "external_BF-SUST", "external_BF-MOZ"])):
        x = np.arange(len(splits))
        for i, (sc, lab, col) in enumerate(schemes):
            v = [float(r[(r.protocol == proto) & (r.scheme == sc) & (r.test_split == s)].auc_file.iloc[0]) for s in splits]
            ax.bar(x + (i - 1) * 0.26, v, 0.26, label=lab, color=col, alpha=0.8)
        ax.axhline(0.5, ls="--", lw=0.8, color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels([SPLITNAME[(proto, s)].replace(" (", "\n(") for s in splits])
        ax.set_title(f"{proto}: trained on {'BF-SUST' if proto == 'P1' else 'MEN'}")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("AUC of coarse window statistics")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    label_panels(axes)
    save(fig, "Fig3", "Label information available from coarse window statistics", "real",
         ["results/phase3/model_visible_shortcut_diagnostics.csv"])


# ----------------------------------------------------------------------------- result figures
def fig_roc():
    models = ["bscan_controlled", "lcnn_lfcc", "lfcc_gmm_controlled", "stats_hgb_controlled"]
    splits = ["internal_test", "external_BF-MOZ", "external_MEN"]
    tiers = {"internal_test": "Tier 1: internal test (BF-SUST)", "external_BF-MOZ": "Tier 2: independent (BF-MOZ)",
             "external_MEN": "Tier 3: cross-dataset (MEN)"}
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), sharey=True)
    sts, srcs = [], []
    for ax, sp in zip(axes, splits):
        for mdl in models:
            name = f"{mdl}__P1__seed42"
            st, path = resolve(name)
            sts.append(st)
            if path is None or not (path / f"predictions_{sp}.csv").exists():
                continue
            f = pd.read_csv(path / f"predictions_{sp}.csv")
            fpr, tpr, _ = roc_curve(f.label, f.score)
            ax.plot(fpr, tpr, color=C[mdl], lw=1.1,
                    label=f"{roc_auc_score(f.label, f.score):.2f}" + ("†" if st != "real" else ""))
            srcs.append(str(path / f"predictions_{sp}.csv"))
        ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="grey")
        ax.set_title(tiers[sp], fontsize=8)
        ax.set_xlabel("False positive rate")
        ax.legend(title="AUC", frameon=False, loc="lower right", handlelength=1.2, title_fontsize=8)
    axes[0].set_ylabel("True positive rate")
    fig.legend([Line2D([0], [0], color=C[m], lw=1.1) for m in models], [LABEL[m] for m in models],
               loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    label_panels(axes)
    save(fig, "Fig4", "ROC curves under P1", combine(sts), srcs)


def fig_cross():
    s = pd.read_csv(ROOT / "results/stats/cross_corpus_auc.csv")
    if s.empty:
        return skip("Fig7", "Cross-corpus AUC matrix", "results/stats/cross_corpus_auc.csv is empty")
    rows = [f"{LABEL[r.model]} ← {r.train_corpus}" for r in s.itertuples()]
    mat = s[["test_BF-SUST", "test_BF-MOZ", "test_MEN"]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 0.4 * len(rows) + 1.2))
    im = ax.imshow(mat, cmap="RdBu", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] == mat[i, j]:
                ax.text(j, i, f"{mat[i, j]:.2f}" + ("†" if s.status.iloc[i] != "real" else ""), ha="center", va="center",
                        fontsize=8, color="white" if abs(mat[i, j] - 0.5) > 0.3 else "black")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["test: BF-SUST", "test: BF-MOZ", "test: MEN"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    fig.colorbar(im, ax=ax, fraction=0.04, label="AUC")
    fig.tight_layout()
    save(fig, "Fig7", "Cross-corpus AUC matrix", combine(list(s.status)), ["results/stats/cross_corpus_auc.csv"])


def fig_repr_ablation():
    s = pd.read_csv(ROOT / "results/stats/multiseed_summary.csv")
    groups = [("Representation", ["mel_only", "lfcc_only", "mfcc_only", "mel_mfcc", "bscan_controlled"]),
              ("Ablation", ["dual_plain", "dual_no_se", "dual_no_tattn", "bscan_controlled"])]
    splits = ["internal_test", "external_BF-MOZ", "external_MEN"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=True)
    sts = []
    for ax, (title, models) in zip(axes, groups):
        x = np.arange(len(splits))
        wdt = 0.8 / len(models)
        for i, mdl in enumerate(models):
            sub = s[(s.model == mdl) & (s.protocol == "P1")].set_index("split")
            v = [sub.auc_mean.get(sp, np.nan) for sp in splits]
            e = [sub.auc_sd.get(sp, np.nan) for sp in splits]
            sts += list(sub.status)
            ax.bar(x + (i - (len(models) - 1) / 2) * wdt, v, wdt, yerr=np.nan_to_num(e), capsize=1.5,
                   color=C.get(mdl, "#999999"), label=LABEL[mdl], alpha=0.85)
        ax.axhline(0.5, ls="--", lw=0.8, color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels([SPLITNAME[("P1", sp)].replace(" (", "\n(") for sp in splits])
        ax.set_title(title + " (P1)")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("AUC (mean ± SD over seeds)")
    fig.tight_layout()
    label_panels(axes)
    save(fig, "Fig9", "Representation study and ablation", combine(sts), ["results/stats/multiseed_summary.csv"])


def _analysis_frame(kind: str) -> tuple[pd.DataFrame, list[str]]:
    reg = load_registry()
    frames, sts = [], []
    for rn in reg["analyses"][kind]["runs"]:
        real = ROOT / "results/runs" / rn / f"{kind}_summary.csv"
        ph = ROOT / "results/runs_PLACEHOLDER" / rn / f"{kind}_summary.csv"
        if real.exists():
            f, st = pd.read_csv(real), "real"
        elif ph.exists():
            f, st = pd.read_csv(ph), "placeholder"
        else:
            continue
        f["run"], f["status"] = rn, st
        frames.append(f)
        sts.append(st)
    return (pd.concat(frames) if frames else pd.DataFrame()), sts


def fig_robustness_probes():
    rb, s1 = _analysis_frame("robustness")
    pr, s2 = _analysis_frame("probes")
    if rb.empty and pr.empty:
        return skip("Fig8", "Robustness and counterfactual probes", "no robustness_summary.csv or probes_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.4))
    if not rb.empty:
        ax = axes[0]
        # conditions in the order of the text and Table 13, top to bottom; models offset so markers do not hide
        conds = [c for c in COND if c in set(rb.condition) and c != "clean" and c in ROBUSTNESS_CODES][::-1]
        runs_a = list(dict.fromkeys(rb.run))
        for k, (rn, g) in enumerate((r, rb[(rb.run == r) & (rb.split == "internal_test")]) for r in runs_a):
            clean = float(g[g.condition == "clean"].auc.iloc[0])
            d = [float(g[g.condition == c].auc.iloc[0]) - clean for c in conds]
            ax.plot(d, np.arange(len(conds)) + (k - (len(runs_a) - 1) / 2) * 0.22, "o", ms=3,
                    color=C.get(rn.split("__")[0], "#999"), label=LABEL[rn.split("__")[0]])
        ax.axvline(0, lw=0.6, color="grey")
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([COND.get(c, c) for c in conds])
        ax.set_xlabel("\u0394AUC vs unperturbed (BF-SUST internal test)")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    if not pr.empty:
        ax = axes[1]
        probes = [c for c in pr.condition.unique() if c != "clean"][::-1]
        runs = list(pr.run.unique())
        wdt = 0.8 / max(1, len(runs))
        for i, rn in enumerate(runs):
            g = pr[(pr.run == rn) & (pr.split == "internal_test")].set_index("condition")
            # flips among the recordings whose model input the probe changes, as in Table 14
            chg = [float(g.share_input_changed.get(p, 0.0)) for p in probes]
            v = [float(g.decision_flip_rate_changed.get(p, np.nan)) * 100 if c > 0 else 0.0 for p, c in zip(probes, chg)]
            ys = np.arange(len(probes)) + (i - (len(runs) - 1) / 2) * wdt
            ax.barh(ys, v, wdt, color=C.get(rn.split("__")[0], "#999"), label=LABEL[rn.split("__")[0]])
            for y, c, val in zip(ys, chg, v):
                if c < 0.5:
                    ax.text((val if val == val else 0) + 1, y, "input unchanged" if c == 0 else f"input changed: {c:.0%}",
                            va="center", fontsize=8)
        ax.set_yticks(range(len(probes)))
        ax.set_yticklabels([COND.get(c, c) for c in probes])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Decisions flipped among changed inputs (%)")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    label_panels(axes)
    save(fig, "Fig8", "Robustness and counterfactual probes", combine(s1 + s2), ["robustness_summary.csv", "probes_summary.csv"])


def fig_errors_calibration():
    ge = ROOT / "results/analysis/error_groups.csv"
    if not ge.exists():
        return skip("Fig6", "Systematic error analysis of BSCAN on MEN", "results/analysis/error_groups.csv missing")
    g = pd.read_csv(ge)
    g = g[g.run == "bscan_controlled__P1__seed42"]
    if g.empty:
        return skip("Fig6", "Systematic error analysis of BSCAN on MEN", "no rows for bscan_controlled__P1__seed42")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), sharey=True)
    level_text = {"<0.05s": "<0.05 s", "0.05-0.3s": "0.05\u20130.3 s", ">0.3s": ">0.3 s"}
    for ax, grouping, title in ((axes[0], "duration_tertile", "Duration"), (axes[1], "snr_proxy_tertile", "SNR proxy"),
                                (axes[2], "trailing_silence", "Trailing silence")):
        sub = g[(g.grouping == grouping) & (g.split == "external_MEN")]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub.fnr * 100, 0.4, color="#D55E00", label="FNR (synthetic missed)")
        ax.bar(x + 0.2, sub.fpr * 100, 0.4, color="#0072B2", label="FPR (bona fide flagged)")
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(sub)))
        # under each bin: the numbers of synthetic and bona fide recordings in it (some bins hold only a few)
        ax.set_xticklabels([f"{level_text.get(lv, lv)}\n{int(ns):,} / {int(nb):,}"
                            for lv, ns, nb in zip(sub.level, sub.n_spoof, sub.n_bonafide)])
        ax.set_title(f"{title} (MEN)")
    axes[0].set_ylabel("Error rate (%)")
    axes[1].set_xlabel("synthetic / bona fide recordings per bin")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    label_panels(axes)
    save(fig, "Fig6", "Systematic error analysis of BSCAN on MEN", combine(list(g.status.unique())),
         ["results/analysis/error_groups.csv"])


def fig_training_curves():
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    st, path = resolve("bscan_controlled__P1__seed42")
    if path is None or not (path / "train_log.csv").exists():
        plt.close(fig)
        return skip("S1_Fig", "Training and validation loss of BSCAN (P1, seed 42)",
                    "train_log.csv of bscan_controlled__P1__seed42 missing")
    t = pd.read_csv(path / "train_log.csv")
    ax.plot(t.epoch, t.train_loss, label="train", color="#0072B2")
    ax.plot(t.epoch, t.val_loss, label="validation", color="#D55E00")
    best = json.loads((path / "metrics.json").read_text()).get("best_epoch")
    if best:
        ax.axvline(best, ls=":", lw=0.8, color="grey")  # retained checkpoint
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "S1_Fig", "Training and validation loss of BSCAN (P1, seed 42)", st, [str(path / "train_log.csv")])


def fig_confusion():
    """Confusion matrices of BSCAN (P1, seed 42) at the validation EER threshold, one per evaluation tier."""
    st, path = resolve("bscan_controlled__P1__seed42")
    if path is None:
        return skip("Fig5", "Confusion matrices of BSCAN across the evaluation tiers",
                    "run bscan_controlled__P1__seed42 missing")
    thr = json.loads((path / "metrics.json").read_text()).get("threshold_logit")
    splits = [("internal_test", "Tier 1: internal test (BF-SUST)"), ("external_BF-MOZ", "Tier 2: independent (BF-MOZ)"),
              ("external_MEN", "Tier 3: cross-dataset (MEN)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    srcs = []
    for ax, (sp, title) in zip(axes, splits):
        f = path / f"predictions_{sp}.csv"
        if thr is None or not f.exists():
            ax.axis("off")
            continue
        d = pd.read_csv(f)
        pred = (d.score >= thr).astype(int)
        cm = np.array([[int(((d.label == 0) & (pred == 0)).sum()), int(((d.label == 0) & (pred == 1)).sum())],
                       [int(((d.label == 1) & (pred == 0)).sum()), int(((d.label == 1) & (pred == 1)).sum())]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / max(1, cm[i].sum()):.1%})", ha="center", va="center",
                        fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["bona fide", "synthetic"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["bona fide", "synthetic"])
        ax.set_xlabel("Decision")
        ax.set_title(title, fontsize=8)
        srcs.append(str(f))
    axes[0].set_ylabel("True class")
    label_panels(axes)
    fig.tight_layout()
    save(fig, "Fig5", "Confusion matrices of BSCAN across the evaluation tiers", st, srcs)


# file name of every generated figure (PLOS/figures/<name>.pdf), = its number in the manuscript; the names of an
# earlier numbering are listed so that their stale files are removed as well
FIGURES = ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6", "Fig7", "Fig8", "Fig9", "S1_Fig"]
OLD_NAMES = ["Fig_confusion"]


def main() -> None:
    # remove the figure files of the previous run (and any listed in the previous manifest) so that only
    # figures regenerated now exist; other files in PLOS/figures/ are left alone
    old = OUT / "figure_manifest.csv"
    names = set(FIGURES) | set(OLD_NAMES) | (set(pd.read_csv(old).figure.astype(str)) if old.exists() else set())
    for n in names:
        (FIG / f"{n}.pdf").unlink(missing_ok=True)
        (SUBMISSION / f"{n}.tif").unlink(missing_ok=True)
    for f in (fig_architecture, fig_corpus_cues, fig_shortcuts, fig_roc, fig_cross, fig_repr_ablation,
              fig_robustness_probes, fig_errors_calibration, fig_training_curves, fig_confusion):
        try:
            f()
        except Exception as exc:  # noqa: BLE001
            MANIFEST.append({"figure": f.__name__, "status": "error", "caption": str(exc), "sources": ""})
            print(f"[figure error] {f.__name__}: {exc}")
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(MANIFEST)
    m.to_csv(OUT / "figure_manifest.csv", index=False)
    print(m[["figure", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
