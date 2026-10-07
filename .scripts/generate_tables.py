"""Build every manuscript table, the numeric LaTeX macros and results/master_results.csv from result files.

Nothing is typed by hand: every number comes from a CSV/JSON produced by the pipeline.
Status of each number is carried through:
    real          -> printed normally
    placeholder   -> printed as \\ph{value} (red, with a dagger) and flagged in the table notes (\\phnote)
    missing       -> printed as \\missing{}
Outputs:
    PLOS/generated/tables/*.tex             LaTeX tables included by PLOS/BSCAN_PLOS_ONE.tex
    PLOS/generated/results_macros.tex       \\res{key} numeric macros and \\cmp{key} comparison words
    results/tables/*.csv                    machine-readable twin of every table
    results/master_results.csv              long-format record of every metric of every run
Table layout follows the PLOS LaTeX template (v3.8): bold title in \\caption above the tabular, legend and
notes below it in a flushleft block, wide tables in adjustwidth{-0.5in}{-0.5in} (never shrunk to fit; the
template body is 6.5 in wide after its \\newgeometry with 1 in margins, so -2.25in would leave the page).

Usage:
    python .scripts/generate_tables.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bscan.registry import expected_runs, load_registry, resolve  # noqa: E402

MANUSCRIPT = ROOT / "PLOS"
GEN = MANUSCRIPT / "generated"
TAB = GEN / "tables"
CSV = ROOT / "results" / "tables"
PAPER_TEX = MANUSCRIPT / "BSCAN_PLOS_ONE.tex"
# tables the manuscript \\inputs (stale ones are removed); None = write every table (no manuscript yet, or a
# flattened single-file manuscript that no longer \\inputs them)
USED_TABLES = (set(re.findall(r"\\input\{generated/tables/([^}]+)\}", PAPER_TEX.read_text())) or None) if PAPER_TEX.exists() else None
SPLIT_LABEL = {"internal_test": "BF-SUST (internal test)", "external_BF-MOZ": "BF-MOZ", "external_MEN": "MEN",
               "external_BF-SUST": "BF-SUST", "external_BF-NEWS": "BF-NEWS (flagged)", "validation": "validation"}
P1S = ["internal_test", "external_BF-MOZ", "external_MEN"]
P2S = ["internal_test", "external_BF-SUST", "external_BF-MOZ"]
PLACEHOLDER_USED: set[str] = set()


def split_label(proto: str, sp: str, short: bool = True) -> str:
    """Name of the corpus seen in a split (P1 internal = BF-SUST, P2 internal = MEN)."""
    corpus = {("P1", "internal_test"): "BF-SUST", ("P2", "internal_test"): "MEN", ("P1", "validation"): "BF-SUST",
              ("P2", "validation"): "MEN"}.get((proto, sp)) or SPLIT_LABEL[sp].split(" (")[0]
    return corpus if short else corpus + (" (internal test)" if sp == "internal_test" else "")


# ----------------------------------------------------------------------------- formatting helpers
def fmt(v, status: str = "real", kind: str = "auc", sd=None) -> str:
    if status == "missing" or v is None or (isinstance(v, float) and np.isnan(v)) or (kind == "text" and v == ""):
        return r"\missing{}"
    if kind in ("eer_pct", "pct"):
        s = f"{100 * v:.1f}" + (f" $\\pm$ {100 * sd:.1f}" if sd is not None and sd == sd else "")
    elif kind == "int":
        s = f"{int(v):,}".replace(",", "{,}")
    elif kind == "ms":
        s = f"{v:.2f}" + (f" $\\pm$ {sd:.2f}" if sd is not None and sd == sd else "")
    elif kind == "rtf":
        s = f"{v:.4f}"
    elif kind == "p":
        s = "$<$0.001" if v < 0.001 else f"{v:.3f}"
    elif kind == "peq":
        s = "$p<0.001$" if v < 0.001 else f"$p={v:.3f}$"
    elif kind == "text":
        s = str(v)
    else:
        s = f"{v:.3f}" + (f" $\\pm$ {sd:.3f}" if sd is not None and sd == sd else "")
    if kind != "text":
        s = re.sub(r"(?<![\w$])-(?=0(?:\.0+)?(?![\d.]))", "", s)  # a value that rounds to zero has no sign (no "-0.000")
        s = re.sub(r"(?<![\w$])-(?=\d)", "$-$", s)  # typographic minus
    if status == "placeholder":
        PLACEHOLDER_USED.add(kind)
        return r"\ph{" + s + "}"
    return s


def write_table(name: str, caption: str, header: list[str], rows: list[list[str]], label: str, note: str = "",
                align: str | None = None, has_placeholder: bool = False, groups: list[tuple[str, int]] | None = None,
                wide: bool = False, colsep: float | None = None) -> None:
    """One table in the PLOS layout.

    caption  "Title. Legend": the first sentence becomes the bold title in \\caption (above the tabular); the
             legend goes below the tabular, followed by `note` and, if needed, the placeholder note
    groups   optional spanning header row [(text, n_columns), ...] above `header`
    wide     wrap in adjustwidth{-0.5in}{-0.5in} (7.5 in instead of the 6.5 in text column of the template body,
             0.5 in into each margin), as the template asks for tables wider than the text column
    colsep   optional \\tabcolsep in pt for this table only (the font size is never reduced)
    """
    TAB.mkdir(parents=True, exist_ok=True)
    if USED_TABLES is not None and name not in USED_TABLES:
        (TAB / f"{name}.tex").unlink(missing_ok=True)
        return
    align = align or ("l" + "c" * (len(header) - 1))
    m = re.search(r"\.\s+", caption)
    title, legend = (caption[:m.start()], caption[m.end():].strip()) if m else (caption.rstrip("."), "")
    lines = [r"\begin{\tablefloat}[!ht]"]
    if wide:
        lines.append(r"\begin{adjustwidth}{-0.5in}{-0.5in}")
    lines += [r"\centering", r"\caption{{\bf " + title + ".}}", r"\label{" + label + "}", r"\tablesetup"]
    if colsep is not None:
        lines.append(r"\setlength{\tabcolsep}{" + f"{colsep:g}pt" + "}")
    lines += [r"\begin{tabular}{" + align + "}", r"\tabletop"]
    if groups:
        assert sum(n for _, n in groups) == len(header), name
        cells, rules, col = [], [], 1
        for text, n in groups:
            cells.append(text if n == 1 else r"\multicolumn{" + str(n) + "}{c}{" + text + "}")
            if text:
                rules.append(r"\cline{" + f"{col}-{col + n - 1}" + "}")
            col += n
        lines.append(" & ".join(cells) + r" \\ " + " ".join(rules))
    lines += [" & ".join(header) + r" \\", r"\tablemid"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\tablebottom", r"\end{tabular}"]
    notes = []
    if legend:
        notes.append(legend if legend.endswith(".") else legend + ".")
    if note:
        notes.append(note)
    if has_placeholder:
        notes.append(r"\phnote")
    if notes:
        lines.append(r"\begin{flushleft} " + " ".join(notes) + r" \end{flushleft}")
    if wide:
        lines.append(r"\end{adjustwidth}")
    lines.append(r"\end{\tablefloat}")
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")


def wrapcol(width: str) -> str:
    """Left-aligned column that wraps at `width` (array package, loaded by the PLOS template)."""
    return r">{\raggedright\arraybackslash}p{" + width + "}"


TABLE_LABEL = {"stats_hgb_controlled": "Recording statistics", "stats_hgb_orig6s": "Recording statistics (orig. prep.)",
               "bscan_orig6s": "BSCAN (orig. prep.)", "dual_plain": "Dual, no SE, no attention",
               "dual_no_tattn": "Dual, no attention", "ssl_xlsr_probe": "XLS-R (frozen) + logistic regression"}
# readable names of the robustness conditions in tab_robustness (the codes stay in the CSV twin)
COND_LABEL = {"clean": "Clean", "noise_snr20": "White noise, 20 dB SNR", "noise_snr10": "White noise, 10 dB SNR",
              "noise_snr5": "White noise, 5 dB SNR", "mp3_64k": "MP3, 64 kbit/s", "mp3_32k": "MP3, 32 kbit/s",
              "aac_32k": "AAC, 32 kbit/s", "telephone": "Telephone channel", "clip_+6dB": "Clipping (+6 dB gain)",
              "speed_0.95": "Speed $\\times$0.95", "speed_1.05": "Speed $\\times$1.05", "pitch_+1st": "Pitch +1 semitone",
              "pitch_-1st": "Pitch $-$1 semitone", "reverb_rt0.3": "Reverberation, RT60 0.3 s",
              "reverb_rt0.6": "Reverberation, RT60 0.6 s"}
# readable names and order of the splits in tab_splits (the codes stay in the CSV twin)
SPLIT_ORDER = ["train", "validation", "internal_test", "external_BF-SUST", "external_BF-MOZ", "external_MEN",
               "external_BF-NEWS"]
SPLIT_TEXT = {"train": "Training", "validation": "Validation", "internal_test": "Internal test",
              "external_BF-SUST": "External test (BF-SUST)", "external_BF-MOZ": "External test (BF-MOZ)",
              "external_MEN": "External test (MEN)", "external_BF-NEWS": "External (BF-NEWS)"}
PROBE_LABEL = {"append_silence_0.44": "append 0.44 s silence", "trim_trailing": "trim trailing silence",
               "add_dc_neg": "add DC offset", "remove_dc": "remove DC offset", "dither_-60dBFS": "add $-$60 dBFS noise"}


def tlabel(reg: dict, model: str) -> str:
    return TABLE_LABEL.get(model, reg["models"][model]["label"])


def any_ph(rows: list[list[str]]) -> bool:
    return any(r"\ph{" in c for r in rows for c in r)


def metrics_of(name: str):
    st, path = resolve(name)
    if path is None:
        return st, None
    return st, json.loads((path / "metrics.json").read_text())


def split_val(name: str, split: str, key: str):
    st, m = metrics_of(name)
    if m is None or split not in m["splits"]:
        return "missing", None
    return st, m["splits"][split].get(key)


# ----------------------------------------------------------------------------- tables
def t_corpora() -> pd.DataFrame:
    c = pd.read_csv(ROOT / "results/phase1/counts.csv")
    f = pd.read_csv(ROOT / "results/phase1/format_stats.csv")
    d = pd.read_csv(ROOT / "results/phase1/duration_stats.csv")
    info = {"BF-SUST": ("BanglaFake (SUST TTS corpus)", "1 / 1", "VITS", "none stated"),
            "BF-MOZ": ("BanglaFake (Common Voice)", "5 / 1", "VITS", "none stated"),
            "BF-NEWS": ("BanglaFake (news), flagged", "unknown / --", "--", "none stated"),
            "MEN": ("Mendeley Bangla dataset (v4)", "75 / unknown", "undocumented", "CC BY 4.0")}
    rows, recs = [], []
    for ds in ["BF-SUST", "BF-MOZ", "BF-NEWS", "MEN"]:
        cb = c[(c.dataset_id == ds) & (c.label_name == "bonafide")]
        cs = c[(c.dataset_id == ds) & (c.label_name == "spoof")]
        nb, ns = (int(cb.files.iloc[0]) if len(cb) else 0), (int(cs.files.iloc[0]) if len(cs) else 0)
        hrs = float(c[c.dataset_id == ds].hours.sum())
        srb = f[(f.dataset_id == ds) & (f.label_name == "bonafide")].sample_rate.tolist()
        srs = f[(f.dataset_id == ds) & (f.label_name == "spoof")].sample_rate.tolist()
        sr = f"{srb[0] / 1000:g} / {srs[0] / 1000:g}" if srs else f"{srb[0] / 1000:g} / --"
        db = d[(d.dataset_id == ds) & (d.label_name == "bonafide")]["mean"].iloc[0]
        dsp = d[(d.dataset_id == ds) & (d.label_name == "spoof")]["mean"]
        dur = f"{db:.2f} / {dsp.iloc[0]:.2f}" if len(dsp) else f"{db:.2f} / --"
        name, spk, gen, lic = info[ds]
        rows.append([ds, fmt(nb, kind="int"), fmt(ns, kind="int") if ns else "0", f"{hrs:.2f}", sr, dur, spk, gen, lic])
        recs.append({"subset": ds, "source": name, "bonafide": nb, "spoof": ns, "hours": hrs, "sr_khz": sr,
                     "mean_dur_s": dur, "speakers": spk, "generator": gen, "licence": lic})
    write_table("tab_corpora", "Corpora and subsets. BF-SUST, BF-MOZ and BF-NEWS are the SUST, Common Voice and news "
                "subsets of BanglaFake; MEN is the Mendeley Bengali corpus (version 4). Counts, durations and sample "
                "rates (SR) were measured from the original archives",
                ["Subset", "Bona fide", "Synthetic", "Hours", "SR (kHz)", "Mean duration (s)", "Speakers", "Generator", "Licence"],
                rows, "tab:corpora", align="lrrrcccll", wide=True,
                note="Pairs of values: bona fide / synthetic. Speakers: bona fide speakers / synthetic voices. The number of "
                     "SUST bona fide speakers is not documented in the release (the BanglaFake paper lists seven bona fide "
                     "speakers in total, five of them from Common Voice). "
                     "MEN stores its synthetic recordings per speaker folder, but the number of distinct synthetic voices is not "
                     "documented. BF-NEWS is labelled bona fide by its folder name only; its labels could not be verified.")
    return pd.DataFrame(recs)


def t_splits() -> pd.DataFrame:
    recs, rows = [], []
    for proto in ("P1", "P2"):
        s = pd.read_csv(ROOT / f"results/phase3/split_counts_{proto}.csv")
        col = f"split_{proto}"
        groups = dict(tuple(s.groupby(col)))
        order = [sp for sp in SPLIT_ORDER if sp in groups] + sorted(sp for sp in groups if sp not in SPLIT_ORDER)
        for i, sp in enumerate(order):
            g = groups[sp]
            nb = int(g[g.label == 0].files.sum())
            ns = int(g[g.label == 1].files.sum())
            role = "training" if sp == "train" else "model/threshold selection" if sp == "validation" else "test"
            shown = "diagnostic only (no metrics)" if sp == "external_BF-NEWS" else role
            rows.append([proto if i == 0 else "", SPLIT_TEXT.get(sp, sp.replace("_", r"\_")),
                         ", ".join(sorted(g.dataset_id.unique())), fmt(nb, kind="int"), fmt(ns, kind="int"), shown])
            recs.append({"protocol": proto, "split": sp, "corpora": ",".join(sorted(g.dataset_id.unique())),
                         "bonafide": nb, "spoof": ns, "role": shown})
    write_table("tab_splits", "Evaluation protocols. P1 trains on BF-SUST with sentence-grouped 70/15/15 splits; P2 "
                "trains on MEN with set-disjoint (hence speaker-disjoint) 9/3/3 sets. Counts are recordings. BF-NEWS (labels "
                "unverified) is scored only as a diagnostic and excluded from every metric",
                ["Protocol", "Split", "Corpora", "Bona fide", "Synthetic", "Role"], rows, "tab:splits", align="lllrrl",
                wide=True)
    return pd.DataFrame(recs)


def t_shortcuts() -> pd.DataFrame:
    r = pd.read_csv(ROOT / "results/phase3/model_visible_shortcut_diagnostics.csv")
    r = r[(r.model == "hgb") & (r.features == "all")]
    rows, recs = [], []
    for scheme, lab in [("orig6s", "Original"), ("trim_repeat", "Trim + repeat-pad"), ("controlled", "Controlled")]:
        row = [lab]
        for proto, splits in (("P1", P1S), ("P2", P2S)):
            for sp in splits:
                v = r[(r.protocol == proto) & (r.scheme == scheme) & (r.test_split == sp)]["auc_file"]
                val = float(v.iloc[0]) if len(v) else np.nan
                row.append(fmt(val))
                recs.append({"scheme": scheme, "protocol": proto, "split": sp, "auc": val})
        rows.append(row)
    hdr = (["Preprocessing"] + [split_label("P1", s) + ("$^{\\ast}$" if s == "internal_test" else "") for s in P1S]
           + [split_label("P2", s) + ("$^{\\ast}$" if s == "internal_test" else "") for s in P2S])
    write_table("tab_shortcuts", "Label information available from coarse window statistics. Recording-level AUC of a "
                "gradient-boosting classifier trained on 11 coarse window descriptors (mainly recording-chain properties: "
                "padding, silence, level, DC offset, clipping, spectral balance) of the protocol's training split",
                hdr, rows, "tab:shortcuts", align="lcccccc", wide=True,
                groups=[("", 1), ("P1: trained on BF-SUST", 3), ("P2: trained on MEN", 3)],
                note="$^{\\ast}$ held-out split of the training corpus. Original, zero-padding preprocessing; Trim + "
                     "repeat-pad, trimming and repeat-padding only; Controlled, trimming, repeat-padding, DC removal and "
                     "dither. Recording score: mean window probability. The "
                     "recording-statistics baseline reported alongside the detectors is the same classifier scored like "
                     "every detector (mean window log-odds), so its AUCs can differ in the third decimal.")
    return pd.DataFrame(recs)


def model_row(summary: pd.DataFrame, model: str, proto: str, splits: list[str], label: str) -> tuple[list[str], list[dict]]:
    row, recs = [label], []
    for sp in splits:
        s = summary[(summary.model == model) & (summary.protocol == proto) & (summary.split == sp)]
        if s.empty:
            row += [r"\missing{}", r"\missing{}"]
            continue
        s = s.iloc[0]
        n = int(s["auc_n"])
        sd_auc = s["auc_sd"] if n > 1 else None
        sd_eer = s["eer_sd"] if n > 1 else None
        row.append(fmt(s["auc_mean"], s["status"], "auc", sd_auc))
        row.append(fmt(s["eer_mean"], s["status"], "eer_pct", sd_eer))
        recs.append({"model": model, "protocol": proto, "split": sp, "auc_mean": s["auc_mean"], "auc_sd": s["auc_sd"],
                     "eer_mean": s["eer_mean"], "eer_sd": s["eer_sd"], "n_seeds": n, "status": s["status"]})
    return row, recs


def t_models(summary: pd.DataFrame, reg: dict, name: str, models: list[str], caption: str, label: str,
             proto: str = "P1", extra_note: str = "", first_col: str = "l", seed42: tuple[str, ...] = (),
             labels: dict | None = None,
             note: str = "Mean $\\pm$ SD over seeds where more than one seed was run; otherwise seed 42.") -> pd.DataFrame:
    """One row per model (mean over its seeds); models in `seed42` get a second row with their seed-42 run, the run
    that the paired comparisons use.  `labels` overrides row names for this table only."""
    splits = P1S if proto == "P1" else P2S
    name_of = lambda m: (labels or {}).get(m, tlabel(reg, m))  # noqa: E731
    rows, recs = [], []
    for m in models:
        r, rc = model_row(summary, m, proto, splits, name_of(m))
        rows.append(r)
        recs += rc
        if m in seed42:
            st, met = metrics_of(f"{m}__{proto}__seed42")
            row = [f"{name_of(m)}, seed 42"]
            for sp in splits:
                v = (met or {}).get("splits", {}).get(sp)
                row += [fmt(v["auc"], st), fmt(v["eer"], st, "eer_pct")] if v else [r"\missing{}", r"\missing{}"]
                if v:
                    recs.append({"model": m, "protocol": proto, "split": sp, "auc_mean": v["auc"], "auc_sd": np.nan,
                                 "eer_mean": v["eer"], "eer_sd": np.nan, "n_seeds": 1, "status": st, "row": "seed 42"})
            rows.append(row)
    hdr = ["Model"] + ["AUC", "EER (\\%)"] * len(splits)
    groups = [("", 1)] + [(split_label(proto, sp, short=False), 2) for sp in splits]
    write_table(name, caption, hdr, rows, label, has_placeholder=any_ph(rows), groups=groups, wide=True,
                align=first_col + "c" * (len(hdr) - 1), note=note + extra_note)
    return pd.DataFrame(recs)


def t_tiers(summary: pd.DataFrame, reg: dict) -> pd.DataFrame:
    """Multi-tier summary of BSCAN (internal test, independent and cross-dataset tiers) under protocol P1."""
    tiers = [("internal_test", "\\textbf{Tier 1:} Internal test (BF-SUST, held out)"),
             ("external_BF-MOZ", "\\textbf{Tier 2:} Independent (BF-MOZ)"),
             ("external_MEN", "\\textbf{Tier 3:} Cross-dataset (MEN)")]
    st42, m42 = metrics_of("bscan_controlled__P1__seed42")
    rows, recs = [], []
    for sp, name in tiers:
        s = summary[(summary.model == "bscan_controlled") & (summary.protocol == "P1") & (summary.split == sp)]
        if s.empty:
            rows.append([name] + [r"\missing{}"] * 7)
            continue
        s = s.iloc[0]
        many = int(s["auc_n"]) > 1
        n = m42["splits"].get(sp, {}).get("n") if m42 else None
        row = [name, fmt(float(n), "real", "int") if n else r"\missing{}"]
        for mt, kind in (("auc", "auc"), ("eer", "eer_pct"), ("accuracy", "pct"), ("f1_macro", "auc"),
                         ("recall_spoof_sensitivity", "pct"), ("specificity_bonafide_recall", "pct")):
            row.append(fmt(s[f"{mt}_mean"], s["status"], kind, s[f"{mt}_sd"] if many else None))
        rows.append(row)
        recs.append({"tier": name, "split": sp, "n": n, "status": s["status"]})
    write_table("tab_tiers", "Performance of BSCAN across the evaluation tiers (protocol P1). Recording-level "
                "metrics; the decision threshold is the validation EER point and is never re-tuned on test data",
                ["Evaluation tier", "Recordings", "AUC", "EER (\\%)", "Accuracy (\\%)", "Macro F1", "Sensitivity (\\%)",
                 "Specificity (\\%)"], rows, "tab:tiers", align=wrapcol("1.15in") + "ccccccc", has_placeholder=any_ph(rows),
                wide=True, colsep=4,
                note="Mean $\\pm$ SD over three seeds (42, 123, 2026). Tier 1 is a held-out partition that played no part "
                     "in model or threshold selection.")
    return pd.DataFrame(recs)


def t_bootstrap(reg: dict) -> pd.DataFrame:
    """Cluster-bootstrap confidence intervals of BSCAN (seed 42) per test set."""
    st, m = metrics_of("bscan_controlled__P1__seed42")
    rows, recs = [], []
    for sp, name in (("internal_test", "Internal test (BF-SUST)"), ("external_BF-MOZ", "Independent (BF-MOZ)"),
                     ("external_MEN", "Cross-dataset (MEN)")):
        v = (m or {}).get("splits", {}).get(sp)
        if not v:
            rows.append([name] + [r"\missing{}"] * 5)
            continue
        ca, ce = v.get("auc_ci95") or [np.nan, np.nan], v.get("eer_ci95") or [np.nan, np.nan]
        rows.append([name, fmt(v.get("n_clusters", np.nan), "real", "int") if v.get("n_clusters") else r"\missing{}",
                     fmt(v["auc"], st), f"[{fmt(ca[0], st)}, {fmt(ca[1], st)}]",
                     fmt(v["eer"], st, "eer_pct"), f"[{fmt(ce[0], st, 'eer_pct')}, {fmt(ce[1], st, 'eer_pct')}]"])
        recs.append({"split": sp, "auc": v["auc"], "auc_ci": ca, "eer": v["eer"], "eer_ci": ce, "status": st})
    write_table("tab_bootstrap", "Cluster-bootstrap 95\\% confidence intervals of BSCAN (seed 42). P1; 2,000 resamples "
                "of sentence groups for BanglaFake and of speakers for MEN; the intervals describe the sampling of "
                "recordings for this one trained model, not the variation between training runs",
                ["Test set", "Clusters", "AUC", "95\\% CI (AUC)", "EER (\\%)", "95\\% CI (EER)"], rows, "tab:bootstrap",
                align="lccccc", has_placeholder=any_ph(rows), wide=True)
    return pd.DataFrame(recs)


def t_cross(reg: dict) -> pd.DataFrame:
    s = pd.read_csv(ROOT / "results/stats/cross_corpus_auc.csv")
    rows, recs = [], []
    for _, r in s.iterrows():
        row = [tlabel(reg, r["model"]), r["train_corpus"]]
        for c in ("BF-SUST", "BF-MOZ", "MEN"):
            v = r.get(f"test_{c}", np.nan)
            row.append(fmt(v, r["status"]) + (r"$^{\ast}$" if c == r["train_corpus"] else ""))
            recs.append({"model": r["model"], "train": r["train_corpus"], "test": c, "auc": v, "status": r["status"]})
        rows.append(row)
    write_table("tab_cross", "Cross-corpus matrix (recording-level AUC). Rows: model and training corpus; columns: test "
                "corpus", ["Model", "Trained on", "BF-SUST", "BF-MOZ", "MEN"], rows, "tab:cross",
                note="$^{\\ast}$ in-corpus test (held-out internal test split).", has_placeholder=any_ph(rows))
    return pd.DataFrame(recs)


def t_paired(reg: dict) -> None:
    """Paired comparisons, one table per protocol (tab_paired: P1, tab_paired_p2: P2); in one table the 57 rows
    are taller than a page.  Each table writes its CSV twin."""
    p = pd.read_csv(ROOT / "results/stats/paired_comparisons.csv")
    p = p[p.protocol.isin(["P1", "P2"]) & p["delta_auc"].notna()] if "delta_auc" in p else p.iloc[0:0]
    legend = ("$\\Delta$AUC = AUC(BSCAN) $-$ AUC(comparator) with 95\\% cluster-bootstrap CI; DeLong $p$-values "
              "Holm-adjusted within each test set; exact McNemar test on decisions at each model's validation threshold")
    note = ("DeLong and McNemar tests treat recordings as independent; a difference is called significant in the "
            "text only if the Holm-adjusted DeLong $p$-value is below 0.05 and the cluster-bootstrap CI excludes zero.")
    for proto, name, label, training in (("P1", "tab_paired", "tab:paired", "training on BF-SUST"),
                                         ("P2", "tab_paired_p2", "tab:paired_p2", "training on MEN")):
        q = p[p.protocol == proto]  # registry order
        rows = [[tlabel(reg, r["run_b"].split("__")[0]), split_label(proto, r["split"]),
                 fmt(r["delta_auc"], r["status"]) + f" [{fmt(r['delta_auc_ci_low'], r['status'])}, {fmt(r['delta_auc_ci_high'], r['status'])}]",
                 fmt(r["delong_p_holm"], r["status"], "p"), fmt(r["mcnemar_p"], r["status"], "p")]
                for _, r in q.iterrows()]
        write_table(name, f"Paired comparisons of BSCAN with each comparator under {proto} ({training}; seed 42, "
                    "same recordings). " + legend,
                    ["Comparator", "Test set", "$\\Delta$AUC [95\\% CI]", "DeLong $p$ (Holm)", "McNemar $p$"],
                    rows, label, align="llccc", has_placeholder=any_ph(rows), wide=True,
                    note=note + " orig.\\ prep., zero-padding (original) preprocessing.")
        q.to_csv(CSV / f"{name}.csv", index=False)


def t_analysis(reg: dict, kind: str) -> pd.DataFrame:
    a = reg["analyses"][kind]
    recs = []
    for rn in a["runs"]:
        st, path = resolve(rn)
        real_file = (ROOT / "results/runs" / rn / f"{kind}_summary.csv")
        ph_file = (ROOT / "results/runs_PLACEHOLDER" / rn / f"{kind}_summary.csv")
        f, fst = (real_file, "real") if real_file.exists() else ((ph_file, "placeholder") if ph_file.exists() else (None, "missing"))
        if f is None:
            continue
        s = pd.read_csv(f)
        s["run"], s["status"] = rn, fst
        recs.append(s)
    df = pd.concat(recs) if recs else pd.DataFrame()
    if df.empty:
        return df
    def cell(v, col, kind_):
        if not len(v) or col not in v.columns or pd.isna(v[col].iloc[0]):
            return r"\missing{}" if not len(v) or col not in v.columns else "--"
        return fmt(float(v[col].iloc[0]), v.status.iloc[0], kind_)

    if kind == "robustness":
        # registry order of the runs, conditions in the order of COND_LABEL (Clean first, then as in the text);
        # the model name goes on the first row of its block
        order = {c: i for i, c in enumerate(COND_LABEL)}
        runs = {r: i for i, r in enumerate(a["runs"])}
        df = df.assign(_r=df.run.map(runs), _o=df.condition.map(order).fillna(len(order))).sort_values(
            ["_r", "_o"], kind="stable").drop(columns=["_r", "_o"])
        rows, any_noop, last = [], False, None
        for (rn, cond), g in df.groupby(["run", "condition"], sort=False):
            noop = cond != "clean" and "share_input_changed" in g.columns and bool((g.share_input_changed == 0).all())
            any_noop |= noop
            label = COND_LABEL.get(cond, cond.replace("_", r"\_")) + (r"$^{\ddagger}$" if noop else "")
            # model name on the first row of its block; a rule between blocks
            row = [("" if last is None else "\\hline ") + tlabel(reg, rn.split("__")[0]) if rn != last else "", label]
            for sp in a["splits"]:
                row.append(cell(g[g.split == sp], "auc", "auc"))
            rows.append(row)
            last = rn
        write_table("tab_robustness", "Robustness to test-time perturbations (recording-level AUC; no retraining). "
                    "Perturbations applied to the 16 kHz recording before the model's own preprocessing; fixed sample "
                    "of 600 recordings per test set (300 per class); seed-42 checkpoints", ["Model", "Condition"] +
                    [SPLIT_LABEL[s].split(" (")[0] for s in a["splits"]], rows, "tab:robustness", align="llccc",
                    has_placeholder=any_ph(rows),
                    # kept to two lines: a third line makes this near-full-page float overfill its page in the PLOS
                    # layout; the text explains why only the clipping changes the (peak-normalised) input
                    note="Clipping: +6 dB gain, hard-clipped at full scale." + (
                             " $^{\\ddagger}$ the model's preprocessing undoes this condition (the input is unchanged "
                             "for every recording)." if any_noop else ""))
    else:
        rows, last = [], None
        for (rn, cond), g in df.groupby(["run", "condition"], sort=False):
            if cond == "clean":
                continue
            v = g[g.split == "internal_test"]
            w = g[g.split == "external_MEN"]
            rows.append([("" if last is None else "\\hline ") + tlabel(reg, rn.split("__")[0]) if rn != last else "",
                         PROBE_LABEL.get(cond, cond.replace("_", r"\_")),
                         cell(v, "mean_score_shift_bonafide", "auc"), cell(v, "mean_score_shift_spoof", "auc"),
                         cell(v, "share_input_changed", "pct"), cell(v, "decision_flip_rate_changed", "pct"),
                         cell(w, "share_input_changed", "pct"), cell(w, "decision_flip_rate_changed", "pct")])
            last = rn
        write_table("tab_probes", "Counterfactual shortcut probes. Mean change of the recording score (logit), share of "
                    "recordings whose model input changes, and share of those whose decision flips when a single cue "
                    "is added or removed at test time",
                    ["Model", "Probe", "$\\Delta$ bona fide", "$\\Delta$ synthetic", "Changed (\\%)",
                     "Flipped (\\%)", "Changed (\\%)", "Flipped (\\%)"], rows, "tab:probes",
                    align=wrapcol("1.3in") + "lcccccc", has_placeholder=any_ph(rows), wide=True, colsep=3,
                    groups=[("", 2), ("BF-SUST (internal test)", 4), ("MEN", 2)],
                    note="Seed-42 checkpoints. $\\Delta$: mean change of the recording score (logit) of bona fide and "
                         "synthetic recordings (BF-SUST internal test). Changed: the windows that reach the model differ from the clean ones; "
                         "a probe the preprocessing removes by construction changes no input. Flipped: among changed "
                         "recordings; -- when none changed. orig.\\ prep., zero-padding (original) preprocessing; the other "
                         "rows use the controlled preprocessing.")
    return df


def t_efficiency(reg: dict) -> pd.DataFrame:
    frames = []
    for f, plat in [("efficiency_cpu_threads1.csv", "CPU (1 thread)"), ("efficiency_mps_threads4.csv", "Apple GPU (MPS)")]:
        p = ROOT / "results/efficiency" / f
        if p.exists():
            d = pd.read_csv(p)
            d["platform"] = plat
            frames.append(d)
    cuda = sorted((ROOT / "results/efficiency").glob("efficiency_cuda_threads*.csv"))
    gpu = "NVIDIA GPU"
    if cuda:
        d = pd.read_csv(cuda[0])
        d["platform"] = "CUDA GPU"
        frames.append(d)
        meta = cuda[0].with_name(cuda[0].stem + "_meta.json")
        if meta.exists():
            gpu = (json.loads(meta.read_text()).get("environment") or {}).get("gpu") or gpu
    df = pd.concat(frames) if frames else pd.DataFrame()
    rows = []
    for _, r in df[df.platform == "CPU (1 thread)"].iterrows():
        mps = df[(df.platform == "Apple GPU (MPS)") & (df.config == r.config)]
        cu = df[(df.platform == "CUDA GPU") & (df.config == r.config)]
        name = tlabel(reg, r.config) if r.config in reg["models"] else r.config
        rows.append([name, fmt(r.params, kind="int"), f"{r.macs_per_window / 1e6:.0f}",
                     fmt(r.b1_mean_ms, kind="ms", sd=r.b1_std_ms), f"{r.b1_p95_ms:.2f}", fmt(r.e2e_rtf, kind="rtf"),
                     fmt(float(mps.b1_mean_ms.iloc[0]), kind="ms") if len(mps) else r"\missing{}",
                     fmt(float(cu.b1_mean_ms.iloc[0]), kind="ms") if len(cu) else r"\missing{}"])
    write_table("tab_efficiency", "Computational cost. Parameters, multiply-accumulate operations per 6 s window, "
                "batch-1 latency (mean $\\pm$ SD and p95) and end-to-end real-time factor on one CPU thread of an "
                "Apple-silicon laptop (arm64, 10 cores, 24 GB; load, resample, segment, features, model; 100 recordings), "
                "and batch-1 latency on GPUs",
                ["Model", "Parameters", "MACs (M)", "Mean (ms)", "p95 (ms)", "RTF", "MPS (ms)", "CUDA (ms)"], rows,
                "tab:efficiency", align="lrrccccc", has_placeholder=any_ph(rows), wide=True,
                groups=[("", 3), ("CPU, 1 thread", 3), ("GPU, batch 1", 2)],
                note=f"CUDA, {gpu} (CUDA backend, Google Colab); MACs, multiply-accumulate operations (millions; thop profiler, "
                     "which also counts batch-normalisation and pooling operations); MPS, Apple GPU "
                     "(Metal Performance Shaders backend); p95, 95th percentile; RTF, real-time factor; SD, standard "
                     "deviation." + (" \\missing{} = not yet measured." if any(r"\missing" in c for r in rows for c in r)
                                     else ""))
    return df


def master(reg: dict) -> pd.DataFrame:
    rows = []
    for run in expected_runs(reg, include_optional=True):
        st, m = metrics_of(run.name)
        if m is None:
            continue
        for sp, v in m["splits"].items():
            for k, val in v.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    ci = v.get(f"{k}_ci95") or [np.nan, np.nan]
                    rows.append({"run": run.name, "model": run.model, "protocol": run.protocol, "seed": run.seed,
                                 "stage": run.stage, "split": sp, "metric": k, "value": val, "ci_low": ci[0],
                                 "ci_high": ci[1], "status": st})
    return pd.DataFrame(rows)


def macros(summary: pd.DataFrame, paired: pd.DataFrame, extra: dict) -> None:
    lines = ["% generated by .scripts/generate_tables.py -- do not edit",
             r"\providecommand{\ph}[1]{\textcolor{red}{#1$^{\dagger}$}}",
             r"\providecommand{\missing}{\textcolor{red}{[MISSING]}}",
             r"\providecommand{\phnote}{\textcolor{red}{$^{\dagger}$Synthetic placeholder, not a result: to be replaced by the corresponding run.}}",
             r"% information only the authors can give; .scripts/final_qc.py blocks submission while any is left",
             r"\providecommand{\authorinput}[1]{\textcolor{red}{[AUTHOR INPUT: #1]}}",
             r"% table layout hooks (PLOS template v3.8); a manuscript may define its own before this file",
             # single-spaced tables: with the manuscript's \doublespacing the paired-comparison table is taller than a page
             r"\providecommand{\tablesetup}{\small\ifdefined\setstretch\setstretch{1}\fi}",
             r"\providecommand{\tablefloat}{table}",
             r"\providecommand{\tabletop}{\hline}", r"\providecommand{\tablemid}{\hline}",
             r"\providecommand{\tablebottom}{\hline}",
             r"% wide tables use adjustwidth (changepage package, loaded by the PLOS template); no-op fallback",
             r"\makeatletter\AtBeginDocument{\@ifundefined{adjustwidth}{\newenvironment{adjustwidth}[2]{}{}}{}}\makeatother",
             r"\makeatletter",
             r"\newcommand{\res}[1]{\@ifundefined{res@#1}{\missing}{\csname res@#1\endcsname}}",
             r"\newcommand{\cmp}[1]{\@ifundefined{cmp@#1}{\missing}{\csname cmp@#1\endcsname}}",
             r"\newcommand{\cmpadj}[1]{\@ifundefined{cmpadj@#1}{\missing}{\csname cmpadj@#1\endcsname}}"]

    def key(*parts):
        return "/".join(str(p).replace("_", "-") for p in parts)

    for _, s in summary.iterrows():
        for mt, kind in (("auc", "auc"), ("eer", "eer_pct"), ("accuracy", "pct"), ("f1_macro", "auc"),
                         ("recall_spoof_sensitivity", "pct"), ("specificity_bonafide_recall", "pct"), ("ece", "auc"),
                         ("brier", "auc")):
            v = s.get(f"{mt}_mean")
            sd = s.get(f"{mt}_sd") if s.get(f"{mt}_n", 0) and s.get(f"{mt}_n", 0) > 1 else None
            lines.append(r"\expandafter\def\csname res@" + key(s.model, s.protocol, s.split, mt) + r"\endcsname{" +
                         fmt(v, s.status, kind) + "}")
            if sd is not None:
                lines.append(r"\expandafter\def\csname res@" + key(s.model, s.protocol, s.split, mt, "sd") +
                             r"\endcsname{" + fmt(sd, s.status, kind) + "}")
    for _, r in paired.iterrows():
        if r.get("delta_auc") is None or r.get("delta_auc") != r.get("delta_auc"):
            continue
        p = r.get("delong_p_holm", np.nan)
        lo, hi = r.get("delta_auc_ci_low", np.nan), r.get("delta_auc_ci_high", np.nan)
        ci_excludes_0 = (lo == lo and hi == hi) and (lo > 0 or hi < 0)
        sig = (p == p and p < 0.05) and ci_excludes_0  # both criteria (DeLong assumes independent recordings)
        adj = ("higher" if r["delta_auc"] > 0 else "lower") if sig else "not significantly different"
        word = {"not significantly different": "not significantly different from", "higher": "higher than",
                "lower": "lower than"}[adj]
        if r["status"] != "real":
            word, adj = r"\ph{" + word + "}", r"\ph{" + adj + "}"
        a, b = r["run_a"].split("__")[0], r["run_b"].split("__")[0]
        lines.append(r"\expandafter\def\csname cmp@" + key(a, b, r["protocol"], r["split"]) + r"\endcsname{" + word + "}")
        lines.append(r"\expandafter\def\csname cmpadj@" + key(a, b, r["protocol"], r["split"]) + r"\endcsname{" + adj + "}")
        lines.append(r"\expandafter\def\csname res@" + key("delta", a, b, r["protocol"], r["split"], "p-eq") +
                     r"\endcsname{" + fmt(p, r["status"], "peq") + "}")
        for side, col in (("lo", "delta_auc_ci_low"), ("hi", "delta_auc_ci_high")):
            lines.append(r"\expandafter\def\csname res@" + key("delta", a, b, r["protocol"], r["split"], "auc", side) +
                         r"\endcsname{" + fmt(r.get(col, np.nan), r["status"]) + "}")
        lines.append(r"\expandafter\def\csname res@" + key("delta", a, b, r["protocol"], r["split"], "auc") +
                     r"\endcsname{" + fmt(r["delta_auc"], r["status"]) + "}")
        lines.append(r"\expandafter\def\csname res@" + key("delta", a, b, r["protocol"], r["split"], "p") +
                     r"\endcsname{" + fmt(p, r["status"], "p") + "}")
    for k, (v, st, kind) in extra.items():
        lines.append(r"\expandafter\def\csname res@" + k + r"\endcsname{" + fmt(v, st, kind) + "}")
    lines.append(r"\makeatother")
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "results_macros.tex").write_text("\n".join(lines) + "\n")


COND_TEXT = {"noise_snr20": "white noise at 20 dB SNR", "noise_snr10": "white noise at 10 dB SNR",
             "noise_snr5": "white noise at 5 dB SNR", "mp3_64k": "MP3 at 64 kbit/s", "mp3_32k": "MP3 at 32 kbit/s",
             "aac_32k": "AAC at 32 kbit/s", "telephone": "the telephone channel",
             "clip_+6dB": "clipping after a +6 dB gain", "speed_0.95": "a speed factor of 0.95",
             "speed_1.05": "a speed factor of 1.05", "pitch_+1st": "a pitch shift of +1 semitone",
             "pitch_-1st": "a pitch shift of $-$1 semitone", "reverb_rt0.3": "reverberation (RT60 0.3 s)",
             "reverb_rt0.6": "reverberation (RT60 0.6 s)"}


def analysis_macros(reg: dict) -> dict:
    """Macros for the robustness and counterfactual-probe analyses.

    rob/<model>/<split>/clean            AUC on the unperturbed subset
    rob/<model>/<split>/<condition>      AUC change (perturbed - clean)
    rob/<model>/<split>/worst, worst-dauc, median-dauc   largest drop, its size, median change, over the
                                         conditions that change the model input (no-ops are excluded)
    rob/<model>/<split>/clip-bona, clip-spoof            share of recordings that clip under clip_+6dB
    probe/<model>/<probe>/<split>/flip, shift-bona, shift-spoof, changed, flip-changed
    """
    out = {}

    def k(*parts):
        return "/".join(str(x).replace("_", "-") for x in parts)

    for kind in ("robustness", "probes"):
        for rn in reg["analyses"][kind]["runs"]:
            real = ROOT / "results/runs" / rn / f"{kind}_summary.csv"
            ph = ROOT / "results/runs_PLACEHOLDER" / rn / f"{kind}_summary.csv"
            f, st = (real, "real") if real.exists() else ((ph, "placeholder") if ph.exists() else (None, "missing"))
            if f is None:
                continue
            d = pd.read_csv(f)
            model = rn.split("__")[0]
            for sp, g in d.groupby("split"):
                g = g.set_index("condition")
                if "clean" not in g.index:
                    continue
                clean = float(g.loc["clean", "auc"])
                if kind == "robustness":
                    out[k("rob", model, sp, "clean")] = (clean, st, "auc")
                    deltas = {c: float(g.loc[c, "auc"]) - clean for c in g.index if c != "clean"}
                    for c, v in deltas.items():
                        out[k("rob", model, sp, c)] = (v, st, "auc")
                    for c in g.index:
                        if c.startswith("clip_") and "share_clipped_bonafide" in g.columns:
                            out[k("rob", model, sp, "clip-bona")] = (float(g.loc[c, "share_clipped_bonafide"]), st, "pct")
                            out[k("rob", model, sp, "clip-spoof")] = (float(g.loc[c, "share_clipped_spoof"]), st, "pct")
                    if "share_input_changed" in g.columns:  # conditions the preprocessing undoes are not robustness tests
                        deltas = {c: v for c, v in deltas.items() if float(g.loc[c, "share_input_changed"]) > 0}
                    if deltas:
                        worst = min(deltas, key=deltas.get)
                        out[k("rob", model, sp, "worst")] = (COND_TEXT.get(worst, worst.replace("_", " ")), st, "text")
                        out[k("rob", model, sp, "worst-dauc")] = (deltas[worst], st, "auc")
                        out[k("rob", model, sp, "median-dauc")] = (float(np.median(list(deltas.values()))), st, "auc")
                        best = max(deltas, key=deltas.get)
                        out[k("rob", model, sp, "best")] = (COND_TEXT.get(best, best.replace("_", " ")), st, "text")
                        out[k("rob", model, sp, "best-dauc")] = (deltas[best], st, "auc")
                else:
                    for c in g.index:
                        if c == "clean":
                            continue
                        out[k("probe", model, c, sp, "flip")] = (float(g.loc[c, "decision_flip_rate"]), st, "pct")
                        out[k("probe", model, c, sp, "shift-bona")] = (float(g.loc[c, "mean_score_shift_bonafide"]), st, "auc")
                        out[k("probe", model, c, sp, "shift-spoof")] = (float(g.loc[c, "mean_score_shift_spoof"]), st, "auc")
                        if "share_input_changed" in g.columns:
                            out[k("probe", model, c, sp, "changed")] = (float(g.loc[c, "share_input_changed"]), st, "pct")
                            if float(g.loc[c, "share_input_changed"]) > 0:  # undefined when the probe changed no input
                                out[k("probe", model, c, sp, "flip-changed")] = (
                                    float(g.loc[c, "decision_flip_rate_changed"]), st, "pct")
    return out


def compute_macros(reg: dict) -> dict:
    """Compute-resource macros from the stored runs (written only when every required run is real).

    compute/gpu              GPU model(s) recorded in environment.json of the neural runs
    compute/gpu-hours        total training time of the registered neural runs (hours)
    compute/gmm-fit-min/<P>  LFCC-GMM fitting time (minutes, single CPU thread)
    """
    out = {}
    neural = [r for r in expected_runs(reg) if reg["models"][r.model]["kind"] == "neural"]
    resolved = [resolve(r.name) for r in neural]
    if neural and all(st == "real" for st, _ in resolved):
        gpus, sec = set(), 0.0
        for _, path in resolved:
            env = json.loads((path / "environment.json").read_text()) if (path / "environment.json").exists() else {}
            gpus.add(str(env.get("gpu") or "unknown"))
            ts = json.loads((path / "train_summary.json").read_text()) if (path / "train_summary.json").exists() else {}
            sec += float(ts.get("train_sec", 0.0)) + float(ts.get("feature_cache_sec", 0.0))
        out["compute/gpu"] = (", ".join(sorted(gpus)), "real", "text")
        out["compute/gpu-hours"] = (sec / 3600.0, "real", "ms")
    for proto in ("P1", "P2"):
        st, path = resolve(f"lfcc_gmm_controlled__{proto}__seed42")
        if st == "real":
            m = json.loads((path / "metrics.json").read_text())
            out[f"compute/gmm-fit-min/{proto}"] = (float(m.get("fit_sec", float("nan"))) / 60.0, "real", "ms")
    return out


def shortcut_macros() -> dict:
    """Macros for the dataset-shortcut findings (Phases 1 and 3; deterministic, always real).

    sc/<corpus>/<cue>                      single-descriptor separability max(AUC, 1-AUC) per corpus
    sc/<protocol>/<scheme>/<features>/<split>   recording-level AUC of the coarse-statistics
                                                gradient-boosting diagnostic (model-visible windows)
    """
    out = {}
    u = pd.read_csv(ROOT / "results/phase1/univariate_separability.csv")
    for ds, desc, k in (("BF-SUST", "trailing_silence_sec", "sust/trailing-silence"), ("BF-SUST", "dc_offset", "sust/dc"),
                        ("BF-SUST", "exact_zero_ratio", "sust/exact-zero"), ("BF-SUST", "activity_ratio", "sust/activity"),
                        ("BF-SUST", "duration_sec", "sust/duration"), ("BF-MOZ", "sample_rate", "moz/sample-rate"),
                        ("BF-MOZ", "dc_offset", "moz/dc"), ("BF-MOZ", "leading_silence_sec", "moz/leading-silence"),
                        ("MEN", "sample_rate", "men/sample-rate"), ("MEN", "clip_ratio", "men/clipping"),
                        ("MEN", "rms_dbfs", "men/level")):
        v = u[(u.dataset_id == ds) & (u.descriptor == desc)].separability
        if len(v):
            out[f"sc/{k}"] = (float(v.iloc[0]), "real", "auc")
    d = pd.read_csv(ROOT / "results/phase3/model_visible_shortcut_diagnostics.csv")
    d = d[d.model == "hgb"]
    for r in d.itertuples():
        key = "/".join(str(x).replace("_", "-") for x in ("sc", r.protocol, r.scheme, r.features, r.test_split))
        if r.auc_file == r.auc_file:
            out[key] = (float(r.auc_file), "real", "auc")
        elif r.mean_p_spoof == r.mean_p_spoof:  # single-class split (BF-NEWS): mean P(spoof)
            out[key + "/mean-p"] = (float(r.mean_p_spoof), "real", "auc")
    return out


def ci_macros() -> dict:
    """95% cluster-bootstrap CI of the AUC of the seed-42 recording-statistics baseline per protocol and test set.

    <model>/<protocol>/<split>/auc-ci     'low--high' (e.g. stats-hgb-controlled/P1/external-MEN/auc-ci)
    """
    out = {}
    for model in ("stats_hgb_controlled", "stats_hgb_orig6s"):
        for proto in ("P1", "P2"):
            st, m = metrics_of(f"{model}__{proto}__seed42")
            for sp, v in ((m or {}).get("splits") or {}).items():
                ci = v.get("auc_ci95")
                if ci and all(c == c for c in ci):
                    lo, hi = (re.sub(r"(?<![\w$])-(?=\d)", "$-$", f"{c:.3f}") for c in ci)
                    out["/".join(x.replace("_", "-") for x in (model, proto, sp)) + "/auc-ci"] = (f"[{lo}, {hi}]", st, "text")
    return out


def run_macros(reg: dict) -> dict:
    """Facts read from the stored runs themselves.

    news/<model>/<protocol>/flagged          share of the BF-NEWS recordings labelled synthetic at the validation
                                             threshold, mean over the model's seeds (written only when all are real)
    train/<model>/<protocol>/seed<N>/best-epoch, .../epochs-run   from train_summary.json (S1 Fig run)
    """
    out, news = {}, {}
    for r in expected_runs(reg, include_optional=True):
        st, m = metrics_of(r.name)
        v = ((m or {}).get("splits") or {}).get("external_BF-NEWS") or {}
        if "frac_pred_spoof" in v:
            news.setdefault((r.model, r.protocol), []).append((float(v["frac_pred_spoof"]), st))
    for (model, proto), vals in news.items():
        if all(s == "real" for _, s in vals):
            out["/".join(x.replace("_", "-") for x in ("news", model, proto, "flagged"))] = (
                float(np.mean([f for f, _ in vals])), "real", "pct")
    # trained neural runs whose lowest validation loss fell in the last permitted epoch
    n_neural = n_cap = 0
    for r in expected_runs(reg):
        if reg["models"][r.model]["kind"] != "neural":
            continue
        ts, cf = ROOT / "results/runs" / r.name / "train_summary.json", ROOT / "results/runs" / r.name / "config.json"
        if not (ts.exists() and cf.exists()):
            continue
        n_neural += 1
        n_cap += json.loads(ts.read_text())["best_epoch"] == json.loads(cf.read_text())["train"]["epochs"]
    if n_neural:
        out["train/n-neural"] = (float(n_neural), "real", "int")
        out["train/n-best-at-cap"] = (float(n_cap), "real", "int")
    run = "bscan_controlled__P1__seed42"
    ts = ROOT / "results/runs" / run / "train_summary.json"
    if ts.exists():
        t = json.loads(ts.read_text())
        for k, name in (("best_epoch", "best-epoch"), ("epochs_run", "epochs-run")):
            if k in t:
                out[f"train/bscan-controlled/P1/seed42/{name}"] = (float(t[k]), "real", "int")
    return out


def main() -> None:
    reg = load_registry()
    CSV.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ROOT / "results/stats/multiseed_summary.csv")
    paired = pd.read_csv(ROOT / "results/stats/paired_comparisons.csv")
    t_corpora().to_csv(CSV / "tab_corpora.csv", index=False)
    t_splits().to_csv(CSV / "tab_splits.csv", index=False)
    t_shortcuts().to_csv(CSV / "tab_shortcuts.csv", index=False)
    t_models(summary, reg, "tab_main", ["bscan_controlled", "lcnn_lfcc", "lfcc_gmm_controlled", "stats_hgb_controlled"],
             "Main comparison under protocol P1 (training on BF-SUST, controlled preprocessing). Recording-level AUC "
             "and EER (neither depends on the decision threshold)", "tab:main").to_csv(CSV / "tab_main.csv", index=False)
    t_models(summary, reg, "tab_repr", ["mel_only", "lfcc_only", "mfcc_only", "mel_mfcc", "bscan_controlled",
                                        "ssl_xlsr_probe"],
             "Representation study under P1. Identical branch architecture, optimiser, budget and preprocessing; "
             "only the input representation differs. Last row: a learned-representation reference",
             "tab:repr", first_col=wrapcol("1.45in"), extra_note=" XLS-R: frozen XLS-R 300M encoder (layer 12 of 24, mean-pooled per window) with a "
             "logistic-regression back end instead of the branch architecture, same windows and threshold rule; one seed."
             ).to_csv(CSV / "tab_repr.csv", index=False)
    t_models(summary, reg, "tab_ablation", ["bscan_controlled", "dual_no_se", "dual_no_tattn", "dual_plain"],
             "Ablation under P1 (each variant trained from scratch). Same data, preprocessing, optimiser and budget as BSCAN",
             "tab:ablation", note="Mean $\\pm$ SD over three seeds (42, 123, 2026).").to_csv(CSV / "tab_ablation.csv", index=False)
    t_models(summary, reg, "tab_preproc", ["bscan_orig6s", "bscan_controlled", "stats_hgb_orig6s", "stats_hgb_controlled"],
             "Effect of the controlled preprocessing (P1). Original preprocessing (orig. prep.): 6 s windows with "
             "zero-padding; controlled: silence trimming, repeat-padding, DC removal and dither", "tab:preproc",
             seed42=("bscan_controlled",),
             labels={"bscan_controlled": "BSCAN (controlled)", "stats_hgb_controlled": "Recording statistics (controlled)"}, extra_note=" The seed-42 row is the run used in the paired comparisons."
             ).to_csv(CSV / "tab_preproc.csv", index=False)
    t_models(summary, reg, "tab_p2", ["bscan_controlled", "mel_only", "lfcc_only", "lcnn_lfcc", "lfcc_gmm_controlled",
                                      "stats_hgb_controlled"],
             "Reverse direction, protocol P2 (training on MEN with set-disjoint, hence speaker-disjoint, splits)", "tab:p2",
             proto="P2").to_csv(CSV / "tab_p2.csv", index=False)
    t_cross(reg).to_csv(CSV / "tab_cross.csv", index=False)
    t_tiers(summary, reg).to_csv(CSV / "tab_tiers.csv", index=False)
    t_bootstrap(reg).to_csv(CSV / "tab_bootstrap.csv", index=False)
    t_paired(reg)  # writes tab_paired.csv and tab_paired_p2.csv itself
    t_analysis(reg, "robustness").to_csv(CSV / "tab_robustness.csv", index=False)
    t_analysis(reg, "probes").to_csv(CSV / "tab_probes.csv", index=False)
    eff = t_efficiency(reg)
    eff.to_csv(CSV / "tab_efficiency.csv", index=False)
    m = master(reg)
    m.to_csv(ROOT / "results" / "master_results.csv", index=False)
    extra = {}
    b = eff[(eff.get("platform") == "CPU (1 thread)") & (eff.get("config") == "bscan_controlled")] if len(eff) else eff
    if len(b):
        extra.update({"eff/bscan/params": (float(b.params.iloc[0]), "real", "int"),
                      "eff/bscan/macs": (float(b.macs_per_window.iloc[0]) / 1e6, "real", "int"),
                      # 32-bit weights: 4 bytes per parameter
                      "eff/bscan/mib": (f"{float(b.params.iloc[0]) * 4 / 2**20:.2f}", "real", "text"),
                      "eff/bscan/cpu-ms": (float(b.b1_mean_ms.iloc[0]), "real", "ms"),
                      "eff/bscan/cpu-p95": (float(b.b1_p95_ms.iloc[0]), "real", "ms"),
                      "eff/bscan/rtf": (float(b.e2e_rtf.iloc[0]), "real", "rtf")})
    extra.update(shortcut_macros())
    extra.update(ci_macros())
    extra.update(run_macros(reg))
    extra.update(analysis_macros(reg))
    extra.update(compute_macros(reg))
    macros(summary, paired, extra)
    stat = m.groupby("status").run.nunique().to_dict() if len(m) else {}
    print(f"tables -> {TAB}; master_results rows {len(m)}; runs by status {stat}; placeholders used: {bool(PLACEHOLDER_USED)}")


if __name__ == "__main__":
    main()
