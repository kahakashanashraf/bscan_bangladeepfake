"""Final consistency and integrity audit of the manuscript PLOS/BSCAN_PLOS_ONE.tex, its generated tables and
macros (PLOS/generated/) and figures (PLOS/figures/).  Writes results/QC_REPORT.md; exit 0 only if
submission-ready.

BLOCKER      the manuscript must not be submitted (placeholders, missing results, integrity problems,
             open \\authorinput{...} items, wording that does not belong in this manuscript)
AUTHOR-INPUT items only the authors can supply (e.g. corresponding author, funding, ethics), listed in full
WARNING      claims or wording to verify by hand
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bscan.registry import expected_runs, load_registry, resolve  # noqa: E402

BLOCK, INPUT, WARN, OK = [], [], [], []
MANUSCRIPT = ROOT / "PLOS"
TEX = MANUSCRIPT / "BSCAN_PLOS_ONE.tex"
BIB = MANUSCRIPT / "refs.bib"  # default; the \\bibliography{...} of the manuscript takes precedence
GEN = MANUSCRIPT / "generated"
FIGDIR = MANUSCRIPT / "figures"
DAS = MANUSCRIPT / "DATA_AVAILABILITY_STATEMENT.txt"  # text for the Editorial Manager field
# unfinished wording in the reference list (fields of refs.bib and the compiled .bbl)
REF_PLACEHOLDER = re.compile(r"to be (added|supplied|confirmed)|\bTODO\b|\bTBD\b|\bplaceholder\b|\bXXX\b", re.I)
MANIFEST = ROOT / "results/figures/figure_manifest.csv"
PLACEHOLDER_MARK = b"SYNTHETIC PLACEHOLDER"  # banner and PDF metadata of placeholder figures (generate_figures.py)
ABSTRACT_MAX_WORDS = 300  # PLOS ONE submission guidelines
TITLE_MAX_CHARS = 250  # PLOS ONE submission guidelines
# wording that must not appear anywhere in the manuscript source (outside the reference list), comments included
# manuscript-hygiene terms: one regular expression per line in a local, untracked file (absent -> no check)
_TERMS = Path(__file__).with_name("qc_forbidden_terms.txt")
FORBIDDEN = [(ln.strip(), 0) for ln in _TERMS.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")] if _TERMS.exists() else []
AUTHORS = ["Kahakashan Ashraf", "Md. Hamid Hosen", "Mahfuzulhoq Chowdhury"]  # the only authors of this work
CLAIM_BLOCK = [r"\bnovel\b", r"state[- ]of[- ]the[- ]art", r"\bSOTA\b", r"\bsuperior\b", r"\bgroundbreaking\b",
               r"\bfirst (study|work|to|attempt|time)\b", r"\bunprecedented\b", r"\boutperforms? all\b"]
CLAIM_WARN = [r"\brobust(ness|ly)?\b", r"\breal[- ]time\b", r"\blightweight\b", r"\bsignificant(ly)?\b",
              r"\bgeneraliz|generalis", r"\bedge\b"]


def check_runs(reg):
    rows = []
    for r in expected_runs(reg):  # required stages B, C, D, LOCAL
        st, path = resolve(r.name)
        rows.append((r.name, st))
        if st != "real":
            BLOCK.append(f"run `{r.name}` is {st.upper()} (required by configs/experiments.yaml stage {r.stage})")
            continue
        m = json.loads((path / "metrics.json").read_text())
        if m.get("threshold_rule") != "eer" or "validation" not in m.get("splits", {}):
            BLOCK.append(f"run `{r.name}`: threshold not derived from the validation split")
        if reg["models"][r.model]["kind"] == "neural":
            cfg = json.loads((path / "config.json").read_text())
            if cfg.get("train", {}).get("seed") != r.seed:
                BLOCK.append(f"run `{r.name}`: seed in config.json does not match the registry")
            if not (path / "environment.json").exists():
                BLOCK.append(f"run `{r.name}`: environment.json missing")
            ts = path / "train_summary.json"
            if r.model == "bscan_controlled" and ts.exists() and json.loads(ts.read_text()).get("params") != 162819:
                BLOCK.append(f"run `{r.name}`: BSCAN parameter count differs from 162,819")
        for f in path.glob("*.csv"):
            if "PLACEHOLDER" in f.read_text()[:2000]:
                BLOCK.append(f"placeholder marker inside real results: {f}")
    n_real = sum(1 for _, s in rows if s == "real")
    (OK if n_real == len(rows) else WARN).append(f"registered runs real: {n_real}/{len(rows)}")
    for kind in ("robustness", "probes"):
        for rn in reg["analyses"][kind]["runs"]:
            if not (ROOT / "results/runs" / rn / f"{kind}_summary.csv").exists():
                BLOCK.append(f"{kind} analysis missing for `{rn}` (real)")


def check_generated():
    for p in sorted(GEN.rglob("*.tex")):
        t = p.read_text()
        body = "\n".join(l for l in t.splitlines() if "providecommand" not in l)
        if r"\ph{" in body:
            BLOCK.append(f"placeholder values in `{p.relative_to(ROOT)}`")
        if r"\missing{}" in body:  # a missing value (generate_tables.fmt); the macro definitions use \missing alone
            BLOCK.append(f"missing values in `{p.relative_to(ROOT)}`")
    for p in sorted(GEN.rglob("*.tex")):
        for i, (pat, flags) in enumerate(FORBIDDEN, 1):
            for m in re.finditer(pat, p.read_text(), flags):  # the term itself is never written to the report
                line = p.read_text().count("\n", 0, m.start()) + 1
                BLOCK.append(f"`{p.relative_to(ROOT)}` line {line} contains local hygiene term #{i}")
    if MANIFEST.exists():
        m = pd.read_csv(MANIFEST)
        for _, r in m[m.status != "real"].iterrows():
            BLOCK.append(f"figure `{r.figure}` status {r.status}")
        status = dict(zip(m.figure.astype(str), m.status))
        for name, st in status.items():
            if st == "real" and not (FIGDIR / f"{name}.pdf").exists():
                BLOCK.append(f"figure `{name}` is listed as real but `PLOS/figures/{name}.pdf` does not exist")
    else:
        BLOCK.append("figure manifest missing (run .scripts/generate_figures.py)")
        status = {}
    for f in sorted(FIGDIR.glob("*")) if FIGDIR.exists() else []:
        if not f.is_file() or f.name.startswith("."):
            continue
        if PLACEHOLDER_MARK in f.read_bytes():
            BLOCK.append(f"figure file `{f.relative_to(ROOT)}` is a synthetic placeholder")
        if f.suffix == ".pdf" and f.stem not in status:
            WARN.append(f"figure file `{f.relative_to(ROOT)}` is not in the figure manifest (not made by generate_figures.py)")
    for rep in ("cross_dataset_analysis.md", "robustness_analysis.md", "statistical_analysis.md"):
        p = ROOT / "results" / "reports" / rep
        if not p.exists() or "†" in p.read_text():
            BLOCK.append(f"`{rep}` missing or contains placeholder values")
    c = ROOT / "results/tables/tab_corpora.csv"
    if c.exists():
        d = pd.read_csv(c)
        tot = int(d.bonafide.sum() + d.spoof.sum())
        (OK if tot == 31092 else BLOCK).append(f"corpus table total recordings = {tot:,} (inventory: 31,092)")


def macro_values() -> dict:
    """key -> definition text for every \\res / \\cmp / \\cmpadj value in the generated macro file."""
    f = GEN / "results_macros.tex"
    vals = {}
    if f.exists():
        for m in re.finditer(r"\\csname (res|cmp|cmpadj)@(.+?)\\endcsname\{(.*)\}\s*$", f.read_text(), re.M):
            vals[(m.group(1), m.group(2))] = m.group(3)
    return vals


def strip_comments(text: str) -> str:
    return "\n".join(re.split(r"(?<!\\)%", l)[0] if not l.strip().startswith("\\expandafter") else ""
                     for l in text.splitlines())


def brace_args(text: str, macro: str) -> list[str]:
    """Arguments of every \\macro{...} (balanced braces)."""
    out = []
    for m in re.finditer(r"\\" + macro + r"\s*\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        out.append(" ".join(text[m.end():i - 1].split()))
    return out


def norm(name: str) -> str:
    return " ".join(name.replace("~", " ").replace("\\", " ").split())


def check_figures_used(t: str) -> None:
    """Every graphics file the manuscript includes exists; generated ones must be real in the manifest."""
    status = dict(zip(*(lambda m: (m.figure.astype(str), m.status))(pd.read_csv(MANIFEST)))) if MANIFEST.exists() else {}
    for g in re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", strip_comments(t)):
        f = MANUSCRIPT / g
        cands = [f] if f.suffix else [f.with_suffix(e) for e in (".pdf", ".png", ".jpg", ".tif", ".eps")]
        hit = next((c for c in cands if c.exists()), None)
        if hit is None:
            BLOCK.append(f"graphics file `{g}` included by the manuscript does not exist")
            continue
        if hit.parent == FIGDIR and hit.suffix == ".pdf":
            st = status.get(hit.stem)
            if st != "real":
                BLOCK.append(f"included figure `{g}` has no 'real' row in the figure manifest (status: {st})")


def check_figure_numbering(t: str) -> None:
    """The k-th figure float includes figures/Fig<k>.<ext>, because PLOS asks for figure files named after the
    printed figure number (Fig1.tif, Fig2.tif, ...)."""
    body = strip_comments(t.split(r"\begin{document}", 1)[-1])
    floats = re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", body, re.S)
    bad = []
    for k, f in enumerate(floats, 1):
        g = re.search(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", f)
        if g and Path(g.group(1)).stem != f"Fig{k}":
            bad.append(f"Fig {k} includes `{g.group(1)}`")
    (BLOCK if bad else OK).append(f"figure files not named after their printed number: {bad}" if bad
                                  else f"figure files Fig1-Fig{len(floats)} match the printed figure numbers")


def check_references(t: str) -> None:
    """No unfinished wording ('to be added', TODO, ...) in the fields of refs.bib or in the compiled .bbl."""
    bibname = re.search(r"\\bibliography\{([^}]*)\}", strip_comments(t))
    bib = MANUSCRIPT / f"{bibname.group(1).split(',')[0].strip()}.bib" if bibname else BIB
    if bib.exists():  # entry fields only: comment lines and the x-verified provenance notes are not printed
        fields = [l for l in bib.read_text().splitlines()
                  if not l.lstrip().startswith("%") and not re.match(r"\s*x-verified\s*=", l)]
        hits = [l.strip() for l in fields if REF_PLACEHOLDER.search(l)]
        (BLOCK if hits else OK).append(f"unfinished wording in {bib.name}: {hits[:5]}" if hits
                                       else f"no unfinished wording in the fields of {bib.name}")
    bbl = TEX.with_suffix(".bbl")
    if bbl.exists():
        hits = [l.strip() for l in bbl.read_text().splitlines() if REF_PLACEHOLDER.search(l)]
        (BLOCK if hits else OK).append(f"unfinished wording in the compiled reference list: {hits[:5]}" if hits
                                       else "no unfinished wording in the compiled reference list")


def check_das() -> None:
    """The Data Availability Statement (pasted into Editorial Manager) must be complete before submission."""
    if not DAS.exists():
        BLOCK.append(f"`{DAS.relative_to(ROOT)}` missing (Data Availability Statement for the submission form)")
        return
    statement = DAS.read_text().split("--- Statement", 1)[-1]  # the header explains the markers
    items = re.findall(r"\[(?:TO BE ADDED|AUTHOR INPUT)[^\]]*\]", statement)
    for a in items:
        INPUT.append(f"Data Availability Statement: {a}")
    (BLOCK if items else OK).append(f"{len(items)} open item(s) in the Data Availability Statement" if items
                                    else "Data Availability Statement complete")


def check_manuscript():
    if not TEX.exists():
        BLOCK.append(f"manuscript `{TEX.relative_to(ROOT)}` missing")
        return
    t = TEX.read_text()
    body = t.split(r"\begin{document}", 1)[-1]
    body = body.split(r"\begin{thebibliography}", 1)[0]  # reference titles are not claims of this paper
    claims_body = body
    body_nocomment = strip_comments(body)
    vals = macro_values()
    used = re.findall(r"\\(res|cmp|cmpadj)\{([^}]*)\}", body_nocomment)
    undefined = sorted({f"\\{k}{{{v}}}" for k, v in used if (k, v) not in vals})
    (BLOCK if undefined else OK).append(f"result macros used in the text but not generated: {undefined[:20]}"
                                        if undefined else f"all {len(set(used))} result macros used in the text are generated")
    ph = sorted({v for k, v in used if (k, v) in vals and r"\ph{" in vals[(k, v)]})
    if ph:
        BLOCK.append(f"{len(ph)} result values in the text are synthetic placeholders (e.g. {ph[:5]})")
    miss = sorted({v for k, v in used if (k, v) in vals and r"\missing" in vals[(k, v)]})
    if miss:
        BLOCK.append(f"{len(miss)} result values in the text are missing (e.g. {miss[:5]})")
    n_fin = len(re.findall(r"\\finalise\{", body_nocomment))
    if n_fin:
        BLOCK.append(f"{n_fin} \\finalise{{...}} note(s): interpretive text still to be written against the final results")
    claims_text = strip_comments(claims_body)
    for pat in CLAIM_BLOCK:
        for m in re.finditer(pat, claims_text, re.I):
            BLOCK.append(f"unsupported-claim wording `{m.group(0)}` in manuscript: '…{claims_text[max(0, m.start() - 60):m.end() + 60]}…'".replace("\n", " "))
    for pat in CLAIM_WARN:
        n = len(re.findall(pat, claims_text, re.I))
        if n:
            WARN.append(f"wording to verify against evidence: /{pat}/ occurs {n}x")
    for marker in (r"\ph{", r"[FINALISE", "PLACEHOLDER"):
        if marker in body_nocomment:
            BLOCK.append(f"manuscript contains `{marker}`")
    # information only the authors can give: listed in full and blocking until replaced
    asks = brace_args(body_nocomment, "authorinput")
    for a in asks:
        INPUT.append(f"\\authorinput{{{a}}}")
    for m in re.finditer(r"\[AUTHOR INPUT[^\]]*\]", body_nocomment):
        INPUT.append(m.group(0))
    n_ask = len(asks) + len(re.findall(r"\[AUTHOR INPUT", body_nocomment))
    (BLOCK if n_ask else OK).append(f"{n_ask} \\authorinput{{...}} item(s) still open (see AUTHOR INPUT NEEDED)"
                                    if n_ask else "no open \\authorinput items")
    # wording that does not belong in this manuscript: whole source incl. preamble and comments, without references
    src = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", t, flags=re.S)
    # the report is shared with the code, so it names neither the terms nor their context, only the term number
    # (line of .scripts/qc_forbidden_terms.txt among the non-comment lines) and the source line
    hits = [(i, src.count("\n", 0, m.start()) + 1) for i, (pat, flags) in enumerate(FORBIDDEN, 1)
            for m in re.finditer(pat, src, flags)]
    for i, line in hits:
        BLOCK.append(f"manuscript source line {line} contains local hygiene term #{i}")
    if FORBIDDEN and not hits:
        OK.append(f"manuscript source free of the {len(FORBIDDEN)} local hygiene terms")
    # line numbers (PLOS template: \linenumbers after the abstract)
    (OK if re.search(r"\\linenumbers\b", body_nocomment) else BLOCK).append(
        "line numbers enabled (\\linenumbers)" if re.search(r"\\linenumbers\b", body_nocomment)
        else "no \\linenumbers in the manuscript body (PLOS requires line numbers)")
    # byline: exactly the three authors, in this order (PLOS front matter: Name\textsuperscript{...}, ...)
    front = body_nocomment.split(r"\section*{Abstract}", 1)[0]
    names = [norm(n) for n in re.findall(r"([^\n,{}\\]*[A-Za-z.][^\n,{}\\]*?)\s*\\textsuperscript\s*\{", front)
             if norm(n) and not re.fullmatch(r"[\d\s.,*]*", norm(n))]
    names = [n for n in names if not n.lower().startswith("with ")]
    if names == AUTHORS:
        OK.append("byline lists exactly the three authors")
    else:
        BLOCK.append(f"byline lists {names}; expected exactly {AUTHORS}")
    abstract = (re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
                or re.search(r"\\section\*\{Abstract\}(.*?)(?=\\section\*|\\linenumbers|\\section\{)", body, re.S))
    if abstract:
        text = re.sub(r"\\(res|cmp|cmpadj)\{([^}]*)\}", lambda m: re.sub(r"\\ph\{(.*)\}", r"\1", vals.get((m.group(1), m.group(2)), "X")),
                      strip_comments(abstract.group(1)))
        words = len(re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?|[{}$]", " ", text).split())
        (OK if words <= ABSTRACT_MAX_WORDS else BLOCK).append(
            f"abstract length ≈ {words} words (PLOS ONE: at most {ABSTRACT_MAX_WORDS})")
    else:
        BLOCK.append("no abstract found (\\section*{Abstract})")
    title = (re.search(r"\{\\Large\s*\\textbf\{((?:[^{}]|\{[^{}]*\})*)\}", strip_comments(t))
             or re.search(r"\\(?:textbf|title)\{(?:\\textbf\{)?(.*?)\}\}?\s*%\s*TITLE", t, re.S))
    if title:
        n = len(" ".join(title.group(1).split()))
        (OK if n <= TITLE_MAX_CHARS else BLOCK).append(f"title length {n} characters (PLOS ONE: at most {TITLE_MAX_CHARS})")
    else:
        WARN.append("title not found (expected {\\Large \\textbf{...}} as in the PLOS template)")
    check_figures_used(t)
    check_figure_numbering(t)
    check_references(t)

    cited = set(k.strip() for c in re.findall(r"\\cite[pt]?\*?\{([^}]*)\}", body_nocomment) for k in c.split(","))
    cited.discard("")
    bibname = re.search(r"\\bibliography\{([^}]*)\}", strip_comments(t))
    bib = MANUSCRIPT / f"{bibname.group(1).split(',')[0].strip()}.bib" if bibname else BIB
    if bib.exists():
        keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text()))
        missing = sorted(cited - keys)
        (BLOCK if missing else OK).append(f"citations missing from {bib.name}: {missing}" if missing else f"all {len(cited)} cited keys present in {bib.name}")
        unverified = [k for k in cited & keys if not re.search(r"@\w+\{" + re.escape(k) + r",[^@]*verified", bib.read_text())]
        if unverified:
            WARN.append(f"cited entries without a 'verified' note: {unverified}")
    elif r"\begin{thebibliography}" in t:  # flattened source with the .bbl pasted in
        keys = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", t))
        missing = sorted(cited - keys)
        (BLOCK if missing else OK).append(f"citations without a \\bibitem: {missing}" if missing else f"all {len(cited)} cited keys have a \\bibitem")
    else:
        BLOCK.append(f"{bib.relative_to(ROOT)} missing")
    log = TEX.with_suffix(".log")
    if log.exists():
        lg = log.read_text(errors="ignore")
        if "Undefined control sequence" in lg or "! LaTeX Error" in lg:
            BLOCK.append("LaTeX errors in the build log")
        n = lg.count("Overfull \\hbox")
        if n:
            WARN.append(f"{n} overfull hboxes in the LaTeX log")
        if "undefined" in lg.lower() and "Citation" in lg:
            BLOCK.append("undefined citations in the LaTeX log")
    else:
        WARN.append("no LaTeX log next to the manuscript (build_all.py without --skip-latex compiles it)")


def main() -> None:
    reg = load_registry()
    check_runs(reg)
    check_generated()
    check_manuscript()
    check_das()
    ready = not BLOCK
    lines = ["# QC report (generated by .scripts/final_qc.py)\n",
             f"**Submission-ready: {'YES' if ready else 'NO'}** — {len(BLOCK)} blocker(s), {len(INPUT)} author-input item(s), {len(WARN)} warning(s).\n"]
    for title, items in (("BLOCKERS", BLOCK), ("AUTHOR INPUT NEEDED", INPUT), ("WARNINGS (verify by hand)", WARN), ("PASSED", OK)):
        lines.append(f"\n## {title}\n")
        lines += [f"- {i}" for i in dict.fromkeys(items)] or ["- none"]
    (ROOT / "results" / "QC_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:2]))
    print(f"details: {ROOT / 'results' / 'QC_REPORT.md'}")
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
