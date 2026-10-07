"""Flatten the PLOS manuscript into one self-contained .tex file (PLOS: the LaTeX source uploaded at acceptance
must be a single file; the template says "do not use \\input, \\externaldocument, or similar commands").

Input   PLOS/BSCAN_PLOS_ONE.tex, PLOS/generated/results_macros.tex, PLOS/generated/tables/*.tex and
        PLOS/BSCAN_PLOS_ONE.bbl (compile the manuscript first, e.g. with .scripts/build_all.py)
Output  PLOS/submission/BSCAN_PLOS_ONE.tex (or --out DIR), compiled there as a check when LaTeX is available

What it does
  1. replaces every \\input{generated/tables/NAME} with the table, with the layout hooks written out
     (\\tablefloat -> table, \\tablesetup -> \\small, \\tabletop/\\tablemid/\\tablebottom -> \\hline);
  2. replaces every \\res{key}, \\cmp{key} and \\cmpadj{key} with its literal value, and drops the macro file;
  3. removes the \\ifdraftfigures ... \\fi blocks, so the file includes no graphics;
  4. pastes the .bbl in place of \\bibliography{...} (or keeps \\bibliography with --keep-bib);
  5. refuses (exit 1) while a placeholder, missing value, \\authorinput or \\finalise is left, unless --draft.

Usage
  python .scripts/flatten_plos.py              # final single-file source
  python .scripts/flatten_plos.py --draft --out /tmp/flat   # trial run while placeholders remain
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "PLOS"
PAPER = "BSCAN_PLOS_ONE"
GEN = MANUSCRIPT / "generated"
HOOKS = {r"\tablefloat": "table", r"\tablesetup": r"\small", r"\tabletop": r"\hline", r"\tablemid": r"\hline",
         r"\tablebottom": r"\hline"}
# definitions that stay in the flattened preamble only while the text still uses them (draft runs)
MARKERS = {
    "ph": r"\providecommand{\ph}[1]{\textcolor{red}{#1$^{\dagger}$}}",
    "missing": r"\providecommand{\missing}{\textcolor{red}{[MISSING]}}",
    "phnote": r"\providecommand{\phnote}{\textcolor{red}{$^{\dagger}$Synthetic placeholder, not a result: to be replaced by the corresponding run.}}",
    "authorinput": r"\providecommand{\authorinput}[1]{\textcolor{red}{[AUTHOR INPUT: #1]}}",
}


def macro_values() -> dict[tuple[str, str], str]:
    vals = {}
    for m in re.finditer(r"\\csname (res|cmp|cmpadj)@(.+?)\\endcsname\{(.*)\}\s*$",
                         (GEN / "results_macros.tex").read_text(), re.M):
        vals[(m.group(1), m.group(2))] = m.group(3)
    return vals


def table_text(name: str) -> str:
    return (GEN / "tables" / f"{name}.tex").read_text().rstrip("\n")


def write_hooks(t: str) -> str:
    """\\begin{\\tablefloat} -> \\begin{table}, \\tablesetup -> \\small, ... (whole control words only)."""
    for hook, val in HOOKS.items():
        t = re.sub(re.escape(hook) + r"(?![A-Za-z])", lambda _m, v=val: v, t)
    return t


def strip_comments(t: str) -> str:
    return "\n".join(re.split(r"(?<!\\)%", line)[0] for line in t.splitlines())


def strip_draft_blocks(t: str) -> str:
    """Remove the draft-figure switch (with the comment line above it) and every \\ifdraftfigures ... \\fi block."""
    t = re.sub(r"(^%[^\n]*\n)?^\\newif\\ifdraftfigures\s*\n", "", t, flags=re.M)
    t = re.sub(r"^\\draftfigures(true|false)\s*\n", "", t, flags=re.M)
    t = re.sub(r"\\ifdraftfigures(?![A-Za-z]).*?\\fi(?![A-Za-z])", "", t, flags=re.S)
    return t


HEADER = """% BSCAN manuscript for PLOS ONE (PLOS LaTeX template v3.8), single-file source written by
% .scripts/flatten_plos.py from PLOS/BSCAN_PLOS_ONE.tex and the generated results; do not edit by hand.
% Figures are uploaded separately; their captions follow the paragraph that first cites them.

"""


def flatten(keep_bib: bool) -> tuple[str, list[str]]:
    src = (MANUSCRIPT / f"{PAPER}.tex").read_text()
    vals = macro_values()
    problems = []

    def value(m):
        key = (m.group(1), m.group(2))
        if key not in vals:
            problems.append(f"\\{key[0]}{{{key[1]}}} is not generated")
            return r"\missing{}"
        return vals[key]

    t = re.sub(r"\\input\{generated/tables/([^}]+)\}", lambda m: table_text(m.group(1)), src)
    t = re.sub(r"\\(res|cmp|cmpadj)\{([^}]*)\}", value, t)
    t = write_hooks(t)  # generated tables and the hand-written ones use the same layout hooks
    t = strip_draft_blocks(t)
    t = HEADER + t[t.index(r"\documentclass"):]  # the working-file header describes \input and the draft switch
    # the macro file is replaced by the few marker definitions that are still used (draft runs only)
    body = t.split(r"\begin{document}", 1)[1]
    keep = [d for name, d in MARKERS.items() if re.search(r"\\" + name + r"(?![A-Za-z])", body)]
    t = re.sub(r"(%[^\n]*\n)?\\input\{generated/results_macros\}\n",
               lambda _m: "\n".join(keep) + ("\n" if keep else ""), t)
    if not keep_bib:
        bbl = MANUSCRIPT / f"{PAPER}.bbl"
        if not bbl.exists():
            sys.exit(f"{bbl.relative_to(ROOT)} missing: compile the manuscript first (or use --keep-bib)")
        t = re.sub(r"\\bibliography\{[^}]*\}", lambda _m: bbl.read_text().strip(), t)
    leftover = re.findall(r"\\input\{[^}]*\}|\\include\{[^}]*\}|\\includegraphics", strip_comments(t))
    if leftover:
        problems.append(f"still present after flattening: {sorted(set(leftover))}")
    for pat, what in ((r"\\ph\{", "synthetic placeholder values"), (r"\\missing", "missing values"),
                      (r"\\authorinput\{", "open \\authorinput items"), (r"\\finalise\{", "\\finalise notes")):
        n = len(re.findall(pat, t.split(r"\begin{document}", 1)[1]))
        if n:
            problems.append(f"{n} {what}")
    return t, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=MANUSCRIPT / "submission")
    ap.add_argument("--keep-bib", action="store_true", help="keep \\bibliography{} (copy refs.bib and the .bst)")
    ap.add_argument("--draft", action="store_true", help="write the file even while placeholders remain")
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    text, problems = flatten(a.keep_bib)
    for p in problems:
        print("PROBLEM:", p)
    if problems and not a.draft:
        sys.exit("not flattened: fix the problems above (or use --draft for a trial run)")
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f"{PAPER}.tex"
    out.write_text(text)
    if a.keep_bib:
        bib = re.search(r"\\bibliography\{([^}]*)\}", text)
        bst = re.search(r"\\bibliographystyle\{([^}]*)\}", text)
        for f in ([f"{b.strip()}.bib" for b in bib.group(1).split(",")] if bib else []) + ([f"{bst.group(1)}.bst"] if bst else []):
            if (MANUSCRIPT / f).exists():
                shutil.copy2(MANUSCRIPT / f, a.out / f)
    print(f"wrote {out}")
    if a.no_compile:
        return
    texbin = os.environ.get("TEXBIN", "")
    pdflatex = str(Path(texbin) / "pdflatex") if texbin and (Path(texbin) / "pdflatex").exists() else shutil.which("pdflatex")
    if not pdflatex:
        print("pdflatex not found: compile check skipped")
        return
    bibtex = str(Path(pdflatex).with_name("bibtex"))
    runs = ([pdflatex], [bibtex], [pdflatex], [pdflatex]) if a.keep_bib else ([pdflatex], [pdflatex], [pdflatex])
    for cmd in runs:
        args = cmd + ([PAPER] if cmd[0] == bibtex else ["-interaction=nonstopmode", PAPER])
        subprocess.run(args, cwd=a.out, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = (a.out / f"{PAPER}.log").read_text(errors="replace") if (a.out / f"{PAPER}.log").exists() else ""
    errors = len(re.findall(r"^! ", log, re.M))
    undefined = len(re.findall(r"undefined", log))
    print(f"compile check: {errors} LaTeX errors, {undefined} 'undefined' warnings")
    for ext in (".aux", ".out", ".blg"):
        (a.out / f"{PAPER}{ext}").unlink(missing_ok=True)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
