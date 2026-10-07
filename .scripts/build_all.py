"""One command from returned Colab results to the tables, figures, reports and PDF of the manuscript.

    python .scripts/build_all.py [--import BSCAN_results_<version>.zip ...] [--no-placeholders] [--skip-latex]

Steps:
  1. import  : unzip each returned Colab bundle; copy its runs into results/runs/ (refusing placeholders),
               efficiency CSVs into results/efficiency/, the CUDA smoke report into results/smoke_cuda/
  2. placeholders for still-missing registered runs (unless --no-placeholders), clearly marked
  3. statistics (bscan.statistics), error analysis and calibration (bscan.analysis)
  4. tables + macros (PLOS/generated/), figures (PLOS/figures/), reports (results/reports/)
  5. PLOS/BSCAN_PLOS_ONE.pdf: pdflatex, bibtex, pdflatex twice, and once more while LaTeX asks for it
     (bibliography style plos2025.bst of the PLOS LaTeX template, kept next to the .tex); then the one-column
     reading copy PLOS/BSCAN_onecolumn.pdf (.scripts/build_onecolumn.py: same body, other page layout, every
     table and figure in place, wide tables on landscape pages).
     Skipped with a message when the manuscript or LaTeX is absent.
     LaTeX tools are taken from the directory in the environment variable TEXBIN if set, otherwise from PATH.
  6. final QC (.scripts/final_qc.py) -> results/QC_REPORT.md; exit status 0 only if submission-ready
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "PLOS"
PAPER = "BSCAN_PLOS_ONE"
PY = sys.executable
ENV = {"PYTHONPATH": str(ROOT / "src")}


def sh(*args, check=True):
    print("$", " ".join(map(str, args)))
    return subprocess.run(list(map(str, args)), cwd=ROOT, env={**os.environ, **ENV}, check=check)


def current_code_version() -> str:
    """The code version of the Colab notebook: hash of src/bscan, configs and tests (build_colab_notebook.payload)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_colab_notebook", ROOT / ".scripts" / "build_colab_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.payload()[2]


def import_bundle(path: Path, allow_other_version: bool = False) -> None:
    want = current_code_version()
    with tempfile.TemporaryDirectory() as td:
        if path.suffix == ".zip":
            zipfile.ZipFile(path).extractall(td)
            base = Path(td)
        else:
            base = path
        # every reported number must come from the code that is released: refuse runs of another code version
        cs = next(base.rglob("colab_session.json"), None)
        got = json.loads(cs.read_text()).get("code_version") if cs else None
        if got != want and not allow_other_version:
            raise SystemExit(f"{path.name}: bundle code version {got} != current source {want}; "
                             "re-run the current notebook (or pass --allow-other-code-version)")
        runs = next((p for p in base.rglob("runs") if p.is_dir()), None)
        n = 0
        if runs:
            for rd in sorted(p for p in runs.iterdir() if p.is_dir()):
                m = rd / "metrics.json"
                if not m.exists():
                    print(f"  skip {rd.name}: no metrics.json")
                    continue
                if json.loads(m.read_text()).get("PLACEHOLDER"):
                    raise SystemExit(f"refusing to import placeholder data: {rd}")
                env = rd / "environment.json"
                rv = json.loads(env.read_text()).get("code_version") if env.exists() else None
                if rv != want and not allow_other_version:
                    raise SystemExit(f"{rd.name}: produced by code version {rv}, not the current {want}")
                shutil.copytree(rd, ROOT / "results/runs" / rd.name, dirs_exist_ok=True)
                n += 1
        for sub, dst in (("efficiency", "results/efficiency"), ("smoke_cuda", "results/smoke_cuda")):
            s = next((p for p in base.rglob(sub) if p.is_dir()), None)
            if s:
                shutil.copytree(s, ROOT / dst, dirs_exist_ok=True)
        cs = next(base.rglob("colab_session.json"), None)
        if cs:
            (ROOT / "results/colab_sessions").mkdir(parents=True, exist_ok=True)
            shutil.copy(cs, ROOT / "results/colab_sessions" / f"{path.stem}.json")
        print(f"imported {n} runs from {path}")


def tex_tool(name: str) -> str | None:
    """Path of a TeX program: $TEXBIN/<name> if TEXBIN is set and contains it, otherwise the one on PATH."""
    texbin = os.environ.get("TEXBIN")
    if texbin and (Path(texbin) / name).exists():
        return str(Path(texbin) / name)
    return shutil.which(name)


def latex(stem: str, clean: tuple[str, ...] = (".aux", ".blg", ".out")) -> None:
    """pdflatex + bibtex + pdflatex x2 in PLOS/; after an error-free build the intermediate files in `clean` are
    removed (the manuscript keeps its .bbl, which PLOS asks to paste into the source, and its .log for QC)."""
    tex = MANUSCRIPT / f"{stem}.tex"
    if not tex.exists():
        print(f"{tex.relative_to(ROOT)} not found: LaTeX build skipped")
        return
    pdflatex, bibtex = tex_tool("pdflatex"), tex_tool("bibtex")
    if not pdflatex or not bibtex:
        print("pdflatex/bibtex not found (set TEXBIN or add them to PATH): LaTeX build skipped")
        return
    style = re.search(r"^[^%\n]*\\bibliographystyle\{([^}]*)\}", tex.read_text(), re.M)
    if style and not (MANUSCRIPT / f"{style.group(1)}.bst").exists():
        print(f"note: {style.group(1)}.bst is not in PLOS/; bibtex will look for it in the TeX installation")
    (MANUSCRIPT / f"{stem}.pdf").unlink(missing_ok=True)  # a stale PDF must not count as a successful build
    pdf = [pdflatex, "-interaction=nonstopmode", stem]
    for cmd in (pdf, [bibtex, stem], pdf, pdf):
        subprocess.run(cmd, cwd=MANUSCRIPT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = MANUSCRIPT / f"{stem}.log"
    for _ in range(3):  # floats that move between passes can leave the cross-references one pass behind
        if "Rerun to get" not in (log.read_text(errors="replace") if log.exists() else ""):
            break
        subprocess.run(pdf, cwd=MANUSCRIPT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok =(MANUSCRIPT / f"{stem}.pdf").exists() and log.exists() and not re.search(r"^!", log.read_text(errors="replace"), re.M)
    print(f"compiled PLOS/{stem}.pdf" if ok else f"LaTeX errors: see PLOS/{stem}.log")
    if ok:
        for ext in clean:
            (MANUSCRIPT / f"{stem}{ext}").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="imports", nargs="*", default=[])
    ap.add_argument("--no-placeholders", action="store_true")
    ap.add_argument("--skip-latex", action="store_true")
    ap.add_argument("--allow-other-code-version", action="store_true",
                    help="import runs made by a different code version (never for reported results)")
    args = ap.parse_args()
    for p in args.imports:
        import_bundle(Path(p).resolve(), args.allow_other_code_version)
    if args.no_placeholders:
        sh(PY, ".scripts/make_placeholder_results.py", "--clean")
    else:
        sh(PY, ".scripts/make_placeholder_results.py")
    sh(PY, "-m", "bscan.statistics")
    sh(PY, "-m", "bscan.analysis")
    sh(PY, ".scripts/generate_tables.py")
    sh(PY, ".scripts/generate_figures.py")
    sh(PY, ".scripts/generate_reports.py")
    if not args.skip_latex:
        latex(PAPER)
        sh(PY, ".scripts/build_onecolumn.py", check=False)  # reading copy; its own LaTeX passes
    r = sh(PY, ".scripts/final_qc.py", check=False)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
