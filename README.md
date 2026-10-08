# BSCAN: Dual-Branch Spectro-Cepstral Residual Attention Network for Bengali Deepfake Audio Detection

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23126423.svg)](https://doi.org/10.5281/zenodo.23126423) [![Dataset DOI](https://img.shields.io/badge/dataset%20DOI-10.34740%2FKAGGLE%2FDSV%2F20278870-blue)](https://doi.org/10.34740/KAGGLE/DSV/20278870)

Kahakashan Ashraf<sup>1</sup>, Md. Hamid Hosen<sup>2</sup>, Mahfuzulhoq Chowdhury<sup>1</sup>

<sup>1</sup> Dept. of CSE, Chittagong University of Engineering and Technology · <sup>2</sup> Dept. of CSE, East Delta University

This repository holds the code, configurations, split definitions and result files of the BSCAN study on Bengali synthetic-speech (deepfake audio) detection. The article is in preparation.

**Companion dataset (Kaggle):** [Bengali Deepfake Forensics Dataset (BSCAN)](https://www.kaggle.com/datasets/kahakashanashraf/bengali-deepfake-forensics-dataset-bscan).

## What the code does

- **Data audit.** It builds a per-file inventory of the two public Bengali corpora (BanglaFake and the Mendeley Bangla audio dataset), checks for exact and near-duplicate recordings, and measures recording-level cues (duration, silence, DC offset, level, sample rate) that separate the classes without any speech content.
- **Leakage-aware protocols.** P1 trains on the SUST subset of BanglaFake with sentence-grouped 70/15/15 splits and tests on BanglaFake (Common Voice) and on the Mendeley corpus. P2 trains on the Mendeley corpus with set-disjoint (hence speaker-disjoint) splits and tests on BanglaFake. Splits are defined by metadata only (`splits/`, `metadata/`).
- **Shortcut diagnostics.** A classifier trained on 11 coarse window statistics (mainly recording-chain properties) measures how much label information survives each preprocessing scheme. The controlled preprocessing (silence trimming, repeat-padding, DC removal, dither) is the default for every model.
- **Models.** BSCAN has two parallel residual branches (Mel spectrogram and LFCC) with squeeze-and-excitation and temporal attention pooling, fused by concatenation. The comparators are single-representation and Mel+MFCC variants, a trained ablation (no SE, no attention, neither), an LCNN-style LFCC model, an LFCC-GMM, the recording-statistics baseline, and a frozen XLS-R probe.
- **Evaluation.** Thresholds come from the validation split only. The code reports recording-level AUC and EER, means and SDs over seeds, cluster-bootstrap confidence intervals, DeLong tests with Holm correction, McNemar tests, calibration and error analyses. It also runs test-time robustness perturbations, counterfactual shortcut probes, and CPU/GPU cost measurements.
- **Outputs.** Every result table and figure of the article, and the model results quoted in its text, are generated from the result files (`.scripts/generate_tables.py`, `.scripts/generate_figures.py`).

Until the neural runs from the Colab notebook are imported, the pipeline fills each missing run with a clearly marked **synthetic placeholder**: these sit in `results/runs_PLACEHOLDER/`, the values are printed in red with †, and the figures carry a watermark. The final QC (`.scripts/final_qc.py`) exits non-zero while any placeholder is left.

## Repository map

```
code/               BSCAN_full_pipeline.ipynb: Colab notebook for all GPU runs and analyses
                    banglafkpart.ipynb: exploratory notebook of an earlier, unpublished analysis with a
                    superseded pipeline (no reported number is computed with it; the training hyperparameters were
                    carried over from it)
src/bscan/          package: audio, features, data, models, train, evaluate, metrics, statistics,
                    analysis, robustness, efficiency, ssl_probe, registry
configs/            model configs; experiments.yaml is the registry of every reported run
tests/smoke_test.py pipeline gate: loading, features, every model family, training, checkpointing,
                    threshold selection, determinism
.scripts/           data audit, CPU baselines, notebook builder, table/figure/report generators,
                    build_all, final_qc, Kaggle release builder
metadata/, splits/  per-file metadata (incl. dataset_inventory.csv) and the leakage-aware splits (P1, P2)
results/            phase1/ and phase3/ (data audit), runs/ (real runs), stats/, analysis/, efficiency/,
                    tables/ (CSV twin of every table), figures/figure_manifest.csv, master_results.csv
```

The following are generated and not tracked in git: `results/runs_PLACEHOLDER/`, `results/reports/` and `results/QC_REPORT.md` (written by `build_all.py`), `release/` (the Kaggle release) and `data/` (the downloaded corpora). The manuscript sources are also outside the repository; `build_all.py` writes the generated tables, result macros and figures to `PLOS/generated/` and `PLOS/figures/`.

## Requirements

- Python 3.12. Local environment: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt` (pinned versions; the CPU steps were run on an Apple M5 laptop).
- GPU steps: Google Colab with a CUDA GPU (T4 or better). The notebook installs `librosa==1.0.0` and `soundfile==0.14.0` and uses Colab's own PyTorch. Each run records its software versions and GPU in `environment.json` and its training time in `train_summary.json`.
- Optional: pdflatex and bibtex to compile the manuscript. Set `TEXBIN` to the directory of the TeX binaries if they are not on `PATH`.

## How to reproduce

1. **GPU runs (Colab).** Open `code/BSCAN_full_pipeline.ipynb` in Google Colab ([open in Colab](https://colab.research.google.com/github/kahakashanashraf/bscan_bangladeepfake/blob/main/code/BSCAN_full_pipeline.ipynb)), select a GPU runtime and choose **Run all**. The notebook then:
   - downloads both corpora from their original hosts and verifies every file against its SHA-256;
   - runs the CUDA smoke test;
   - trains and evaluates every registered run (`configs/experiments.yaml`), then runs the robustness, probe and latency analyses.

   The notebook takes the code from this repository at a fixed commit and stops unless it matches the notebook's code version (a SHA-256 hash of `src/bscan`, `configs` and `tests`). Results go to Google Drive after every run, so a disconnected session resumes where it stopped. At the end the notebook downloads `BSCAN_results_<version>.zip`. After a code change, publish the code, set `PUBLIC_COMMIT` in `.scripts/build_colab_notebook.py` and rebuild the notebook with `python .scripts/build_colab_notebook.py`; `--embedded` instead writes a notebook that carries the source files itself (for unpublished code).
2. **Tables, figures and checks (local).**
   ```bash
   .venv/bin/python .scripts/build_all.py --import BSCAN_results_<version>.zip
   ```
   This imports the runs (it refuses bundles of another code version and any placeholder data), then computes the statistics and analyses. It generates the tables, result macros, figures and reports, compiles the manuscript when its source and LaTeX are present, and writes `results/QC_REPORT.md`. Use `--skip-latex` to skip the LaTeX step.
3. **Kaggle release.** Run `.venv/bin/python .scripts/build_kaggle_release.py`. It builds `release/BSCAN_Bengali_Audio_Deepfake_Benchmark/` with metadata, checksums, splits, descriptors, audits, per-recording scores and the download/verify scripts. It contains **no audio**. Publish it as a new version of the Kaggle dataset ([kahakashanashraf/bengali-deepfake-forensics-dataset-bscan](https://www.kaggle.com/datasets/kahakashanashraf/bengali-deepfake-forensics-dataset-bscan)) by following the upload notes in the script's docstring, then cite the DOI that Kaggle issues for that version.

The data-audit steps that come before the GPU runs need the corpora in `data/`. Their outputs are already in `results/phase1/`, `results/phase3/`, `metadata/`, `splits/` and `results/runs/` (CPU baselines). To re-run them, use this order (each script's docstring gives its arguments):
`build_inventory.py` → `summarize_inventory.py` → `duplicate_check.py` → `leakage_check.py` → `model_visible_shortcuts.py` → `run_stats_baseline.py` → `run_gmm_baseline.py`. Run the smoke test with `python -m tests.smoke_test`.

## Data and licences

- **BanglaFake** (Fahad, Asif & Sikder, arXiv:2505.10885) is obtained from `huggingface.co/datasets/sifat1221/banglaFake`. It names no specific licence: its Hugging Face and GitHub repositories carry none, and its paper says only "open license". Its audio is therefore not redistributed here; the notebook and the Kaggle download script fetch it and verify the checksums.
- **Mendeley Bangla Audio Dataset, version 4** (Dipto, Ayan & Faria, DOI 10.17632/4ftmwt86vr.4; its audio is byte-identical to version 1) is listed under CC BY 4.0. Versions 3 and 4 add a dataset usage agreement that, among other terms, allows academic and research use only, forbids redistribution without the provider's written consent, and forbids any re-identification of speakers. Its audio is not redistributed; fetch it from Mendeley Data and follow those terms.
- Data files produced by this project (metadata, splits, descriptors, audits and scores) are released under CC BY 4.0. The code is released under the MIT licence (`LICENSE`).

Please cite the article and both source datasets. The companion dataset of this repository is: Ashraf K, Hosen MH, Chowdhury M. Bengali Deepfake Forensics Dataset (BSCAN), version 7. Kaggle; 2026. doi:[10.34740/KAGGLE/DSV/20278870](https://doi.org/10.34740/KAGGLE/DSV/20278870). The code of the article is release v1.0.0, archived at Zenodo: doi:[10.5281/zenodo.23126424](https://doi.org/10.5281/zenodo.23126424) (all versions: [10.5281/zenodo.23126423](https://doi.org/10.5281/zenodo.23126423)).
