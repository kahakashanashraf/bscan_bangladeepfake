"""Optional learned-representation comparator: frozen XLS-R (300M) + logistic regression (Colab GPU).

Pre-declared design (no tuning on any test split):
  * windows produced by the same `controlled` preprocessing and seeds as every other model
  * encoder: facebook/wav2vec2-xls-r-300m, frozen, per-window input normalisation (zero mean,
    unit variance), hidden state of transformer layer 12 (of 24), mean-pooled over time -> 1024-d
  * classifier: standardisation + L2 logistic regression (C = 1.0), fitted on TRAIN windows only
  * recording score = mean window log-odds; threshold = validation EER point (bscan.evaluate)
Output: results/runs/ssl_xlsr_probe__{protocol}__seed{seed}/ in the standard format.

Usage (Colab, GPU):
    python -m bscan.ssl_probe --protocol P1 --seed 42 --audio_root ... --metadata ... --cache ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audio import load_audio, segment
from .evaluate import file_table, split_metrics
from .metrics import select_threshold
from .utils import environment_info, save_json, seed_everything, stable_seed

MODEL_ID = "facebook/wav2vec2-xls-r-300m"
REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"  # pinned Hugging Face commit (last change 2022-08-10)
LAYER = 12


def embed_split(files: pd.DataFrame, root: Path, model, device, batch: int = 16) -> tuple[np.ndarray, pd.DataFrame]:
    import torch

    feats, rows, buf, meta = [], [], [], []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-7)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            h = model(x, output_hidden_states=True).hidden_states[LAYER]
        feats.append(h.float().mean(dim=1).cpu().numpy())
        rows.extend(meta)
        buf.clear()
        meta.clear()

    for r in files.itertuples():
        x, _ = load_audio(root / r.audio_path)
        for s in segment(x, scheme="controlled", seed=stable_seed(r.audio_path)):
            buf.append(s.audio)
            meta.append({"audio_path": r.audio_path, "window": s.index, "pad_fraction": s.pad_fraction,
                         "label": r.label, "dataset_id": r.dataset_id, "speaker_id": r.speaker_id,
                         "text_group": r.text_group})
            if len(buf) == batch:
                flush()
    flush()
    return np.concatenate(feats), pd.DataFrame(rows)


def main() -> None:
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import Wav2Vec2Model

    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="P1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metadata", default="metadata/unified_metadata.csv")
    ap.add_argument("--audio_root", default="data")
    ap.add_argument("--cache", default="data/cache/ssl")
    ap.add_argument("--out_dir", default="results/runs")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(args.metadata, dtype={"utterance_id": str}, low_memory=False)
    col = f"split_{args.protocol}"
    rd = Path(args.out_dir) / f"ssl_xlsr_probe__{args.protocol}__seed{args.seed}"
    rd.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) / args.protocol
    cache.mkdir(parents=True, exist_ok=True)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID, revision=REVISION).to(device).eval()
    emb = {}
    for sp in [s for s in meta[col].unique() if s != "unused"]:
        fx, fi = cache / f"{sp}.npy", cache / f"{sp}.csv"
        if not fx.exists():
            X, idx = embed_split(meta[meta[col] == sp].sort_values("audio_path"), Path(args.audio_root), model, device)
            np.save(fx, X)
            idx.to_csv(fi, index=False)
        emb[sp] = (np.load(fx), pd.read_csv(fi, low_memory=False))
        print(f"{sp}: {emb[sp][0].shape}")
    Xtr, itr = emb["train"]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(Xtr, itr["label"])
    tables = {}
    for sp, (X, idx) in emb.items():
        if sp == "train":
            continue
        logit = clf.decision_function(X)
        idx.assign(logit=logit).to_csv(rd / f"windows_{sp}.csv.gz", index=False)
        tables[sp] = file_table(idx, logit)
        tables[sp].to_csv(rd / f"predictions_{sp}.csv", index=False)
    thr = select_threshold(tables["validation"]["label"].to_numpy(), tables["validation"]["score"].to_numpy(), "eer")
    res = {"model": f"{MODEL_ID} (frozen, layer {LAYER}, mean-pooled) + logistic regression", "scheme": "controlled",
           "protocol": args.protocol, "seed": args.seed, "threshold_rule": "eer", "threshold_logit": thr, "splits": {}}
    for sp, f in tables.items():
        m = split_metrics(f, thr, 2000, args.seed)
        m.pop("_reliability", None)
        res["splits"][sp] = m
    save_json(res, rd / "metrics.json")
    import transformers

    save_json({"model_id": MODEL_ID, "revision": REVISION, "layer": LAYER, "protocol": args.protocol, "seed": args.seed,
               "transformers": transformers.__version__}, rd / "config.json")
    save_json(environment_info(), rd / "environment.json")
    print(json.dumps({k: {kk: v.get(kk) for kk in ("auc", "eer", "accuracy")} for k, v in res["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
