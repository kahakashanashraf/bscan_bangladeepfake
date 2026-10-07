"""Experiment configuration: YAML files with inheritance, turned into a typed dataclass.

A config file may contain `base: other.yaml`; keys are merged recursively (child wins).
Every run writes its fully resolved config next to its results.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    protocol: str = "P1"                       # P1: train BF-SUST; P2: train MEN
    metadata_csv: str = "metadata/unified_metadata.csv"
    audio_root: str = "data"
    cache_dir: str = "data/cache"
    scheme: str = "controlled"                 # orig6s | trim_repeat | controlled
    window_sec: float = 6.0
    overlap: float = 0.10
    # splits missing from the protocol are skipped (P1 has external_MEN, P2 has external_BF-SUST)
    eval_splits: tuple = ("validation", "internal_test", "external_BF-MOZ", "external_MEN", "external_BF-SUST",
                          "external_BF-NEWS")
    max_files_per_split: int | None = None     # for smoke tests only
    num_workers: int = 2


@dataclass
class ModelCfg:
    name: str = "bscan"                        # bscan | lcnn
    branches: tuple = ("mel", "lfcc")
    hidden_dim: int = 64
    use_se: bool = True
    use_temporal_attn: bool = True
    dropout: float = 0.3


@dataclass
class TrainCfg:
    seed: int = 42
    batch_size: int = 16                       # the original run used 16 (cell 54)
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    patience: int = 5                          # early stopping on validation loss
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    grad_clip: float = 1.0
    amp: bool = True                           # mixed precision on CUDA only
    deterministic: bool = True
    device: str = "auto"


@dataclass
class EvalCfg:
    threshold_rule: str = "eer"                # threshold chosen on validation only
    aggregate: str = "mean_logit"              # per-recording score
    n_boot: int = 2000


@dataclass
class ExperimentCfg:
    name: str = "bscan_controlled"
    out_dir: str = "results/runs"
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    def run_dir(self) -> Path:
        return Path(self.out_dir) / f"{self.name}__{self.data.protocol}__seed{self.train.seed}"

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    return _merge(_load_yaml(path.parent / base), raw) if base else raw


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> ExperimentCfg:
    raw = _load_yaml(Path(path)) if path else {}
    if overrides:
        raw = _merge(raw, overrides)
    cfg = ExperimentCfg()
    for section, klass in (("data", DataCfg), ("model", ModelCfg), ("train", TrainCfg), ("eval", EvalCfg)):
        vals = raw.get(section, {})
        for k in vals:
            if k not in klass.__dataclass_fields__:
                raise KeyError(f"unknown config key {section}.{k}")
        tuples = {k: tuple(v) for k, v in vals.items() if isinstance(v, list)}
        setattr(cfg, section, klass(**{**vals, **tuples}))
    for k in ("name", "out_dir"):
        if k in raw:
            setattr(cfg, k, raw[k])
    return cfg


def parse_overrides(pairs: list[str]) -> dict:
    """Turn ['train.seed=123', 'data.scheme=orig6s'] into a nested dict (YAML-typed values)."""
    out: dict[str, Any] = {}
    for p in pairs:
        key, val = p.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out
