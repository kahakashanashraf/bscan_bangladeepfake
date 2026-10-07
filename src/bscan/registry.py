"""Experiment registry (configs/experiments.yaml) and run resolution.

A run is resolved in this order:
    results/runs/{name}/metrics.json          -> status "real"
    results/runs_PLACEHOLDER/{name}/...        -> status "placeholder" (synthetic, clearly marked)
    otherwise                                  -> status "missing"
Everything downstream (tables, figures, reports, manuscript macros, QC) carries the status, so a
placeholder value can never be mistaken for a result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = ROOT / "results" / "runs"
PLACEHOLDER_DIR = ROOT / "results" / "runs_PLACEHOLDER"


@dataclass(frozen=True)
class Run:
    model: str
    protocol: str
    seed: int
    stage: str

    @property
    def name(self) -> str:
        return f"{self.model}__{self.protocol}__seed{self.seed}"


def load_registry(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "configs" / "experiments.yaml").read_text())


def expected_runs(reg: dict | None = None, stages: tuple[str, ...] = ("B", "C", "D", "LOCAL"),
                  include_optional: bool = False) -> list[Run]:
    reg = reg or load_registry()
    out = []
    for st, items in reg["runs"].items():
        if st not in stages and not (include_optional and st == "OPTIONAL"):
            continue
        for it in items:
            for s in it["seeds"]:
                out.append(Run(it["model"], it["protocol"], int(s), st))
    return out


def resolve(name: str, real_dir: Path = REAL_DIR, placeholder_dir: Path = PLACEHOLDER_DIR) -> tuple[str, Path | None]:
    r = real_dir / name
    if (r / "metrics.json").exists():
        m = json.loads((r / "metrics.json").read_text())
        if m.get("PLACEHOLDER"):
            raise RuntimeError(f"placeholder data found inside the real results folder: {r}")
        return "real", r
    p = placeholder_dir / name
    if (p / "metrics.json").exists():
        return "placeholder", p
    return "missing", None


def status_table(reg: dict | None = None, include_optional: bool = False) -> list[dict]:
    rows = []
    for run in expected_runs(reg, include_optional=include_optional):
        st, path = resolve(run.name)
        rows.append({"run": run.name, "model": run.model, "protocol": run.protocol, "seed": run.seed,
                     "stage": run.stage, "status": st, "path": str(path) if path else ""})
    return rows
