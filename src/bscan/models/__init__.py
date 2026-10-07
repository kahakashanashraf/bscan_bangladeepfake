"""Model definitions and the config-driven factory."""

from .baselines import LCNN
from .bscan import BSCAN, count_parameters

__all__ = ["BSCAN", "LCNN", "build_model", "count_parameters", "model_features"]


def build_model(mcfg):
    if mcfg.name == "bscan":
        return BSCAN(branches=tuple(mcfg.branches), hidden_dim=mcfg.hidden_dim, use_se=mcfg.use_se,
                     use_temporal_attn=mcfg.use_temporal_attn, dropout=mcfg.dropout)
    if mcfg.name == "lcnn":
        if len(mcfg.branches) != 1:
            raise ValueError("LCNN takes exactly one feature (model.branches)")
        return LCNN(feature=mcfg.branches[0])
    raise ValueError(f"unknown model {mcfg.name}")


def model_features(mcfg) -> tuple:
    """Feature names the model consumes (used to build the feature cache)."""
    return tuple(mcfg.branches)
