"""
Lens configurations for the source-science robustness study.

Three configs vary Einstein radius, mass axis_ratio/angle, external shear,
and the Sersic-lens-light parameters. Source is held constant across all
configs (same SersicCore for Sersic-truth tests, same extracted MGE for
MGE-truth tests) so the only thing varying between configs is the lens.
"""

from __future__ import annotations
from typing import Dict


CONFIGS: Dict[str, Dict] = {
    "config_0": {
        "name": "config_0",
        "label": "config 0 — existing (E_R=1.6)",
        "mass": {
            "einstein_radius": 1.6,
            "axis_ratio": 0.9,
            "angle": 45.0,
        },
        "shear": {"gamma_1": 0.05, "gamma_2": 0.05},
        "bulge_sersic": {
            "intensity": 1.0,
            "effective_radius": 0.8,
            "sersic_index": 4.0,
            "axis_ratio": 0.9,
            "angle": 45.0,
        },
    },
    "config_1": {
        "name": "config_1",
        "label": "config 1 — compact (E_R=1.0)",
        "mass": {
            "einstein_radius": 1.0,
            "axis_ratio": 0.7,
            "angle": 80.0,
        },
        "shear": {"gamma_1": -0.03, "gamma_2": 0.04},
        "bulge_sersic": {
            "intensity": 0.6,
            "effective_radius": 0.4,
            "sersic_index": 3.0,
            "axis_ratio": 0.75,
            "angle": 30.0,
        },
    },
    "config_2": {
        "name": "config_2",
        "label": "config 2 — extended (E_R=2.0)",
        "mass": {
            "einstein_radius": 2.0,
            "axis_ratio": 0.95,
            "angle": 10.0,
        },
        "shear": {"gamma_1": 0.08, "gamma_2": -0.02},
        "bulge_sersic": {
            "intensity": 1.5,
            "effective_radius": 1.2,
            "sersic_index": 4.5,
            "axis_ratio": 0.95,
            "angle": 60.0,
        },
    },
}


def get_config(name: str) -> Dict:
    if name not in CONFIGS:
        raise KeyError(f"Unknown config {name!r}; available: {list(CONFIGS)}")
    return CONFIGS[name]


# Source truth — same SersicCore across all configs for the Sersic-source-truth
# tests. (Tests using MGE source truth load
# `source_science/results/mge_truth_source.json` instead.)
SERSIC_SOURCE_TRUTH = {
    "centre": (0.0, 0.0),
    "axis_ratio": 0.8,
    "angle": 60.0,
    "intensity": 4.0,
    "effective_radius": 0.1,
    "sersic_index": 1.0,
}
