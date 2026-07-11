"""Compatibility exports for the modular pipeline implementation.

The executable phase logic lives in ``pipeline.evaluator`` and environment/config
construction lives in ``pipeline.training_engine``. This module intentionally owns
no model persistence or serving lifecycle behavior.
"""

from mars_lite.pipeline.dataset_builder import build_feature_set
from mars_lite.pipeline.evaluator import (
    phase_p0,
    phase_pbt,
    phase_regime,
    phase_train,
    phase_wf,
    report_comparison,
)
from mars_lite.pipeline.training_engine import (
    build_env_kwargs,
    build_post_processor,
    train_ppo,
)

__all__ = [
    "build_env_kwargs",
    "build_feature_set",
    "build_post_processor",
    "phase_p0",
    "phase_pbt",
    "phase_regime",
    "phase_train",
    "phase_wf",
    "report_comparison",
    "train_ppo",
]
