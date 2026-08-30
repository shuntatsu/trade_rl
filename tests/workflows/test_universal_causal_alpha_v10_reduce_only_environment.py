from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.workflows.universal_causal_alpha_v10_stage_entry import (
    _V10ReduceOnlyEnvironment,
)


class _Environment:
    def __init__(self) -> None:
        self.masks: list[np.ndarray] = []

    def set_next_hybrid_reduce_only_mask(self, mask: np.ndarray) -> None:
        self.masks.append(np.asarray(mask).copy())

    def step(self, action: np.ndarray):
        return action.copy(), 0.0, True, False, {}


class _Policy:
    def __init__(self, reduce_only: bool) -> None:
        self.last_step_trace_metadata = {"reduce_only": reduce_only}


def test_v10_hierarchical_environment_forwards_reduce_only_metadata_before_step() -> (
    None
):
    environment = _Environment()
    wrapper = _V10ReduceOnlyEnvironment(environment, _Policy(True))

    wrapper.step(np.asarray([0.10], dtype=np.float32))

    assert len(environment.masks) == 1
    assert environment.masks[0].dtype == np.dtype(np.bool_)
    assert environment.masks[0].tolist() == [True]


def test_v10_hierarchical_environment_defaults_normal_actions_to_false() -> None:
    environment = _Environment()
    wrapper = _V10ReduceOnlyEnvironment(environment, _Policy(False))

    wrapper.step(np.asarray([0.10], dtype=np.float32))

    assert environment.masks[0].tolist() == [False]


def test_v10_hierarchical_environment_delegates_other_attributes() -> None:
    environment = _Environment()
    environment.marker = SimpleNamespace(value="kept")
    wrapper = _V10ReduceOnlyEnvironment(environment, _Policy(False))

    assert wrapper.marker.value == "kept"
