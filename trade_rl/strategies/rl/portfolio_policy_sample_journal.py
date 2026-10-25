"""Opt-in write-on-forward journal for joint portfolio policy samples."""

from __future__ import annotations

import math
import re
import threading
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes

PORTFOLIO_POLICY_SAMPLE_JOURNAL_SCHEMA = "portfolio_policy_sample_journal_v1"
_POLICY_SAMPLE_SCHEMA = "portfolio_policy_sample_v2"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTIVE_POLICIES: weakref.WeakKeyDictionary[object, weakref.ReferenceType[object]] = (
    weakref.WeakKeyDictionary()
)
_ACTIVE_POLICIES_LOCK = threading.Lock()


def _required_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _required_digest(value: object, *, field: str) -> str:
    if type(value) is not str or not _DIGEST_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _space_number(value: object) -> int | float | str:
    if isinstance(value, bool):
        return bool(value)
    if not isinstance(value, (int, float)):
        raise TypeError("space bounds must be numeric values")
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "+Infinity" if number > 0 else "-Infinity"
    return number


def _space_array(value: object) -> object:
    import numpy as np

    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        return _space_number(scalar)
    return [_space_array(item) for item in array.tolist()]


def _space_descriptor(space: object) -> dict[str, object]:
    import numpy as np

    shape_value = getattr(space, "shape", None)
    dtype_value = getattr(space, "dtype", None)
    if shape_value is None or dtype_value is None:
        raise TypeError("observation subspaces must expose numeric shape and dtype")
    shape = tuple(int(size) for size in shape_value)
    dtype = np.dtype(dtype_value)
    if dtype.kind not in "biuf":
        raise TypeError("observation subspaces must have numeric or boolean dtype")
    descriptor: dict[str, object] = {
        "kind": f"{type(space).__module__}.{type(space).__qualname__}",
        "shape": list(shape),
        "dtype": dtype.name,
    }
    if hasattr(space, "low") and hasattr(space, "high"):
        descriptor["low"] = _space_array(getattr(space, "low"))
        descriptor["high"] = _space_array(getattr(space, "high"))
    if hasattr(space, "n"):
        descriptor["n"] = int(getattr(space, "n"))
    return descriptor


def _cpu_tensor_values(
    tensor: Any, *, torch: Any, field: str
) -> tuple[str, tuple[int, ...], object]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{field} must be a Torch tensor")
    if tensor.layout != torch.strided:
        raise TypeError(f"{field} must use strided tensor storage")
    dtype_name = str(tensor.dtype)
    if tensor.is_floating_point():
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{field} must be finite")
    elif dtype_name != "torch.bool" and not dtype_name.startswith(
        ("torch.int", "torch.uint")
    ):
        raise TypeError(f"{field} must use a real numeric or boolean tensor dtype")
    detached = tensor.detach().to(device="cpu").contiguous().clone()
    return dtype_name.removeprefix("torch."), tuple(tensor.shape), detached.tolist()


def _strict_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


class PortfolioPolicySampleJournal:
    """Capture actual SB3 policy inputs and sampled outputs before environment step.

    The caller-supplied context, transform, and policy-version fields are labels,
    not attestations of the Dataset, transformed observations, model weights, or
    checkpoint lineage. Always close the journal to remove its transient hook.
    """

    def __init__(
        self,
        model: object,
        *,
        root: str | Path,
        symbols: tuple[str, ...],
        context_digest: str,
        transform_digest: str,
        policy_version: str,
        policy_version_semantics: str,
    ) -> None:
        import gymnasium as gym
        import numpy as np
        import torch
        from stable_baselines3.common.base_class import BaseAlgorithm
        from stable_baselines3.common.policies import MultiInputActorCriticPolicy

        if not isinstance(model, BaseAlgorithm):
            raise TypeError("model must be a Stable-Baselines3 algorithm")
        policy = model.policy
        if not isinstance(policy, MultiInputActorCriticPolicy):
            raise TypeError("model policy must be MultiInputActorCriticPolicy")
        observation_space = policy.observation_space
        action_space = policy.action_space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("policy observation space must be a Dict")
        if not isinstance(action_space, gym.spaces.MultiDiscrete):
            raise TypeError("policy action space must be MultiDiscrete")
        if (
            type(symbols) is not tuple
            or not symbols
            or any(type(symbol) is not str or not symbol.strip() for symbol in symbols)
        ):
            raise TypeError("symbols must be a non-empty tuple of names")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique and ordered")
        resolved_context = _required_digest(context_digest, field="context_digest")
        resolved_transform = _required_digest(
            transform_digest, field="transform_digest"
        )
        resolved_version = _required_text(policy_version, field="policy_version")
        resolved_version_semantics = _required_text(
            policy_version_semantics, field="policy_version_semantics"
        )

        nvec = np.asarray(action_space.nvec)
        start = np.asarray(action_space.start)
        if nvec.ndim != 1 or nvec.shape != (len(symbols),):
            raise ValueError("MultiDiscrete action dimensions must match symbols")
        if not np.array_equal(nvec, np.full(len(symbols), 4, dtype=nvec.dtype)):
            raise ValueError("each symbol must have exactly four action codes")
        if not np.array_equal(start, np.zeros(len(symbols), dtype=start.dtype)):
            raise ValueError("MultiDiscrete action codes must start at zero")

        observation_keys = tuple(observation_space.spaces)
        if not observation_keys or any(
            type(key) is not str for key in observation_keys
        ):
            raise ValueError("Dict observation keys must be non-empty strings")
        observation_spaces = {
            key: observation_space.spaces[key] for key in observation_keys
        }
        descriptors = {
            key: _space_descriptor(observation_spaces[key]) for key in observation_keys
        }
        shapes: dict[str, tuple[int, ...]] = {}
        for key in observation_keys:
            shape = observation_spaces[key].shape
            if shape is None:
                raise TypeError("observation subspaces must expose a shape")
            shapes[key] = tuple(int(size) for size in shape)
        action_space_descriptor = {
            "kind": f"{type(action_space).__module__}.{type(action_space).__qualname__}",
            "nvec": [int(value) for value in nvec.tolist()],
            "start": [int(value) for value in start.tolist()],
            "dtype": np.dtype(action_space.dtype).name,
        }

        self.root = Path(root)
        self.symbols = symbols
        self.context_digest = resolved_context
        self.transform_digest = resolved_transform
        self.policy_version = resolved_version
        self.policy_version_semantics = resolved_version_semantics
        self._torch = torch
        self._policy = policy
        self._global_hook_module = torch.nn.modules.module
        self._observation_keys = observation_keys
        self._observation_shapes = shapes
        self._state = "open"
        self._sample_count = 0
        self._handle: Any | None = None
        self._hook_callback: Any | None = None
        self._hook_layout_at_attach: tuple[tuple[tuple[int, int], ...], ...] = ()

        with _ACTIVE_POLICIES_LOCK:
            existing_ref = _ACTIVE_POLICIES.get(policy)
            existing = None if existing_ref is None else existing_ref()
            if existing is not None:
                raise ValueError("a policy sample journal is already attached")
            if self._current_forward_hook_layout() != ((), (), (), (), ()):
                raise RuntimeError(
                    "pre-existing forward hooks (local or global) are unsupported"
                )
            _ACTIVE_POLICIES[policy] = weakref.ref(self)

        try:
            if self.root.exists() or self.root.is_symlink():
                raise FileExistsError(f"journal root already exists: {self.root}")
            self.root.mkdir(parents=True, exist_ok=False)
            samples_dir = self.root / "samples"
            samples_dir.mkdir(exist_ok=False)
            manifest = {
                "schema": PORTFOLIO_POLICY_SAMPLE_JOURNAL_SCHEMA,
                "sample_schema": _POLICY_SAMPLE_SCHEMA,
                "symbols": list(symbols),
                "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
                "observation_space": {
                    "kind": f"{type(observation_space).__module__}.{type(observation_space).__qualname__}",
                    "fields": descriptors,
                    "field_order": list(observation_keys),
                },
                "action_space": action_space_descriptor,
                "context_digest": resolved_context,
                "transform_digest": resolved_transform,
                "policy_version": resolved_version,
                "policy_version_semantics": resolved_version_semantics,
                "samples_directory": "samples",
            }
            atomic_write_bytes(
                self.root / "manifest.json", canonical_json_bytes(manifest)
            )
            self._hook_callback = cast(Any, self._on_forward)
            self._handle = policy.register_forward_hook(self._hook_callback)
            expected_layout = (
                ((self._handle.id, id(self._hook_callback)),),
                (),
                (),
                (),
                (),
            )
            if self._current_forward_hook_layout() != expected_layout:
                raise RuntimeError(
                    "forward-hook layout changed during journal attachment"
                )
            self._hook_layout_at_attach = expected_layout
        except BaseException as original_error:
            cleanup_error: BaseException | None = None
            try:
                self._remove_owned_hook()
            except BaseException as error:
                cleanup_error = error
            finally:
                self._release_policy_registration()
            if cleanup_error is not None:
                raise original_error from cleanup_error
            raise

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def is_poisoned(self) -> bool:
        return self._state == "poisoned"

    @property
    def is_closed(self) -> bool:
        return self._state == "closed"

    def _release_policy_registration(self) -> None:
        with _ACTIVE_POLICIES_LOCK:
            registered_ref = _ACTIVE_POLICIES.get(self._policy)
            if registered_ref is not None and registered_ref() is self:
                del _ACTIVE_POLICIES[self._policy]

    @staticmethod
    def _hook_mapping_layout(
        hooks: object, *, field: str
    ) -> tuple[tuple[int, int], ...]:
        if not isinstance(hooks, Mapping):
            raise RuntimeError(f"{field} forward-hook layout is unavailable")
        entries: list[tuple[int, int]] = []
        for hook_id, hook in hooks.items():
            if type(hook_id) is not int:
                raise RuntimeError(f"{field} forward-hook identifiers are invalid")
            entries.append((hook_id, id(hook)))
        return tuple(entries)

    def _current_forward_hook_layout(
        self,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        return (
            self._hook_mapping_layout(
                getattr(self._policy, "_forward_hooks", None),
                field="policy hooks",
            ),
            self._hook_mapping_layout(
                getattr(self._policy, "_forward_hooks_with_kwargs", None),
                field="policy kwargs hooks",
            ),
            self._hook_mapping_layout(
                getattr(self._policy, "_forward_hooks_always_called", None),
                field="policy always-called hooks",
            ),
            self._hook_mapping_layout(
                getattr(self._global_hook_module, "_global_forward_hooks", None),
                field="global hooks",
            ),
            self._hook_mapping_layout(
                getattr(
                    self._global_hook_module,
                    "_global_forward_hooks_always_called",
                    None,
                ),
                field="global always-called hooks",
            ),
        )

    def _validate_forward_hook_layout(self) -> None:
        if self._current_forward_hook_layout() != self._hook_layout_at_attach:
            raise RuntimeError(
                "policy or global forward-hook layout changed after journal attachment"
            )

    def _remove_owned_hook(self) -> None:
        handle = self._handle
        if handle is not None:
            handle.remove()
            self._handle = None

    def _on_forward(self, module: Any, inputs: tuple[Any, ...], output: object) -> None:
        if self._state != "open":
            raise RuntimeError(f"policy sample journal is {self._state}")
        try:
            self._validate_forward_hook_layout()
            records = self._records_for_forward(module, inputs, output)
            for record in records:
                sample_id = self._sample_count + 1
                record["sample_id"] = sample_id
                destination = self.root / "samples" / f"sample-{sample_id:012d}.json"
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(
                        f"sample record already exists: {destination}"
                    )
                atomic_write_bytes(destination, canonical_json_bytes(record))
                self._sample_count = sample_id
        except BaseException:
            self._state = "poisoned"
            raise

    def _records_for_forward(
        self, module: Any, inputs: tuple[Any, ...], output: object
    ) -> list[dict[str, object]]:
        torch = self._torch
        if module is not self._policy:
            raise RuntimeError("forward hook ran on an unexpected module")
        if len(inputs) != 1 or not isinstance(inputs[0], Mapping):
            raise TypeError("policy forward must receive one Dict observation")
        observations = inputs[0]
        if set(observations) != set(self._observation_keys):
            raise ValueError("policy input keys do not match its Dict space")

        input_rows: dict[str, tuple[str, tuple[int, ...], list[object]]] = {}
        batch_size: int | None = None
        for key in self._observation_keys:
            dtype, shape, values = _cpu_tensor_values(
                observations[key], torch=torch, field=f"input {key}"
            )
            expected_shape = self._observation_shapes[key]
            if len(shape) != len(expected_shape) + 1 or shape[1:] != expected_shape:
                raise ValueError(f"input {key} shape does not match its Dict space")
            if batch_size is None:
                batch_size = shape[0]
            elif shape[0] != batch_size:
                raise ValueError("policy input tensors have inconsistent batch sizes")
            if batch_size <= 0:
                raise ValueError("policy input batch must be non-empty")
            if not isinstance(values, list) or len(values) != batch_size:
                raise ValueError(f"input {key} must have a leading batch dimension")
            input_rows[key] = (dtype, expected_shape, values)

        if batch_size is None:
            raise ValueError("policy input batch is empty")
        if not isinstance(output, (tuple, list)) or len(output) != 3:
            raise TypeError("policy forward must return action, value, and log-prob")
        actions, values_tensor, log_probs_tensor = output
        action_dtype, action_shape, action_values = _cpu_tensor_values(
            actions, torch=torch, field="sampled actions"
        )
        if not action_dtype.startswith(("int", "uint")):
            raise TypeError("sampled action codes must use an integer tensor dtype")
        if action_shape != (batch_size, len(self.symbols)):
            raise ValueError("sampled action shape does not match batch and symbols")
        if not isinstance(action_values, list) or len(action_values) != batch_size:
            raise ValueError("sampled actions must have a leading batch dimension")

        value_dtype, value_shape, value_values = _cpu_tensor_values(
            values_tensor, torch=torch, field="critic values"
        )
        log_prob_dtype, log_prob_shape, log_prob_values = _cpu_tensor_values(
            log_probs_tensor, torch=torch, field="joint log probabilities"
        )
        if not value_dtype.startswith(("float", "bfloat")) or value_shape != (
            batch_size,
            1,
        ):
            raise ValueError("critic values must be finite floating (batch, 1) tensors")
        if not log_prob_dtype.startswith(("float", "bfloat")) or log_prob_shape != (
            batch_size,
        ):
            raise ValueError(
                "joint log probabilities must be finite floating batch vectors"
            )
        if not isinstance(value_values, list) or len(value_values) != batch_size:
            raise ValueError("critic values must have a leading batch dimension")
        if not isinstance(log_prob_values, list) or len(log_prob_values) != batch_size:
            raise ValueError(
                "joint log probabilities must have a leading batch dimension"
            )

        records: list[dict[str, object]] = []
        for batch_index in range(batch_size):
            row_actions = action_values[batch_index]
            if not isinstance(row_actions, list) or len(row_actions) != len(
                self.symbols
            ):
                raise ValueError("each sampled action row must match the symbol roster")
            action_codes: list[int] = []
            for code in row_actions:
                if type(code) is not int or not 0 <= code < 4:
                    raise ValueError("sampled action codes must be integers in [0, 4)")
                action_codes.append(code)
            row_value = value_values[batch_index]
            if not isinstance(row_value, list) or len(row_value) != 1:
                raise ValueError("each critic-value row must contain one prediction")
            record_inputs: dict[str, object] = {}
            for key in self._observation_keys:
                dtype, shape, rows = input_rows[key]
                record_inputs[key] = {
                    "dtype": dtype,
                    "shape": list(shape),
                    "values": rows[batch_index],
                }
            records.append(
                {
                    "schema": _POLICY_SAMPLE_SCHEMA,
                    "batch_index": batch_index,
                    "forward_batch_size": batch_size,
                    "forward_output_metadata": {
                        "actions": {
                            "dtype": action_dtype,
                            "shape": list(action_shape),
                        },
                        "value_prediction": {
                            "dtype": value_dtype,
                            "shape": list(value_shape),
                        },
                        "joint_log_probability": {
                            "dtype": log_prob_dtype,
                            "shape": list(log_prob_shape),
                        },
                    },
                    "symbols": list(self.symbols),
                    "inputs": record_inputs,
                    "raw_action_codes": action_codes,
                    "joint_log_probability": _strict_float(
                        log_prob_values[batch_index], field="joint log probability"
                    ),
                    "value_prediction": _strict_float(
                        row_value[0], field="critic value prediction"
                    ),
                    "policy_version": self.policy_version,
                    "policy_version_semantics": self.policy_version_semantics,
                    "context_digest": self.context_digest,
                    "transform_digest": self.transform_digest,
                }
            )
        return records

    def close(self) -> None:
        if self._state == "closed":
            return
        try:
            self._remove_owned_hook()
        finally:
            self._state = "closed"
            self._release_policy_registration()

    def __enter__(self) -> PortfolioPolicySampleJournal:
        if self._state != "open":
            raise RuntimeError(f"policy sample journal is {self._state}")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


__all__ = [
    "PORTFOLIO_POLICY_SAMPLE_JOURNAL_SCHEMA",
    "PortfolioPolicySampleJournal",
]
