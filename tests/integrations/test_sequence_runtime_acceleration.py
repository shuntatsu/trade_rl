from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import trade_rl.integrations.compact_rollout_buffer as compact_rollout
import trade_rl.integrations.sb3_training as sb3_training
from trade_rl.integrations.compact_rollout_buffer import IndexBackedDictRolloutBuffer
from trade_rl.rl.training import ResidualTrainingConfig


def _config(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 16,
        "gamma": 0.99,
        "seeds": (0,),
        "n_steps": 4,
        "batch_size": 4,
        "n_epochs": 1,
        "observation_encoder": "flat_mlp",
        "device": "cuda",
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


def test_sequence_runtime_settings_are_identity_bound_and_sequence_only() -> None:
    baseline = _config(
        observation_encoder=("hierarchical_sequence_v2"),
        policy="MultiInputPolicy",
    )
    accelerated = _config(
        observation_encoder=("hierarchical_sequence_v2"),
        policy="MultiInputPolicy",
        sequence_compile=True,
        sequence_compile_mode="reduce-overhead",
        sequence_transfer_mode="pinned_non_blocking",
    )

    assert accelerated.sequence_compile is True
    assert accelerated.sequence_compile_mode == "reduce-overhead"
    assert accelerated.sequence_transfer_mode == "pinned_non_blocking"
    assert accelerated.digest_payload() != baseline.digest_payload()

    with pytest.raises(ValueError, match="sequence_compile.*observation_encoder"):
        _config(sequence_compile=True)
    with pytest.raises(ValueError, match="sequence_transfer_mode.*observation_encoder"):
        _config(sequence_transfer_mode="pinned_non_blocking")
    with pytest.raises(ValueError, match="sequence_compile_mode"):
        _config(
            observation_encoder=("hierarchical_sequence_v2"),
            policy="MultiInputPolicy",
            sequence_compile=True,
            sequence_compile_mode="unsafe-mode",
        )
    with pytest.raises(ValueError, match="sequence_transfer_mode"):
        _config(
            observation_encoder=("hierarchical_sequence_v2"),
            policy="MultiInputPolicy",
            sequence_transfer_mode="background-magic",
        )


def test_maintained_full_cuda_configs_enable_sequence_runtime() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    direct = json.loads(
        (
            repository_root / "examples/binance-multitimeframe/training-full.json"
        ).read_text(encoding="utf-8")
    )["training"]
    walk_forward_config = json.loads(
        (
            repository_root / "examples/binance-multitimeframe/walk-forward-full.json"
        ).read_text(encoding="utf-8")
    )
    run_file = walk_forward_config["candidates"][0]["run_file"]
    walk_forward = json.loads(
        (repository_root / "examples/binance-multitimeframe" / run_file).read_text(
            encoding="utf-8"
        )
    )["training"]

    for training in (direct, walk_forward):
        assert training["sequence_compile"] is False
        assert training["sequence_compile_mode"] == "reduce-overhead"
        assert training["sequence_transfer_mode"] == "pinned_non_blocking"


class _FakeCpuTensor:
    def __init__(self) -> None:
        self.contiguous_calls = 0
        self.shape = (1, 2, 1)
        self.dtype = "float16"

    def is_contiguous(self) -> bool:
        return True

    def contiguous(self) -> _FakeCpuTensor:
        self.contiguous_calls += 1
        return self


class _FakePinnedTensor:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.shape = (1, 2, 1)
        self.dtype = "float16"
        self.copy_source: object | None = None
        self.to_device: object | None = None
        self.non_blocking: bool | None = None
        self.gpu_result = object()

    def copy_(self, source: object) -> _FakePinnedTensor:
        self.calls.append("copy")
        self.copy_source = source
        return self

    def to(self, device: object, *, non_blocking: bool = False) -> object:
        self.calls.append("to")
        self.to_device = device
        self.non_blocking = non_blocking
        return self.gpu_result


class _FakeCudaEvent:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def record(self, stream: object) -> None:
        assert stream == "current_cuda_stream"
        self.calls.append("record")

    def synchronize(self) -> None:
        self.calls.append("synchronize")


class _FakeCuda:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.events: list[_FakeCudaEvent] = []

    def Event(self) -> _FakeCudaEvent:  # noqa: N802 - mirror torch.cuda API
        event = _FakeCudaEvent(self.calls)
        self.events.append(event)
        return event

    def current_stream(self, device: object) -> str:
        assert getattr(device, "type", None) == "cuda"
        return "current_cuda_stream"


class _FakeTransferTorch:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.cpu_tensor = _FakeCpuTensor()
        self.pinned_tensors: list[_FakePinnedTensor] = []
        self.asarray_value: object | None = None
        self.empty_like_args: list[tuple[object, object, bool]] = []
        self.cuda_device = SimpleNamespace(type="cuda")
        self.cuda = _FakeCuda(self.calls)

    def device(self, value: object) -> object:
        assert value == "cuda"
        return self.cuda_device

    def as_tensor(self, value: object) -> _FakeCpuTensor:
        self.asarray_value = value
        return self.cpu_tensor

    def empty_like(
        self,
        source: object,
        *,
        device: object,
        pin_memory: bool,
    ) -> _FakePinnedTensor:
        self.calls.append("allocate")
        self.empty_like_args.append((source, device, pin_memory))
        pinned = _FakePinnedTensor(self.calls)
        self.pinned_tensors.append(pinned)
        return pinned


def test_pinned_sequence_transfer_uses_non_blocking_cuda_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTransferTorch()
    monkeypatch.setattr(compact_rollout, "torch", fake_torch)
    buffer = object.__new__(IndexBackedDictRolloutBuffer)
    buffer.device = "cuda"
    buffer.sequence_transfer_mode = "pinned_non_blocking"
    values = np.asarray([[[1.0], [2.0]]], dtype=np.float16)

    result = buffer._sequence_to_torch(values, staging_key="sequence_1h_values")

    pinned = fake_torch.pinned_tensors[0]
    assert result is pinned.gpu_result
    assert fake_torch.asarray_value is values
    assert fake_torch.empty_like_args == [(fake_torch.cpu_tensor, "cpu", True)]
    assert pinned.copy_source is fake_torch.cpu_tensor
    assert pinned.to_device is fake_torch.cuda_device
    assert pinned.non_blocking is True
    assert fake_torch.calls == ["allocate", "copy", "to", "record"]


def test_pinned_sequence_staging_is_reused_across_rollout_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTransferTorch()
    monkeypatch.setattr(compact_rollout, "torch", fake_torch)
    buffer = object.__new__(IndexBackedDictRolloutBuffer)
    buffer.device = "cuda"
    buffer.sequence_transfer_mode = "pinned_non_blocking"
    buffer._compact_keys = ()
    buffer.buffer_size = 1
    buffer.n_envs = 1
    buffer.action_dim = 1
    values = np.asarray([[[1.0], [2.0]]], dtype=np.float16)

    first = buffer._sequence_to_torch(values, staging_key="sequence_1h_values")
    buffer.reset()
    second = buffer._sequence_to_torch(values, staging_key="sequence_1h_values")

    assert first is second
    assert len(fake_torch.pinned_tensors) == 1
    assert len(fake_torch.cuda.events) == 2
    assert fake_torch.calls == [
        "allocate",
        "copy",
        "to",
        "record",
        "synchronize",
        "copy",
        "to",
        "record",
    ]


def test_pinned_sequence_staging_allocates_once_per_sequence_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTransferTorch()
    monkeypatch.setattr(compact_rollout, "torch", fake_torch)
    buffer = object.__new__(IndexBackedDictRolloutBuffer)
    buffer.device = "cuda"
    buffer.sequence_transfer_mode = "pinned_non_blocking"
    values = np.asarray([[[1.0], [2.0]]], dtype=np.float16)

    buffer._sequence_to_torch(values, staging_key="sequence_1h_values")
    buffer._sequence_to_torch(values, staging_key="sequence_1h_available")
    buffer._sequence_to_torch(values, staging_key="sequence_1h_values")

    assert len(fake_torch.pinned_tensors) == 2
    assert len(fake_torch.empty_like_args) == 2


def test_synchronous_sequence_transfer_preserves_existing_to_torch_path() -> None:
    buffer = object.__new__(IndexBackedDictRolloutBuffer)
    buffer.sequence_transfer_mode = "synchronous"
    sentinel = object()
    calls: list[object] = []

    def to_torch(value: object, *args: Any, **kwargs: Any) -> object:
        del args, kwargs
        calls.append(value)
        return sentinel

    buffer.to_torch = to_torch  # type: ignore[method-assign]
    values = np.asarray([1.0], dtype=np.float32)

    assert buffer._sequence_to_torch(values) is sentinel
    assert calls == [values]


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def compile(self, **kwargs: object) -> None:
        self.calls.append(dict(kwargs))


class _FakeCompileTorch:
    __version__ = "2.3.1"

    def __init__(self, device_type: str) -> None:
        self.device_type = device_type

    def device(self, value: object) -> object:
        del value
        return SimpleNamespace(type=self.device_type)


def _runtime_config(**overrides: object) -> object:
    values: dict[str, object] = {
        "observation_encoder": "invalid_legacy_combination",
        "sequence_compile": True,
        "sequence_compile_mode": "reduce-overhead",
        "sequence_transfer_mode": "pinned_non_blocking",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sequence_compile_targets_only_feature_extractor_and_records_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    configure = getattr(sb3_training, "_configure_sequence_runtime", None)
    assert callable(configure)
    extractor = _FakeExtractor()
    model = SimpleNamespace(
        device="cuda",
        policy=SimpleNamespace(features_extractor=extractor),
    )

    evidence = configure(
        _FakeCompileTorch("cuda"),
        model,
        _runtime_config(),
    )

    assert extractor.calls == [
        {
            "mode": "reduce-overhead",
            "fullgraph": False,
            "dynamic": False,
        }
    ]
    assert evidence == {
        "compile_enabled": True,
        "compile_mode": "reduce-overhead",
        "compile_target": "_FakeExtractor",
        "fullgraph": False,
        "dynamic": False,
        "inductor_compile_threads": None,
        "sequence_transfer_mode": "pinned_non_blocking",
        "torch_version": "2.3.1",
        "schema_version": "sequence_runtime_v2",
    }


def test_disabled_sequence_compile_does_not_touch_feature_extractor() -> None:
    configure = getattr(sb3_training, "_configure_sequence_runtime", None)
    assert callable(configure)
    extractor = _FakeExtractor()
    model = SimpleNamespace(
        device="cpu",
        policy=SimpleNamespace(features_extractor=extractor),
    )

    evidence = configure(
        _FakeCompileTorch("cpu"),
        model,
        _runtime_config(
            sequence_compile=False,
            sequence_transfer_mode="synchronous",
        ),
    )

    assert extractor.calls == []
    assert evidence["compile_enabled"] is False
    assert evidence["compile_target"] is None
    assert evidence["sequence_transfer_mode"] == "synchronous"


def test_requested_sequence_compile_fails_closed_without_cuda() -> None:
    configure = getattr(sb3_training, "_configure_sequence_runtime", None)
    assert callable(configure)
    model = SimpleNamespace(
        device="cpu",
        policy=SimpleNamespace(features_extractor=_FakeExtractor()),
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        configure(_FakeCompileTorch("cpu"), model, _runtime_config())
