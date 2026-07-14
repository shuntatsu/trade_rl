"""Deterministic Stable-Baselines3 actor exports with parity verification."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import nn

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.domain.policies import PolicyEnsembleManifest

EXPORT_MANIFEST_NAME: Final = "export.json"
EXPORT_SCHEMA: Final = "policy_export_v1"
_SUPPORTED_ALGORITHMS: Final = frozenset({"ppo", "sac", "td3", "tqc"})


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_model(algorithm: str, checkpoint_path: Path) -> Any:
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(checkpoint_path), device="cpu")
    if algorithm == "sac":
        from stable_baselines3 import SAC

        return SAC.load(str(checkpoint_path), device="cpu")
    if algorithm == "td3":
        from stable_baselines3 import TD3

        return TD3.load(str(checkpoint_path), device="cpu")
    if algorithm == "tqc":
        from sb3_contrib import TQC

        return TQC.load(str(checkpoint_path), device="cpu")
    raise ValueError("unsupported policy export algorithm")


class _DeterministicActor(nn.Module):
    def __init__(self, policy: Any) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.policy._predict(observation, deterministic=True)


@dataclass(frozen=True, slots=True)
class ExportRecord:
    format: str
    status: str
    path: str | None
    digest: str | None
    size_bytes: int | None
    max_abs_error: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.format not in {"onnx", "torchscript"}:
            raise ValueError("unsupported policy export format")
        if self.status not in {"verified", "unsupported"}:
            raise ValueError("unsupported policy export status")
        if self.status == "verified":
            if self.path is None or self.digest is None or self.size_bytes is None:
                raise ValueError("verified export requires a file identity")
            require_sha256(self.digest, field="export.digest")
            if self.size_bytes < 0:
                raise ValueError("export size must be non-negative")
            if self.max_abs_error is None or not np.isfinite(self.max_abs_error):
                raise ValueError("verified export requires a finite parity error")
            if self.reason is not None:
                raise ValueError("verified export cannot contain an error reason")
        else:
            if any(
                value is not None
                for value in (
                    self.path,
                    self.digest,
                    self.size_bytes,
                    self.max_abs_error,
                )
            ):
                raise ValueError("unsupported export cannot declare a file")
            if not self.reason:
                raise ValueError("unsupported export requires a reason")


@dataclass(frozen=True, slots=True)
class ExportManifest:
    digest: str
    source_checkpoint_digest: str
    algorithm: str
    observation_size: int
    action_size: int
    action_spec_digest: str
    normalizer_digest: str | None
    tolerance: float
    exports: tuple[ExportRecord, ...]
    schema_version: str = EXPORT_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.digest, field="export_manifest.digest")
        require_sha256(
            self.source_checkpoint_digest,
            field="source_checkpoint_digest",
        )
        require_sha256(self.action_spec_digest, field="action_spec_digest")
        if self.normalizer_digest is not None:
            require_sha256(self.normalizer_digest, field="normalizer_digest")
        if self.algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError("unsupported export algorithm")
        if self.observation_size <= 0 or self.action_size <= 0:
            raise ValueError("export dimensions must be positive")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("export tolerance must be finite and positive")
        if not self.exports:
            raise ValueError("export manifest must contain export records")
        if len({item.format for item in self.exports}) != len(self.exports):
            raise ValueError("export formats must be unique")
        if not any(item.status == "verified" for item in self.exports):
            raise ValueError(
                "export manifest must contain at least one verified export"
            )
        if self.schema_version != EXPORT_SCHEMA:
            raise ValueError("unsupported export manifest schema")
        expected = content_digest(self.digest_payload())
        if self.digest != expected:
            raise ValueError("export manifest digest does not match content")

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_size": self.action_size,
            "action_spec_digest": self.action_spec_digest,
            "algorithm": self.algorithm,
            "exports": self.exports,
            "normalizer_digest": self.normalizer_digest,
            "observation_size": self.observation_size,
            "schema_version": self.schema_version,
            "source_checkpoint_digest": self.source_checkpoint_digest,
            "tolerance": self.tolerance,
        }


def _corpus(observation_size: int) -> np.ndarray:
    linear = np.linspace(-0.5, 0.5, observation_size, dtype=np.float32)
    rng = np.random.default_rng(0)
    stochastic = rng.normal(0.0, 0.75, size=(8, observation_size)).astype(np.float32)
    stochastic = np.clip(stochastic, -3.0, 3.0)
    alternating = np.where(
        np.arange(observation_size) % 2 == 0,
        1.0,
        -1.0,
    ).astype(np.float32)
    sparse = np.zeros(observation_size, dtype=np.float32)
    sparse[:: max(1, observation_size // 16)] = 1.0
    return np.concatenate(
        (
            np.stack(
                (
                    np.zeros(observation_size, dtype=np.float32),
                    np.full(observation_size, 0.25, dtype=np.float32),
                    linear,
                    -linear,
                    alternating,
                    sparse,
                ),
                axis=0,
            ),
            stochastic,
        ),
        axis=0,
    )


def _expected_actions(model: Any, corpus: np.ndarray, action_size: int) -> np.ndarray:
    actions, _ = model.predict(corpus, deterministic=True)
    resolved = np.asarray(actions, dtype=np.float32).reshape(corpus.shape[0], -1)
    if resolved.shape != (corpus.shape[0], action_size):
        raise ValueError("SB3 actor output does not match the export action size")
    if not np.isfinite(resolved).all():
        raise ValueError("SB3 actor output must be finite")
    return resolved


def _parity_error(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    action_size: int,
) -> float:
    resolved = np.asarray(actual, dtype=np.float32).reshape(expected.shape[0], -1)
    if resolved.shape != (expected.shape[0], action_size):
        raise ValueError("exported actor output shape mismatch")
    if not np.isfinite(resolved).all():
        raise ValueError("exported actor output must be finite")
    return float(np.max(np.abs(resolved - expected), initial=0.0))


def _export_torchscript(
    actor: nn.Module,
    path: Path,
    corpus: np.ndarray,
) -> np.ndarray:
    example = torch.from_numpy(corpus)
    with torch.no_grad():
        traced = torch.jit.trace(actor, example, strict=False)
        traced.save(str(path))
        restored = torch.jit.load(str(path), map_location="cpu")
        return restored(example).detach().cpu().numpy()


def _export_onnx(actor: nn.Module, path: Path, corpus: np.ndarray) -> np.ndarray:
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "ONNX export requires the trade-rl[export] dependencies"
        ) from error
    example = torch.from_numpy(corpus[:1])
    torch.onnx.export(
        actor,
        (example,),
        str(path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(["action"], {"observation": corpus})
    return np.asarray(outputs[0], dtype=np.float32)


def _verified_record(
    *,
    format_name: str,
    path: Path,
    relative_path: str,
    error: float,
) -> ExportRecord:
    return ExportRecord(
        format=format_name,
        status="verified",
        path=relative_path,
        digest=_file_digest(path),
        size_bytes=path.stat().st_size,
        max_abs_error=error,
    )


def export_policy_actor(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    algorithm: str,
    observation_size: int,
    action_size: int,
    action_spec_digest: str,
    normalizer_digest: str | None,
    onnx: bool,
    torchscript: bool,
    tolerance: float,
) -> ExportManifest:
    """Export and verify one deterministic actor from an SB3 checkpoint."""

    algorithm = algorithm.lower()
    if algorithm not in _SUPPORTED_ALGORITHMS:
        raise ValueError("unsupported policy export algorithm")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"policy checkpoint is missing: {checkpoint_path}")
    if not onnx and not torchscript:
        raise ValueError("at least one policy export format must be requested")
    if observation_size <= 0 or action_size <= 0:
        raise ValueError("export dimensions must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("export tolerance must be finite and positive")
    require_sha256(action_spec_digest, field="action_spec_digest")
    if normalizer_digest is not None:
        require_sha256(normalizer_digest, field="normalizer_digest")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(algorithm, checkpoint_path)
    policy = model.policy.to("cpu")
    policy.set_training_mode(False)
    actor = _DeterministicActor(policy).eval()
    corpus = _corpus(observation_size)
    expected = _expected_actions(model, corpus, action_size)
    records: list[ExportRecord] = []

    if torchscript:
        path = output_dir / "policy.torchscript.pt"
        try:
            actual = _export_torchscript(actor, path, corpus)
            max_error = _parity_error(expected, actual, action_size=action_size)
            if max_error > tolerance:
                raise ValueError(
                    f"TorchScript parity error {max_error} exceeds tolerance {tolerance}"
                )
            records.append(
                _verified_record(
                    format_name="torchscript",
                    path=path,
                    relative_path=path.name,
                    error=max_error,
                )
            )
        except Exception as export_error:
            path.unlink(missing_ok=True)
            records.append(
                ExportRecord(
                    format="torchscript",
                    status="unsupported",
                    path=None,
                    digest=None,
                    size_bytes=None,
                    max_abs_error=None,
                    reason=str(export_error),
                )
            )

    if onnx:
        path = output_dir / "policy.onnx"
        try:
            actual = _export_onnx(actor, path, corpus)
            max_error = _parity_error(expected, actual, action_size=action_size)
            if max_error > tolerance:
                raise ValueError(
                    f"ONNX parity error {max_error} exceeds tolerance {tolerance}"
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        records.append(
            _verified_record(
                format_name="onnx",
                path=path,
                relative_path=path.name,
                error=max_error,
            )
        )

    source_digest = _file_digest(checkpoint_path)
    temporary_payload = {
        "action_size": action_size,
        "action_spec_digest": action_spec_digest,
        "algorithm": algorithm,
        "exports": tuple(records),
        "normalizer_digest": normalizer_digest,
        "observation_size": observation_size,
        "schema_version": EXPORT_SCHEMA,
        "source_checkpoint_digest": source_digest,
        "tolerance": tolerance,
    }
    manifest = ExportManifest(
        digest=content_digest(temporary_payload),
        source_checkpoint_digest=source_digest,
        algorithm=algorithm,
        observation_size=observation_size,
        action_size=action_size,
        action_spec_digest=action_spec_digest,
        normalizer_digest=normalizer_digest,
        tolerance=tolerance,
        exports=tuple(records),
    )
    _atomic_write(output_dir / EXPORT_MANIFEST_NAME, canonical_json_bytes(manifest))
    return manifest


def export_ensemble_members(
    *,
    root: Path,
    ensemble: PolicyEnsembleManifest,
    algorithm: str,
    onnx: bool,
    torchscript: bool,
    tolerance: float,
) -> tuple[ExportManifest, ...]:
    """Export every ensemble member; any required ONNX failure rejects the run."""

    if ensemble.observation_size is None or ensemble.action_spec_digest is None:
        raise ValueError("ensemble identity is incomplete for policy export")
    manifests: list[ExportManifest] = []
    for index, member in enumerate(ensemble.members):
        member_root = root / "members" / f"member-{index:03d}"
        checkpoint = member_root / "policy.zip"
        if _file_digest(checkpoint) != member.checkpoint_digest:
            raise ValueError("ensemble checkpoint digest mismatch before export")
        manifests.append(
            export_policy_actor(
                checkpoint_path=checkpoint,
                output_dir=member_root,
                algorithm=algorithm,
                observation_size=ensemble.observation_size,
                action_size=ensemble.action_size,
                action_spec_digest=ensemble.action_spec_digest,
                normalizer_digest=ensemble.normalizer_digest,
                onnx=onnx,
                torchscript=torchscript,
                tolerance=tolerance,
            )
        )
    return tuple(manifests)
