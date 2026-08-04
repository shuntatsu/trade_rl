"""Three-update training smoke test for CUDA runtime validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

TINY_TRAINING_SMOKE_SCHEMA: Final = "tiny_cuda_training_smoke_v1"
TINY_TRAINING_UPDATES: Final = 3


@dataclass(frozen=True, slots=True)
class TinyTrainingSmokeResult:
    """Evidence from exactly three optimizer updates."""

    updates: int
    losses: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    parameter_delta_l2: float
    device: str
    device_type: str
    dtype: str
    torch_version: str
    cuda_version: str | None
    device_name: str
    compute_capability: str | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    schema_version: str = TINY_TRAINING_SMOKE_SCHEMA

    def __post_init__(self) -> None:
        if self.updates != TINY_TRAINING_UPDATES:
            raise ValueError("tiny training smoke must execute exactly three updates")
        if len(self.losses) != self.updates:
            raise ValueError("loss count must match update count")
        if len(self.gradient_norms) != self.updates:
            raise ValueError("gradient norm count must match update count")
        if any(not math.isfinite(value) for value in self.losses):
            raise ValueError("losses must be finite")
        if any(
            not math.isfinite(value) or value <= 0.0 for value in self.gradient_norms
        ):
            raise ValueError("gradient norms must be finite and positive")
        if not math.isfinite(self.parameter_delta_l2) or self.parameter_delta_l2 <= 0.0:
            raise ValueError("parameters must change during the smoke test")
        for field, value in (
            ("peak_allocated_bytes", self.peak_allocated_bytes),
            ("peak_reserved_bytes", self.peak_reserved_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field} must be a non-negative integer or None")
        if self.schema_version != TINY_TRAINING_SMOKE_SCHEMA:
            raise ValueError("unsupported tiny training smoke schema")

    def payload(self) -> dict[str, object]:
        """Return canonical JSON-compatible evidence."""

        return asdict(self)


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for the training smoke test") from error
    return torch


def run_tiny_training_smoke(
    *,
    device: str = "cuda:0",
    require_cuda: bool = True,
    seed: int = 7,
) -> TinyTrainingSmokeResult:
    """Run exactly three tiny optimizer updates and validate finite device state."""

    torch = _import_torch()
    resolved_device = torch.device(device)
    if require_cuda and resolved_device.type != "cuda":
        raise RuntimeError("CUDA device is required for this smoke test")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required but torch.cuda is unavailable")

    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(resolved_device)
        torch.cuda.reset_peak_memory_stats(resolved_device)

    dtype = torch.float64
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 2),
    ).to(device=resolved_device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(16, 8, device=resolved_device, dtype=dtype)
    targets = torch.randn(16, 2, device=resolved_device, dtype=dtype)
    initial_parameters = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )

    losses: list[float] = []
    gradient_norms: list[float] = []
    for update in range(TINY_TRAINING_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite loss at update {update + 1}")
        loss.backward()

        gradient_squared = 0.0
        for name, parameter in model.named_parameters():
            if parameter.device != resolved_device:
                raise RuntimeError(f"device mismatch for parameter {name}")
            if parameter.dtype != dtype:
                raise RuntimeError(f"dtype mismatch for parameter {name}")
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(f"missing gradient for parameter {name}")
            if not bool(torch.isfinite(gradient).all().item()):
                raise RuntimeError(f"non-finite gradient for parameter {name}")
            gradient_squared += float(torch.sum(gradient * gradient).item())

        optimizer.step()
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        losses.append(float(loss.item()))
        gradient_norms.append(math.sqrt(gradient_squared))

    parameter_delta_squared = 0.0
    for initial, parameter in zip(
        initial_parameters,
        model.parameters(),
        strict=True,
    ):
        if not bool(torch.isfinite(parameter).all().item()):
            raise RuntimeError("non-finite parameter after optimizer update")
        delta = parameter.detach() - initial
        parameter_delta_squared += float(torch.sum(delta * delta).item())

    if resolved_device.type == "cuda":
        device_index = resolved_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_index)
        device_name = torch.cuda.get_device_name(device_index)
        compute_capability = f"{major}.{minor}"
        peak_allocated = int(torch.cuda.max_memory_allocated(resolved_device))
        peak_reserved = int(torch.cuda.max_memory_reserved(resolved_device))
    else:
        device_name = "cpu"
        compute_capability = None
        peak_allocated = None
        peak_reserved = None

    return TinyTrainingSmokeResult(
        updates=TINY_TRAINING_UPDATES,
        losses=tuple(losses),
        gradient_norms=tuple(gradient_norms),
        parameter_delta_l2=math.sqrt(parameter_delta_squared),
        device=str(resolved_device),
        device_type=resolved_device.type,
        dtype="float64",
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        device_name=device_name,
        compute_capability=compute_capability,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution for test harnesses; production smoke should omit this.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke test and write canonical evidence to stdout or a file."""

    arguments = _parser().parse_args(argv)
    try:
        result = run_tiny_training_smoke(
            device=arguments.device,
            require_cuda=not arguments.allow_cpu,
        )
    except Exception as error:
        print(f"CUDA_SMOKE_FAIL: {error}", file=sys.stderr)
        return 1

    serialized = (
        json.dumps(
            result.payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    if arguments.output is None:
        sys.stdout.write(serialized)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TINY_TRAINING_SMOKE_SCHEMA",
    "TINY_TRAINING_UPDATES",
    "TinyTrainingSmokeResult",
    "main",
    "run_tiny_training_smoke",
]
