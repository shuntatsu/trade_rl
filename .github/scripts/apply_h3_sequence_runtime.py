from __future__ import annotations

import json
from pathlib import Path


def replace_once(source: str, old: str, new: str, *, seam: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{seam} changed: expected one exact match, got {count}")
    return source.replace(old, new)


training_path = Path("trade_rl/rl/training.py")
training = training_path.read_text(encoding="utf-8")
training = replace_once(
    training,
    """    sequence_dropout: float = 0.05
    max_policy_parameters: int = 12_000_000
""",
    """    sequence_dropout: float = 0.05
    sequence_compile: bool = False
    sequence_compile_mode: str = "reduce-overhead"
    sequence_transfer_mode: str = "synchronous"
    max_policy_parameters: int = 12_000_000
""",
    seam="sequence runtime fields",
)
training = replace_once(
    training,
    """        if not isinstance(self.sequence_encoder, bool):
            raise ValueError("sequence_encoder must be a boolean")
        if self.sequence_capacity not in {"standard", "compact"}:
""",
    """        if not isinstance(self.sequence_encoder, bool):
            raise ValueError("sequence_encoder must be a boolean")
        if not isinstance(self.sequence_compile, bool):
            raise ValueError("sequence_compile must be a boolean")
        if self.sequence_compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
        }:
            raise ValueError(
                "sequence_compile_mode must be default, reduce-overhead, or max-autotune"
            )
        if self.sequence_transfer_mode not in {
            "synchronous",
            "pinned_non_blocking",
        }:
            raise ValueError(
                "sequence_transfer_mode must be synchronous or pinned_non_blocking"
            )
        if (
            not self.sequence_compile
            and self.sequence_compile_mode != "reduce-overhead"
        ):
            raise ValueError(
                "sequence_compile_mode is inactive when sequence_compile is false"
            )
        if self.sequence_capacity not in {"standard", "compact"}:
""",
    seam="sequence runtime validation",
)
training = replace_once(
    training,
    """                    ("sequence_attention_layers", self.sequence_attention_layers, 2),
                    ("sequence_dropout", self.sequence_dropout, 0.05),
                ),
""",
    """                    ("sequence_attention_layers", self.sequence_attention_layers, 2),
                    ("sequence_dropout", self.sequence_dropout, 0.05),
                    ("sequence_compile", self.sequence_compile, False),
                    (
                        "sequence_compile_mode",
                        self.sequence_compile_mode,
                        "reduce-overhead",
                    ),
                    (
                        "sequence_transfer_mode",
                        self.sequence_transfer_mode,
                        "synchronous",
                    ),
                ),
""",
    seam="inactive sequence runtime defaults",
)
training = replace_once(
    training,
    """            "sequence_attention_layers": self.sequence_attention_layers,
            "sequence_dropout": self.sequence_dropout,
            "max_policy_parameters": self.max_policy_parameters,
""",
    """            "sequence_attention_layers": self.sequence_attention_layers,
            "sequence_dropout": self.sequence_dropout,
            "sequence_compile": self.sequence_compile,
            "sequence_compile_mode": self.sequence_compile_mode,
            "sequence_transfer_mode": self.sequence_transfer_mode,
            "max_policy_parameters": self.max_policy_parameters,
""",
    seam="sequence runtime digest",
)
training_path.write_text(training, encoding="utf-8")


buffer_path = Path("trade_rl/integrations/compact_rollout_buffer.py")
buffer = buffer_path.read_text(encoding="utf-8")
buffer = replace_once(
    buffer,
    """import numpy as np
from gymnasium import spaces
""",
    """import numpy as np
import torch
from gymnasium import spaces
""",
    seam="compact rollout torch import",
)
buffer = replace_once(
    buffer,
    """_FLOAT16_MAX = float(np.finfo(np.float16).max)


@dataclass""",
    """_FLOAT16_MAX = float(np.finfo(np.float16).max)
_SEQUENCE_TRANSFER_MODES = frozenset({"synchronous", "pinned_non_blocking"})


def _validate_sequence_transfer_mode(value: str) -> str:
    if value not in _SEQUENCE_TRANSFER_MODES:
        raise ValueError(
            "sequence_transfer_mode must be synchronous or pinned_non_blocking"
        )
    return value


@dataclass""",
    seam="sequence transfer validation helper",
)
buffer = replace_once(
    buffer,
    """        *,
        sequence_reconstructor: SequenceRolloutReconstructor | None = None,
    ) -> None:
""",
    """        *,
        sequence_reconstructor: SequenceRolloutReconstructor | None = None,
        sequence_transfer_mode: str = "synchronous",
    ) -> None:
""",
    seam="rollout constructor transfer parameter",
)
buffer = replace_once(
    buffer,
    """        self.sequence_reconstructor = sequence_reconstructor
        super().__init__(
""",
    """        self.sequence_reconstructor = sequence_reconstructor
        self.sequence_transfer_mode = _validate_sequence_transfer_mode(
            sequence_transfer_mode
        )
        super().__init__(
""",
    seam="rollout transfer state",
)
buffer = replace_once(
    buffer,
    """    def bind_sequence_reconstructor(
        self, reconstructor: SequenceRolloutReconstructor
    ) -> None:
        if not isinstance(reconstructor, SequenceRolloutReconstructor):
            raise TypeError("sequence reconstructor has an invalid type")
        self.sequence_reconstructor = reconstructor

    def reset(self) -> None:
""",
    """    def bind_sequence_reconstructor(
        self,
        reconstructor: SequenceRolloutReconstructor,
        *,
        sequence_transfer_mode: str | None = None,
    ) -> None:
        if not isinstance(reconstructor, SequenceRolloutReconstructor):
            raise TypeError("sequence reconstructor has an invalid type")
        self.sequence_reconstructor = reconstructor
        if sequence_transfer_mode is not None:
            self.sequence_transfer_mode = _validate_sequence_transfer_mode(
                sequence_transfer_mode
            )

    def reset(self) -> None:
""",
    seam="resume transfer binding",
)
buffer = replace_once(
    buffer,
    """    def _materialize_sequence_observations(
        self, reconstructor: SequenceRolloutReconstructor
    ) -> dict[str, Any]:
""",
    """    def _sequence_to_torch(self, value: np.ndarray) -> Any:
        """Convert one reconstructed sequence tensor using the configured path."""

        mode = _validate_sequence_transfer_mode(self.sequence_transfer_mode)
        if mode == "synchronous":
            return self.to_torch(value)
        device = torch.device(self.device)
        if device.type != "cuda":
            raise RuntimeError(
                "pinned_non_blocking sequence transfer requires a CUDA device"
            )
        cpu_tensor = torch.as_tensor(value)
        if not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()
        pinned = torch.empty_like(cpu_tensor, device="cpu", pin_memory=True)
        pinned.copy_(cpu_tensor)
        return pinned.to(device, non_blocking=True)

    def _materialize_sequence_observations(
        self, reconstructor: SequenceRolloutReconstructor
    ) -> dict[str, Any]:
""",
    seam="sequence transfer helper",
)
buffer = replace_once(
    buffer,
    """        with measure_sequence_tensor_conversion():
            cached = {key: self.to_torch(value) for key, value in reconstructed.items()}
""",
    """        with measure_sequence_tensor_conversion():
            cached = {
                key: self._sequence_to_torch(value)
                for key, value in reconstructed.items()
            }
""",
    seam="sequence materialization transfer",
)
buffer_path.write_text(buffer, encoding="utf-8")


sb3_path = Path("trade_rl/integrations/sb3_training.py")
sb3 = sb3_path.read_text(encoding="utf-8")
sb3 = replace_once(
    sb3,
    """    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "sequence_encoder_autocast": ("bfloat16" if bf16_supported else "disabled"),
    }


""",
    """    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "sequence_encoder_autocast": ("bfloat16" if bf16_supported else "disabled"),
    }


def _configure_sequence_runtime(
    torch: Any,
    model: Any,
    config: ResidualTrainingConfig,
) -> dict[str, object]:
    """Apply the identity-bound sequence runtime after construction or load."""

    compile_enabled = bool(config.sequence_compile)
    compile_target: str | None = None
    if compile_enabled:
        resolved_device = torch.device(model.device)
        if resolved_device.type != "cuda":
            raise RuntimeError("sequence_compile requires a resolved CUDA device")
        extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
        compile_module = getattr(extractor, "compile", None)
        if not callable(compile_module):
            raise RuntimeError(
                "sequence feature extractor does not support in-place compile"
            )
        compile_module(
            mode=config.sequence_compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        compile_target = type(extractor).__name__
    return {
        "compile_enabled": compile_enabled,
        "compile_mode": config.sequence_compile_mode,
        "compile_target": compile_target,
        "fullgraph": False,
        "dynamic": False,
        "sequence_transfer_mode": config.sequence_transfer_mode,
        "torch_version": str(torch.__version__),
        "schema_version": "sequence_runtime_v1",
    }


""",
    seam="sequence runtime configurator",
)
sb3 = replace_once(
    sb3,
    """                    rollout_kwargs["rollout_buffer_kwargs"] = {
                        "sequence_reconstructor": sequence_reconstructor
                    }
""",
    """                    rollout_kwargs["rollout_buffer_kwargs"] = {
                        "sequence_reconstructor": sequence_reconstructor,
                        "sequence_transfer_mode": config.sequence_transfer_mode,
                    }
""",
    seam="fresh rollout transfer mode",
)
sb3 = replace_once(
    sb3,
    """                    binder(sequence_reconstructor)
                    model.rollout_buffer_kwargs = {
                        "sequence_reconstructor": sequence_reconstructor
                    }
""",
    """                    binder(
                        sequence_reconstructor,
                        sequence_transfer_mode=config.sequence_transfer_mode,
                    )
                    model.rollout_buffer_kwargs = {
                        "sequence_reconstructor": sequence_reconstructor,
                        "sequence_transfer_mode": config.sequence_transfer_mode,
                    }
""",
    seam="resume rollout transfer mode",
)
sb3 = replace_once(
    sb3,
    """            torch_runtime = _configure_torch_cuda_runtime(torch, config.device)

            parameter_count = sum(
""",
    """            torch_runtime = _configure_torch_cuda_runtime(torch, config.device)
            sequence_runtime = _configure_sequence_runtime(torch, model, config)

            parameter_count = sum(
""",
    seam="apply sequence runtime",
)
sb3 = replace_once(
    sb3,
    """                        "torch_runtime": torch_runtime,
                        "rollout_buffer": (
""",
    """                        "torch_runtime": torch_runtime,
                        "sequence_runtime": sequence_runtime,
                        "rollout_buffer": (
""",
    seam="sequence runtime architecture evidence",
)
sb3_path.write_text(sb3, encoding="utf-8")


for config_path in (
    Path("examples/binance-multitimeframe/training-full.json"),
    Path("examples/binance-multitimeframe/walk-forward-full.json"),
):
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if config_path.name == "training-full.json":
        training_payload = payload["training"]
    else:
        training_payload = payload["candidates"][0]["run"]["training"]
    training_payload["sequence_compile"] = True
    training_payload["sequence_compile_mode"] = "reduce-overhead"
    training_payload["sequence_transfer_mode"] = "pinned_non_blocking"
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
