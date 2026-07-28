from __future__ import annotations

from pathlib import Path


SMOKE = Path("examples/binance-multitimeframe/run_gpu_training_smoke.py")
CUDA_WORKFLOW = Path(".github/workflows/finalize-pr227-gpu-verification.yml")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label} anchor count was {text.count(old)}, expected 1")
    return text.replace(old, new, 1)


def patch_smoke() -> None:
    text = SMOKE.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '            "device": "cuda",\n            "n_envs": 4,',
        '            "device": "cuda",\n'
        '            "cuda_runtime_mode": (\n'
        '                "performance" if accelerated else "deterministic"\n'
        '            ),\n'
        '            "n_envs": 4,',
        label="explicit CUDA runtime mode",
    )
    helper = '''

_TORCH_RUNTIME_FIELDS = {
    "mode",
    "deterministic_algorithms",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "cudnn_tf32",
    "float32_matmul_precision",
    "matmul_tf32",
    "sequence_encoder_autocast",
}


def _load_torch_runtime(
    member_root: Path,
    *,
    expected_mode: str,
) -> dict[str, object]:
    path = member_root / "model-architecture.json"
    if not path.is_file():
        raise RuntimeError("model architecture evidence is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model architecture evidence must be a JSON object")
    if payload.get("schema_version") != "policy_architecture_v2":
        raise ValueError("model architecture schema is unsupported")
    raw_runtime = payload.get("torch_runtime")
    if not isinstance(raw_runtime, dict):
        raise ValueError("torch runtime evidence must be a mapping")
    if set(raw_runtime) != _TORCH_RUNTIME_FIELDS:
        raise ValueError("torch runtime evidence fields are invalid")
    if expected_mode not in {"deterministic", "performance"}:
        raise ValueError("expected CUDA runtime mode is invalid")
    if raw_runtime.get("mode") != expected_mode:
        raise ValueError("torch runtime mode does not match the requested mode")
    for field in (
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_tf32",
        "matmul_tf32",
    ):
        if not isinstance(raw_runtime.get(field), bool):
            raise ValueError(f"torch runtime {field} must be a boolean")
    expected_flags = (
        {
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cudnn_tf32": False,
            "float32_matmul_precision": "highest",
            "matmul_tf32": False,
        }
        if expected_mode == "deterministic"
        else {
            "deterministic_algorithms": False,
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "cudnn_tf32": True,
            "float32_matmul_precision": "high",
            "matmul_tf32": True,
        }
    )
    for field, expected in expected_flags.items():
        if raw_runtime.get(field) != expected:
            raise ValueError(f"torch runtime {field} does not match {expected_mode} mode")
    if raw_runtime.get("sequence_encoder_autocast") not in {
        "bfloat16",
        "disabled",
    }:
        raise ValueError("torch runtime sequence encoder autocast is invalid")
    return dict(raw_runtime)
'''
    text = replace_once(
        text,
        "\n\ndef run_gpu_training_smoke(\n",
        helper + "\n\ndef run_gpu_training_smoke(\n",
        label="torch runtime helper",
    )
    text = replace_once(
        text,
        "    training_performance = _load_training_performance(member_root)\n",
        "    training_performance = _load_training_performance(member_root)\n"
        "    expected_cuda_mode = str(config.training.cuda_runtime_mode)\n"
        "    torch_runtime = _load_torch_runtime(\n"
        "        member_root, expected_mode=expected_cuda_mode\n"
        "    )\n",
        label="initial runtime evidence",
    )
    text = replace_once(
        text,
        "    resumed_training_performance = _load_training_performance(resumed_member_root)\n",
        "    resumed_training_performance = _load_training_performance(resumed_member_root)\n"
        "    resumed_torch_runtime = _load_torch_runtime(\n"
        "        resumed_member_root, expected_mode=expected_cuda_mode\n"
        "    )\n",
        label="resume runtime evidence",
    )
    text = replace_once(
        text,
        '        "behavior_cloning_epochs": config.training.behavior_cloning_epochs,\n',
        '        "behavior_cloning_epochs": config.training.behavior_cloning_epochs,\n'
        '        "cuda_runtime": torch_runtime,\n',
        label="initial evidence payload",
    )
    text = replace_once(
        text,
        '            "checkpoint": str(resume_checkpoint),\n',
        '            "checkpoint": str(resume_checkpoint),\n'
        '            "cuda_runtime": resumed_torch_runtime,\n',
        label="resume evidence payload",
    )
    text = replace_once(
        text,
        '"gpu_sequence_target_oracle_bc_training_smoke_v7"',
        '"gpu_sequence_target_oracle_bc_training_smoke_v8"',
        label="GPU smoke schema",
    )
    SMOKE.write_text(text, encoding="utf-8")


def patch_workflow() -> None:
    text = CUDA_WORKFLOW.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'assert evidence["schema"] == "gpu_sequence_target_oracle_bc_training_smoke_v7"',
        'assert evidence["schema"] == "gpu_sequence_target_oracle_bc_training_smoke_v8"',
        label="workflow schema",
    )
    text = replace_once(
        text,
        '          assert evidence["resolved_device"] == "cuda"\n',
        '''          assert evidence["resolved_device"] == "cuda"
          cuda_runtime = evidence["cuda_runtime"]
          assert cuda_runtime["mode"] == "performance"
          assert cuda_runtime["deterministic_algorithms"] is False
          assert cuda_runtime["cudnn_benchmark"] is True
          assert cuda_runtime["cudnn_deterministic"] is False
          assert cuda_runtime["cudnn_tf32"] is True
          assert cuda_runtime["matmul_tf32"] is True
          assert cuda_runtime["float32_matmul_precision"] == "high"
''',
        label="workflow initial runtime assertions",
    )
    text = replace_once(
        text,
        '          resumed = resume["performance"]["training_artifact"]\n',
        '''          resumed_cuda_runtime = resume["cuda_runtime"]
          assert resumed_cuda_runtime == cuda_runtime
          resumed = resume["performance"]["training_artifact"]
''',
        label="workflow resume runtime assertions",
    )
    CUDA_WORKFLOW.write_text(text, encoding="utf-8")


def main() -> None:
    patch_smoke()
    patch_workflow()


if __name__ == "__main__":
    main()
