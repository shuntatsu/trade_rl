from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 8 operations anchor in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def add_tests() -> None:
    replace_once(
        "tests/examples/test_run_gpu_training_smoke.py",
        '''    assert config.training.behavior_cloning_epochs == 1
''',
        '''    assert config.training.behavior_cloning_epochs == 1
    assert config.training.checkpoint_interval_steps == 64
    assert config.training.max_checkpoints == 2
''',
    )
    append_once(
        "tests/examples/test_docker_training_assets.py",
        "test_gpu_nightly_contract_measures_vram_throughput_and_resume",
        '''

def test_gpu_nightly_contract_measures_vram_throughput_and_resume() -> None:
    workflow = (ROOT / ".github" / "workflows" / "gpu-nightly.yml").read_text(
        encoding="utf-8"
    )
    smoke = (
        ROOT
        / "examples"
        / "binance-multitimeframe"
        / "run_gpu_training_smoke.py"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "self-hosted" in workflow
    assert "nvidia" in workflow.lower()
    assert "run_gpu_training_smoke.py" in workflow
    assert "gpu-training-smoke.json" in workflow
    assert "peak_gpu_memory_mib" in smoke
    assert "throughput_steps_per_second" in smoke
    assert "resume_checkpoint" in smoke
    assert "gpu_sequence_target_oracle_bc_training_smoke_v5" in smoke


def test_ci_explicitly_runs_recovery_and_structured_serving_smokes() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "Recovery and structured serving smoke" in workflow
    assert "test_backend_resumes_ppo_checkpoint_to_requested_total" in workflow
    assert "test_structured_sb3_loader_rebuilds_native_sequence_observation" in workflow
    assert "agent/causal-sequence-feature-encoder" in workflow
''',
    )
    append_once(
        "tests/examples/test_docker_training_assets.py",
        "test_architecture_docs_state_current_research_and_runtime_boundaries",
        '''

def test_architecture_docs_state_current_research_and_runtime_boundaries() -> None:
    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    research = (ROOT / "docs" / "MULTITIMEFRAME_RESEARCH.md").read_text(
        encoding="utf-8"
    )
    runbook = (ROOT / "docs" / "operations" / "docker-gpu-full-training.md").read_text(
        encoding="utf-8"
    )
    combined = "\n".join((architecture, research, runbook))
    for phrase in (
        "approximate portfolio teacher",
        "shared per-asset actor",
        "index-backed rollout",
        "180 OOS days",
        "fresh confirmation",
        "structured sequence serving",
        "Production remains NO-GO",
    ):
        assert phrase in combined
    assert "exact optimal Oracle" not in combined
''',
    )


def add_implementation() -> None:
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''import subprocess
import sys
''',
        '''import subprocess
import sys
import time
from copy import deepcopy
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''            "max_rollout_buffer_bytes": 268_435_456,
            "seeds": [0],
''',
        '''            "max_rollout_buffer_bytes": 268_435_456,
            "checkpoint_interval_steps": max(1, timesteps // 2),
            "max_checkpoints": 2,
            "seeds": [0],
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''def _run_authoritative_training(
    *, config: Path, dataset: Path, artifacts: Path
) -> dict[str, Any]:
''',
        '''def _gpu_memory_mib() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return 0
    values: list[int] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.isdigit():
            values.append(int(stripped))
    return sum(values)


def _run_authoritative_training(
    *, config: Path, dataset: Path, artifacts: Path, run_id: str
) -> tuple[dict[str, Any], dict[str, float]]:
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''        "gpu-training-smoke",
    ]
''',
        '''        run_id,
    ]
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "authoritative training workflow failed: " + completed.stderr.strip()
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
''',
        '''    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    peak_gpu_memory_mib = 0
    while process.poll() is None:
        peak_gpu_memory_mib = max(peak_gpu_memory_mib, _gpu_memory_mib())
        time.sleep(0.2)
    stdout, stderr = process.communicate()
    duration_seconds = max(time.perf_counter() - started, 1e-9)
    peak_gpu_memory_mib = max(peak_gpu_memory_mib, _gpu_memory_mib())
    if process.returncode != 0:
        raise RuntimeError("authoritative training workflow failed: " + stderr.strip())
    lines = [line for line in stdout.splitlines() if line.strip()]
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''    return dict(result)
''',
        '''    resolved = dict(result)
    actual_timesteps = int(resolved.get("actual_timesteps", 0))
    return resolved, {
        "duration_seconds": duration_seconds,
        "peak_gpu_memory_mib": float(peak_gpu_memory_mib),
        "throughput_steps_per_second": actual_timesteps / duration_seconds,
    }
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''    result = _run_authoritative_training(
        config=config_path,
        dataset=dataset_path,
        artifacts=work_root / "artifacts",
    )
''',
        '''    result, first_metrics = _run_authoritative_training(
        config=config_path,
        dataset=dataset_path,
        artifacts=work_root / "artifacts",
        run_id="gpu-training-smoke",
    )
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''    if serving_support.get("status") != "unsupported":
        raise RuntimeError("structured smoke must fail closed for flat serving")
    checkpoint = artifact_path / "members" / "member-000" / "policy.zip"
    evidence: dict[str, object] = {
''',
        '''    if serving_support.get("status") != "supported":
        raise RuntimeError("structured smoke must publish native serving support")
    policy = artifact_path / "members" / "member-000" / "policy.zip"
    checkpoint_manifests = sorted(
        (artifact_path / "members" / "member-000" / "checkpoints").glob(
            "step-*/checkpoint.json"
        )
    )
    if not checkpoint_manifests:
        raise RuntimeError("GPU smoke did not publish a resumable checkpoint")
    resume_checkpoint = checkpoint_manifests[0].parent
    resume_payload = deepcopy(config_payload)
    resume_payload["resume_checkpoints"] = {"0": str(resume_checkpoint)}
    resume_config_path = work_root / "training-resume.json"
    resume_config_path.write_text(
        json.dumps(resume_payload, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    resumed, resume_metrics = _run_authoritative_training(
        config=resume_config_path,
        dataset=dataset_path,
        artifacts=work_root / "artifacts-resumed",
        run_id="gpu-training-smoke-resumed",
    )
    resumed_artifact = Path(str(resumed["artifact_path"]))
    if not resumed_artifact.is_absolute():
        resumed_artifact = ROOT / resumed_artifact
    resume_evidence_path = (
        resumed_artifact / "members" / "member-000" / "resume.json"
    )
    if not resume_evidence_path.is_file():
        raise RuntimeError("GPU smoke resume evidence is missing")
    evidence: dict[str, object] = {
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''        "checkpoint": {
            "digest": ensemble["members"][0]["checkpoint_digest"],
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
''',
        '''        "checkpoint": {
            "digest": ensemble["members"][0]["checkpoint_digest"],
            "path": str(policy),
            "size_bytes": policy.stat().st_size,
        },
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''        "requested_timesteps": config.training.timesteps,
        "resolved_device": ensemble["resolved_device"],
        "schema": "gpu_sequence_target_oracle_bc_training_smoke_v4",
''',
        '''        "requested_timesteps": config.training.timesteps,
        "resolved_device": ensemble["resolved_device"],
        "performance": first_metrics,
        "resume": {
            "actual_timesteps": int(resumed["actual_timesteps"]),
            "checkpoint": str(resume_checkpoint),
            "evidence": json.loads(resume_evidence_path.read_text(encoding="utf-8")),
            "performance": resume_metrics,
        },
        "schema": "gpu_sequence_target_oracle_bc_training_smoke_v5",
''',
    )

    (ROOT / ".github/workflows/gpu-nightly.yml").write_text(
        '''name: GPU Structured Training Verification

on:
  workflow_dispatch:
    inputs:
      timesteps:
        description: PPO timesteps for the CUDA smoke
        required: false
        default: "4096"

permissions:
  contents: read

concurrency:
  group: gpu-structured-training-verification
  cancel-in-progress: false

jobs:
  cuda-structured-resume:
    runs-on: [self-hosted, linux, x64, gpu, nvidia]
    timeout-minutes: 180
    env:
      TRADE_RL_GIT_COMMIT: ${{ github.sha }}
      TRADE_RL_GIT_DIRTY: "false"
      OMP_NUM_THREADS: "2"
      MKL_NUM_THREADS: "2"
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
          enable-cache: true
      - run: uv sync --extra dev --extra train-sb3
      - name: CUDA device evidence
        run: nvidia-smi
      - name: Structured BC PPO checkpoint resume smoke
        run: >-
          uv run python examples/binance-multitimeframe/run_gpu_training_smoke.py
          --work-root var/gpu-nightly
          --timesteps ${{ inputs.timesteps }}
      - name: Verify measured evidence
        shell: bash
        run: |
          python - <<'PY'
          import json
          from pathlib import Path
          evidence = json.loads(Path("var/gpu-nightly/gpu-training-smoke.json").read_text())
          assert evidence["schema"] == "gpu_sequence_target_oracle_bc_training_smoke_v5"
          assert evidence["resolved_device"] == "cuda"
          assert evidence["serving_support"]["status"] == "supported"
          assert evidence["performance"]["peak_gpu_memory_mib"] > 0
          assert evidence["performance"]["throughput_steps_per_second"] > 0
          assert evidence["resume"]["actual_timesteps"] == evidence["actual_timesteps"]
          assert evidence["resume"]["evidence"]["schema_version"] == "training_resume_v1"
          PY
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: gpu-structured-training-evidence
          path: var/gpu-nightly
''',
        encoding="utf-8",
    )

    replace_once(
        ".github/workflows/ci.yml",
        '''      - agent/causal-training-hardening
''',
        '''      - agent/causal-training-hardening
      - agent/causal-sequence-feature-encoder
''',
    )
    replace_once(
        ".github/workflows/ci.yml",
        '''      - name: Tests and coverage
        run: uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json
''',
        '''      - name: Recovery and structured serving smoke
        run: >-
          uv run pytest -q
          tests/integrations/test_sb3_training.py::test_backend_resumes_ppo_checkpoint_to_requested_total
          tests/serving/test_sb3_loader.py::test_structured_sb3_loader_rebuilds_native_sequence_observation

      - name: Tests and coverage
        run: uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json
''',
    )

    append_once(
        "docs/ARCHITECTURE.md",
        "## Final causal sequence hardening",
        '''

## Final causal sequence hardening

The maintained policy uses a shared per-asset actor over contextual asset tokens. The critic remains portfolio-level. PPO uses an index-backed rollout: persistent rollout state contains decision indices and current state, while overlapping native sequences are reconstructed from the immutable causal dataset only for sampled minibatches.

The BC teacher is an approximate portfolio teacher, not an exact optimal Oracle. It uses a bounded beam and executor-compatible minimum-notional and partial-fill semantics. Model artifacts record the approximation contract.

Structured sequence serving is native SB3 serving. The runtime restores the flat and sequence normalizers, validates symbols, ordered features, global features, cadence and sequence layout, rebuilds the Dict observation from a bounded rolling dataset, and leaves live order routing outside the policy artifact.

Production remains NO-GO until the maintained GPU verification, 180 OOS days, a fresh confirmation interval and paper-trading reconciliation all pass.
''',
    )
    append_once(
        "docs/MULTITIMEFRAME_RESEARCH.md",
        "## Material evidence protocol",
        '''

## Material evidence protocol

The maintained preset requires six sealed walk-forward folds covering 180 OOS days. A positive mean alone is insufficient: the circular block-bootstrap lower bound on mean daily log growth must be positive, selection stability must hold across folds, and the final artifact preserves the fixed multi-seed ensemble rather than selecting a lucky seed.

A fresh confirmation period beginning after the development cutoff is separate from walk-forward evidence. The final retrained weights do not receive production approval without at least 30 fresh sealed days, positive return, acceptable drawdown and matching policy identity.

The approximate portfolio teacher, shared per-asset actor, index-backed rollout and structured sequence serving contracts are evaluated independently. Feature, network and BC ablations remain research evidence rather than assumptions.

Production remains NO-GO until these gates and paper trading pass.
''',
    )
    append_once(
        "docs/operations/docker-gpu-full-training.md",
        "## GPU verification and checkpoint recovery",
        '''

## GPU verification and checkpoint recovery

Run the manually dispatched `GPU Structured Training Verification` workflow on a self-hosted NVIDIA runner before a full generation. It records peak GPU memory, throughput, structured serving support and a checkpoint-resume run in `gpu-training-smoke.json`.

For interrupted single-seed work, add a `resume_checkpoints` object to the training JSON, mapping the seed string to the checkpoint step directory. The loader validates seed, algorithm, environment digest, training-config digest and observed timestep before continuing. Dataset-bound sequence reconstructors are never serialized into `policy.zip`; they are rebound from the current validated dataset after loading.

A successful CUDA smoke is a systems gate, not profitability evidence. Production remains NO-GO until 180 OOS days, fresh confirmation and paper-trading reconciliation are complete.
''',
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task8_operations.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
