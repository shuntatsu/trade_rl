from pathlib import Path

path = Path("trade_rl/integrations/sb3_training.py")
text = path.read_text()

class_marker = "\n\nclass StableBaselines3Backend(_StableBaselines3TeacherPipeline):\n"
if "def _publish_final_training_checkpoint(" not in text:
    if class_marker not in text:
        raise SystemExit("StableBaselines3Backend marker not found")
    block = r'''


def _publish_final_training_checkpoint(
    *,
    model: Any,
    output_root: Path,
    config: Any,
    seed: int,
    environment_digest: str,
    target_total_timesteps: int,
) -> Any:
    """Publish the exact completed policy as the retained Stage A checkpoint."""

    observed_timestep = getattr(model, "num_timesteps", None)
    if (
        isinstance(target_total_timesteps, bool)
        or not isinstance(target_total_timesteps, int)
        or target_total_timesteps <= 0
    ):
        raise ValueError("target_total_timesteps must be a positive integer")
    if (
        isinstance(observed_timestep, bool)
        or not isinstance(observed_timestep, int)
        or observed_timestep < target_total_timesteps
    ):
        raise RuntimeError("model has not reached the target training horizon")
    algorithm = getattr(config, "algorithm", None)
    digest_payload = getattr(config, "digest_payload", None)
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("training algorithm identity is unavailable")
    if not callable(digest_payload):
        raise TypeError("training config must expose digest_payload")
    from trade_rl.rl.checkpointing import publish_checkpoint

    return publish_checkpoint(
        model=model,
        checkpoint_root=Path(output_root) / "checkpoints",
        algorithm=algorithm,
        seed=seed,
        requested_timestep=target_total_timesteps,
        observed_timestep=observed_timestep,
        environment_digest=environment_digest,
        training_config_digest=content_digest(digest_payload()),
    )
'''
    text = text.replace(class_marker, block + class_marker, 1)

call_anchor = '''                write_training_performance_evidence(\n                    output_path.parent / "training-performance.json",\n                    performance_evidence,\n                )\n            output_path.parent.mkdir(parents=True, exist_ok=True)\n'''
call_replacement = '''                write_training_performance_evidence(\n                    output_path.parent / "training-performance.json",\n                    performance_evidence,\n                )\n            _publish_final_training_checkpoint(\n                model=model,\n                output_root=output_path.parent,\n                config=config,\n                seed=seed,\n                environment_digest=str(identity["environment_digest"]),\n                target_total_timesteps=target_total_timesteps,\n            )\n            output_path.parent.mkdir(parents=True, exist_ok=True)\n'''
if "target_total_timesteps=target_total_timesteps" not in text:
    if call_anchor not in text:
        raise SystemExit("final checkpoint call anchor not found")
    text = text.replace(call_anchor, call_replacement, 1)

path.write_text(text)
compile(text, str(path), "exec")
