from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def update_algorithm_configs() -> None:
    path = Path("trade_rl/rl/algorithm_configs.py")
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "    learning_rate: float\n    batch_size: int\n",
        "    learning_rate: float\n"
        "    learning_rate_schedule: str\n"
        "    learning_rate_final_ratio: float\n"
        "    batch_size: int\n",
        "algorithm common schedule fields",
    )
    text = replace_once(
        text,
        '        "learning_rate": source.learning_rate,\n'
        '        "batch_size": source.batch_size,\n',
        '        "learning_rate": source.learning_rate,\n'
        '        "learning_rate_schedule": source.learning_rate_schedule,\n'
        '        "learning_rate_final_ratio": source.learning_rate_final_ratio,\n'
        '        "batch_size": source.batch_size,\n',
        "PPO schedule payload",
    )
    text = replace_once(
        text,
        "        learning_rate=source.learning_rate,\n"
        "        batch_size=source.batch_size,\n",
        "        learning_rate=source.learning_rate,\n"
        "        learning_rate_schedule=source.learning_rate_schedule,\n"
        "        learning_rate_final_ratio=source.learning_rate_final_ratio,\n"
        "        batch_size=source.batch_size,\n",
        "off-policy common schedule payload",
    )
    explicit = (
        "            learning_rate=source.learning_rate,\n"
        "            batch_size=source.batch_size,\n"
    )
    count = text.count(explicit)
    if count != 2:
        raise RuntimeError(
            f"explicit off-policy schedules: expected two matches, found {count}"
        )
    text = text.replace(
        explicit,
        "            learning_rate=source.learning_rate,\n"
        "            learning_rate_schedule=source.learning_rate_schedule,\n"
        "            learning_rate_final_ratio=source.learning_rate_final_ratio,\n"
        "            batch_size=source.batch_size,\n",
    )
    path.write_text(text, encoding="utf-8")


def update_training_config() -> None:
    path = Path("trade_rl/rl/training.py")
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "    lagrangian_probe_max_steps_per_episode: int = 0\n\n"
        "    def __post_init__(self) -> None:\n",
        "    lagrangian_probe_max_steps_per_episode: int = 0\n"
        '    learning_rate_schedule: str = "constant"\n'
        "    learning_rate_final_ratio: float = 0.1\n"
        "    tensorboard_enabled: bool = False\n"
        "    tensorboard_log_interval: int = 1\n\n"
        "    def __post_init__(self) -> None:\n",
        "training schedule fields",
    )
    text = replace_once(
        text,
        '            ("n_epochs", self.n_epochs),\n'
        '            ("buffer_size", self.buffer_size),\n',
        '            ("n_epochs", self.n_epochs),\n'
        '            ("tensorboard_log_interval", self.tensorboard_log_interval),\n'
        '            ("buffer_size", self.buffer_size),\n',
        "tensorboard interval validation",
    )
    text = replace_once(
        text,
        "        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:\n"
        '            raise ValueError("learning_rate must be finite and positive")\n'
        "        if not math.isfinite(self.gae_lambda) or not 0.0 < self.gae_lambda <= 1.0:\n",
        "        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:\n"
        '            raise ValueError("learning_rate must be finite and positive")\n'
        '        if self.learning_rate_schedule not in {"constant", "linear", "cosine"}:\n'
        "            raise ValueError(\n"
        '                "learning_rate_schedule must be constant, linear, or cosine"\n'
        "            )\n"
        "        if (\n"
        "            not math.isfinite(self.learning_rate_final_ratio)\n"
        "            or not 0.0 < self.learning_rate_final_ratio <= 1.0\n"
        "        ):\n"
        '            raise ValueError("learning_rate_final_ratio must be within (0, 1]")\n'
        "        if not isinstance(self.tensorboard_enabled, bool):\n"
        '            raise ValueError("tensorboard_enabled must be a boolean")\n'
        "        if not math.isfinite(self.gae_lambda) or not 0.0 < self.gae_lambda <= 1.0:\n",
        "schedule validation",
    )
    text = replace_once(
        text,
        '            "learning_rate": self.learning_rate,\n'
        '            "learning_starts": self.learning_starts,\n',
        '            "learning_rate": self.learning_rate,\n'
        '            "learning_rate_final_ratio": self.learning_rate_final_ratio,\n'
        '            "learning_rate_schedule": self.learning_rate_schedule,\n'
        '            "learning_starts": self.learning_starts,\n',
        "schedule digest",
    )
    text = replace_once(
        text,
        '            "target_kl": self.target_kl,\n'
        '            "timesteps": self.timesteps,\n',
        '            "target_kl": self.target_kl,\n'
        '            "tensorboard_enabled": self.tensorboard_enabled,\n'
        '            "tensorboard_log_interval": self.tensorboard_log_interval,\n'
        '            "timesteps": self.timesteps,\n',
        "tensorboard digest",
    )
    path.write_text(text, encoding="utf-8")


def update_checkpointing() -> None:
    path = Path("trade_rl/rl/checkpointing.py")
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, "import tempfile\n", "import uuid\n", "uuid import")

    helpers = '''

def _expected_algorithm_identity_digest(
    algorithm_identity: dict[str, object] | None,
) -> str | None:
    return None if algorithm_identity is None else content_digest(algorithm_identity)


def _same_checkpoint_identity(
    manifest: CheckpointManifest,
    *,
    algorithm: str,
    seed: int,
    requested_timestep: int,
    observed_timestep: int,
    environment_digest: str,
    training_config_digest: str,
    algorithm_identity: dict[str, object] | None,
) -> bool:
    return (
        manifest.requested_timestep == requested_timestep
        and manifest.observed_timestep == observed_timestep
        and manifest.algorithm == algorithm
        and manifest.seed == seed
        and manifest.environment_digest == environment_digest
        and manifest.training_config_digest == training_config_digest
        and manifest.algorithm_identity == algorithm_identity
        and manifest.algorithm_identity_digest
        == _expected_algorithm_identity_digest(algorithm_identity)
    )


def _same_checkpoint_run_identity(
    manifest: CheckpointManifest,
    *,
    algorithm: str,
    seed: int,
    observed_timestep: int,
    environment_digest: str,
    training_config_digest: str,
    algorithm_identity: dict[str, object] | None,
) -> bool:
    return (
        manifest.observed_timestep == observed_timestep
        and manifest.algorithm == algorithm
        and manifest.seed == seed
        and manifest.environment_digest == environment_digest
        and manifest.training_config_digest == training_config_digest
        and manifest.algorithm_identity == algorithm_identity
        and manifest.algorithm_identity_digest
        == _expected_algorithm_identity_digest(algorithm_identity)
    )


def _checkpoint_destination(
    checkpoint_root: Path,
    *,
    algorithm: str,
    seed: int,
    requested_timestep: int,
    observed_timestep: int,
    environment_digest: str,
    training_config_digest: str,
    algorithm_identity: dict[str, object] | None,
) -> tuple[Path, CheckpointManifest | None]:
    primary = checkpoint_root / f"step-{observed_timestep:012d}"
    if not primary.exists():
        return primary, None
    existing = load_checkpoint_manifest(primary / CHECKPOINT_MANIFEST_NAME)
    if _same_checkpoint_identity(
        existing,
        algorithm=algorithm,
        seed=seed,
        requested_timestep=requested_timestep,
        observed_timestep=observed_timestep,
        environment_digest=environment_digest,
        training_config_digest=training_config_digest,
        algorithm_identity=algorithm_identity,
    ):
        return primary, existing
    if not _same_checkpoint_run_identity(
        existing,
        algorithm=algorithm,
        seed=seed,
        observed_timestep=observed_timestep,
        environment_digest=environment_digest,
        training_config_digest=training_config_digest,
        algorithm_identity=algorithm_identity,
    ):
        raise ValueError("checkpoint destination already has conflicting identity")

    fallback = checkpoint_root / (
        f"step-{observed_timestep:012d}-requested-{requested_timestep:012d}"
    )
    if not fallback.exists():
        return fallback, None
    fallback_existing = load_checkpoint_manifest(fallback / CHECKPOINT_MANIFEST_NAME)
    if not _same_checkpoint_identity(
        fallback_existing,
        algorithm=algorithm,
        seed=seed,
        requested_timestep=requested_timestep,
        observed_timestep=observed_timestep,
        environment_digest=environment_digest,
        training_config_digest=training_config_digest,
        algorithm_identity=algorithm_identity,
    ):
        raise ValueError("checkpoint destination already has conflicting identity")
    return fallback, fallback_existing
'''
    marker = "\n\ndef publish_checkpoint("
    if text.count(marker) != 1:
        raise RuntimeError("publish checkpoint marker is not unique")
    text = text.replace(marker, helpers + marker, 1)

    publish_start = text.index("def publish_checkpoint(")
    publish_end = text.index("\ndef _required_integer", publish_start)
    publish = '''def publish_checkpoint(
    *,
    model: SavablePolicy,
    checkpoint_root: Path,
    algorithm: str,
    seed: int,
    requested_timestep: int,
    observed_timestep: int,
    environment_digest: str,
    training_config_digest: str,
) -> CheckpointManifest:
    """Publish one checkpoint atomically with full run and algorithm identity."""

    if requested_timestep <= 0 or observed_timestep < requested_timestep:
        raise ValueError("checkpoint timestep identity is invalid")
    checkpoint_root = Path(checkpoint_root)
    algorithm_identity = _model_algorithm_identity(model)
    destination, existing = _checkpoint_destination(
        checkpoint_root,
        algorithm=algorithm,
        seed=seed,
        requested_timestep=requested_timestep,
        observed_timestep=observed_timestep,
        environment_digest=environment_digest,
        training_config_digest=training_config_digest,
        algorithm_identity=algorithm_identity,
    )
    if existing is not None:
        return existing
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    staging = checkpoint_root / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        save_target = staging / "policy"
        save_policy_without_runtime_state(model, str(save_target))
        policy_path = save_target.with_suffix(".zip")
        if not policy_path.is_file():
            raise FileNotFoundError("checkpoint model save did not create policy.zip")
        policy_digest = _file_digest(policy_path)
        algorithm_identity_digest = _expected_algorithm_identity_digest(
            algorithm_identity
        )
        payload: dict[str, object] = {
            "algorithm": algorithm,
            "environment_digest": environment_digest,
            "observed_timestep": observed_timestep,
            "policy_digest": policy_digest,
            "policy_file": CHECKPOINT_POLICY_NAME,
            "requested_timestep": requested_timestep,
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
            "seed": seed,
            "training_config_digest": training_config_digest,
        }
        if algorithm_identity is not None:
            payload["algorithm_identity"] = algorithm_identity
            payload["algorithm_identity_digest"] = algorithm_identity_digest
        manifest = CheckpointManifest(
            digest=content_digest(payload),
            algorithm=algorithm,
            seed=seed,
            requested_timestep=requested_timestep,
            observed_timestep=observed_timestep,
            environment_digest=environment_digest,
            training_config_digest=training_config_digest,
            policy_digest=policy_digest,
            policy_path=destination / CHECKPOINT_POLICY_NAME,
            algorithm_identity=algorithm_identity,
            algorithm_identity_digest=algorithm_identity_digest,
        )
        (staging / CHECKPOINT_MANIFEST_NAME).write_bytes(
            canonical_json_bytes(
                {
                    **asdict(manifest),
                    "policy_path": CHECKPOINT_POLICY_NAME,
                }
            )
        )
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if checkpoint_root.is_dir() and not tuple(checkpoint_root.iterdir()):
            checkpoint_root.rmdir()
        raise
'''
    text = text[:publish_start] + publish + text[publish_end:]

    callback_start = text.index("def build_checkpoint_callback(")
    callback_end = text.index("\n\n__all__ =", callback_start)
    callback = '''def planned_checkpoint_steps(
    *,
    total_timesteps: int,
    interval_steps: int,
    max_checkpoints: int,
) -> tuple[int, ...]:
    """Select deterministic requested steps across the full training horizon."""

    if (
        isinstance(total_timesteps, bool)
        or not isinstance(total_timesteps, int)
        or total_timesteps <= 0
    ):
        raise ValueError("total_timesteps must be a positive integer")
    if (
        isinstance(interval_steps, bool)
        or not isinstance(interval_steps, int)
        or interval_steps < 0
    ):
        raise ValueError("interval_steps must be a non-negative integer")
    if (
        isinstance(max_checkpoints, bool)
        or not isinstance(max_checkpoints, int)
        or max_checkpoints <= 0
    ):
        raise ValueError("max_checkpoints must be a positive integer")
    if interval_steps == 0:
        return ()
    candidates = tuple(range(interval_steps, total_timesteps, interval_steps))
    if len(candidates) <= max_checkpoints:
        return candidates
    if max_checkpoints == 1:
        return (candidates[-1],)
    positions = tuple(
        round(index * (len(candidates) - 1) / (max_checkpoints - 1))
        for index in range(max_checkpoints)
    )
    return tuple(candidates[position] for position in positions)


def build_checkpoint_callback(
    *,
    checkpoint_root: Path,
    algorithm: str,
    seed: int,
    interval_steps: int,
    max_checkpoints: int,
    total_timesteps: int,
    starting_timestep: int = 0,
    environment_digest: str,
    training_config_digest: str,
) -> Any:
    """Build full-horizon checkpoint and sampled Studio telemetry callbacks."""

    all_planned = planned_checkpoint_steps(
        total_timesteps=total_timesteps,
        interval_steps=interval_steps,
        max_checkpoints=max_checkpoints,
    )
    if (
        isinstance(starting_timestep, bool)
        or not isinstance(starting_timestep, int)
        or starting_timestep < 0
        or starting_timestep > total_timesteps
    ):
        raise ValueError("starting_timestep must be within the training horizon")
    planned = tuple(step for step in all_planned if step > starting_timestep)

    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    checkpoint_root = Path(checkpoint_root)
    telemetry_callback = build_training_telemetry_callback(
        path=checkpoint_root.parent / "telemetry" / "training-telemetry.jsonl",
        seed=seed,
    )
    if not planned:
        return telemetry_callback

    class AtomicCheckpointCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.cursor = 0

        def _on_step(self) -> bool:
            observed = int(self.model.num_timesteps)
            while self.cursor < len(planned) and observed >= planned[self.cursor]:
                requested = planned[self.cursor]
                publish_checkpoint(
                    model=self.model,
                    checkpoint_root=checkpoint_root,
                    algorithm=algorithm,
                    seed=seed,
                    requested_timestep=requested,
                    observed_timestep=observed,
                    environment_digest=environment_digest,
                    training_config_digest=training_config_digest,
                )
                self.cursor += 1
            return True

    return CallbackList([AtomicCheckpointCallback(), telemetry_callback])
'''
    text = text[:callback_start] + callback + text[callback_end:]
    text = replace_once(
        text,
        '    "load_checkpoint_manifest",\n    "publish_checkpoint",\n',
        '    "load_checkpoint_manifest",\n'
        '    "planned_checkpoint_steps",\n'
        '    "publish_checkpoint",\n',
        "checkpoint export",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    update_algorithm_configs()
    update_training_config()
    update_checkpointing()


if __name__ == "__main__":
    main()
