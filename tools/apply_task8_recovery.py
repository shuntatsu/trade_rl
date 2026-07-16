from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 8 recovery anchor in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def add_tests() -> None:
    append_once(
        "tests/rl/test_checkpointing.py",
        "test_checkpoint_save_excludes_runtime_sequence_reconstructor",
        '''

class RuntimeStateModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.reconstructor = object()
        self.rollout_buffer_kwargs = {
            "sequence_reconstructor": self.reconstructor,
            "retained": "value",
        }
        self.saved_kwargs = None

    def save(self, target: str) -> None:
        self.saved_kwargs = dict(self.rollout_buffer_kwargs)
        super().save(target)


def test_checkpoint_save_excludes_runtime_sequence_reconstructor(
    tmp_path: Path,
) -> None:
    model = RuntimeStateModel()
    publish_checkpoint(
        model=model,
        checkpoint_root=tmp_path / "checkpoints",
        algorithm="ppo",
        seed=0,
        requested_timestep=1,
        observed_timestep=1,
        environment_digest="e" * 64,
        training_config_digest="a" * 64,
    )
    assert model.saved_kwargs == {"retained": "value"}
    assert model.rollout_buffer_kwargs["sequence_reconstructor"] is model.reconstructor
''',
    )
    append_once(
        "tests/rl/test_rollout_memory.py",
        "test_index_backed_buffer_requires_runtime_reconstructor_only_when_sampling",
        '''

def test_index_backed_buffer_requires_runtime_reconstructor_only_when_sampling() -> None:
    from trade_rl.integrations.compact_rollout_buffer import (
        IndexBackedDictRolloutBuffer,
    )

    buffer = IndexBackedDictRolloutBuffer(
        2,
        _sequence_observation_space(),
        spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        device="cpu",
        n_envs=1,
    )
    assert buffer.sequence_reconstructor is None
    with pytest.raises(RuntimeError, match="reconstructor"):
        buffer._get_samples(np.array([0], dtype=np.int64))
''',
    )
    append_once(
        "tests/workflows/test_training_run_config.py",
        "test_training_config_parses_seed_checkpoint_resume_mapping",
        '''

def test_training_config_parses_seed_checkpoint_resume_mapping(tmp_path) -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["resume_checkpoints"] = {"0": "resume/step-1"}
    config = TrainingRunConfig.from_mapping(raw).resolve_artifact_paths(tmp_path)
    assert config.resume_checkpoints == ((0, tmp_path / "resume/step-1"),)


def test_training_config_rejects_resume_seed_outside_ensemble() -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["resume_checkpoints"] = {"7": "resume/step-1"}
    with pytest.raises(ValueError, match="resume checkpoint seed"):
        TrainingRunConfig.from_mapping(raw)
''',
    )
    append_once(
        "tests/integrations/test_sb3_training.py",
        "test_backend_resumes_ppo_checkpoint_to_requested_total",
        '''

def test_backend_resumes_ppo_checkpoint_to_requested_total(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from trade_rl.rl.checkpointing import publish_checkpoint

    config = ResidualTrainingConfig(
        timesteps=2,
        gamma=0.99,
        seeds=(0,),
        n_steps=1,
        n_envs=1,
        batch_size=1,
        n_epochs=1,
        asset_set_encoder=False,
        device="cpu",
    )

    class CheckpointSource:
        def save(self, target: str) -> None:
            Path(target).with_suffix(".zip").write_bytes(b"resume-policy")

    manifest = publish_checkpoint(
        model=CheckpointSource(),
        checkpoint_root=tmp_path / "resume",
        algorithm="ppo",
        seed=0,
        requested_timestep=1,
        observed_timestep=1,
        environment_digest=ENVIRONMENT_DIGEST,
        training_config_digest=content_digest(config.digest_payload()),
    )
    events: list[object] = []

    class FakeParameter:
        def numel(self) -> int:
            return 2

    class FakePolicy:
        def parameters(self):
            return (FakeParameter(),)

    class FakeResumePPO:
        device = "cpu"

        def __init__(self, policy, environment, **kwargs):
            self.policy = FakePolicy()
            self.num_timesteps = 0
            self.rollout_buffer_kwargs = {}

        @classmethod
        def load(cls, path, env=None, device=None):
            events.append(("load", Path(path), device, env is not None))
            model = cls("MlpPolicy", env)
            model.num_timesteps = 1
            return model

        def learn(self, *, total_timesteps, callback, reset_num_timesteps=True):
            events.append(("learn", total_timesteps, reset_num_timesteps))
            self.num_timesteps += total_timesteps
            return self

        def save(self, target: str) -> None:
            Path(target).with_suffix(".zip").write_bytes(b"resumed-policy")

    monkeypatch.setattr("stable_baselines3.PPO", FakeResumePPO)
    monkeypatch.setattr(
        "trade_rl.rl.checkpointing.build_checkpoint_callback",
        lambda **kwargs: object(),
    )
    result = StableBaselines3Backend(
        lambda: TrainingProbe([]),
        resume_checkpoint_artifacts={0: manifest.policy_path.parent},
    ).train(
        seed=0,
        config=config,
        output_path=tmp_path / "output" / "policy.zip",
    )
    assert result.actual_timesteps == 2
    assert events[0][0] == "load"
    assert ("learn", 1, False) in events
    resume_payload = (tmp_path / "output" / "resume.json").read_text(
        encoding="utf-8"
    )
    assert manifest.digest in resume_payload
''',
    )


def add_implementation() -> None:
    replace_once(
        "trade_rl/integrations/compact_rollout_buffer.py",
        '''        *,
        sequence_reconstructor: SequenceRolloutReconstructor,
    ) -> None:
''',
        '''        *,
        sequence_reconstructor: SequenceRolloutReconstructor | None = None,
    ) -> None:
''',
    )
    replace_once(
        "trade_rl/integrations/compact_rollout_buffer.py",
        '''        self.sequence_reconstructor = sequence_reconstructor
        super().__init__(
''',
        '''        self.sequence_reconstructor = sequence_reconstructor
        super().__init__(
''',
    )
    replace_once(
        "trade_rl/integrations/compact_rollout_buffer.py",
        '''    def reset(self) -> None:
''',
        '''    def bind_sequence_reconstructor(
        self, reconstructor: SequenceRolloutReconstructor
    ) -> None:
        if not isinstance(reconstructor, SequenceRolloutReconstructor):
            raise TypeError("sequence reconstructor has an invalid type")
        self.sequence_reconstructor = reconstructor

    def reset(self) -> None:
''',
    )
    replace_once(
        "trade_rl/integrations/compact_rollout_buffer.py",
        '''        observations.update(self.sequence_reconstructor.reconstruct(decision_indices))
''',
        '''        reconstructor = self.sequence_reconstructor
        if reconstructor is None:
            raise RuntimeError("sequence rollout reconstructor is not bound")
        observations.update(reconstructor.reconstruct(decision_indices))
''',
    )

    replace_once(
        "trade_rl/rl/checkpointing.py",
        '''def publish_checkpoint(
''',
        '''def save_policy_without_runtime_state(model: SavablePolicy, target: str) -> None:
    """Save without serializing dataset-bound rollout reconstruction objects."""

    missing = object()
    original = getattr(model, "rollout_buffer_kwargs", missing)
    if isinstance(original, dict) and "sequence_reconstructor" in original:
        sanitized = {
            key: value
            for key, value in original.items()
            if key != "sequence_reconstructor"
        }
        setattr(model, "rollout_buffer_kwargs", sanitized)
    try:
        model.save(target)
    finally:
        if original is not missing:
            setattr(model, "rollout_buffer_kwargs", original)


def publish_checkpoint(
''',
    )
    replace_once(
        "trade_rl/rl/checkpointing.py",
        '''        model.save(str(save_target))
''',
        '''        save_policy_without_runtime_state(model, str(save_target))
''',
    )
    replace_once(
        "trade_rl/rl/checkpointing.py",
        '''    "publish_checkpoint",
''',
        '''    "publish_checkpoint",
    "save_policy_without_runtime_state",
''',
    )

    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''    def __init__(
        self,
        environment_factory: Callable[[], Any],
        *,
        verbose: int = 0,
        resume_replay_artifact: Path | None = None,
    ) -> None:
''',
        '''    def __init__(
        self,
        environment_factory: Callable[[], Any],
        *,
        verbose: int = 0,
        resume_replay_artifact: Path | None = None,
        resume_checkpoint_artifacts: Mapping[int, Path] | None = None,
    ) -> None:
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''        self.resume_replay_artifact = resume_replay_artifact
        self._oracle_target_cache: dict[tuple[str, int, int, str], np.ndarray] = {}
''',
        '''        self.resume_replay_artifact = resume_replay_artifact
        self.resume_checkpoint_artifacts = dict(resume_checkpoint_artifacts or {})
        self._oracle_target_cache: dict[tuple[str, int, int, str], np.ndarray] = {}
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''            parameter_count = sum(
''',
        '''            resume_manifest = None
            resume_root = self.resume_checkpoint_artifacts.get(seed)
            if resume_root is not None:
                from trade_rl.rl.checkpointing import load_checkpoint_manifest

                manifest_path = Path(resume_root)
                if manifest_path.is_dir():
                    manifest_path = manifest_path / "checkpoint.json"
                resume_manifest = load_checkpoint_manifest(manifest_path)
                expected_training_digest = content_digest(config.digest_payload())
                if resume_manifest.algorithm != config.algorithm:
                    raise ValueError("checkpoint algorithm mismatch")
                if resume_manifest.seed != seed:
                    raise ValueError("checkpoint seed mismatch")
                if resume_manifest.environment_digest != identity["environment_digest"]:
                    raise ValueError("checkpoint environment identity mismatch")
                if resume_manifest.training_config_digest != expected_training_digest:
                    raise ValueError("checkpoint training configuration mismatch")
                algorithm_class: Any
                if config.algorithm == "ppo":
                    algorithm_class = PPO
                elif config.algorithm == "sac":
                    algorithm_class = SAC
                elif config.algorithm == "td3":
                    algorithm_class = TD3
                else:
                    from sb3_contrib import TQC

                    algorithm_class = TQC
                model = algorithm_class.load(
                    str(resume_manifest.policy_path),
                    env=environment,
                    device=config.device,
                )
                if int(model.num_timesteps) != resume_manifest.observed_timestep:
                    raise ValueError("checkpoint timestep identity mismatch")
                if config.sequence_encoder:
                    if sequence_reconstructor is None:
                        raise RuntimeError("sequence reconstructor was not resolved")
                    rollout_buffer = getattr(model, "rollout_buffer", None)
                    binder = getattr(rollout_buffer, "bind_sequence_reconstructor", None)
                    if not callable(binder):
                        raise ValueError("checkpoint rollout buffer cannot bind sequences")
                    binder(sequence_reconstructor)
                    model.rollout_buffer_kwargs = {
                        "sequence_reconstructor": sequence_reconstructor
                    }

            parameter_count = sum(
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''            if config.behavior_cloning_epochs > 0:
''',
        '''            if config.behavior_cloning_epochs > 0 and resume_manifest is None:
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''            model.learn(total_timesteps=config.timesteps, callback=callback)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_target = output_path.with_suffix("")
            model.save(str(save_target))
''',
        '''            remaining_timesteps = config.timesteps
            if resume_manifest is not None:
                remaining_timesteps = max(
                    0, config.timesteps - resume_manifest.observed_timestep
                )
            if remaining_timesteps > 0:
                learn_kwargs: dict[str, object] = {
                    "total_timesteps": remaining_timesteps,
                    "callback": callback,
                }
                if resume_manifest is not None:
                    learn_kwargs["reset_num_timesteps"] = False
                model.learn(**learn_kwargs)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if resume_manifest is not None:
                (output_path.parent / "resume.json").write_bytes(
                    canonical_json_bytes(
                        {
                            "checkpoint_digest": resume_manifest.digest,
                            "checkpoint_observed_timestep": (
                                resume_manifest.observed_timestep
                            ),
                            "remaining_timesteps": remaining_timesteps,
                            "schema_version": "training_resume_v1",
                        }
                    )
                )
            save_target = output_path.with_suffix("")
            from trade_rl.rl.checkpointing import save_policy_without_runtime_state

            save_policy_without_runtime_state(model, str(save_target))
''',
    )

    replace_once(
        "trade_rl/workflows/training_run.py",
        '''from trade_rl.rl.training import ResidualTrainingConfig, train_residual_ensemble
''',
        '''from trade_rl.rl.checkpointing import load_checkpoint_manifest
from trade_rl.rl.training import ResidualTrainingConfig, train_residual_ensemble
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''    factor_artifact: Path | None = None
    export_onnx: bool = False
''',
        '''    factor_artifact: Path | None = None
    resume_checkpoints: tuple[tuple[int, Path], ...] = ()
    export_onnx: bool = False
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''        if self.action.alpha_enabled != (self.alpha_artifact is not None):
''',
        '''        resume_seeds = tuple(seed for seed, _ in self.resume_checkpoints)
        if len(set(resume_seeds)) != len(resume_seeds):
            raise ValueError("resume checkpoint seeds must be unique")
        if any(seed not in self.training.seeds for seed in resume_seeds):
            raise ValueError("resume checkpoint seed is outside training seeds")
        if self.action.alpha_enabled != (self.alpha_artifact is not None):
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''        raw_alpha_artifact = payload.get("alpha_artifact")
        raw_factor_artifact = payload.get("factor_artifact")
''',
        '''        raw_alpha_artifact = payload.get("alpha_artifact")
        raw_factor_artifact = payload.get("factor_artifact")
        raw_resume_checkpoints = payload.get("resume_checkpoints", {})
        if not isinstance(raw_resume_checkpoints, dict):
            raise ValueError("resume_checkpoints must be a JSON object")
        resume_checkpoints: list[tuple[int, Path]] = []
        for raw_seed, raw_path in raw_resume_checkpoints.items():
            if not isinstance(raw_seed, str) or not raw_seed.isdigit():
                raise ValueError("resume checkpoint seed keys must be integers")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("resume checkpoint paths must be non-empty strings")
            resume_checkpoints.append((int(raw_seed), Path(raw_path)))
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''            factor_artifact=(
                None if raw_factor_artifact is None else Path(raw_factor_artifact)
            ),
''',
        '''            factor_artifact=(
                None if raw_factor_artifact is None else Path(raw_factor_artifact)
            ),
            resume_checkpoints=tuple(sorted(resume_checkpoints)),
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''        return replace(
            self,
            alpha_artifact=resolved(self.alpha_artifact),
            factor_artifact=resolved(self.factor_artifact),
        )
''',
        '''        return replace(
            self,
            alpha_artifact=resolved(self.alpha_artifact),
            factor_artifact=resolved(self.factor_artifact),
            resume_checkpoints=tuple(
                (seed, resolved(path) or path)
                for seed, path in self.resume_checkpoints
            ),
        )
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''            "reward": asdict(self.reward),
            "schema_version": self.schema_version,
''',
        '''            "reward": asdict(self.reward),
            "resume_checkpoint_digests": {
                str(seed): load_checkpoint_manifest(
                    path / "checkpoint.json" if path.is_dir() else path
                ).digest
                for seed, path in self.resume_checkpoints
            },
            "schema_version": self.schema_version,
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''            backend=StableBaselines3Backend(
                _environment_factory(
                    dataset,
                    config,
                    normalizer=normalizer,
                    sequence_normalizer=sequence_normalizer,
                )
            ),
''',
        '''            backend=StableBaselines3Backend(
                _environment_factory(
                    dataset,
                    config,
                    normalizer=normalizer,
                    sequence_normalizer=sequence_normalizer,
                ),
                resume_checkpoint_artifacts=dict(config.resume_checkpoints),
            ),
''',
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task8_recovery.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
