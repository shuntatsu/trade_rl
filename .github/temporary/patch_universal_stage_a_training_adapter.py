from pathlib import Path

path = Path("trade_rl/workflows/universal_stage_a.py")
text = path.read_text()

if "from collections.abc import Mapping\n" not in text:
    text = text.replace(
        "from __future__ import annotations\n\n",
        "from __future__ import annotations\n\nfrom collections.abc import Mapping\nfrom pathlib import Path\n\n",
        1,
    )
if "from trade_rl.domain.common import require_sha256\n" not in text:
    text = text.replace(
        "from trade_rl.artifacts.hashing import content_digest\n",
        "from trade_rl.artifacts.hashing import content_digest\n"
        "from trade_rl.domain.common import require_sha256\n",
        1,
    )
if "from trade_rl.rl.checkpointing import checkpoint_manifests\n" not in text:
    text = text.replace(
        "from trade_rl.rl.training import ResidualTrainingConfig\n",
        "from trade_rl.rl.checkpointing import checkpoint_manifests\n"
        "from trade_rl.rl.training import ResidualTrainingConfig\n",
        1,
    )

marker = "\n\n@dataclass(frozen=True, slots=True)\nclass UniversalStageAPlan:\n"
if "def build_universal_stage_a_candidate_from_training(" not in text:
    if marker not in text:
        raise SystemExit("UniversalStageAPlan marker not found")
    block = r'''


def build_universal_stage_a_candidate_from_training(
    *,
    architecture: UniversalArchitectureName | str,
    training_config: ResidualTrainingConfig,
    training_manifest: Mapping[str, object],
    output_root: str | Path,
) -> UniversalStageACandidate:
    """Bind one completed Universal training run to exact Stage A checkpoints."""

    resolved_architecture = UniversalArchitectureName(architecture)
    if not isinstance(training_config, ResidualTrainingConfig):
        raise TypeError("Universal Stage A training_config must be ResidualTrainingConfig")
    manifest = dict(training_manifest)
    if manifest.get("schema_version") != "universal_training_run_v1":
        raise ValueError("Universal Stage A training manifest schema mismatch")
    if manifest.get("architecture_name") != resolved_architecture.value:
        raise ValueError("Universal Stage A training architecture mismatch")

    config_digest = content_digest(training_config.digest_payload())
    if manifest.get("training_config_digest") != config_digest:
        raise ValueError("Universal Stage A training config digest mismatch")
    run_digest = manifest.get("run_digest")
    if not isinstance(run_digest, str):
        raise ValueError("Universal Stage A training run digest is unavailable")
    require_sha256(run_digest, field="Universal Stage A training run_digest")
    run_payload = {key: value for key, value in manifest.items() if key != "run_digest"}
    if run_digest != content_digest(run_payload):
        raise ValueError("Universal Stage A training run digest mismatch")

    raw_members = manifest.get("members")
    if not isinstance(raw_members, list | tuple):
        raise TypeError("Universal Stage A training members must be a sequence")
    members = tuple(raw_members)
    seeds = tuple(training_config.seeds)
    if len(members) != len(seeds):
        raise ValueError("Universal Stage A training seed closure mismatch")

    checkpoint_digests: list[tuple[int, str]] = []
    policy_architecture_digest: str | None = None
    root = Path(output_root)
    for expected_seed, raw_member in zip(seeds, members, strict=True):
        if not isinstance(raw_member, Mapping):
            raise TypeError("Universal Stage A training member must be a mapping")
        member = dict(raw_member)
        if member.get("seed") != expected_seed:
            raise ValueError("Universal Stage A training member seed mismatch")
        actual_timesteps = member.get("actual_timesteps")
        if (
            isinstance(actual_timesteps, bool)
            or not isinstance(actual_timesteps, int)
            or actual_timesteps < training_config.timesteps
        ):
            raise ValueError("Universal Stage A training member is incomplete")
        environment_digest = member.get("environment_digest")
        architecture_digest = member.get("architecture_digest")
        if not isinstance(environment_digest, str):
            raise ValueError("Universal Stage A member environment digest is unavailable")
        if not isinstance(architecture_digest, str):
            raise ValueError("Universal Stage A member architecture digest is unavailable")
        require_sha256(
            environment_digest,
            field="Universal Stage A member environment_digest",
        )
        require_sha256(
            architecture_digest,
            field="Universal Stage A member architecture_digest",
        )

        manifests = checkpoint_manifests(root / f"seed-{expected_seed}" / "checkpoints")
        final = tuple(
            item
            for item in manifests
            if item.seed == expected_seed
            and item.requested_timestep == training_config.timesteps
            and item.observed_timestep == actual_timesteps
        )
        if len(final) != 1:
            raise ValueError(
                "Universal Stage A requires exactly one final checkpoint per seed"
            )
        checkpoint = final[0]
        if checkpoint.algorithm != training_config.algorithm:
            raise ValueError("Universal Stage A final checkpoint algorithm mismatch")
        if checkpoint.training_config_digest != config_digest:
            raise ValueError("Universal Stage A final checkpoint config mismatch")
        if checkpoint.environment_digest != environment_digest:
            raise ValueError("Universal Stage A final checkpoint environment mismatch")
        identity = checkpoint.algorithm_identity
        if not isinstance(identity, Mapping):
            raise ValueError("Universal Stage A final checkpoint policy identity is missing")
        policy = identity.get("policy")
        if not isinstance(policy, Mapping):
            raise ValueError("Universal Stage A final checkpoint policy identity is missing")
        checkpoint_architecture = policy.get("policy_architecture_digest")
        if checkpoint_architecture != architecture_digest:
            raise ValueError("Universal Stage A policy architecture identity mismatch")
        if policy_architecture_digest is None:
            policy_architecture_digest = architecture_digest
        elif policy_architecture_digest != architecture_digest:
            raise ValueError("Universal Stage A policy architecture differs across seeds")
        require_sha256(checkpoint.digest, field="Universal Stage A checkpoint digest")
        checkpoint_digests.append((expected_seed, checkpoint.digest))

    if policy_architecture_digest is None:  # pragma: no cover - seeds are non-empty
        raise RuntimeError("Universal Stage A policy architecture identity disappeared")
    stage_a_candidate = StageACandidate.create(
        candidate_id=resolved_architecture.value,
        candidate_config_digest=config_digest,
        final_training_completion_digest=run_digest,
        policy_identity=policy_architecture_digest,
        checkpoint_digests=tuple(checkpoint_digests),
    )
    return UniversalStageACandidate(
        architecture=resolved_architecture,
        stage_a_candidate=stage_a_candidate,
        training_config=training_config,
    )
'''
    text = text.replace(marker, block + marker, 1)

old = '__all__ = ["UniversalStageACandidate", "UniversalStageAPlan"]\n'
new = '''__all__ = [
    "UniversalStageACandidate",
    "UniversalStageAPlan",
    "build_universal_stage_a_candidate_from_training",
]
'''
if old in text:
    text = text.replace(old, new, 1)
elif '"build_universal_stage_a_candidate_from_training"' not in text.split("__all__ =", 1)[-1]:
    raise SystemExit("Universal Stage A __all__ marker not found")

path.write_text(text)
compile(text, str(path), "exec")
