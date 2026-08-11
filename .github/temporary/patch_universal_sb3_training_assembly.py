from pathlib import Path

path = Path("trade_rl/workflows/universal_training_runner.py")
text = path.read_text()

if "from trade_rl.integrations.sb3_training import StableBaselines3Backend\n" not in text:
    anchor = "from trade_rl.data.contracts import (\n"
    if anchor not in text:
        raise SystemExit("runner data contracts import anchor not found")
    text = text.replace(
        anchor,
        "from trade_rl.integrations.sb3_training import StableBaselines3Backend\n"
        "from trade_rl.integrations.universal_pretraining import (\n"
        "    UniversalPretrainingBundle,\n"
        "    build_universal_pretraining_hook,\n"
        ")\n"
        "from trade_rl.workflows.universal_teacher_runtime import (\n"
        "    build_universal_oracle_batches,\n"
        "    build_universal_pretraining_bundle_from_batches,\n"
        ")\n"
        + anchor,
        1,
    )

marker = "\n\ndef train_universal_seeds(\n"
if "def assemble_universal_sb3_training_backend(" not in text:
    if marker not in text:
        raise SystemExit("train_universal_seeds marker not found")
    block = r'''


def assemble_universal_sb3_training_backend(
    *,
    routed_environment_factory: UniversalRoutedEnvironmentFactory,
    training: Any,
    fold_train_range: tuple[int, int],
    normalizer_digest: str,
    feature_schema_digest: str,
    verbose: int = 0,
) -> tuple[StableBaselines3Backend, UniversalPretrainingBundle]:
    """Assemble the maintained U4 Oracle -> BC/critic -> SB3 training path."""

    if not isinstance(routed_environment_factory, UniversalRoutedEnvironmentFactory):
        raise TypeError(
            "routed_environment_factory must be a UniversalRoutedEnvironmentFactory"
        )
    behavior_cloning_epochs = getattr(training, "behavior_cloning_epochs", None)
    if (
        isinstance(behavior_cloning_epochs, bool)
        or not isinstance(behavior_cloning_epochs, int)
        or behavior_cloning_epochs <= 0
    ):
        raise ValueError("Universal U4 requires behavior cloning to be enabled")
    if getattr(training, "behavior_cloning_teacher", None) != "oracle":
        raise ValueError("Universal U4 requires the Oracle behavior-cloning teacher")
    behavior_cloning_seed = getattr(training, "behavior_cloning_seed", None)
    if (
        isinstance(behavior_cloning_seed, bool)
        or not isinstance(behavior_cloning_seed, int)
        or not 0 <= behavior_cloning_seed <= 0xFFFFFFFF
    ):
        raise ValueError(
            "Universal U4 requires an explicit uint32 behavior_cloning_seed"
        )
    n_envs = getattr(training, "n_envs", None)
    if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError("Universal U4 requires a positive n_envs")
    gamma = getattr(training, "gamma", None)
    if isinstance(gamma, bool) or not isinstance(gamma, int | float):
        raise ValueError("Universal U4 gamma must be numeric")
    gamma_value = float(gamma)
    if not 0.0 < gamma_value <= 1.0:
        raise ValueError("Universal U4 gamma must be in (0, 1]")
    validation_fraction = getattr(
        training, "behavior_cloning_validation_fraction", None
    )
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, int | float)
        or not 0.0 <= float(validation_fraction) < 0.5
    ):
        raise ValueError(
            "Universal U4 behavior_cloning_validation_fraction must be in [0, 0.5)"
        )
    if isinstance(verbose, bool) or not isinstance(verbose, int) or verbose < 0:
        raise ValueError("Universal U4 verbose must be a non-negative integer")
    provider = routed_environment_factory.instrument_context_provider
    if not callable(provider):
        raise ValueError("Universal U4 requires an instrument context provider")

    batches = build_universal_oracle_batches(
        train_symbols=routed_environment_factory.train_symbols,
        bindings=routed_environment_factory.bindings,
        concrete_environment_factory=(
            routed_environment_factory.concrete_environment_factory
        ),
        fold_train_range=fold_train_range,
        behavior_cloning_seed=behavior_cloning_seed,
        n_envs=n_envs,
    )
    bundle = build_universal_pretraining_bundle_from_batches(
        train_symbols=routed_environment_factory.train_symbols,
        bindings=routed_environment_factory.bindings,
        batches=batches,
        concrete_environment_factory=(
            routed_environment_factory.concrete_environment_factory
        ),
        instrument_context_provider=provider,
        partition_digest=routed_environment_factory.partition_digest,
        training_contract_digest=routed_environment_factory.training_contract_digest,
        run_seed=routed_environment_factory.run_seed,
        gamma=gamma_value,
        validation_fraction=float(validation_fraction),
        normalizer_digest=normalizer_digest,
        feature_schema_digest=feature_schema_digest,
    )
    backend = StableBaselines3Backend(
        routed_environment_factory,
        verbose=verbose,
        universal_pretraining_hook=build_universal_pretraining_hook(bundle),
    )
    return backend, bundle
'''
    text = text.replace(marker, block + marker, 1)

all_marker = "__all__ = [\n"
if all_marker not in text:
    raise SystemExit("runner __all__ marker not found")
if '    "assemble_universal_sb3_training_backend",\n' not in text:
    text = text.replace(
        all_marker,
        all_marker + '    "assemble_universal_sb3_training_backend",\n',
        1,
    )

path.write_text(text)
compile(text, str(path), "exec")
