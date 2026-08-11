from pathlib import Path

path = Path("trade_rl/workflows/universal_teacher_runtime.py")
text = path.read_text()

import_anchor = "from trade_rl.integrations.universal_pretraining import (\n"
if "build_episode_oracle_batch_for_environment" not in text:
    text = text.replace(
        import_anchor,
        "from trade_rl.integrations.sb3_runtime import (\n"
        "    build_episode_oracle_batch_for_environment,\n"
        ")\n"
        + import_anchor,
        1,
    )

marker = "\n\ndef build_universal_pretraining_bundle_from_batches(\n"
if "def build_universal_oracle_batches(" not in text:
    if marker not in text:
        raise SystemExit("teacher bundle marker not found")
    block = r'''


def build_universal_oracle_batches(
    *,
    train_symbols: Sequence[str],
    bindings: Sequence[InstrumentDatasetBinding],
    concrete_environment_factory: Callable[[InstrumentDatasetBinding], Any],
    fold_train_range: tuple[int, int],
    behavior_cloning_seed: int,
    n_envs: int,
) -> dict[str, EpisodeOracleBatch]:
    """Solve paired Oracle episodes for every train symbol inside one fold."""

    symbols = tuple(train_symbols)
    binding_values = tuple(bindings)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Universal Oracle train_symbols must be non-empty and unique")
    if tuple(binding.concrete_symbol for binding in binding_values) != symbols:
        raise ValueError("Universal Oracle bindings must follow train_symbols exactly")
    if any(binding.split != "train" for binding in binding_values):
        raise ValueError("Universal Oracle accepts train bindings only")
    if not callable(concrete_environment_factory):
        raise TypeError("Universal Oracle concrete environment factory must be callable")
    start, stop = fold_train_range
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
    ):
        raise ValueError("Universal Oracle fold_train_range is invalid")
    if (
        isinstance(behavior_cloning_seed, bool)
        or not isinstance(behavior_cloning_seed, int)
        or not 0 <= behavior_cloning_seed <= 0xFFFFFFFF
    ):
        raise ValueError("Universal Oracle behavior_cloning_seed is invalid")
    if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError("Universal Oracle n_envs must be a positive integer")

    batches: dict[str, EpisodeOracleBatch] = {}
    for symbol, binding in zip(symbols, binding_values, strict=True):
        environment = concrete_environment_factory(binding)
        try:
            batch = build_episode_oracle_batch_for_environment(
                environment,
                train_range=(start, stop),
                seed=behavior_cloning_seed,
                n_envs=n_envs,
            )
        finally:
            environment.close()
        if batch.dataset_id != binding.source_dataset_id:
            raise ValueError("Universal Oracle batch dataset identity mismatch")
        batches[symbol] = batch
    return batches
'''
    text = text.replace(marker, block + marker, 1)

old_all = '''__all__ = [\n    "build_universal_pretraining_bundle_from_batches",\n    "build_universal_symbol_teacher_environment",\n]\n'''
new_all = '''__all__ = [\n    "build_universal_oracle_batches",\n    "build_universal_pretraining_bundle_from_batches",\n    "build_universal_symbol_teacher_environment",\n]\n'''
if old_all in text:
    text = text.replace(old_all, new_all, 1)
elif '"build_universal_oracle_batches"' not in text.split("__all__ =", 1)[-1]:
    raise SystemExit("teacher runtime __all__ marker not found")

path.write_text(text)
compile(text, str(path), "exec")
