from pathlib import Path

causal = Path("trade_rl/workflows/universal_causal_alpha_teacher.py")
text = causal.read_text(encoding="utf-8")
old = """from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaRidgeConfig,
"""
new = """from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
"""
if text.count(old) != 1:
    raise SystemExit("causal horizon import target drifted")
text = text.replace(old, new)

old = """        kwargs = {**base, **overrides, "name": name}
        candidate = _candidate(**kwargs)  # type: ignore[arg-type]
"""
new = """        kwargs = {**base, **overrides}
        candidate = _candidate(
            name=name,
            ridge_strength=float(kwargs["ridge_strength"]),
            horizon_mix=CausalAlphaHorizonMix(kwargs["horizon_mix"]),
            score_scale=float(kwargs["score_scale"]),
            entry_threshold=float(kwargs["entry_threshold"]),
            exit_threshold=float(kwargs["exit_threshold"]),
            no_trade_band=float(kwargs["no_trade_band"]),
            max_target_delta=float(kwargs["max_target_delta"]),
        )
"""
if text.count(old) != 1:
    raise SystemExit("causal candidate construction target drifted")
causal.write_text(text, encoding="utf-8")

runtime = Path("trade_rl/workflows/universal_teacher_runtime.py")
text = runtime.read_text(encoding="utf-8")
old = """from trade_rl.rl.universal_single_instrument_env import (
    EpisodeRoutedSingleInstrumentEnv,
    InstrumentContextProvider,
)
from trade_rl.workflows.universal_training import collect_universal_episode_teacher
"""
new = """from trade_rl.rl.universal_single_instrument_env import (
    EpisodeRoutedSingleInstrumentEnv,
    InstrumentContextProvider,
)
from trade_rl.workflows.universal_causal_alpha_teacher import (
    UniversalCausalAlphaTeacherPackage,
    latest_complete_episode_split,
)
from trade_rl.workflows.universal_training import collect_universal_episode_teacher
"""
if text.count(old) != 1:
    raise SystemExit("runtime causal import target drifted")
text = text.replace(old, new)

old = """    feature_schema_digest: str,
    teacher_kind: str = "oracle",
) -> UniversalPretrainingBundle:
"""
new = """    feature_schema_digest: str,
    teacher_kind: str = "oracle",
    causal_teacher_package: UniversalCausalAlphaTeacherPackage | None = None,
) -> UniversalPretrainingBundle:
"""
if text.count(old) != 1:
    raise SystemExit("runtime bundle signature target drifted")
text = text.replace(old, new)

old = """    if set(batches) != set(symbols):
        raise ValueError("Universal teacher batches must exactly match train_symbols")
    if (
"""
new = """    if set(batches) != set(symbols):
        raise ValueError("Universal teacher batches must exactly match train_symbols")
    if teacher_kind not in {"oracle", "trend_baseline", "causal_alpha_ridge"}:
        raise ValueError("Universal teacher kind is unsupported")
    if teacher_kind == "causal_alpha_ridge":
        if causal_teacher_package is None:
            raise ValueError("Universal causal teacher package is required")
        if causal_teacher_package.train_symbols != symbols:
            raise ValueError("Universal causal teacher package symbol scope mismatch")
        if set(causal_teacher_package.batches) != set(symbols):
            raise ValueError("Universal causal teacher package batch scope mismatch")
        if any(
            causal_teacher_package.batches[symbol] is not batches[symbol]
            for symbol in symbols
        ):
            raise ValueError("Universal causal teacher package batches must be reused")
    elif causal_teacher_package is not None:
        raise ValueError("Universal causal teacher package requires causal_alpha_ridge")
    if (
"""
if text.count(old) != 1:
    raise SystemExit("runtime package validation target drifted")
text = text.replace(old, new)

old = """        concrete_environment = concrete_environment_factory(binding)
        close_concrete = getattr(concrete_environment, "close", None)
        if not callable(close_concrete):
            raise TypeError("Universal teacher concrete environment must be closable")
        try:
            candidate_teacher_digest = (
                oracle_teacher_config_for_environment(concrete_environment).digest
                if teacher_kind == "oracle"
                else causal_trend_teacher_digest(concrete_environment)
            )
        finally:
            close_concrete()
        if candidate_teacher_digest != batch.teacher_config_digest:
            raise ValueError("Universal teacher config identity mismatch")
"""
new = """        if teacher_kind == "causal_alpha_ridge":
            if causal_teacher_package is None:
                raise RuntimeError("Universal causal teacher package disappeared")
            candidate_teacher_digest = causal_teacher_package.teacher_config_digest
        else:
            concrete_environment = concrete_environment_factory(binding)
            close_concrete = getattr(concrete_environment, "close", None)
            if not callable(close_concrete):
                raise TypeError("Universal teacher concrete environment must be closable")
            try:
                candidate_teacher_digest = (
                    oracle_teacher_config_for_environment(concrete_environment).digest
                    if teacher_kind == "oracle"
                    else causal_trend_teacher_digest(concrete_environment)
                )
            finally:
                close_concrete()
        if candidate_teacher_digest != batch.teacher_config_digest:
            raise ValueError("Universal teacher config identity mismatch")
"""
if text.count(old) != 1:
    raise SystemExit("runtime teacher digest target drifted")
text = text.replace(old, new)

old = """        split = behavior_cloning_split(
            collected.dataset,
            validation_fraction=validation_fraction,
        )
"""
new = """        if teacher_kind == "causal_alpha_ridge":
            if not batch.contracts:
                raise ValueError("Universal causal teacher batch has no episodes")
            split = latest_complete_episode_split(
                collected.dataset,
                holdout_episode_id=batch.contracts[-1].episode_index,
            )
        else:
            split = behavior_cloning_split(
                collected.dataset,
                validation_fraction=validation_fraction,
            )
"""
if text.count(old) != 1:
    raise SystemExit("runtime split target drifted")
text = text.replace(old, new)
runtime.write_text(text, encoding="utf-8")
