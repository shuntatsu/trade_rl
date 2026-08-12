from pathlib import Path

causal = Path("trade_rl/workflows/universal_causal_alpha_teacher.py")
text = causal.read_text(encoding="utf-8")

old = "from dataclasses import dataclass\nfrom types import SimpleNamespace\nfrom typing import Any, Mapping\n"
new = "from dataclasses import dataclass\nfrom functools import partial\nfrom types import SimpleNamespace\nfrom typing import Any, Mapping\n"
if text.count(old) != 1:
    raise SystemExit("causal import target drifted")
text = text.replace(old, new)

old = "from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding\n"
new = """from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
"""
if text.count(old) != 1:
    raise SystemExit("risk import target drifted")
text = text.replace(old, new)

marker = "\ndef _train_range(\n"
if text.count(marker) != 1:
    raise SystemExit("package insertion marker drifted")
package_types = r'''

@dataclass(frozen=True, slots=True)
class UniversalCausalAlphaTeacherPackage:
    """One immutable train-only teacher identity shared across Universal consumers."""

    train_symbols: tuple[str, ...]
    batches: Mapping[str, EpisodeOracleBatch]
    partitions: Mapping[str, CausalAlphaEpisodePartition]
    samples: Mapping[str, CausalAlphaSymbolSamples]
    selection: CausalAlphaSelectionEvidence
    selected_candidate_digest: str
    teacher_config_digest: str
    batch_evidence: Mapping[str, CausalAlphaBatchEvidence]
    digest: str = ""

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("causal alpha package train_symbols must be unique")
        mappings = {
            "batches": dict(self.batches),
            "partitions": dict(self.partitions),
            "samples": dict(self.samples),
            "batch_evidence": dict(self.batch_evidence),
        }
        for field, values in mappings.items():
            if set(values) != set(symbols):
                raise ValueError(
                    f"causal alpha package {field} must exactly match train_symbols"
                )
        if self.selection.selected_candidate_digest != self.selected_candidate_digest:
            raise ValueError("causal alpha package selected candidate identity drifted")
        for field, value in (
            ("selected_candidate_digest", self.selected_candidate_digest),
            ("teacher_config_digest", self.teacher_config_digest),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"causal alpha package {field} is invalid")
        for symbol in symbols:
            batch = mappings["batches"][symbol]
            if getattr(batch, "teacher_config_digest", None) != self.teacher_config_digest:
                raise ValueError("causal alpha package batch teacher identity drifted")
            for field in ("partitions", "samples", "batch_evidence"):
                digest = getattr(mappings[field][symbol], "digest", None)
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError(
                        f"causal alpha package {field} digest is unavailable"
                    )
        expected = content_digest(
            {
                "batch_digests": {
                    symbol: getattr(mappings["batches"][symbol], "digest", None)
                    for symbol in symbols
                },
                "batch_evidence_digests": {
                    symbol: mappings["batch_evidence"][symbol].digest
                    for symbol in symbols
                },
                "partition_digests": {
                    symbol: mappings["partitions"][symbol].digest for symbol in symbols
                },
                "sample_digests": {
                    symbol: mappings["samples"][symbol].digest for symbol in symbols
                },
                "schema_version": "universal_causal_alpha_teacher_package_v1",
                "selected_candidate_digest": self.selected_candidate_digest,
                "selection_digest": self.selection.digest,
                "teacher_config_digest": self.teacher_config_digest,
                "train_symbols": symbols,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha teacher package digest mismatch")
        object.__setattr__(self, "train_symbols", symbols)
        for field, values in mappings.items():
            object.__setattr__(self, field, values)
        object.__setattr__(self, "digest", expected)


def _candidate(
    *,
    name: str,
    ridge_strength: float,
    horizon_mix: CausalAlphaHorizonMix,
    score_scale: float,
    entry_threshold: float,
    exit_threshold: float,
    no_trade_band: float,
    max_target_delta: float,
) -> CausalAlphaCandidateConfig:
    return CausalAlphaCandidateConfig(
        name=name,
        ridge=CausalAlphaRidgeConfig(ridge_strength=ridge_strength),
        controller=CausalAlphaControllerConfig(
            horizon_mix=horizon_mix,
            score_scale=score_scale,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            no_trade_band=no_trade_band,
            max_target_delta=max_target_delta,
        ),
    )


def default_causal_alpha_candidate_grid(
    risk_config: PreTradeRiskConfig,
) -> tuple[CausalAlphaCandidateConfig, ...]:
    """Return the maintained bounded one-factor-at-a-time causal teacher grid."""

    if not isinstance(risk_config, PreTradeRiskConfig):
        raise TypeError("causal alpha default grid requires PreTradeRiskConfig")
    no_trade = float(risk_config.no_trade_band)
    max_delta = min(0.25, float(risk_config.max_abs_weight))
    if max_delta <= 0.0:
        raise ValueError("causal alpha max target delta cannot be resolved")
    base = dict(
        ridge_strength=0.01,
        horizon_mix=CausalAlphaHorizonMix.EQUAL,
        score_scale=50.0,
        entry_threshold=0.003,
        exit_threshold=0.001,
        no_trade_band=no_trade,
        max_target_delta=max_delta,
    )
    variants: tuple[tuple[str, dict[str, object]], ...] = (
        ("baseline", {}),
        ("ridge-strong", {"ridge_strength": 0.1}),
        ("horizon-24h", {"horizon_mix": CausalAlphaHorizonMix.H24}),
        ("horizon-72h", {"horizon_mix": CausalAlphaHorizonMix.H72}),
        ("scale-low", {"score_scale": 25.0}),
        ("scale-high", {"score_scale": 100.0}),
        (
            "threshold-low",
            {"entry_threshold": 0.0015, "exit_threshold": 0.0005},
        ),
        (
            "threshold-high",
            {"entry_threshold": 0.006, "exit_threshold": 0.002},
        ),
        ("no-trade-low", {"no_trade_band": no_trade * 0.5}),
        (
            "no-trade-high",
            {"no_trade_band": min(float(risk_config.max_abs_weight), no_trade * 2.0)},
        ),
        ("delta-low", {"max_target_delta": max_delta * 0.5}),
        (
            "delta-high",
            {
                "max_target_delta": min(
                    float(risk_config.max_abs_weight), max_delta * 2.0
                )
            },
        ),
    )
    result: list[CausalAlphaCandidateConfig] = []
    observed: set[str] = set()
    for name, overrides in variants:
        kwargs = {**base, **overrides, "name": name}
        candidate = _candidate(**kwargs)  # type: ignore[arg-type]
        if candidate.digest not in observed:
            observed.add(candidate.digest)
            result.append(candidate)
    if len(result) < 8:
        raise ValueError("causal alpha default grid collapsed unexpectedly")
    return tuple(result)
'''
text = text.replace(marker, package_types + marker)

old = """def build_causal_alpha_episode_batch(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    partition: CausalAlphaEpisodePartition,
    ridge_config: CausalAlphaRidgeConfig,
    controller_config: CausalAlphaControllerConfig,
) -> tuple[EpisodeOracleBatch, CausalAlphaBatchEvidence]:
"""
new = """def build_causal_alpha_episode_batch(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    partition: CausalAlphaEpisodePartition,
    ridge_config: CausalAlphaRidgeConfig,
    controller_config: CausalAlphaControllerConfig,
    teacher_config_digest: str | None = None,
) -> tuple[EpisodeOracleBatch, CausalAlphaBatchEvidence]:
"""
if text.count(old) != 1:
    raise SystemExit("batch signature target drifted")
text = text.replace(old, new)

old = """    batch = EpisodeOracleBatch(
        dataset_id=block.dataset_id,
        teacher_config_digest=evidence.digest,
"""
new = """    resolved_teacher_config_digest = (
        evidence.digest if teacher_config_digest is None else teacher_config_digest
    )
    if not isinstance(resolved_teacher_config_digest, str) or len(resolved_teacher_config_digest) != 64:
        raise ValueError("causal alpha teacher_config_digest must be SHA-256")
    batch = EpisodeOracleBatch(
        dataset_id=block.dataset_id,
        teacher_config_digest=resolved_teacher_config_digest,
"""
if text.count(old) != 1:
    raise SystemExit("batch teacher digest target drifted")
text = text.replace(old, new)

marker = "\ndef build_causal_alpha_episode_batch(\n"
if text.count(marker) != 1:
    raise SystemExit("package builder marker drifted")
package_builder = r'''

def build_universal_causal_alpha_teacher_package(
    *,
    train_symbols: tuple[str, ...],
    bindings: tuple[InstrumentDatasetBinding, ...],
    concrete_environment_factory: Any,
    instrument_context_provider: Any,
    fold_train_range: tuple[int, int],
    feature_schema_digest: str,
    episode_hours: float | None = None,
    candidates: tuple[CausalAlphaCandidateConfig, ...] | None = None,
) -> UniversalCausalAlphaTeacherPackage:
    """Build the causal teacher exactly once for all Universal consumers."""

    symbols = tuple(train_symbols)
    binding_values = tuple(bindings)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("causal alpha package train_symbols must be unique")
    if tuple(binding.concrete_symbol for binding in binding_values) != symbols:
        raise ValueError("causal alpha package bindings must follow train_symbols")
    if any(binding.split != "train" for binding in binding_values):
        raise ValueError("causal alpha package accepts train bindings only")
    if not callable(concrete_environment_factory):
        raise TypeError("causal alpha concrete environment factory must be callable")

    partitions: dict[str, CausalAlphaEpisodePartition] = {}
    samples: dict[str, CausalAlphaSymbolSamples] = {}
    risk_configs: list[PreTradeRiskConfig] = []
    observed_episode_hours: list[float] = []
    for symbol, binding in zip(symbols, binding_values, strict=True):
        environment = concrete_environment_factory(binding)
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("causal alpha concrete environment must be closable")
        try:
            partitions[symbol] = build_chronological_episode_partition(
                environment,
                train_range=fold_train_range,
            )
            samples[symbol] = build_causal_alpha_symbol_samples(
                environment=environment,
                binding=binding,
                instrument_context_provider=instrument_context_provider,
                train_range=fold_train_range,
                feature_schema_digest=feature_schema_digest,
            )
            risk_config = getattr(
                getattr(environment, "pre_trade_risk", None), "config", None
            )
            if not isinstance(risk_config, PreTradeRiskConfig):
                raise TypeError("causal alpha environment risk config is unavailable")
            risk_configs.append(risk_config)
            environment_episode_hours = getattr(
                getattr(environment, "config", None), "episode_hours", None
            )
            if isinstance(environment_episode_hours, bool) or not isinstance(
                environment_episode_hours, int | float
            ):
                raise ValueError("causal alpha environment episode_hours is unavailable")
            observed_episode_hours.append(float(environment_episode_hours))
        finally:
            close()
    validate_universal_causal_alpha_partitions(
        train_symbols=symbols,
        partitions=partitions,
    )
    if len({content_digest(config) for config in risk_configs}) != 1:
        raise ValueError("causal alpha train-symbol risk configs differ")
    if len(set(observed_episode_hours)) != 1:
        raise ValueError("causal alpha train-symbol episode horizons differ")
    resolved_episode_hours = (
        observed_episode_hours[0] if episode_hours is None else float(episode_hours)
    )
    if not np.isfinite(resolved_episode_hours) or resolved_episode_hours <= 0.0:
        raise ValueError("causal alpha package episode_hours must be positive")
    if any(
        abs(value - resolved_episode_hours) > 1e-12 for value in observed_episode_hours
    ):
        raise ValueError("causal alpha requested episode_hours differs from environment")

    candidate_values = (
        default_causal_alpha_candidate_grid(risk_configs[0])
        if candidates is None
        else tuple(candidates)
    )
    if not candidate_values:
        raise ValueError("causal alpha candidate grid must be non-empty")
    binding_by_symbol = {
        binding.concrete_symbol: binding for binding in binding_values
    }
    selection = evaluate_causal_alpha_selection(
        train_symbols=symbols,
        samples=samples,
        partitions=partitions,
        candidates=candidate_values,
        environment_factories={
            symbol: partial(concrete_environment_factory, binding_by_symbol[symbol])
            for symbol in symbols
        },
        episode_hours=resolved_episode_hours,
    )
    selected_evidence = tuple(
        item
        for item in selection.candidates
        if item.candidate.digest == selection.selected_candidate_digest
    )
    if len(selected_evidence) != 1:
        raise RuntimeError("causal alpha selected candidate cannot be resolved")
    selected = selected_evidence[0].candidate
    teacher_config_digest = content_digest(
        {
            "feature_schema_digest": feature_schema_digest,
            "schema_version": "universal_causal_alpha_teacher_config_v1",
            "selected_candidate_digest": selected.digest,
            "selection_digest": selection.digest,
        }
    )
    batches: dict[str, EpisodeOracleBatch] = {}
    batch_evidence: dict[str, CausalAlphaBatchEvidence] = {}
    for symbol in symbols:
        batch, evidence = build_causal_alpha_episode_batch(
            symbol=symbol,
            train_symbols=symbols,
            samples=samples,
            partition=partitions[symbol],
            ridge_config=selected.ridge,
            controller_config=selected.controller,
            teacher_config_digest=teacher_config_digest,
        )
        batches[symbol] = batch
        batch_evidence[symbol] = evidence
    return UniversalCausalAlphaTeacherPackage(
        train_symbols=symbols,
        batches=batches,
        partitions=partitions,
        samples=samples,
        selection=selection,
        selected_candidate_digest=selected.digest,
        teacher_config_digest=teacher_config_digest,
        batch_evidence=batch_evidence,
    )
'''
text = text.replace(marker, package_builder + marker)

old = '    "CausalAlphaSelectionEvidence",\n    "CausalAlphaSymbolSamples",\n'
new = '    "CausalAlphaSelectionEvidence",\n    "CausalAlphaSymbolSamples",\n    "UniversalCausalAlphaTeacherPackage",\n'
if text.count(old) != 1:
    raise SystemExit("package type export target drifted")
text = text.replace(old, new)
old = '    "build_causal_alpha_symbol_samples",\n    "build_chronological_episode_partition",\n'
new = '    "build_causal_alpha_symbol_samples",\n    "build_chronological_episode_partition",\n    "build_universal_causal_alpha_teacher_package",\n    "default_causal_alpha_candidate_grid",\n'
if text.count(old) != 1:
    raise SystemExit("package function export target drifted")
text = text.replace(old, new)
causal.write_text(text, encoding="utf-8")

runner = Path("trade_rl/workflows/universal_training_runner.py")
text = runner.read_text(encoding="utf-8")
old = "from trade_rl.workflows.universal_teacher_runtime import (\n"
new = """from trade_rl.workflows.universal_causal_alpha_teacher import (
    UniversalCausalAlphaTeacherPackage,
    build_universal_causal_alpha_teacher_package,
)
from trade_rl.workflows.universal_teacher_runtime import (
"""
if text.count(old) != 1:
    raise SystemExit("runner causal import target drifted")
text = text.replace(old, new)

old = """    oracle_batches: Mapping[str, EpisodeOracleBatch] | None = None,
    verbose: int = 0,
) -> tuple[StableBaselines3Backend, UniversalPretrainingBundle]:
"""
new = """    oracle_batches: Mapping[str, EpisodeOracleBatch] | None = None,
    causal_teacher_package: UniversalCausalAlphaTeacherPackage | None = None,
    verbose: int = 0,
) -> tuple[StableBaselines3Backend, UniversalPretrainingBundle]:
"""
if text.count(old) != 1:
    raise SystemExit("runner signature target drifted")
text = text.replace(old, new)

old = '''    if teacher_kind not in {"oracle", "trend_baseline"}:
        raise ValueError("Universal U4 behavior-cloning teacher is unsupported")
'''
new = '''    if teacher_kind not in {"oracle", "trend_baseline", "causal_alpha_ridge"}:
        raise ValueError("Universal U4 behavior-cloning teacher is unsupported")
'''
if text.count(old) != 1:
    raise SystemExit("runner teacher kind target drifted")
text = text.replace(old, new)

old = """    if oracle_batches is None:
        batches = build_universal_teacher_batches(
            teacher_kind=teacher_kind,
            train_symbols=routed_environment_factory.train_symbols,
            bindings=routed_environment_factory.bindings,
            concrete_environment_factory=(
                routed_environment_factory.concrete_environment_factory
            ),
            fold_train_range=fold_train_range,
            behavior_cloning_seed=behavior_cloning_seed,
            n_envs=n_envs,
        )
    else:
        batches = dict(oracle_batches)
        if set(batches) != set(routed_environment_factory.train_symbols):
            raise ValueError(
                "Universal U4 oracle_batches must exactly match train_symbols"
            )
        if any(not isinstance(batch, EpisodeOracleBatch) for batch in batches.values()):
            raise TypeError(
                "Universal U4 oracle_batches must contain EpisodeOracleBatch"
            )
    bundle = build_universal_pretraining_bundle_from_batches(
"""
new = """    resolved_causal_package = causal_teacher_package
    if teacher_kind == "causal_alpha_ridge":
        if oracle_batches is not None:
            raise ValueError(
                "Universal causal alpha training cannot accept legacy oracle_batches"
            )
        if resolved_causal_package is None:
            resolved_causal_package = build_universal_causal_alpha_teacher_package(
                train_symbols=routed_environment_factory.train_symbols,
                bindings=routed_environment_factory.bindings,
                concrete_environment_factory=(
                    routed_environment_factory.concrete_environment_factory
                ),
                instrument_context_provider=provider,
                fold_train_range=fold_train_range,
                feature_schema_digest=feature_schema_digest,
            )
        batches = dict(resolved_causal_package.batches)
    else:
        if resolved_causal_package is not None:
            raise ValueError(
                "Universal causal teacher package requires causal_alpha_ridge"
            )
        if oracle_batches is None:
            batches = build_universal_teacher_batches(
                teacher_kind=teacher_kind,
                train_symbols=routed_environment_factory.train_symbols,
                bindings=routed_environment_factory.bindings,
                concrete_environment_factory=(
                    routed_environment_factory.concrete_environment_factory
                ),
                fold_train_range=fold_train_range,
                behavior_cloning_seed=behavior_cloning_seed,
                n_envs=n_envs,
            )
        else:
            batches = dict(oracle_batches)
            if set(batches) != set(routed_environment_factory.train_symbols):
                raise ValueError(
                    "Universal U4 oracle_batches must exactly match train_symbols"
                )
            if any(
                not isinstance(batch, EpisodeOracleBatch) for batch in batches.values()
            ):
                raise TypeError(
                    "Universal U4 oracle_batches must contain EpisodeOracleBatch"
                )
    bundle = build_universal_pretraining_bundle_from_batches(
"""
if text.count(old) != 1:
    raise SystemExit("runner batch routing target drifted")
text = text.replace(old, new)

old = """        feature_schema_digest=feature_schema_digest,
        teacher_kind=teacher_kind,
    )
"""
new = """        feature_schema_digest=feature_schema_digest,
        teacher_kind=teacher_kind,
        causal_teacher_package=resolved_causal_package,
    )
"""
if text.count(old) != 1:
    raise SystemExit("runner bundle package target drifted")
text = text.replace(old, new)
runner.write_text(text, encoding="utf-8")
