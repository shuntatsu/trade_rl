from pathlib import Path

path = Path("trade_rl/workflows/universal_training_runner.py")
text = path.read_text()
if "def build_universal_bindings(" in text:
    raise SystemExit(0)
text = text.replace(
    "from functools import partial\n",
    "from functools import partial\nfrom pathlib import Path\n",
    1,
)
text = text.replace(
    "from trade_rl.artifacts.hashing import content_digest\n",
    "from trade_rl.artifacts.atomic_write import atomic_write_bytes\n"
    "from trade_rl.artifacts.codec import canonical_json_bytes\n"
    "from trade_rl.artifacts.hashing import content_digest\n"
    "from trade_rl.artifacts.verified_file import file_digest\n",
    1,
)
marker = "\n\n__all__ = [\n"
if marker not in text:
    raise SystemExit("universal_training_runner __all__ marker not found")
block = r'''


def build_universal_bindings(
    *,
    datasets: Mapping[str, Any],
    contracts: Mapping[str, Any],
    catalog: Any,
    train_symbols: Sequence[str],
) -> tuple[InstrumentDatasetBinding, ...]:
    """Bind each concrete train dataset to immutable metadata and descriptor evidence."""

    symbols = tuple(train_symbols)
    if set(datasets) != set(symbols) or set(contracts) != set(symbols):
        raise ValueError("Universal bindings must close exactly over train_symbols")
    raw_metadata = getattr(catalog, "per_symbol_metadata_digests", None)
    if not isinstance(raw_metadata, (tuple, list)):
        raise TypeError("Universal catalog metadata digests are unavailable")
    metadata_by_symbol = {str(symbol): str(digest) for symbol, digest in raw_metadata}
    missing = set(symbols) - set(metadata_by_symbol)
    if missing:
        raise ValueError(
            f"Universal catalog metadata digests are missing train symbols: {sorted(missing)}"
        )

    bindings: list[InstrumentDatasetBinding] = []
    for symbol in symbols:
        dataset_id = getattr(datasets[symbol], "dataset_id", None)
        if not isinstance(dataset_id, str):
            raise ValueError(f"Universal dataset identity is unavailable for {symbol}")
        contract = contracts[symbol]
        payload_method = getattr(contract, "canonical_payload", None)
        if not callable(payload_method):
            raise TypeError("Universal instrument contract must expose canonical_payload")
        bindings.append(
            InstrumentDatasetBinding(
                concrete_symbol=symbol,
                source_dataset_id=dataset_id,
                symbol_dataset_digest=dataset_id,
                execution_metadata_digest=metadata_by_symbol[symbol],
                instrument_descriptor_digest=content_digest(payload_method()),
                split="train",
            )
        )
    return tuple(bindings)


def train_universal_seeds(
    *,
    runtime: Any,
    training: Any,
    backend: Any,
    output_root: Path,
    architecture_name: str,
) -> dict[str, object]:
    """Train every configured member and publish one non-research-success manifest."""

    seeds = tuple(getattr(training, "seeds", ()))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Universal training seeds must be non-empty and unique")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Universal training seeds must be non-negative integers")
    digest_payload = getattr(training, "digest_payload", None)
    if not callable(digest_payload):
        raise TypeError("Universal training config must expose digest_payload")
    training_config_digest = content_digest(digest_payload())
    output_root.mkdir(parents=True, exist_ok=True)

    members: list[dict[str, object]] = []
    for seed in seeds:
        policy_path = output_root / f"seed-{seed}" / "policy.zip"
        result = backend.train(seed=seed, config=training, output_path=policy_path)
        checkpoint_path = Path(getattr(result, "checkpoint_path", policy_path))
        if checkpoint_path != policy_path or not policy_path.is_file():
            raise RuntimeError("Universal backend did not publish the requested policy path")
        environment_digest = getattr(result, "environment_digest", None)
        if not isinstance(environment_digest, str):
            raise ValueError("Universal backend environment digest is unavailable")
        architecture_digest = getattr(result, "architecture_digest", None)
        actual_timesteps = getattr(result, "actual_timesteps", None)
        if isinstance(actual_timesteps, bool) or not isinstance(actual_timesteps, int):
            raise ValueError("Universal backend actual_timesteps is unavailable")
        members.append(
            {
                "seed": seed,
                "policy_file": policy_path.relative_to(output_root).as_posix(),
                "policy_digest": file_digest(policy_path, field="Universal policy"),
                "environment_digest": environment_digest,
                "architecture_digest": architecture_digest,
                "actual_timesteps": actual_timesteps,
            }
        )

    payload: dict[str, object] = {
        "schema_version": "universal_training_run_v1",
        "architecture_name": architecture_name,
        "train_symbols": list(getattr(runtime, "train_symbols")),
        "catalog_digest": getattr(runtime, "catalog_digest"),
        "partition_digest": getattr(runtime, "partition_digest"),
        "split_manifest_digest": getattr(runtime, "split_manifest_digest"),
        "feature_schema_digest": getattr(runtime, "feature_schema_digest"),
        "statistics_digest": getattr(runtime, "statistics_digest"),
        "instrument_context_schema_digest": getattr(
            runtime, "instrument_context_schema_digest"
        ),
        "training_contract_digest": getattr(runtime, "training_contract_digest"),
        "pretraining_artifact_digest": getattr(
            runtime, "pretraining_artifact_digest", None
        ),
        "training_config_digest": training_config_digest,
        "members": members,
        "research_success": False,
        "research_success_reason": "sealed zero-shot evidence not evaluated by training runner",
    }
    run_digest = content_digest(payload)
    manifest = {**payload, "run_digest": run_digest}
    atomic_write_bytes(
        output_root / "universal-training.json",
        canonical_json_bytes(manifest),
    )
    return manifest
'''
text = text.replace(marker, block + marker, 1)
text = text.replace(
    '    "UniversalRoutedEnvironmentFactory",\n',
    '    "UniversalRoutedEnvironmentFactory",\n'
    '    "build_universal_bindings",\n'
    '    "train_universal_seeds",\n',
    1,
)
path.write_text(text)
compile(text, str(path), "exec")
