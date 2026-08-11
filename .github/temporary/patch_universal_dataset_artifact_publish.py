from pathlib import Path

path = Path("trade_rl/workflows/universal_training_runner.py")
text = path.read_text()
if "def publish_universal_train_dataset_artifacts(" in text:
    raise SystemExit(0)
text = text.replace(
    "from trade_rl.data.artifact import load_market_dataset_artifact\n",
    "from trade_rl.data.artifact import (\n"
    "    load_market_dataset_artifact,\n"
    "    publish_market_dataset_artifact,\n"
    ")\n",
    1,
)
marker = "\n\n@dataclass(frozen=True, slots=True)\nclass UniversalDatasetArtifactEnvironmentFactory:\n"
if marker not in text:
    raise SystemExit("artifact factory marker not found")
block = r'''


def publish_universal_train_dataset_artifacts(
    datasets: Mapping[str, Any],
    *,
    train_symbols: Sequence[str],
    artifact_root: Path,
) -> dict[str, Path]:
    """Publish or strictly reuse immutable train-only single-symbol datasets."""

    symbols = tuple(train_symbols)
    if not symbols or len(set(symbols)) != len(symbols) or any(not symbol for symbol in symbols):
        raise ValueError("Universal train_symbols must be non-empty and unique")
    if set(datasets) != set(symbols):
        raise ValueError("Universal dataset artifact scope must exactly match train_symbols")
    artifact_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for symbol in symbols:
        dataset = datasets[symbol]
        if tuple(getattr(dataset, "symbols", ())) != (symbol,):
            raise ValueError("Universal dataset artifact symbol identity mismatch")
        dataset_id = getattr(dataset, "dataset_id", None)
        if not isinstance(dataset_id, str):
            raise ValueError("Universal dataset artifact identity is unavailable")
        destination = artifact_root / symbol
        if destination.exists():
            existing = load_market_dataset_artifact(destination)
            if (
                tuple(getattr(existing, "symbols", ())) != (symbol,)
                or getattr(existing, "dataset_id", None) != dataset_id
            ):
                raise ValueError("Universal existing dataset artifact identity mismatch")
        else:
            publish_market_dataset_artifact(destination, dataset)
        paths[symbol] = destination
    return paths
'''
text = text.replace(marker, block + marker, 1)
text = text.replace(
    '    "concrete_action_spec_digest",\n',
    '    "concrete_action_spec_digest",\n'
    '    "publish_universal_train_dataset_artifacts",\n',
    1,
)
path.write_text(text)
compile(text, str(path), "exec")
