from pathlib import Path

path = Path("trade_rl/workflows/universal_training_runner.py")
text = path.read_text()
if "class UniversalDatasetArtifactEnvironmentFactory" in text:
    raise SystemExit(0)
import_anchor = "from trade_rl.data.contracts import (\n"
if import_anchor not in text:
    raise SystemExit("dataset artifact import anchor missing")
text = text.replace(
    import_anchor,
    "from trade_rl.data.artifact import load_market_dataset_artifact\n" + import_anchor,
    1,
)
action_anchor = "from trade_rl.rl.actions import ACTION_SCHEMA, ActionMode, ActionSpec\n"
if action_anchor not in text:
    raise SystemExit("runner action import anchor missing")
text = text.replace(
    action_anchor,
    action_anchor
    + "from trade_rl.risk.portfolio import PortfolioRiskModel\n"
    + "from trade_rl.risk.pretrade import PreTradeRisk\n"
    + "from trade_rl.rl.environment import ResidualMarketEnv\n",
    1,
)
context_anchor = "from trade_rl.rl.universal_instrument_context import CausalInstrumentContextProvider\n"
if context_anchor not in text:
    raise SystemExit("runner context import anchor missing")
text = text.replace(
    context_anchor,
    context_anchor + "from trade_rl.strategies.trend import TrendStrategy\n",
    1,
)
marker = "\n\n@dataclass(frozen=True, slots=True)\nclass UniversalRoutedEnvironmentFactory:\n"
if marker not in text:
    raise SystemExit("routed factory marker missing")
block = r'''


def _build_universal_concrete_environment(
    dataset: Any,
    *,
    run_config: Any,
    normalizers: tuple[Any, Any],
) -> ResidualMarketEnv:
    flat_normalizer, sequence_normalizer = normalizers
    return ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(run_config.trend),
        alpha_enabled=False,
        factor_count=0,
        action_spec=run_config.action,
        pre_trade_risk=PreTradeRisk(run_config.risk),
        portfolio_risk=PortfolioRiskModel(run_config.portfolio_risk),
        normalizer=flat_normalizer,
        sequence_normalizer=sequence_normalizer,
        config=run_config.environment,
    )


@dataclass(frozen=True, slots=True)
class UniversalDatasetArtifactEnvironmentFactory:
    """Load one concrete dataset lazily from immutable artifacts inside each worker."""

    dataset_artifact_paths: Mapping[str, Path]
    run_config: Any
    normalizers: Mapping[str, tuple[Any, Any]]

    def __post_init__(self) -> None:
        paths = {str(symbol): Path(value) for symbol, value in self.dataset_artifact_paths.items()}
        normalizers = dict(self.normalizers)
        if not paths or set(paths) != set(normalizers):
            raise ValueError("Universal dataset artifact paths and normalizers must share scope")
        if any(not symbol for symbol in paths):
            raise ValueError("Universal dataset artifact symbols must be non-empty")
        object.__setattr__(self, "dataset_artifact_paths", paths)
        object.__setattr__(self, "normalizers", normalizers)

    def __call__(self, binding: InstrumentDatasetBinding) -> Any:
        if not isinstance(binding, InstrumentDatasetBinding):
            raise TypeError("binding must be an InstrumentDatasetBinding")
        if binding.split != "train":
            raise ValueError("Universal training factory accepts train bindings only")
        symbol = binding.concrete_symbol
        try:
            dataset_path = self.dataset_artifact_paths[symbol]
            normalizers = self.normalizers[symbol]
        except KeyError as error:
            raise ValueError("Universal dataset artifact binding is outside factory scope") from error
        dataset = load_market_dataset_artifact(dataset_path)
        if tuple(getattr(dataset, "symbols", ())) != (symbol,):
            raise ValueError("Universal dataset artifact symbol identity mismatch")
        dataset_id = getattr(dataset, "dataset_id", None)
        if dataset_id != binding.source_dataset_id or dataset_id != binding.symbol_dataset_digest:
            raise ValueError("Universal dataset artifact identity mismatch")
        return _build_universal_concrete_environment(
            dataset,
            run_config=self.run_config,
            normalizers=normalizers,
        )
'''
text = text.replace(marker, block + marker, 1)
text = text.replace(
    '    "UniversalRoutedEnvironmentFactory",\n',
    '    "UniversalDatasetArtifactEnvironmentFactory",\n'
    '    "UniversalRoutedEnvironmentFactory",\n',
    1,
)
path.write_text(text)
compile(text, str(path), "exec")
