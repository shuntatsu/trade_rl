from pathlib import Path

path = Path("trade_rl/evaluation/experiments/bootstrap/config.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one replacement, found {count}: {old!r}")
    text = text.replace(old, new, 1)


replace_once(
    "from trade_rl.artifacts.hashing import content_digest\n",
    "from trade_rl.artifacts.hashing import content_digest\n"
    "from trade_rl.data.build import ExecutionEconomicsProfile\n",
)
replace_once(
    '_SCHEMA_VERSION = "canonical_m2_bootstrap_config_v1"\n'
    "_TOP_LEVEL_KEYS = frozenset(\n"
    "    {\n"
    '        "schema_version",\n'
    '        "research_question",\n'
    '        "market",\n'
    '        "symbols",\n'
    '        "base_timeframe",\n'
    '        "feature_timeframes",\n'
    '        "data_start",\n'
    '        "data_stop_exclusive",\n'
    '        "baseline",\n'
    '        "ppo_seeds",\n'
    '        "allowed_factors",\n'
    '        "max_experiments",\n'
    '        "n_bootstrap",\n'
    '        "bootstrap_seed",\n'
    "    }\n"
    ")\n",
    '_SCHEMA_VERSION_V1 = "canonical_m2_bootstrap_config_v1"\n'
    '_SCHEMA_VERSION_V2 = "canonical_m2_bootstrap_config_v2"\n'
    "_TOP_LEVEL_KEYS_V1 = frozenset(\n"
    "    {\n"
    '        "schema_version",\n'
    '        "research_question",\n'
    '        "market",\n'
    '        "symbols",\n'
    '        "base_timeframe",\n'
    '        "feature_timeframes",\n'
    '        "data_start",\n'
    '        "data_stop_exclusive",\n'
    '        "baseline",\n'
    '        "ppo_seeds",\n'
    '        "allowed_factors",\n'
    '        "max_experiments",\n'
    '        "n_bootstrap",\n'
    '        "bootstrap_seed",\n'
    "    }\n"
    ")\n"
    '_TOP_LEVEL_KEYS_V2 = frozenset((*_TOP_LEVEL_KEYS_V1, "execution_economics"))\n',
)
replace_once(
    "    bootstrap_seed: int\n"
    "    schema_version: str = _SCHEMA_VERSION\n\n"
    "    def __post_init__(self) -> None:\n"
    "        if self.schema_version != _SCHEMA_VERSION:\n"
    '            raise ValueError("schema_version does not match canonical M2 contract")\n',
    "    bootstrap_seed: int\n"
    "    execution_economics: ExecutionEconomicsProfile | None = None\n"
    "    schema_version: str = _SCHEMA_VERSION_V1\n\n"
    "    def __post_init__(self) -> None:\n"
    "        if self.schema_version not in {_SCHEMA_VERSION_V1, _SCHEMA_VERSION_V2}:\n"
    '            raise ValueError("schema_version does not match canonical M2 contract")\n'
    "        if self.schema_version == _SCHEMA_VERSION_V1:\n"
    "            if self.execution_economics is not None:\n"
    '                raise ValueError("v1 bootstrap config must not include execution_economics")\n'
    "        elif self.execution_economics is None:\n"
    '            raise ValueError("v2 bootstrap config requires execution_economics")\n'
    "        if self.execution_economics is not None and not isinstance(\n"
    "            self.execution_economics, ExecutionEconomicsProfile\n"
    "        ):\n"
    '            raise ValueError("execution_economics must be an ExecutionEconomicsProfile")\n',
)
replace_once(
    '''    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "research_question": self.research_question,
            "market": self.market.value,
            "symbols": list(self.symbols),
            "base_timeframe": self.base_timeframe,
            "feature_timeframes": list(self.feature_timeframes),
            "data_start": self.data_start.astimezone(UTC).isoformat(),
            "data_stop_exclusive": self.data_stop_exclusive.astimezone(UTC).isoformat(),
            "baseline": _baseline_payload(self.baseline),
            "ppo_seeds": list(self.ppo_seeds),
            "allowed_factors": [factor.value for factor in self.allowed_factors],
            "max_experiments": self.max_experiments,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_seed": self.bootstrap_seed,
        }
''',
    '''    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "research_question": self.research_question,
            "market": self.market.value,
            "symbols": list(self.symbols),
            "base_timeframe": self.base_timeframe,
            "feature_timeframes": list(self.feature_timeframes),
            "data_start": self.data_start.astimezone(UTC).isoformat(),
            "data_stop_exclusive": self.data_stop_exclusive.astimezone(UTC).isoformat(),
            "baseline": _baseline_payload(self.baseline),
            "ppo_seeds": list(self.ppo_seeds),
            "allowed_factors": [factor.value for factor in self.allowed_factors],
            "max_experiments": self.max_experiments,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_seed": self.bootstrap_seed,
        }
        if self.execution_economics is not None:
            payload["execution_economics"] = self.execution_economics.to_payload()
        return payload
''',
)
replace_once(
    "def _parse_config(raw: Mapping[str, object]) -> CanonicalM2BootstrapConfig:\n"
    '    _expect_exact_keys(raw, _TOP_LEVEL_KEYS, field="bootstrap config")\n'
    '    if raw.get("schema_version") != _SCHEMA_VERSION:\n'
    '        raise ValueError("schema_version does not match canonical M2 contract")\n',
    "def _parse_config(raw: Mapping[str, object]) -> CanonicalM2BootstrapConfig:\n"
    '    schema_version = _require_text(raw.get("schema_version"), field="schema_version")\n'
    "    if schema_version == _SCHEMA_VERSION_V1:\n"
    '        _expect_exact_keys(raw, _TOP_LEVEL_KEYS_V1, field="bootstrap config")\n'
    "        execution_economics = None\n"
    "    elif schema_version == _SCHEMA_VERSION_V2:\n"
    '        _expect_exact_keys(raw, _TOP_LEVEL_KEYS_V2, field="bootstrap config")\n'
    "        execution_economics = ExecutionEconomicsProfile.from_payload(\n"
    '            raw.get("execution_economics"),\n'
    '            field="execution_economics",\n'
    "        )\n"
    "    else:\n"
    '        raise ValueError("schema_version does not match canonical M2 contract")\n',
)
replace_once(
    '        bootstrap_seed=_int_value(raw.get("bootstrap_seed"), field="bootstrap_seed"),\n'
    "    )\n",
    '        bootstrap_seed=_int_value(raw.get("bootstrap_seed"), field="bootstrap_seed"),\n'
    "        execution_economics=execution_economics,\n"
    "        schema_version=schema_version,\n"
    "    )\n",
)
path.write_text(text, encoding="utf-8")
