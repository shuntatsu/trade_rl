from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path.cwd()

EXECUTION_DEFAULTS = {
    "fee_rate": 0.0005,
    "maker_fee_rate": 0.0,
    "taker_fee_rate": 0.0,
    "spread_rate": 0.0002,
    "impact_rate": 0.0001,
    "multiplier": 1.0,
    "max_participation_rate": 0.05,
    "slippage_std": 0.0,
    "tail_slippage_probability": 0.0,
    "tail_slippage_multiplier": 5.0,
    "random_seed": 0,
    "minimum_notional": 0.0,
    "lot_size": 0.0,
    "tick_size": 0.0,
    "allow_short": True,
    "borrow_rate_multiplier": 1.0,
    "max_leverage": 1.0,
    "maintenance_margin_rate": 0.25,
    "collateral_haircut": 1.0,
    "margin_mode": "cross",
    "order_latency_bars": 0,
    "order_type": "market",
    "limit_offset_rate": 0.0005,
    "path_mode": "conservative",
    "processing_bar_volume_capacity": True,
    "partial_fill_carry": True,
    "trigger_volume_fractions": [1.0, 0.5, 0.25, 0.0],
}

red_path = ROOT / "tests/workflows/test_training_config_explicit_semantics.py"
red_path.write_text(
    '''from __future__ import annotations

from dataclasses import fields

import pytest

from tests.training_config_support import complete_execution_config
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.training_run import TrainingRunConfig


def _mapping() -> dict[str, object]:
    return {
        "schema_version": "training_run_config_v4",
        "training": {
            "timesteps": 8,
            "gamma": 0.99,
            "seeds": [0],
            "n_steps": 8,
            "batch_size": 8,
            "policy_actor_head": "standard_continuous_v1",
            "hierarchical_gate_temperature": 1.0,
            "behavior_cloning_gate_loss_weight": 1.0,
            "behavior_cloning_target_loss_weight": 1.0,
            "behavior_cloning_composed_loss_weight": 1.0,
            "behavior_cloning_gate_change_threshold": 0.05,
            "behavior_cloning_max_positive_class_weight": 20.0,
            "behavior_cloning_min_gate_precision": 0.0,
            "behavior_cloning_min_gate_recall": 0.0,
            "behavior_cloning_max_active_target_rmse": 1.0,
            "behavior_cloning_min_activity_ratio": 0.0,
            "behavior_cloning_max_activity_ratio": 1.0,
            "behavior_cloning_min_causal_holdout_trades": 0,
            "behavior_cloning_max_causal_holdout_regret": 0.0,
            "behavior_cloning_causal_holdout_bootstrap_resamples": 2_000,
            "behavior_cloning_causal_holdout_confidence_level": 0.95,
        },
        "environment": {
            "episode_bars": 4,
            "decision_every": 1,
            "initial_capital": 1_000.0,
        },
        "risk": {},
        "reward": {},
        "trend": {"fast_lookback": 1, "base_lookback": 2, "slow_lookback": 3},
        "action": {"alpha_enabled": False, "n_factors": 0},
    }


def test_complete_execution_fixture_tracks_public_execution_fields() -> None:
    expected = {item.name for item in fields(ExecutionCostConfig) if item.init}
    assert set(complete_execution_config()) == expected


def test_training_config_rejects_v3_with_migration_message() -> None:
    raw = _mapping()
    raw["schema_version"] = "training_run_config_v3"
    raw["execution"] = complete_execution_config()
    environment = dict(raw["environment"])
    environment["require_full_reward_preroll"] = True
    raw["environment"] = environment

    with pytest.raises(ValueError, match="training_run_config_v4"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_omitted_execution_section() -> None:
    raw = _mapping()
    environment = dict(raw["environment"])
    environment["require_full_reward_preroll"] = True
    raw["environment"] = environment

    with pytest.raises(ValueError, match="execution"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_omitted_execution_field() -> None:
    raw = _mapping()
    execution = complete_execution_config()
    del execution["maker_fee_rate"]
    raw["execution"] = execution
    environment = dict(raw["environment"])
    environment["require_full_reward_preroll"] = True
    raw["environment"] = environment

    with pytest.raises(ValueError, match="maker_fee_rate"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_rejects_omitted_reward_preroll_contract() -> None:
    raw = _mapping()
    raw["execution"] = complete_execution_config()

    with pytest.raises(ValueError, match="require_full_reward_preroll"):
        TrainingRunConfig.from_mapping(raw)


def test_training_config_accepts_complete_explicit_semantics() -> None:
    raw = _mapping()
    raw["execution"] = complete_execution_config()
    environment = dict(raw["environment"])
    environment["require_full_reward_preroll"] = True
    raw["environment"] = environment

    config = TrainingRunConfig.from_mapping(raw)

    assert config.schema_version == "training_run_config_v4"
    assert config.environment.require_full_reward_preroll is True
    assert config.environment.execution_cost == ExecutionCostConfig()
''',
    encoding="utf-8",
)

helper = ROOT / "tests/training_config_support.py"
helper.write_text(
    '''from __future__ import annotations

from typing import Any


_EXECUTION_CONFIG_V4: dict[str, object] = {
    "fee_rate": 0.0005,
    "maker_fee_rate": 0.0,
    "taker_fee_rate": 0.0,
    "spread_rate": 0.0002,
    "impact_rate": 0.0001,
    "multiplier": 1.0,
    "max_participation_rate": 0.05,
    "slippage_std": 0.0,
    "tail_slippage_probability": 0.0,
    "tail_slippage_multiplier": 5.0,
    "random_seed": 0,
    "minimum_notional": 0.0,
    "lot_size": 0.0,
    "tick_size": 0.0,
    "allow_short": True,
    "borrow_rate_multiplier": 1.0,
    "max_leverage": 1.0,
    "maintenance_margin_rate": 0.25,
    "collateral_haircut": 1.0,
    "margin_mode": "cross",
    "order_latency_bars": 0,
    "order_type": "market",
    "limit_offset_rate": 0.0005,
    "path_mode": "conservative",
    "processing_bar_volume_capacity": True,
    "partial_fill_carry": True,
    "trigger_volume_fractions": [1.0, 0.5, 0.25, 0.0],
}


def complete_execution_config(**overrides: Any) -> dict[str, object]:
    config = dict(_EXECUTION_CONFIG_V4)
    config.update(overrides)
    return config


__all__ = ["complete_execution_config"]
''',
    encoding="utf-8",
)

p = ROOT / "trade_rl/workflows/training_run.py"
text = p.read_text(encoding="utf-8")
text = text.replace(
    "from dataclasses import asdict, dataclass, field, replace",
    "from dataclasses import asdict, dataclass, field, fields as dataclass_fields, replace",
)
text = text.replace(
    'TRAINING_RUN_CONFIG_SCHEMA = "training_run_config_v3"',
    'TRAINING_RUN_CONFIG_SCHEMA = "training_run_config_v4"',
)
text = text.replace("_REQUIRED_V3_TRAINING_FIELDS", "_REQUIRED_V4_TRAINING_FIELDS")
text = text.replace("_require_v3_training_fields", "_require_v4_training_fields")
text = text.replace("training_run_config_v3 training", "training_run_config_v4 training")
text = text.replace(
    ")\n\n\ndef _require_v4_training_fields",
    ")\n_REQUIRED_V4_EXECUTION_FIELDS = frozenset(\n"
    "    item.name for item in dataclass_fields(ExecutionCostConfig) if item.init\n"
    ")\n_LEGACY_TRAINING_RUN_CONFIG_SCHEMAS = frozenset(\n"
    '    {"training_run_config_v1", "training_run_config_v2", "training_run_config_v3"}\n'
    ")\n\n\ndef _require_v4_training_fields",
    1,
)
text = text.replace(
    'if self.schema_version in {"training_run_config_v1", "training_run_config_v2"}:',
    "if self.schema_version in _LEGACY_TRAINING_RUN_CONFIG_SCHEMAS:",
)
text = text.replace(
    'if schema_version in {"training_run_config_v1", "training_run_config_v2"}:',
    "if schema_version in _LEGACY_TRAINING_RUN_CONFIG_SCHEMAS:",
)
text = text.replace(
    '                "action",\n            },\n            optional={\n                "execution",',
    '                "action",\n                "execution",\n            },\n            optional={',
)
text = text.replace(
    '''        execution_data = require_dataclass_fields(
            _mapping(payload.get("execution"), field="execution"),
            ExecutionCostConfig,
            field="execution",
        )
        execution = ExecutionCostConfig(**execution_data)''',
    '''        execution_data = _tuple_fields(
            require_exact_fields(
                _mapping(payload["execution"], field="execution"),
                required=_REQUIRED_V4_EXECUTION_FIELDS,
                optional=set(),
                field="execution",
            ),
            "trigger_volume_fractions",
        )
        execution = ExecutionCostConfig(**execution_data)''',
)
text = text.replace(
    '''        if "require_full_reward_preroll" not in environment_mapping:
            environment_data["require_full_reward_preroll"] = True''',
    '''        if "require_full_reward_preroll" not in environment_mapping:
            raise ValueError(
                "environment has missing required fields: "
                "require_full_reward_preroll"
            )''',
)
p.write_text(text, encoding="utf-8")

p = ROOT / "trade_rl/integrations/sb3_training.py"
p.write_text(
    p.read_text(encoding="utf-8").replace(
        "training_run_config_v3", "training_run_config_v4"
    ),
    encoding="utf-8",
)


def migrate_json(value: object) -> int:
    count = 0
    if isinstance(value, dict):
        if value.get("schema_version") == "training_run_config_v3":
            value["schema_version"] = "training_run_config_v4"
            environment = value.get("environment")
            if not isinstance(environment, dict):
                raise RuntimeError("v3 config lacks environment object")
            environment["require_full_reward_preroll"] = True
            execution = value.get("execution")
            if not isinstance(execution, dict):
                execution = {}
            value["execution"] = {
                name: execution.get(name, default)
                for name, default in EXECUTION_DEFAULTS.items()
            }
            count += 1
        for child in value.values():
            count += migrate_json(child)
    elif isinstance(value, list):
        for child in value:
            count += migrate_json(child)
    return count


json_count = 0
for path in sorted((ROOT / "examples").rglob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    migrated = migrate_json(payload)
    if migrated:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        json_count += migrated

config_test_files = [
    ROOT / "tests/e2e/test_research_to_serving_v2.py",
    ROOT / "tests/workflows/test_explicit_sealed_ledger_mode.py",
    ROOT / "tests/workflows/test_market_walk_forward.py",
    ROOT / "tests/workflows/test_pure_growth_training_contract.py",
    ROOT / "tests/workflows/test_signal_digest.py",
    ROOT / "tests/workflows/test_symbol_triplet_stage_orchestrator.py",
    ROOT / "tests/workflows/test_symbol_triplet_stage_training.py",
    ROOT / "tests/workflows/test_training_preroll_contract.py",
    ROOT / "tests/workflows/test_training_run.py",
    ROOT / "tests/workflows/test_training_run_config.py",
    ROOT / "tests/workflows/test_training_run_transfer_config.py",
    ROOT / "tests/workflows/test_walk_forward_manifest_provenance.py",
]


def literal_key(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


for path in config_test_files:
    source = path.read_text(encoding="utf-8").replace(
        "training_run_config_v3", "training_run_config_v4"
    )
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    insertions: list[tuple[int, str]] = []
    inserted_execution = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        pairs = [
            (literal_key(key_node), value_node)
            for key_node, value_node in zip(node.keys, node.values, strict=True)
        ]
        if not any(
            key == "schema_version"
            and isinstance(value, ast.Constant)
            and value.value == "training_run_config_v4"
            for key, value in pairs
        ):
            continue
        keys = {key for key, _ in pairs}
        if "execution" not in keys:
            risk_key_node = next(
                key_node for key_node in node.keys if literal_key(key_node) == "risk"
            )
            indent = " " * risk_key_node.col_offset
            insertions.append(
                (
                    risk_key_node.lineno - 1,
                    f'{indent}"execution": complete_execution_config(),\n',
                )
            )
            inserted_execution = True
        environment_value = next(
            value for key, value in pairs if key == "environment"
        )
        if not isinstance(environment_value, ast.Dict):
            raise RuntimeError(f"{path}: environment is not a dict literal")
        environment_keys = {literal_key(key_node) for key_node in environment_value.keys}
        if "require_full_reward_preroll" not in environment_keys:
            if environment_value.end_lineno is None:
                raise RuntimeError("missing end_lineno")
            closing_line = environment_value.end_lineno - 1
            base_indent = len(lines[closing_line]) - len(lines[closing_line].lstrip(" "))
            insertions.append(
                (
                    closing_line,
                    " " * (base_indent + 4)
                    + '"require_full_reward_preroll": True,\n',
                )
            )
    for index, addition in sorted(insertions, reverse=True):
        lines.insert(index, addition)
    source = "".join(lines)
    if (
        inserted_execution
        and "from tests.training_config_support import complete_execution_config"
        not in source
    ):
        parsed = ast.parse(source)
        imports = [
            node
            for node in parsed.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        import_line = max((node.end_lineno or node.lineno) for node in imports)
        parts = source.splitlines(keepends=True)
        parts.insert(
            import_line,
            "from tests.training_config_support import complete_execution_config\n",
        )
        source = "".join(parts)
    path.write_text(source, encoding="utf-8")

replacements = {
    ROOT / "tests/integrations/test_sb3_training.py": [
        ("training_run_config_v3", "training_run_config_v4")
    ],
    ROOT / "tests/workflows/test_training_run.py": [
        ("test_v3_requires_explicit_actor_head", "test_v4_requires_explicit_actor_head"),
        ('match="training_run_config_v3"', 'match="training_run_config_v4"'),
    ],
    ROOT / "tests/workflows/test_training_run_config.py": [
        (
            "test_training_config_requires_explicit_v2_schema",
            "test_training_config_requires_explicit_schema_version",
        )
    ],
    ROOT / "tests/workflows/test_training_preroll_contract.py": [
        (
            '"""Explicit opt-out is rejected while omission resolves to the maintained mode."""',
            '"""Training refuses an explicitly disabled full-preroll contract."""',
        )
    ],
}
for path, pairs in replacements.items():
    text = path.read_text(encoding="utf-8")
    for old, new in pairs:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")

readme = ROOT / "README.md"
text = readme.read_text(encoding="utf-8").replace(
    "training_run_config_v3", "training_run_config_v4"
)
text = text.replace(
    "QuickstartはPipeline確認用ですが、Reward dataclassの既定値変更で意味が静かに変わらないよう、hybrid reward値をJSONへ明示しています。",
    "Quickstartを含む維持対象設定は、Executionの全Fieldと`require_full_reward_preroll: true`を明示し、dataclass既定値の変更で学習意味が静かに変わることを防ぎます。",
)
readme.write_text(text, encoding="utf-8")

architecture = ROOT / "docs/ARCHITECTURE.md"
text = architecture.read_text(encoding="utf-8").replace(
    "training_run_config_v3", "training_run_config_v4"
)
text = text.replace(
    "`training_run_config_v4`は`observation_encoder`と階層Actor契約を明示します。",
    "`training_run_config_v4`は`observation_encoder`と階層Actor契約に加え、Executionの全Fieldと完全なReward prerollを明示します。",
)
architecture.write_text(text, encoding="utf-8")

configuration = ROOT / "docs/CONFIGURATION.md"
text = configuration.read_text(encoding="utf-8")
text = text.replace("# Training Configuration v3", "# Training Configuration v4")
text = text.replace("training_run_config_v3", "training_run_config_v4")
text = text.replace(
    '  "environment": {},',
    '  "environment": {"require_full_reward_preroll": true},',
)
text = text.replace(
    '  "execution": {},',
    '  "execution": {"all_ExecutionCostConfig_fields": "required"},',
)
text = text.replace(
    "`training_run_config_v1`は自動変換しません。明示的に拒否します。",
    "`training_run_config_v1`、`training_run_config_v2`、`training_run_config_v3`は自動変換せず拒否します。v4ではTop-level `execution`、Executionの全Field、`environment.require_full_reward_preroll: true`が必須です。",
)
configuration.write_text(text, encoding="utf-8")

current_documentation = ROOT / "tests/test_current_documentation_contract.py"
current_documentation.write_text(
    current_documentation.read_text(encoding="utf-8").replace(
        "training_run_config_v3", "training_run_config_v4"
    ),
    encoding="utf-8",
)

old_documentation_test = ROOT / "tests/test_maintained_documentation_v3_contract.py"
new_documentation_test = ROOT / "tests/test_maintained_documentation_v4_contract.py"
text = old_documentation_test.read_text(encoding="utf-8")
text = text.replace("v3", "v4")
text = text.replace("training_run_config_v3", "training_run_config_v4")
text = text.replace("# Training Configuration v3", "# Training Configuration v4")
text = text.replace(
    'assert "training_run_config_v2" not in readme',
    'assert "training_run_config_v3" not in readme',
)
new_documentation_test.write_text(text, encoding="utf-8")
old_documentation_test.unlink()

(ROOT / ".github/workflows/dev-source-snapshot.yml").unlink(missing_ok=True)
Path(__file__).unlink()
print(f"migrated {json_count} maintained JSON training configs")
