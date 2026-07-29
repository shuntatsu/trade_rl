from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any


REQUIRED_V3_DEFAULTS: dict[str, object] = {
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
}


def _replace_once(text: str, old: str, new: str, *, field: str) -> str:
    if old not in text:
        raise RuntimeError(f"{field} fragment was not found")
    return text.replace(old, new, 1)


def _migrate_mapping(value: Any) -> bool:
    changed = False
    if isinstance(value, dict):
        if value.get("schema_version") == "training_run_config_v2":
            training = value.get("training")
            if not isinstance(training, dict):
                raise RuntimeError("training_run_config_v2 has no training mapping")
            value["schema_version"] = "training_run_config_v3"
            encoder = str(training.get("observation_encoder", "asset_set")).strip().lower()
            training.setdefault(
                "policy_actor_head",
                "hierarchical_gate_target_v1"
                if encoder == "hierarchical_sequence_v2"
                else "standard_continuous_v1",
            )
            for name, default in REQUIRED_V3_DEFAULTS.items():
                training.setdefault(name, default)
            changed = True
        for nested in value.values():
            changed = _migrate_mapping(nested) or changed
    elif isinstance(value, list):
        for nested in value:
            changed = _migrate_mapping(nested) or changed
    return changed


def migrate_json(root: Path) -> None:
    for base in (root / "examples", root / "tests"):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if _migrate_mapping(payload):
                path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )


def fix_sequence_encoder(root: Path) -> None:
    path = root / "trade_rl/rl/sequence_policy.py"
    text = path.read_text(encoding="utf-8")
    class_start = text.index("class CausalTimeframeEncoder")
    sequence_start = text.index("    def forward_sequence(", class_start)
    forward_start = text.index("    def forward(\n", sequence_start)
    forward_end = text.index("\n\n@dataclass", forward_start)
    new_forward = '''    def forward(
        self, value: torch.Tensor, available: torch.Tensor | None = None
    ) -> torch.Tensor:
        if available is None:
            encoded = self.forward_sequence(value)
            return self.projection(encoded[:, -1])
        if available.shape != value.shape[:2]:
            raise ValueError("availability mask must match batch and time dimensions")
        mask = available.to(dtype=torch.bool)
        positions = torch.arange(value.shape[1], device=value.device).expand_as(mask)
        indices = positions.masked_fill(~mask, -1).max(dim=1).values
        valid = indices >= 0
        if not torch.any(valid):
            return (value.sum(dim=(1, 2)).unsqueeze(1) * 0.0).expand(
                -1, self.latent_dim
            )
        valid_values = value[valid]
        encoded = self.forward_sequence(valid_values)
        valid_indices = indices[valid]
        selected = encoded[
            torch.arange(encoded.shape[0], device=value.device), valid_indices
        ]
        projected = self.projection(selected)
        output = projected.new_zeros((value.shape[0], self.latent_dim))
        batch_indices = torch.arange(value.shape[0], device=value.device)[valid]
        return output.index_copy(0, batch_indices, projected)
'''
    path.write_text(text[:forward_start] + new_forward + text[forward_end:], encoding="utf-8")


def fix_sequence_tests(root: Path) -> None:
    path = root / "tests/rl/test_sequence_policy_core.py"
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        "    assert projection_input_shapes == [torch.Size((3, 8))]\n",
        "    assert projection_input_shapes == [torch.Size((2, 8))]\n",
        field="projection input shape",
    )
    text = _replace_once(
        text,
        '''    torch.testing.assert_close(
        case.optimized_input_gradient,
        case.legacy_input_gradient,
        rtol=1e-9,
        atol=1e-10,
    )
''',
        '''    torch.testing.assert_close(
        case.optimized_input_gradient[:2],
        case.legacy_input_gradient[:2],
        rtol=1e-9,
        atol=1e-10,
    )
''',
        field="float64 valid input gradient",
    )
    text = _replace_once(
        text,
        '''    _assert_gradient_semantics(
        case.optimized_input_gradient,
        case.legacy_input_gradient,
    )
''',
        '''    _assert_gradient_semantics(
        case.optimized_input_gradient[:2],
        case.legacy_input_gradient[:2],
    )
''',
        field="float32 valid input gradient",
    )
    path.write_text(text, encoding="utf-8")


def fix_sequence_diagnostics(root: Path) -> None:
    path = root / "tests/rl/test_sequence_diagnostics.py"
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        '        "active": spaces.Box(0.0, 1.0, shape=(3,)),\n    }\n',
        '        "active": spaces.Box(0.0, 1.0, shape=(3,)),\n'
        '        "current_weights": spaces.Box(-1.0, 1.0, shape=(3,)),\n'
        '    }\n',
        field="diagnostic observation space",
    )
    text = _replace_once(
        text,
        '''        "active": torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    }
''',
        '''        "active": torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        "current_weights": torch.zeros(2, 3, dtype=torch.float32),
    }
''',
        field="diagnostic observation",
    )
    path.write_text(text, encoding="utf-8")


def fix_parallel_sequence_fixtures(root: Path) -> None:
    path = root / "tests/integrations/test_parallel_sequence_environments.py"
    text = path.read_text(encoding="utf-8")
    replacements = (
        ("        per_symbol_width=6,\n", "        per_symbol_width=11,\n", "layout width"),
        (
            "    current = np.arange(layout.size, dtype=np.float32)\n",
            '''    current = np.linspace(-0.5, 0.5, layout.size, dtype=np.float32)
    current[: layout.n_symbols * layout.per_symbol_width].reshape(
        layout.n_symbols, layout.per_symbol_width
    )[:, layout.current_weight_column] = np.asarray((0.25, -0.5))
''',
            "current weight fixture",
        ),
        (
            '    for key in ("current_snapshot", "asset_state", "global_state", "active"):\n',
            '''    for key in (
        "current_snapshot",
        "asset_state",
        "global_state",
        "active",
        "current_weights",
    ):
''',
            "structured key comparison",
        ),
        (
            '''            "current_snapshot": spaces.Box(
                -np.inf, np.inf, shape=(1, 1), dtype=np.float32
            ),
''',
            '''            "current_snapshot": spaces.Box(
                -np.inf, np.inf, shape=(1, 1), dtype=np.float32
            ),
            "current_weights": spaces.Box(
                -1.0, 1.0, shape=(1,), dtype=np.float32
            ),
''',
            "full structured space",
        ),
        (
            '''    assert tuple(environment.observation_space.spaces) == (
        "current_snapshot",
        "decision_index",
    )
''',
            '''    assert tuple(environment.observation_space.spaces) == (
        "current_snapshot",
        "current_weights",
        "decision_index",
    )
''',
            "compact structured space",
        ),
        (
            '''        "current_snapshot": np.asarray(indices, dtype=np.float32).reshape(-1, 1, 1),
    }
''',
            '''        "current_snapshot": np.asarray(indices, dtype=np.float32).reshape(-1, 1, 1),
        "current_weights": np.zeros((len(indices), 1), dtype=np.float32),
    }
''',
            "compact batch",
        ),
    )
    for old, new, field in replacements:
        text = _replace_once(text, old, new, field=field)
    path.write_text(text, encoding="utf-8")


def cleanup(root: Path) -> None:
    for relative in (
        ".github/workflows/temporary-task9-format.yml",
        ".github/workflows/tmp-task9-sequence-gradient.yml",
    ):
        (root / relative).unlink(missing_ok=True)
    shutil.rmtree(root / "docs/superpowers", ignore_errors=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: task9_remediate.py ROOT")
    root = Path(sys.argv[1]).resolve()
    cleanup(root)
    migrate_json(root)
    fix_sequence_encoder(root)
    fix_sequence_tests(root)
    fix_sequence_diagnostics(root)
    fix_parallel_sequence_fixtures(root)


if __name__ == "__main__":
    main()
