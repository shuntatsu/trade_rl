"""Align focused tests and the temporary export gate with the v2 contract."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _replace_required(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count == 0:
        if new in text:
            return text
        raise RuntimeError(f"missing v2 cleanup anchor: {label}")
    return text.replace(old, new)


def _migrate_identity_test() -> None:
    path = ROOT / "tests" / "rl" / "test_sequence_architecture_identity.py"
    text = path.read_text(encoding="utf-8")
    text = _replace_required(
        text,
        "replace(base, attention_layers=3)",
        "replace(base, timeframe_attention_layers=3)",
        label="identity layer replacement",
    )
    path.write_text(text, encoding="utf-8")


def _migrate_feature_extractor_kwargs() -> None:
    path = ROOT / "tests" / "rl" / "test_sequence_policy_core.py"
    text = path.read_text(encoding="utf-8")
    dict_heads = re.compile(
        r'(?P<indent>[ \t]+)"attention_heads": (?P<value>[^,\n]+),'
    )
    dict_layers = re.compile(
        r'(?P<indent>[ \t]+)"attention_layers": (?P<value>[^,\n]+),'
    )
    call_heads = re.compile(
        r"(?P<indent>[ \t]+)attention_heads=(?P<value>[^,\n]+),"
    )
    call_layers = re.compile(
        r"(?P<indent>[ \t]+)attention_layers=(?P<value>[^,\n]+),"
    )
    text = dict_heads.sub(
        lambda match: (
            f'{match.group("indent")}"timeframe_attention_heads": '
            f'{match.group("value")},\n'
            f'{match.group("indent")}"asset_attention_heads": '
            f'{match.group("value")},'
        ),
        text,
    )
    text = dict_layers.sub(
        lambda match: (
            f'{match.group("indent")}"timeframe_attention_layers": '
            f'{match.group("value")},\n'
            f'{match.group("indent")}"asset_attention_layers": '
            f'{match.group("value")},'
        ),
        text,
    )
    text = call_heads.sub(
        lambda match: (
            f'{match.group("indent")}timeframe_attention_heads='
            f'{match.group("value")},\n'
            f'{match.group("indent")}asset_attention_heads='
            f'{match.group("value")},'
        ),
        text,
    )
    text = call_layers.sub(
        lambda match: (
            f'{match.group("indent")}timeframe_attention_layers='
            f'{match.group("value")},\n'
            f'{match.group("indent")}asset_attention_layers='
            f'{match.group("value")},'
        ),
        text,
    )
    path.write_text(text, encoding="utf-8")


def _migrate_active_field_expectations() -> None:
    path = ROOT / "tests" / "rl" / "test_training_config_active_fields.py"
    text = path.read_text(encoding="utf-8")
    replacements = {
        "test_disabled_sequence_encoder_rejects_non_default_sequence_parameters": (
            "test_non_sequence_observation_encoder_rejects_sequence_parameters"
        ),
        'match="sequence_d_model.*sequence_encoder"': (
            'match="sequence_d_model.*observation_encoder"'
        ),
        'match="sequence_capacity.*sequence_encoder"': (
            'match="sequence_tcn_capacity.*observation_encoder"'
        ),
        'match="sequence_capacity"': 'match="sequence_tcn_capacity"',
        "test_disabled_asset_set_encoder_rejects_non_default_embedding_parameters": (
            "test_non_asset_set_observation_encoder_rejects_embedding_parameters"
        ),
        'match="asset_embedding_dim.*asset_set_encoder"': (
            'match="asset_embedding_dim.*observation_encoder"'
        ),
    }
    for old, new in replacements.items():
        text = _replace_required(text, old, new, label=old)
    path.write_text(text, encoding="utf-8")


def _restore_export_gate_until_structured_export_exists() -> None:
    path = ROOT / "trade_rl" / "workflows" / "training_run.py"
    text = path.read_text(encoding="utf-8")
    block = '''        if self.training.observation_encoder == "hierarchical_sequence_v2" and (
            self.export_onnx or self.export_torchscript
        ):
            raise ValueError(
                "structured sequence policies do not support flat ONNX/TorchScript export"
            )
'''
    if block not in text:
        anchor = "        if self.git_commit is not None and not self.git_commit:\n"
        if text.count(anchor) != 1:
            raise RuntimeError("training run export gate anchor is not unique")
        text = text.replace(anchor, block + anchor, 1)
    path.write_text(text, encoding="utf-8")


def _normalize_mypy_loop_variables() -> None:
    replacements = {
        ROOT / "trade_rl" / "rl" / "training.py": (
            "        for field_name, value in (\n"
            "            (\n"
            "                \"sequence_timeframe_gate_bias\",\n"
            "                self.sequence_timeframe_gate_bias,\n"
            "            ),\n"
            "            (\"sequence_asset_gate_bias\", self.sequence_asset_gate_bias),\n"
            "        ):\n"
            "            if not math.isfinite(value):\n"
            "                raise ValueError(f\"{field_name} must be finite\")\n",
            "        for field_name, gate_value in (\n"
            "            (\n"
            "                \"sequence_timeframe_gate_bias\",\n"
            "                self.sequence_timeframe_gate_bias,\n"
            "            ),\n"
            "            (\"sequence_asset_gate_bias\", self.sequence_asset_gate_bias),\n"
            "        ):\n"
            "            if not math.isfinite(gate_value):\n"
            "                raise ValueError(f\"{field_name} must be finite\")\n",
        ),
        ROOT / "trade_rl" / "rl" / "sequence_policy.py": (
            "        for field_name, value in (\n"
            "            (\"timeframe_gate_bias\", self.timeframe_gate_bias),\n"
            "            (\"asset_gate_bias\", self.asset_gate_bias),\n"
            "        ):\n"
            "            if not math.isfinite(value):\n"
            "                raise ValueError(f\"{field_name} must be finite\")\n",
            "        for field_name, gate_value in (\n"
            "            (\"timeframe_gate_bias\", self.timeframe_gate_bias),\n"
            "            (\"asset_gate_bias\", self.asset_gate_bias),\n"
            "        ):\n"
            "            if not math.isfinite(gate_value):\n"
            "                raise ValueError(f\"{field_name} must be finite\")\n",
        ),
    }
    for path, (old, new) in replacements.items():
        text = path.read_text(encoding="utf-8")
        text = _replace_required(text, old, new, label=f"MyPy loop in {path.name}")
        path.write_text(text, encoding="utf-8")


def _simplify_cli_observation_encoder() -> None:
    path = ROOT / "trade_rl" / "cli" / "app.py"
    text = path.read_text(encoding="utf-8")
    expression = re.compile(
        r"observation_encoder=\(\"invalid_legacy_combination\" "
        r"if \(False\) and \(not args\.no_asset_set_encoder\) else "
        r"\"hierarchical_sequence_v2\" if \(False\) else "
        r"\"asset_set\" if \(not args\.no_asset_set_encoder\) else \"flat_mlp\"\),"
    )
    text, count = expression.subn(
        'observation_encoder=(\n'
        '            "asset_set" if not args.no_asset_set_encoder else "flat_mlp"\n'
        '        ),',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("CLI observation encoder expression was not normalized")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    _migrate_identity_test()
    _migrate_feature_extractor_kwargs()
    _migrate_active_field_expectations()
    _restore_export_gate_until_structured_export_exists()
    _normalize_mypy_loop_variables()
    _simplify_cli_observation_encoder()


if __name__ == "__main__":
    main()
