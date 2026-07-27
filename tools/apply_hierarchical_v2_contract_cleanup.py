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
    heads = re.compile(
        r'(?P<indent>[ \t]+)"attention_heads": (?P<value>[^,\n]+),'
    )
    layers = re.compile(
        r'(?P<indent>[ \t]+)"attention_layers": (?P<value>[^,\n]+),'
    )
    text = heads.sub(
        lambda match: (
            f'{match.group("indent")}"timeframe_attention_heads": '
            f'{match.group("value")},\n'
            f'{match.group("indent")}"asset_attention_heads": '
            f'{match.group("value")},'
        ),
        text,
    )
    text = layers.sub(
        lambda match: (
            f'{match.group("indent")}"timeframe_attention_layers": '
            f'{match.group("value")},\n'
            f'{match.group("indent")}"asset_attention_layers": '
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


def main() -> None:
    _migrate_identity_test()
    _migrate_feature_extractor_kwargs()
    _migrate_active_field_expectations()
    _restore_export_gate_until_structured_export_exists()


if __name__ == "__main__":
    main()
