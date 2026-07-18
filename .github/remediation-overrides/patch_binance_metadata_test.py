from __future__ import annotations

from pathlib import Path

path = Path("tests/workflows/test_binance_metadata_modes.py")
source = path.read_text(encoding="utf-8")
source = source.replace(
    "from trade_rl.data.contracts import InstrumentExecutionRule\n",
    "",
    1,
)
source = source.replace(
    '''from trade_rl.workflows.binance_metadata_modes import (\n    BinanceHistoricalSignedScope,\n    BinanceMetadataMode,\n    BinanceMetadataResolutionProvider,\n    resolution_from_historical_signed,\n    resolve_conservative_static,\n    resolve_frozen_snapshot,\n)\n''',
    '''from tests.binance_signed_helpers import (\n    ISSUED,\n    TRUSTED_KEYS,\n    TRUSTED_NOW,\n    signed_rule_history_document,\n)\nfrom trade_rl.workflows.binance_metadata_modes import (\n    BinanceMetadataMode,\n    BinanceMetadataResolutionProvider,\n    load_verified_binance_rule_history,\n    resolution_from_historical_signed,\n    resolve_conservative_static,\n    resolve_frozen_snapshot,\n)\n''',
    1,
)
start_marker = (
    "def test_historical_signed_resolution_preserves_effective_history() -> None:\n"
)
end_marker = (
    "\n\ndef test_conservative_static_requires_versioned_payload_and_positive_stress"
)
start = source.find(start_marker)
end = source.find(end_marker, start)
if start < 0 or end < 0:
    raise SystemExit("historical signed metadata test markers are missing")
replacement = '''def test_historical_signed_resolution_preserves_verified_effective_history(\n    tmp_path: Path,\n) -> None:\n    document = signed_rule_history_document()\n    verified = load_verified_binance_rule_history(\n        document,\n        trusted_keys=TRUSTED_KEYS,\n        trusted_now=TRUSTED_NOW,\n    )\n\n    resolution = resolution_from_historical_signed(\n        verified,\n        start_time=START,\n        end_time=END,\n    )\n    resolution.write_artifacts(tmp_path)\n\n    assert resolution.mode is BinanceMetadataMode.HISTORICAL_SIGNED\n    assert resolution.execution_rule_histories == verified.execution_rule_histories\n    assert resolution.identity_evidence["authentication"] == "ed25519"\n    assert resolution.identity_evidence["point_in_time"] is True\n    assert resolution.identity_evidence["limitations"] == ()\n    assert resolution.identity_evidence["source_uri"] == verified.source_uri\n    assert resolution.identity_evidence["as_of"] == ISSUED.isoformat()\n    assert (tmp_path / "exchange-info.raw.json").read_bytes() == verified.signed_document\n'''
path.write_text(source[:start] + replacement + source[end:], encoding="utf-8")
