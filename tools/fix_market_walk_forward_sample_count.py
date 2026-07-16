from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "trade_rl/workflows/market_walk_forward.py"


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    old = '''def _sequence_normalizer_payload(
    normalizer: SequenceFeatureNormalizer,
) -> dict[str, object]:
    return {
'''
    new = '''def _sequence_normalizer_payload(
    normalizer: SequenceFeatureNormalizer,
) -> dict[str, object]:
    sample_count = normalizer.sample_count
    if sample_count is None:
        raise RuntimeError("sequence normalizer sample counts are unavailable")
    return {
'''
    if old not in text:
        raise RuntimeError("market walk-forward sample-count anchor is missing")
    text = text.replace(old, new, 1)
    text = text.replace(
        "normalizer.sample_count[key]",
        "sample_count[key]",
        1,
    )
    TARGET.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
