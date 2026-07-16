from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 4 follow-up anchor in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        "tests/rl/test_sequence_normalization.py",
        '''    available = dataset.feature_available.copy()
    available[:, :, 3] = False
    missing = replace(dataset, feature_available=available, dataset_id="e" * 64)
''',
        '''    available = dataset.feature_available.copy()
    available[:, :, 3] = False
    staleness = dataset.feature_staleness.copy()
    staleness[:, :, 3] = 1.0
    missing = replace(
        dataset,
        feature_available=available,
        feature_staleness=staleness,
        dataset_id="e" * 64,
    )
''',
    )
    replace_once(
        "tests/rl/test_observation_v2.py",
        "    assert rows[0, offset + 14] == np.log1p(1.0)\n",
        '''    np.testing.assert_allclose(
        rows[0, offset + 14], np.log1p(1.0), rtol=0.0, atol=1e-7
    )
''',
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        '''    def digest_payload(self) -> dict[str, object]:
        return {
''',
        '''    def digest_payload(self) -> dict[str, object]:
        sample_count = self.sample_count
        if sample_count is None:
            raise RuntimeError("sequence sample counts were not initialized")
        return {
''',
    )
    replace_once(
        "trade_rl/rl/sequence_normalization.py",
        '''            "sample_count": {
                key: tuple(int(value) for value in self.sample_count[key])
                for key in self.feature_names
            },
''',
        '''            "sample_count": {
                key: tuple(int(value) for value in sample_count[key])
                for key in self.feature_names
            },
''',
    )


if __name__ == "__main__":
    main()
