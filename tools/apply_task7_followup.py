from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 7 follow-up anchor in {path}: {old[:140]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once("tests/serving/test_sb3_loader.py", "    n = 16\n", "    n = 128\n")
    replace_once(
        "tests/serving/test_sb3_loader.py",
        "        index=8,\n        current_flat=np.zeros(layout.size, dtype=np.float32),\n",
        "        index=100,\n        current_flat=np.zeros(layout.size, dtype=np.float32),\n",
    )
    replace_once(
        "tests/serving/test_sb3_loader.py",
        "        index=8,\n        current_flat=np.zeros(layout.size, dtype=np.float32),\n",
        "        index=100,\n        current_flat=np.zeros(layout.size, dtype=np.float32),\n",
    )
    replace_once(
        "tests/serving/test_sb3_loader.py",
        "            index=8,\n            current_flat=np.zeros(layout.size, dtype=np.float32),\n",
        "            index=100,\n            current_flat=np.zeros(layout.size, dtype=np.float32),\n",
    )
    replace_once(
        "tests/serving/test_sb3_loader.py",
        "            index=8,\n            current_flat=np.zeros(layout.size, dtype=np.float32),\n",
        "            index=100,\n            current_flat=np.zeros(layout.size, dtype=np.float32),\n",
    )

    replace_once(
        "trade_rl/serving/runtime.py",
        '''from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from collections.abc import Mapping
from typing import Any, Protocol
''',
        '''from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Protocol
''',
    )
    replace_once(
        "trade_rl/serving/runtime.py",
        '''    def predict(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.zeros(self.action_size, dtype=np.float32)
''',
        '''    def predict(self, observation: PolicyObservation) -> np.ndarray:
        del observation
        return np.zeros(self.action_size, dtype=np.float32)
''',
    )

    replace_once(
        "trade_rl/serving/sequence_normalizer.py",
        '''def write_sequence_feature_normalizer(
    root: Path,
    normalizer: SequenceFeatureNormalizer,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
''',
        '''def write_sequence_feature_normalizer(
    root: Path,
    normalizer: SequenceFeatureNormalizer,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    sample_count = normalizer.sample_count
    if sample_count is None:
        raise RuntimeError("sequence normalizer sample counts are unavailable")
''',
    )
    replace_once(
        "trade_rl/serving/sequence_normalizer.py",
        '''            key: tuple(int(value) for value in normalizer.sample_count[key])
''',
        '''            key: tuple(int(value) for value in sample_count[key])
''',
    )
    replace_once(
        "trade_rl/serving/sequence_normalizer.py",
        '''def load_sequence_feature_normalizer(root: Path) -> SequenceFeatureNormalizer:
''',
        '''def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def load_sequence_feature_normalizer(root: Path) -> SequenceFeatureNormalizer:
''',
    )
    replace_once(
        "trade_rl/serving/sequence_normalizer.py",
        '''            train_start=int(raw_range[0]),
            train_end=int(raw_range[1]),
''',
        '''            train_start=_integer(raw_range[0], field="train_range[0]"),
            train_end=_integer(raw_range[1], field="train_range[1]"),
''',
    )
    replace_once(
        "trade_rl/serving/sequence_normalizer.py",
        '''            minimum_samples_per_channel=int(raw["minimum_samples_per_channel"]),
            clip=float(raw["clip"]),
            epsilon=float(raw.get("epsilon", 1e-8)),
''',
        '''            minimum_samples_per_channel=_integer(
                raw["minimum_samples_per_channel"],
                field="minimum_samples_per_channel",
            ),
            clip=_number(raw["clip"], field="clip"),
            epsilon=_number(raw.get("epsilon", 1e-8), field="epsilon"),
''',
    )

    replace_once(
        "trade_rl/integrations/sb3_serving.py",
        '''    def _validate_dataset(self, dataset: MarketDataset) -> None:
        expected_symbols = tuple(self.dataset_reference.get("symbols", ()))
        expected_features = tuple(self.dataset_reference.get("feature_names", ()))
        expected_globals = tuple(self.dataset_reference.get("global_feature_names", ()))
''',
        '''    @staticmethod
    def _reference_names(value: object, *, field: str) -> tuple[str, ...]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"dataset reference {field} must be a list of strings")
        return tuple(value)

    def _validate_dataset(self, dataset: MarketDataset) -> None:
        expected_symbols = self._reference_names(
            self.dataset_reference.get("symbols"), field="symbols"
        )
        expected_features = self._reference_names(
            self.dataset_reference.get("feature_names"), field="feature_names"
        )
        expected_globals = self._reference_names(
            self.dataset_reference.get("global_feature_names"),
            field="global_feature_names",
        )
''',
    )
    replace_once(
        "trade_rl/integrations/sb3_serving.py",
        '''    def smoke_observation(self) -> dict[str, np.ndarray]:
        return {
            key: np.zeros(space.shape, dtype=space.dtype)
            for key, space in self.observation_space.spaces.items()
        }
''',
        '''    def smoke_observation(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for key, space in self.observation_space.spaces.items():
            if space.shape is None or space.dtype is None:
                raise ValueError("structured observation space must declare shape and dtype")
            result[key] = np.zeros(space.shape, dtype=space.dtype)
        return result
''',
    )

    replace_once(
        "trade_rl/workflows/training_run.py",
        '''def _sequence_normalizer_payload(
    normalizer: SequenceFeatureNormalizer,
) -> dict[str, object]:
    return {
''',
        '''def _sequence_normalizer_payload(
    normalizer: SequenceFeatureNormalizer,
) -> dict[str, object]:
    sample_count = normalizer.sample_count
    if sample_count is None:
        raise RuntimeError("sequence normalizer sample counts are unavailable")
    return {
''',
    )
    replace_once(
        "trade_rl/workflows/training_run.py",
        '''            key: tuple(int(value) for value in normalizer.sample_count[key])
''',
        '''            key: tuple(int(value) for value in sample_count[key])
''',
    )


if __name__ == "__main__":
    main()
