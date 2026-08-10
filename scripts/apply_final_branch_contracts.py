#!/usr/bin/env python3
"""Apply the final episode-aware behavior-cloning split contract."""

from __future__ import annotations

from pathlib import Path


PATH = Path("trade_rl/integrations/behavior_cloning.py")

OLD = """    expected_validation_count = (
        0
        if config.validation_fraction == 0.0
        else max(
            1,
            int(math.floor(sample_count * config.validation_fraction)),
        )
    )
    if validation_indices.size != expected_validation_count:
        raise ValueError(
            "explicit behavior-cloning split disagrees with validation_fraction"
        )
    return train_indices, validation_indices
"""

NEW = """    if hasattr(dataset, "episode_ids"):
        expected_split = behavior_cloning_split(
            dataset,
            validation_fraction=config.validation_fraction,
        )
        for name, actual, expected in (
            ("training", train_indices, expected_split.train_indices),
            ("validation", validation_indices, expected_split.validation_indices),
            ("purged", purged_indices, expected_split.purged_indices),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(
                    "explicit behavior-cloning split disagrees with "
                    f"episode-aware {name} split"
                )
        return train_indices, validation_indices

    expected_validation_count = (
        0
        if config.validation_fraction == 0.0
        else max(
            1,
            int(math.floor(sample_count * config.validation_fraction)),
        )
    )
    if validation_indices.size != expected_validation_count:
        raise ValueError(
            "explicit behavior-cloning split disagrees with validation_fraction"
        )
    return train_indices, validation_indices
"""


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    if text.count(OLD) != 1:
        raise SystemExit("behavior-cloning explicit split marker changed")
    PATH.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
