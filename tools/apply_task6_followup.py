from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 6 follow-up anchor in {path}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        "tests/examples/test_binance_multitimeframe_full_assets.py",
        '''def test_full_runner_requires_same_representative_seed_across_folds(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    stability = namespace["_selection_stability_passed"]
    folds = [
        {
            "selected_configuration": "ppo-15m-target",
            "selected_seed": seed,
            "candidate_aggregates": [
                {"configuration": "ppo-15m-target", "eligible": True}
            ],
        }
        for seed in (0, 1)
    ]

    assert stability(folds) is False
''',
        '''def test_full_runner_treats_seed_as_nuisance_not_selected_hyperparameter(
    tmp_path: Path,
) -> None:
    namespace = _runner_namespace()
    stability = namespace["_selection_stability_passed"]
    folds = [
        {
            "selected_configuration": "ppo-15m-target",
            "selected_seed": seed,
            "candidate_aggregates": [
                {"configuration": "ppo-15m-target", "eligible": True}
            ],
        }
        for seed in (0, 1)
    ]

    assert stability(folds) is True
''',
    )
    replace_once(
        "tests/examples/test_binance_multitimeframe_full_assets.py",
        '''def test_selected_walk_forward_recipe_freezes_representative_seed(
    tmp_path: Path,
) -> None:
''',
        '''def test_selected_walk_forward_recipe_preserves_seed_ensemble(
    tmp_path: Path,
) -> None:
''',
    )
    replace_once(
        "tests/examples/test_binance_multitimeframe_full_assets.py",
        '''    name, seed, path = select_recipe(walk_forward_path, config_path, output)

    assert name == "ppo-15m-target"
    assert seed == 2
    assert path == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["training"]["seeds"] == [2]
''',
        '''    name, seeds, path = select_recipe(walk_forward_path, config_path, output)

    assert name == "ppo-15m-target"
    assert seeds == (0, 1, 2)
    assert path == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["training"]["seeds"] == [0, 1, 2]
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''    sequence_payload = SequenceObservationBuilder().schema_payload(dataset)
    sequence_observation_count = 0
    for window in sequence_payload["windows"]:
''',
        '''    sequence_payload = SequenceObservationBuilder().schema_payload(dataset)
    raw_sequence_windows = sequence_payload.get("windows")
    if not isinstance(raw_sequence_windows, (tuple, list)):
        raise RuntimeError("sequence schema windows must be ordered")
    sequence_observation_count = 0
    for window in raw_sequence_windows:
''',
    )


if __name__ == "__main__":
    main()
