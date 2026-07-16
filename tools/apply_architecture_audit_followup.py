from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing audit follow-up anchor in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        "trade_rl/rl/environment.py",
        '''        pending_target_discarded = bool(
            time_limit_reached
            and self.config.signal_delay_decisions == 1
            and self._pending_hybrid_target is not None
        )
        discarded_pending_target = (
            None
            if not pending_target_discarded
            else self._pending_hybrid_target.copy()
        )
''',
        '''        pending_hybrid_target = self._pending_hybrid_target
        pending_target_discarded = bool(
            time_limit_reached
            and self.config.signal_delay_decisions == 1
            and pending_hybrid_target is not None
        )
        discarded_pending_target = (
            None
            if not pending_target_discarded or pending_hybrid_target is None
            else pending_hybrid_target.copy()
        )
''',
    )
    replace_once(
        "trade_rl/rl/policies.py",
        '''        per_dimension = self.distribution.log_prob(gaussian_actions)
''',
        '''        distribution = self.distribution
        if distribution is None:
            raise RuntimeError("masked action distribution is not initialized")
        per_dimension = distribution.log_prob(gaussian_actions)
''',
    )
    replace_once(
        "tests/serving/test_sb3_loader.py",
        '''                "dataset_id": dataset.dataset_id,
                "feature_names": list(dataset.feature_names),
''',
        '''                "dataset_id": dataset.dataset_id,
                "feature_config_digest": dataset.feature_config_digest,
                "feature_names": list(dataset.feature_names),
''',
    )


if __name__ == "__main__":
    main()
