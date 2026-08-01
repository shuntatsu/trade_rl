from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def _update_training_config() -> None:
    path = ROOT / "trade_rl/rl/training_run_config.py"
    source = path.read_text(encoding="utf-8")

    source = _replace_once(
        source,
        '''            "alpha_artifact_digest": _signal_artifact_digest(
                self.alpha_artifact, kind="alpha"
            ),
            "environment": asdict(self.environment),
            "factor_artifact_digest": _signal_artifact_digest(
                self.factor_artifact, kind="factor"
            ),
''',
        '''            "alpha_artifact_digest": self.alpha_artifact_digest,
            "environment": asdict(self.environment),
            "factor_artifact_digest": self.factor_artifact_digest,
''',
        label="training config identity payload",
    )

    marker = "    def resolve_artifact_paths(self, base: Path) -> TrainingRunConfig:\n"
    properties = '''    @property
    def alpha_artifact_digest(self) -> str | None:
        """Return the validated alpha artifact digest, when configured."""

        return _signal_artifact_digest(self.alpha_artifact, kind="alpha")

    @property
    def factor_artifact_digest(self) -> str | None:
        """Return the validated factor artifact digest, when configured."""

        return _signal_artifact_digest(self.factor_artifact, kind="factor")

'''
    source = _replace_once(
        source,
        marker,
        properties + marker,
        label="training config property insertion",
    )
    path.write_text(source, encoding="utf-8")


def _update_training_workflow() -> None:
    path = ROOT / "trade_rl/workflows/training_run.py"
    source = path.read_text(encoding="utf-8")

    source = _replace_once(
        source,
        '''from trade_rl.rl.training_run_config import (
    TrainingRunConfig,
    _signal_artifact_digest,
)
''',
        "from trade_rl.rl.training_run_config import TrainingRunConfig\n",
        label="training workflow private import",
    )
    source = _replace_once(
        source,
        '''                "alpha_artifact_digest": _signal_artifact_digest(
                    config.alpha_artifact, kind="alpha"
                ),
                "environment": asdict(config.environment),
                "factor_artifact_digest": _signal_artifact_digest(
                    config.factor_artifact, kind="factor"
                ),
''',
        '''                "alpha_artifact_digest": config.alpha_artifact_digest,
                "environment": asdict(config.environment),
                "factor_artifact_digest": config.factor_artifact_digest,
''',
        label="training workflow environment identity",
    )
    path.write_text(source, encoding="utf-8")


def main() -> None:
    _update_training_config()
    _update_training_workflow()
    Path(__file__).unlink()


if __name__ == "__main__":
    main()
