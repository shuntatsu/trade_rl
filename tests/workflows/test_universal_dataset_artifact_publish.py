from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.artifacts.hashing import content_digest


def _digest(label: str) -> str:
    return content_digest(label)


def _dataset(symbol: str) -> SimpleNamespace:
    return SimpleNamespace(symbols=(symbol,), dataset_id=_digest(f"dataset:{symbol}"))


def test_publish_universal_train_dataset_artifacts_is_train_only_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.universal_training_runner as module
    from trade_rl.workflows.universal_training_runner import (
        publish_universal_train_dataset_artifacts,
    )

    published: list[tuple[str, Path]] = []

    def publish(path: Path, dataset: SimpleNamespace) -> None:
        published.append((dataset.symbols[0], path))

    monkeypatch.setattr(module, "publish_market_dataset_artifact", publish)
    paths = publish_universal_train_dataset_artifacts(
        {"AAAUSDT": _dataset("AAAUSDT"), "BBBUSDT": _dataset("BBBUSDT")},
        train_symbols=("AAAUSDT", "BBBUSDT"),
        artifact_root=tmp_path,
    )

    assert paths == {
        "AAAUSDT": tmp_path / "AAAUSDT",
        "BBBUSDT": tmp_path / "BBBUSDT",
    }
    assert published == [
        ("AAAUSDT", tmp_path / "AAAUSDT"),
        ("BBBUSDT", tmp_path / "BBBUSDT"),
    ]


def test_publish_universal_train_dataset_artifacts_reuses_only_matching_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.universal_training_runner as module
    from trade_rl.workflows.universal_training_runner import (
        publish_universal_train_dataset_artifacts,
    )

    path = tmp_path / "AAAUSDT"
    path.mkdir()
    monkeypatch.setattr(
        module,
        "load_market_dataset_artifact",
        lambda _path: _dataset("AAAUSDT"),
    )
    monkeypatch.setattr(
        module,
        "publish_market_dataset_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must reuse")),
    )

    result = publish_universal_train_dataset_artifacts(
        {"AAAUSDT": _dataset("AAAUSDT")},
        train_symbols=("AAAUSDT",),
        artifact_root=tmp_path,
    )
    assert result == {"AAAUSDT": path}

    monkeypatch.setattr(
        module,
        "load_market_dataset_artifact",
        lambda _path: SimpleNamespace(
            symbols=("AAAUSDT",), dataset_id=_digest("different")
        ),
    )
    with pytest.raises(ValueError, match="identity"):
        publish_universal_train_dataset_artifacts(
            {"AAAUSDT": _dataset("AAAUSDT")},
            train_symbols=("AAAUSDT",),
            artifact_root=tmp_path,
        )
