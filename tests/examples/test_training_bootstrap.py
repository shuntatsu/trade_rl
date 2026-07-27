from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "binance-multitimeframe"
        / "training_bootstrap.py"
    )
    spec = importlib.util.spec_from_file_location("training_bootstrap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_checks_cache_before_full_entrypoint(tmp_path: Path) -> None:
    module = _load_module()
    calls: list[str] = []

    result = module.run_bootstrap(
        cache_root=tmp_path,
        check_cache=lambda _root: calls.append("check"),
        full_entrypoint=lambda: calls.append("run") or 7,
    )

    assert result == 7
    assert calls == ["check", "run"]


def test_bootstrap_does_not_start_training_when_cache_is_incomplete(
    tmp_path: Path,
) -> None:
    module = _load_module()
    calls: list[str] = []

    def fail_check(_root: Path) -> None:
        raise FileNotFoundError("missing=2")

    with pytest.raises(FileNotFoundError, match="market-data-sync"):
        module.run_bootstrap(
            cache_root=tmp_path,
            check_cache=fail_check,
            full_entrypoint=lambda: calls.append("run") or 0,
        )

    assert calls == []


def test_bootstrap_forwards_shared_cache_root_to_full_entrypoint(
    tmp_path: Path,
) -> None:
    module = _load_module()

    argv = module.ensure_cache_root_argument(["training_bootstrap.py"], tmp_path)

    assert argv == [
        "training_bootstrap.py",
        "--cache-root",
        str(tmp_path),
    ]


def test_bootstrap_preserves_explicit_cache_root(tmp_path: Path) -> None:
    module = _load_module()
    explicit = tmp_path / "explicit"

    argv = module.ensure_cache_root_argument(
        ["training_bootstrap.py", "--cache-root", str(explicit)],
        tmp_path,
    )

    assert argv == ["training_bootstrap.py", "--cache-root", str(explicit)]
