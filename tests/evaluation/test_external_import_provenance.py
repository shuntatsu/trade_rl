"""Opt-in import-origin evidence must bind actually loaded regular modules."""

from __future__ import annotations

import importlib.machinery
import sys
from hashlib import sha256
from types import ModuleType

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.runs.external_imports import verify_loaded_package_imports
from trade_rl.evaluation.runs.external_provenance import (
    ExternalPackageInput,
    PackageFilePin,
    PackageSourcePin,
)


@pytest.fixture
def source(tmp_path, monkeypatch):
    package = "external_import_test"
    root = tmp_path / package
    root.mkdir()
    files = {"__init__.py": b"VALUE = 1\n", "policy.py": b"VALUE = 2\n"}
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    pin = PackageSourcePin(
        repository="example/research",
        commit="a" * 40,
        tree="b" * 40,
        package=package,
        wheel_sha256="c" * 64,
        wheel_size_bytes=1,
        files=tuple(
            PackageFilePin(name, len(raw), sha256(raw).hexdigest())
            for name, raw in sorted(files.items())
        ),
    )
    item = ExternalPackageInput(pin, tmp_path / "external.whl", root)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    import importlib

    importlib.import_module(package + ".policy")
    yield item
    for name in tuple(sys.modules):
        if name == package or name.startswith(package + "."):
            sys.modules.pop(name, None)


def verify(source, **kwargs):
    return verify_loaded_package_imports(
        source,
        expected_pin_digest=content_digest(source.pin),
        required_modules=(source.pin.package, source.pin.package + ".policy"),
        **kwargs,
    )


def test_actual_imports_and_repeat_are_bound(source):
    result = verify(source)
    assert result.package_pin_digest == content_digest(source.pin)
    assert result.loaded_modules == (
        (source.pin.package, "__init__.py"),
        (source.pin.package + ".policy", "policy.py"),
    )
    assert verify(source, expected_observation_digest=result.digest) == result
    assert str(source.installed_package) not in repr(result)


@pytest.mark.parametrize(
    "kind",
    [
        "file",
        "spec",
        "loader",
        "path",
        "name",
        "missing",
        "extra",
        "bytes",
        "module_type",
    ],
)
def test_wrong_loaded_origin_or_bytes_refused(source, monkeypatch, kind):
    name = source.pin.package + ".policy"
    module = sys.modules[name]
    if kind == "file":
        monkeypatch.setattr(
            module, "__file__", str(source.installed_package / "__init__.py")
        )
    elif kind == "spec":
        monkeypatch.setattr(
            module.__spec__, "origin", str(source.installed_package / "__init__.py")
        )
    elif kind == "loader":
        monkeypatch.setattr(module.__spec__, "loader", object())
    elif kind == "path":
        monkeypatch.setattr(sys.modules[source.pin.package], "__path__", ["other"])
    elif kind == "name":
        monkeypatch.setattr(module, "__name__", "alias")
    elif kind == "missing":
        monkeypatch.delitem(sys.modules, name)
    elif kind == "extra":
        monkeypatch.setitem(
            sys.modules, source.pin.package + ".extra", ModuleType("extra")
        )
    elif kind == "bytes":
        (source.installed_package / "policy.py").write_bytes(b"VALUE = 9\n")
    else:
        monkeypatch.setitem(sys.modules, name, object())
    with pytest.raises(ValueError):
        verify(source)


def test_required_roster_and_expected_digest_refused(source):
    for roster in (
        (),
        ("foreign",),
        (source.pin.package + ".policy", source.pin.package),
        (source.pin.package, source.pin.package),
    ):
        with pytest.raises(ValueError):
            verify_loaded_package_imports(
                source,
                expected_pin_digest=content_digest(source.pin),
                required_modules=roster,
            )
    with pytest.raises(ValueError):
        verify_loaded_package_imports(
            source, expected_pin_digest="f" * 64, required_modules=(source.pin.package,)
        )
    with pytest.raises(ValueError):
        verify(source, expected_observation_digest="f" * 64)


def test_wrong_loader_path_is_not_accepted(source, monkeypatch):
    module = sys.modules[source.pin.package + ".policy"]
    loader = importlib.machinery.SourceFileLoader(
        module.__name__, str(source.installed_package / "__init__.py")
    )
    monkeypatch.setattr(module.__spec__, "loader", loader)
    monkeypatch.setattr(module, "__loader__", loader)
    with pytest.raises(ValueError):
        verify(source)
