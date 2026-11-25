"""Generated wheel/source closure and provenance admission, without executing it."""

from __future__ import annotations

import stat
import zipfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.runs import provenance as legacy
from trade_rl.evaluation.runs.external_provenance import (
    ExternalPackageInput,
    PackageFilePin,
    PackageSourcePin,
    verify_external_source_provenance,
)


def _hash(data: bytes) -> str:
    return sha256(data).hexdigest()


def _write_wheel(path: Path, package: str, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        for name, raw in files.items():
            wheel.writestr(f"{package}/{name}", raw)
        wheel.writestr(f"{package}-0.1.dist-info/METADATA", b"Version: 0.1\n")
        wheel.writestr(f"{package}-0.1.dist-info/WHEEL", b"Wheel-Version: 1.0\n")
        wheel.writestr(f"{package}-0.1.dist-info/RECORD", b"")


def _pin(tmp: Path, package: str, *, repo: str) -> ExternalPackageInput:
    install = tmp / "site-packages" / package
    install.mkdir(parents=True)
    files = {
        "__init__.py": b"VALUE = 1\n",
        "resources/contract.json": b'{"schema":"generated"}\n',
    }
    for name, raw in files.items():
        path = install / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
    wheel = tmp / f"{package}-0.1-py3-none-any.whl"
    _write_wheel(wheel, package, files)
    return ExternalPackageInput(
        pin=PackageSourcePin(
            repository=repo,
            commit="a" * 40,
            tree="b" * 40,
            package=package,
            wheel_sha256=_hash(wheel.read_bytes()),
            wheel_size_bytes=wheel.stat().st_size,
            files=tuple(
                PackageFilePin(name, len(raw), _hash(raw))
                for name, raw in sorted(files.items())
            ),
        ),
        wheel=wheel,
        installed_package=install,
    )


@pytest.fixture
def inputs(tmp_path):
    return (
        _pin(tmp_path, "trade_rl", repo="example/core"),
        _pin(tmp_path, "research_test", repo="example/research"),
    )


def _verify(core, research, *, expected=None, config=None, **overrides):
    config = {"fee": 0.001} if config is None else config
    args = {
        "expected_core_pin_digest": content_digest(core.pin),
        "expected_research_pin_digest": content_digest(research.pin),
        "expected_runtime_config_digest": content_digest(config),
        "expected_provenance_digest": expected,
    }
    args.update(overrides)
    return verify_external_source_provenance(
        core, research, runtime_config=config, **args
    )


def test_two_source_closures_and_recheck_are_bound(inputs):
    core, research = inputs
    record = _verify(core, research)
    assert record.schema_version == "external_source_provenance_v1"
    assert record.core == core.pin
    assert record.research == research.pin
    assert record.runtime_config_digest == content_digest({"fee": 0.001})
    assert record.runtime_environment_digest == content_digest(
        legacy.runtime_environment_manifest()
    )
    assert record.digest == content_digest(record)
    assert _verify(core, research, expected=record.digest) == record
    assert str(core.wheel) not in repr(record)
    assert _verify(core, research, config={"fee": 0.002}).digest != record.digest
    with pytest.raises(ValueError, match="provenance digest"):
        _verify(core, research, config={"fee": 0.002}, expected=record.digest)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("kind", ["changed", "missing", "extra", "resource", "wheel"])
def test_recheck_rejects_source_drift(inputs, which, kind):
    record = _verify(*inputs)
    selected = inputs[which]
    root = selected.installed_package
    if kind == "changed":
        (root / "__init__.py").write_bytes(b"VALUE = 2\n")
    elif kind == "missing":
        (root / "__init__.py").unlink()
    elif kind == "extra":
        (root / "unreviewed.py").write_bytes(b"EXTRA = 1\n")
    elif kind == "resource":
        (root / "resources/contract.json").write_bytes(b"{}\n")
    else:
        selected.wheel.write_bytes(selected.wheel.read_bytes() + b"x")
    with pytest.raises((ValueError, OSError)):
        _verify(*inputs, expected=record.digest)


def test_wrong_reviewed_pins_and_config_refused_before_files(inputs):
    core, research = inputs
    research.wheel.unlink()
    for name in (
        "expected_core_pin_digest",
        "expected_research_pin_digest",
        "expected_runtime_config_digest",
    ):
        with pytest.raises(ValueError, match="digest"):
            _verify(core, research, **{name: "f" * 64})


def test_independently_pinned_manifest_must_match_wheel_even_if_wheel_hash_matches(
    inputs,
):
    core, research = inputs
    with zipfile.ZipFile(research.wheel, "a") as wheel:
        wheel.writestr("research_test/new.py", b"unreviewed\n")
    altered = replace(
        research,
        pin=replace(
            research.pin,
            wheel_sha256=_hash(research.wheel.read_bytes()),
            wheel_size_bytes=research.wheel.stat().st_size,
        ),
    )
    with pytest.raises(ValueError, match="roster"):
        _verify(core, altered)


@pytest.mark.parametrize(
    "name",
    [
        "../evil.py",
        "/absolute",
        "a//b",
        "a\\b",
        "a/./b",
        "AUX.txt",
        "a:stream",
        "__pycache__/x.py",
    ],
)
def test_unsafe_source_paths_refused(name):
    with pytest.raises(ValueError):
        PackageFilePin(name, 1, "a" * 64)


def test_noncanonical_and_aliased_manifests_refused(inputs):
    pin = inputs[0].pin
    for files in (
        pin.files[::-1],
        (*pin.files, pin.files[-1]),
        tuple(
            sorted(
                (*pin.files, PackageFilePin("__INIT__.py", 1, "a" * 64)),
                key=lambda f: f.path,
            )
        ),
        tuple(
            sorted(
                (*pin.files, PackageFilePin("resources", 1, "a" * 64)),
                key=lambda f: f.path,
            )
        ),
    ):
        with pytest.raises(ValueError):
            replace(pin, files=files)


def test_symlinked_sources_refused(inputs, tmp_path):
    core, research = inputs
    source = research.installed_package / "__init__.py"
    other = tmp_path / "same.py"
    other.write_bytes(source.read_bytes())
    source.unlink()
    try:
        source.symlink_to(other)
    except OSError:
        pytest.skip("symlink creation not supported by this platform")
    with pytest.raises((ValueError, OSError)):
        _verify(core, research)


@pytest.mark.parametrize(
    "member", ["research_test/../other.py", "other.py", "research_test/__init__.py"]
)
def test_bad_wheel_members_rejected_even_with_updated_wheel_digest(inputs, member):
    core, research = inputs
    # Duplicate member intentionally tests rejection of ambiguous archive paths.
    with zipfile.ZipFile(research.wheel, "a") as wheel:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            wheel.writestr(member, b"foreign\n")
    altered = replace(
        research,
        pin=replace(
            research.pin,
            wheel_sha256=_hash(research.wheel.read_bytes()),
            wheel_size_bytes=research.wheel.stat().st_size,
        ),
    )
    with pytest.raises(ValueError):
        _verify(core, altered)


def test_zip_symlink_refused(inputs):
    core, research = inputs
    with zipfile.ZipFile(research.wheel, "a") as wheel:
        info = zipfile.ZipInfo("research_test/link")
        info.create_system = 3
        info.external_attr = stat.S_IFLNK << 16
        wheel.writestr(info, "__init__.py")
    altered = replace(
        research,
        pin=replace(
            research.pin,
            wheel_sha256=_hash(research.wheel.read_bytes()),
            wheel_size_bytes=research.wheel.stat().st_size,
        ),
    )
    with pytest.raises(ValueError):
        _verify(core, altered)


def test_legacy_manifest_does_not_gain_external_authority(inputs):
    core, research = inputs
    before = legacy._implementation_manifest(core.installed_package)
    (research.installed_package / "__init__.py").write_bytes(b"VALUE = 2\n")
    assert legacy._implementation_manifest(core.installed_package) == before
    assert before == {
        "schema_version": "candidate_run_implementation_v1",
        "files": [{"path": "__init__.py", "sha256": _hash(b"VALUE = 1\n")}],
    }
    with pytest.raises(ValueError):
        _verify(core, research)


def test_runtime_drift_refuses_previously_verified_record(inputs, monkeypatch):
    from trade_rl.evaluation.runs import external_provenance

    record = _verify(*inputs)
    monkeypatch.setattr(
        external_provenance,
        "runtime_environment_manifest",
        lambda: {"machine": "different"},
    )
    with pytest.raises(ValueError, match="provenance digest"):
        _verify(*inputs, expected=record.digest)


def test_installed_bytecode_cache_refused(inputs):
    _, research = inputs
    cache = research.installed_package / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"cache")
    with pytest.raises(ValueError, match="path"):
        _verify(*inputs)


def test_wheel_must_have_single_complete_metadata_roster(inputs):
    core, research = inputs
    with zipfile.ZipFile(research.wheel, "w") as wheel:
        for row in research.pin.files:
            wheel.writestr(
                f"research_test/{row.path}",
                (research.installed_package / row.path).read_bytes(),
            )
        wheel.writestr("research_test-0.1.dist-info/METADATA", b"Version: 0.1\n")
    altered = replace(
        research,
        pin=replace(
            research.pin,
            wheel_sha256=_hash(research.wheel.read_bytes()),
            wheel_size_bytes=research.wheel.stat().st_size,
        ),
    )
    with pytest.raises(ValueError, match="metadata roster"):
        _verify(core, altered)
