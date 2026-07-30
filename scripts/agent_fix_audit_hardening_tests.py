from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def replace_count(path: str, old: str, new: str, *, expected: int) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise RuntimeError(
            f"expected {expected} matches in {path}, found {count}: {old[:80]!r}"
        )
    target.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    replace_once(
        "tests/integrations/test_sb3_checkpoint_assembly.py",
        '''from __future__ import annotations\n\nfrom pathlib import Path\nfrom types import SimpleNamespace\n''',
        '''from __future__ import annotations\n\nfrom collections.abc import Iterator\nfrom contextlib import contextmanager\nfrom pathlib import Path\nfrom types import SimpleNamespace\n''',
    )
    replace_once(
        "tests/integrations/test_sb3_checkpoint_assembly.py",
        '''    payload.update(changes)\n    return SimpleNamespace(**payload)\n\n\n@pytest.mark.parametrize(\n''',
        '''    payload.update(changes)\n    return SimpleNamespace(**payload)\n\n\n@contextmanager\ndef _passthrough_policy_copy(manifest: SimpleNamespace) -> Iterator[Path]:\n    yield manifest.policy_path\n\n\n@pytest.mark.parametrize(\n''',
    )
    replace_count(
        "tests/integrations/test_sb3_checkpoint_assembly.py",
        '''    monkeypatch.setattr(assembly_module, "load_checkpoint_manifest", lambda _: manifest)\n    monkeypatch.setattr(\n        assembly_module,\n        "validate_checkpoint_algorithm_identity",\n''',
        '''    monkeypatch.setattr(assembly_module, "load_checkpoint_manifest", lambda _: manifest)\n    monkeypatch.setattr(\n        assembly_module,\n        "verified_checkpoint_policy_copy",\n        _passthrough_policy_copy,\n    )\n    monkeypatch.setattr(\n        assembly_module,\n        "validate_checkpoint_algorithm_identity",\n''',
        expected=3,
    )

    replace_once(
        "tests/integrations/test_sb3_checkpoint_transfer.py",
        '''from __future__ import annotations\n\nfrom pathlib import Path\nfrom types import SimpleNamespace\n''',
        '''from __future__ import annotations\n\nfrom collections.abc import Iterator\nfrom contextlib import contextmanager\nfrom pathlib import Path\nfrom types import SimpleNamespace\n''',
    )
    replace_once(
        "tests/integrations/test_sb3_checkpoint_transfer.py",
        '''    payload.update(changes)\n    return SimpleNamespace(**payload)\n\n\ndef test_transfer_loader_accepts_new_environment_with_compatible_architecture(\n''',
        '''    payload.update(changes)\n    return SimpleNamespace(**payload)\n\n\n@contextmanager\ndef _passthrough_policy_copy(manifest: SimpleNamespace) -> Iterator[Path]:\n    yield manifest.policy_path\n\n\ndef test_transfer_loader_accepts_new_environment_with_compatible_architecture(\n''',
    )
    replace_once(
        "tests/integrations/test_sb3_checkpoint_transfer.py",
        '''    monkeypatch.setattr(assembly_module, "load_checkpoint_manifest", lambda _: manifest)\n    monkeypatch.setattr(\n        assembly_module,\n        "checkpoint_identity_payload_for_model",\n''',
        '''    monkeypatch.setattr(assembly_module, "load_checkpoint_manifest", lambda _: manifest)\n    monkeypatch.setattr(\n        assembly_module,\n        "verified_checkpoint_policy_copy",\n        _passthrough_policy_copy,\n    )\n    monkeypatch.setattr(\n        assembly_module,\n        "checkpoint_identity_payload_for_model",\n''',
    )

    replace_once(
        "tests/rl/test_checkpoint_trust_boundary.py",
        '''from trade_rl.rl.checkpointing import load_checkpoint_manifest, publish_checkpoint\n''',
        '''from trade_rl.rl.checkpointing import (\n    load_checkpoint_manifest,\n    publish_checkpoint,\n    verified_checkpoint_policy_copy,\n)\n''',
    )
    replace_once(
        "tests/rl/test_checkpoint_trust_boundary.py",
        '''def test_checkpoint_policy_must_not_be_a_symlink(tmp_path: Path) -> None:\n    root = _publish(tmp_path)\n    policy_path = root / "policy.zip"\n    external = tmp_path / "external-policy.zip"\n    policy_path.replace(external)\n    policy_path.symlink_to(external)\n\n    with pytest.raises(ValueError, match="symlink"):\n        load_checkpoint_manifest(root / "checkpoint.json")\n''',
        '''def test_checkpoint_policy_must_not_be_a_symlink(tmp_path: Path) -> None:\n    root = _publish(tmp_path)\n    policy_path = root / "policy.zip"\n    external = tmp_path / "external-policy.zip"\n    policy_path.replace(external)\n    policy_path.symlink_to(external)\n\n    with pytest.raises(ValueError, match="symlink"):\n        load_checkpoint_manifest(root / "checkpoint.json")\n\n\ndef test_checkpoint_deserialization_uses_a_private_verified_copy(\n    tmp_path: Path,\n) -> None:\n    root = _publish(tmp_path)\n    manifest = load_checkpoint_manifest(root / "checkpoint.json")\n\n    with verified_checkpoint_policy_copy(manifest) as verified:\n        assert verified != manifest.policy_path\n        assert verified.read_bytes() == manifest.policy_path.read_bytes()\n        private_root = verified.parent\n\n    assert not private_root.exists()\n''',
    )

    for relative in (
        "scripts/agent_fix_audit_hardening_tests.py",
        ".github/workflows/agent-audit-render.yml",
    ):
        path = ROOT / relative
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
