"""Fail closed on mixed uv toolchains before re-enabling Dependabot (#1175)."""

from __future__ import annotations

import re
import subprocess
import tomllib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.architecture.workflow_action_refs import setup_uv_action_matches
from tools import ppo_checkpoint_actions as checkpoint

ROOT = Path(__file__).resolve().parents[2]
CURRENT_UV = "0.13.10"
# Immutable reviewed/execution sources may require a historical uv version.
SOURCE_BOUND_WORKFLOWS = {
    "ci-original-runtime-probe.yml": 1,
    "ppo-normalization-execution.yml": 3,
    "ppo-4h-gemini-review.yml": 2,
}


def test_repository_requires_the_current_compatible_uv() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["uv"]["required-version"] == f"=={CURRENT_UV}"


def test_setup_uv_pins_current_source_or_obeys_immutable_reviewed_source() -> None:
    total = 0
    source_bound_counts: dict[str, int] = {}
    workflows = ROOT / ".github" / "workflows"
    for path in sorted((*workflows.glob("*.yml"), *workflows.glob("*.yaml"))):
        source = path.read_text(encoding="utf-8")
        for match in setup_uv_action_matches(source, path):
            # - uses: and a separate uses: under - name: are both valid
            # GitHub Actions syntax; mutable tags must not evade inspection.
            assert re.fullmatch(r"[0-9a-f]{40}", match.group("ref")), path
            # A new step may start with - run:, - if: or - id:; a new job
            # may also follow directly. Neither may lend this action a pin.
            action_indent = len(match.group("indent")) + (
                len(match.group("dash") or "")
            )
            following = source[match.end() :]
            step_boundary = re.search(rf"(?m)^ {{0,{action_indent - 2}}}\S", following)
            if step_boundary:
                following = following[: step_boundary.start()]
            with_marker = re.search(
                rf"(?m)^ {{{action_indent}}}with:[ \t]*$", following
            )
            assert with_marker is not None, path
            with_options = following[with_marker.end() :]
            options_boundary = re.search(
                rf"(?m)^ {{0,{action_indent}}}\S", with_options
            )
            if options_boundary:
                with_options = with_options[: options_boundary.start()]
            option_indent = action_indent + 2
            if path.name in SOURCE_BOUND_WORKFLOWS:
                assert re.search(
                    rf"(?m)^ {{{option_indent}}}working-directory: target$",
                    with_options,
                ), path
                assert (
                    re.search(rf"(?m)^ {{{option_indent}}}version:", with_options)
                    is None
                ), path
                source_bound_counts[path.name] = (
                    source_bound_counts.get(path.name, 0) + 1
                )
            else:
                assert re.search(
                    rf'(?m)^ {{{option_indent}}}version: "{CURRENT_UV}"$',
                    with_options,
                ), path
            total += 1
    assert source_bound_counts == SOURCE_BOUND_WORKFLOWS
    assert total == 19  # same inventory as the exact Action SHA contract


def test_setup_uv_version_cannot_be_borrowed_from_an_unnamed_following_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unpinned action must fail even if a later - run: step sets env.version."""
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    original_read = Path.read_text
    source = original_read(workflow, encoding="utf-8")
    before = (
        '          version: "0.13.10"\n'
        "          enable-cache: true\n"
        "\n"
        "      - name: Install\n"
    )
    after = (
        "          enable-cache: true\n"
        "\n"
        '      - run: echo "later step"\n'
        "        env:\n"
        '          version: "0.13.10"\n'
        "\n"
        "      - name: Install\n"
    )
    assert before in source
    unsafe_source = source.replace(before, after, 1)

    def fake_read_text(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if path == workflow:
            return unsafe_source
        return original_read(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    with pytest.raises(AssertionError):
        test_setup_uv_pins_current_source_or_obeys_immutable_reviewed_source()


def test_existing_commented_setup_uv_still_requires_its_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    original_read = Path.read_text
    source = original_read(workflow, encoding="utf-8")
    action = "        uses: astral-sh/setup-uv@1db35dd8eee11c5894872e0b383ae6c406cf738d"
    assert action in source
    mutated = source.replace(action + "\n", action + " # reviewed Action\n", 1)

    def read_text(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if path == workflow:
            return mutated
        return original_read(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)
    # A valid pin with a YAML comment remains in the exact step inventory.
    test_setup_uv_pins_current_source_or_obeys_immutable_reviewed_source()
    mutated = mutated.replace(
        '          version: "0.13.10"\n', '          version: "0.10.0"\n', 1
    )
    with pytest.raises(AssertionError):
        test_setup_uv_pins_current_source_or_obeys_immutable_reviewed_source()


@pytest.mark.parametrize("trailing_comment", ["", " # action review pin"])
def test_setup_uv_check_covers_yaml_suffix_as_well_as_yml(
    monkeypatch: pytest.MonkeyPatch,
    trailing_comment: str,
) -> None:
    """A new *.yaml workflow or YAML comment must not bypass the pin contract."""
    workflows = ROOT / ".github" / "workflows"
    new_path = workflows / "unpinned-ci.yaml"
    assert not new_path.exists()
    original_glob = Path.glob
    original_read = Path.read_text

    def fake_glob(path: Path, pattern: str) -> Iterator[Path]:
        if path == workflows and pattern == "*.yaml":
            return iter((new_path,))
        return original_glob(path, pattern)

    def fake_read_text(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if path == new_path:
            return (
                "jobs:\n  check:\n    steps:\n"
                "      - uses: astral-sh/setup-uv@1db35dd8eee11c5894872e0b383ae6c406cf738d"
                + trailing_comment
                + "\n"
                '        with:\n          version: "0.10.0"\n'
            )
        return original_read(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "glob", fake_glob)
    monkeypatch.setattr(Path, "read_text", fake_read_text)
    with pytest.raises(AssertionError):
        test_setup_uv_pins_current_source_or_obeys_immutable_reviewed_source()


@pytest.mark.parametrize(
    ("reported", "allowed"),
    [
        ("0.13.10", True),
        ("0.13.10 (x86_64-unknown-linux-gnu)", True),
        ("0.10.0", False),
        ("0.13.7", False),
        ("0.13.9", False),
        ("0.13.100", False),
    ],
)
def test_checkpoint_runtime_provenance_preserves_exact_version_lineage(
    reported: str, allowed: bool
) -> None:
    completed = subprocess.CompletedProcess(
        args=["uv", "--version"], returncode=0, stdout=f"uv {reported}\n", stderr=""
    )
    with patch.object(checkpoint.subprocess, "run", return_value=completed) as call:
        if allowed:
            assert checkpoint._runtime_provenance()["uv_version"] == CURRENT_UV
        else:
            with pytest.raises(ValueError, match="actual uv executable"):
                checkpoint._runtime_provenance()
        assert call.call_args.args[0] == ["uv", "--version"]


@pytest.mark.parametrize(
    "workflow",
    ["ppo-feature-checkpoint.yml", "ppo-checkpoint-runtime-preflight.yml"],
)
def test_manual_checkpoint_version_guard_accepts_platform_suffix(
    workflow: str,
) -> None:
    source = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    assert "grep -Eq '^uv 0\\.13\\.10([[:space:]]|$)'" in source
    assert "grep -Fx 'uv 0.10.0'" not in source


@pytest.mark.parametrize("guard", ["version", "action"])
@pytest.mark.parametrize(
    "step",
    [
        '      - uses: "astral-sh/setup-uv@v7"',
        "      - uses: 'astral-sh/setup-uv@v7'",
        '      - name: uv\n        uses: "astral-sh/setup-uv@v7"',
        "      - name: uv\n        uses: 'astral-sh/setup-uv@v7'",
        '      - "uses": "astral-sh/setup-uv@v7"',
        "      - {uses: 'astral-sh/setup-uv@v7'}",
        "      - uses: astral-sh/setup-uv@v7",
        '      - uses: "Astral-sh/Setup-uv@v7"',
        '      - uses: "astral-sh/setup-uv@1db35dd8eee11c5894872e0b383ae6c406cf738d"\n'
        '        with:\n          version: "0.13.10"',
    ],
)
def test_additional_setup_uv_action_cannot_hide_in_yaml_scalar_syntax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, guard: str, step: str
) -> None:
    from tests.architecture import test_ci_python_runtime as runtime

    original = ROOT / ".github" / "workflows"
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    for path in (*original.glob("*.yml"), *original.glob("*.yaml")):
        (workflows / path.name).write_bytes(path.read_bytes())
    (workflows / "extra.yml").write_text(
        "name: additional action\non: workflow_dispatch\njobs:\n  check:\n"
        "    runs-on: ubuntu-latest\n    steps:\n" + step + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    check = (
        test_setup_uv_pins_current_source_or_obeys_immutable_reviewed_source
        if guard == "version"
        else runtime.test_setup_uv_action_commit_is_consistent_across_all_workflows
    )
    with pytest.raises(AssertionError):
        check()


@pytest.mark.parametrize("guard", ["version", "action"])
@pytest.mark.parametrize("quote", ["'", '"'])
def test_quoted_existing_setup_uv_action_stays_in_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, guard: str, quote: str
) -> None:
    from tests.architecture import test_ci_python_runtime as runtime

    original = ROOT / ".github" / "workflows"
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    for path in (*original.glob("*.yml"), *original.glob("*.yaml")):
        (workflows / path.name).write_bytes(path.read_bytes())
    ci = workflows / "ci.yml"
    source = ci.read_text(encoding="utf-8")
    action = runtime.SETUP_UV
    old = f"uses: {action}\n"
    assert old in source
    ci.write_text(
        source.replace(old, f"uses: {quote}{action}{quote} # reviewed pin\n", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    check = (
        test_setup_uv_pins_current_source_or_obeys_immutable_reviewed_source
        if guard == "version"
        else runtime.test_setup_uv_action_commit_is_consistent_across_all_workflows
    )
    check()


@pytest.mark.parametrize("quote", ["'", '"'])
def test_quoted_setup_uv_cannot_change_version_or_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quote: str
) -> None:
    from tests.architecture import test_ci_python_runtime as runtime

    original = ROOT / ".github" / "workflows"
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    for path in (*original.glob("*.yml"), *original.glob("*.yaml")):
        (workflows / path.name).write_bytes(path.read_bytes())
    ci = workflows / "ci.yml"
    source = ci.read_text(encoding="utf-8")
    old = f"uses: {runtime.SETUP_UV}\n"
    assert old in source
    good = source.replace(old, f"uses: {quote}{runtime.SETUP_UV}{quote}\n", 1)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    ci.write_text(
        good.replace('version: "0.13.10"', 'version: "0.10.0"', 1), encoding="utf-8"
    )
    with pytest.raises(AssertionError):
        test_setup_uv_pins_current_source_or_obeys_immutable_reviewed_source()
    ci.write_text(
        good.replace(
            f"{quote}{runtime.SETUP_UV}{quote}",
            f"{quote}astral-sh/setup-uv@{'a' * 40}{quote}",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        runtime.test_setup_uv_action_commit_is_consistent_across_all_workflows()
