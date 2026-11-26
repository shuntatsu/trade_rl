"""Fail-closed, network-free checks for registered source/document semantics.

No market archive, economic stage, GitHub API, generated report, or model is read.
Only explicitly registered synthetic-safe prospective harness modules are imported.
A passing check verifies the listed invariants, not every sentence of any document.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class ContractViolation(ValueError):
    """A normative document contradicts its source-bound, checked contract."""


@dataclass(frozen=True)
class ProtocolBinding:
    protocol_id: str
    source: str
    spec: str


# Add a tuple when an active pre-registered harness and spec are introduced.
# Partial source/doc introductions are rejected; unintroduced pairs are allowed.
PROTOCOLS: tuple[ProtocolBinding, ...] = ()

_DIGEST_LINE = re.compile(r"^Protocol digest: `([0-9a-f]{64})`[ \t]*$", re.MULTILINE)


def _read(root: Path, path: str) -> str:
    try:
        return (root / path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ContractViolation(
            f"{path}: cannot read canonical source/document"
        ) from error


def check_registered_protocol_coverage(root: Path) -> None:
    """New source-derived digest harnesses must opt into the semantic gate.

    Only inspect AST. Never import or execute an unregistered protocol.
    """
    known = {binding.source for binding in PROTOCOLS}
    for source in sorted((root / "tools").glob("*.py")):
        text = _read(root, source.relative_to(root).as_posix())
        try:
            tree = ast.parse(text, filename=source.name)
        except SyntaxError as error:
            raise ContractViolation(f"{source.name}: invalid source syntax") from error
        declares_digest = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "protocol_digest"
            for node in tree.body
        )
        if declares_digest and source.relative_to(root).as_posix() not in known:
            raise ContractViolation(
                f"{source.relative_to(root)}: unregistered protocol_digest harness; "
                "register source/spec and source-bound oracles before merging"
            )


def _installed_digest(root: Path, binding: ProtocolBinding) -> tuple[str, str]:
    """Import only registered synthetic-safe harnesses from this exact checkout."""
    module = importlib.import_module(
        binding.source.removesuffix(".py").replace("/", ".")
    )
    source = getattr(module, "__file__", None)
    if source is None or Path(source).resolve() != (root / binding.source).resolve():
        raise ContractViolation(
            f"{binding.source}: source import resolved outside checkout"
        )
    if getattr(module, "SPEC_RELPATH", None) != binding.spec:
        raise ContractViolation(
            f"{binding.source}: spec path differs from source binding"
        )
    return module.PROTOCOL_ID, module.protocol_digest()


def check_protocol_digest(
    root: Path,
    binding: ProtocolBinding,
    resolver: Callable[[ProtocolBinding], tuple[str, str]] | None = None,
) -> bool:
    """Check active spec/harness identity and source+spec digest, with no stage execution."""
    source, spec = root / binding.source, root / binding.spec
    if not source.is_file() and not spec.is_file():
        return False
    if not source.is_file():
        raise ContractViolation(f"{binding.spec}: missing source {binding.source}")
    if not spec.is_file():
        raise ContractViolation(f"{binding.source}: missing spec {binding.spec}")
    doc = _read(root, binding.spec)
    lines = doc.splitlines()
    if not lines or f"`{binding.protocol_id}`" not in lines[0]:
        raise ContractViolation(f"{binding.spec}: protocol_id heading mismatch")
    matches = _DIGEST_LINE.findall(doc)
    if len(matches) != 1:
        raise ContractViolation(
            f"{binding.spec}: require exactly one 64-hex digest line"
        )
    actual_id, digest = (resolver or (lambda b: _installed_digest(root, b)))(binding)
    if actual_id != binding.protocol_id:
        raise ContractViolation(f"{binding.source}: protocol_id mismatch")
    if not isinstance(digest, str) or digest != matches[0]:
        raise ContractViolation(f"{binding.spec}: protocol digest mismatch")
    return True


def _verify_asof_equity(full: object, truncated: object, observed_rows: int) -> None:
    """Equal point-in-time information must never produce different past equity."""
    import numpy as np

    def finite_marks(value: object) -> np.ndarray:
        # The synthetic oracle consumes native real vectors, not coercion
        # protocols that may erase missingness or the imaginary component.
        error = "trend_breadth: as-of equity must contain finite binary64-compatible real marks"
        # The native replay emits binary64 equity. Refuse integer inputs
        # whose precision cannot be guaranteed before NumPy promotes them.
        exact_integer_limit = 2**53
        if type(value) is np.ndarray:
            owner: object = value
            while True:
                # Exact types avoid arbitrary owner attribute access and
                # reject subclasses carrying additional array semantics.
                if type(owner) is np.ndarray:
                    owner = np.ndarray.base.__get__(owner)
                elif type(owner) is memoryview:
                    owner = owner.obj
                else:
                    break
            # Do not follow arbitrary array-interface wrappers or their base
            # getters: an opaque owner may retain a hidden masked array.
            # Ordinary owned arrays and exact raw-byte buffers are sufficient
            # for the synthetic native equity contract.
            if (
                owner is not None
                and type(owner) is not bytes
                and type(owner) is not bytearray
            ):
                raise ContractViolation(error)
        elif isinstance(value, (list, tuple)) and type(value) in (list, tuple):
            sequence: tuple[object, ...] = tuple(value)
            for mark in sequence:
                if type(mark) not in (int, float):
                    if not isinstance(mark, np.generic):
                        raise ContractViolation(error)
                    dtype = np.generic.dtype.__get__(mark)
                    if (
                        type(mark) is not dtype.type
                        or dtype.kind not in "iuf"
                        or dtype.itemsize > 8
                    ):
                        raise ContractViolation(error)
                if (
                    isinstance(mark, (int, np.integer))
                    and abs(int(mark)) > exact_integer_limit
                ):
                    raise ContractViolation(error)
            value = sequence
        else:
            raise ContractViolation(error)
        try:
            marks = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractViolation(error) from exc
        if (
            marks.dtype.kind not in "iuf"
            or marks.dtype.itemsize > 8
            or not np.isfinite(marks).all()
        ):
            raise ContractViolation(error)
        # Every integer dtype narrower than 64 bits fits exactly in binary64.
        # Compare the precision bounds only where they are representable.
        if marks.dtype.kind in "iu" and marks.dtype.itemsize == 8:
            if np.any(marks > exact_integer_limit) or (
                marks.dtype.kind == "i" and np.any(marks < -exact_integer_limit)
            ):
                raise ContractViolation(error)
        return marks.astype(np.float64, copy=True)

    if type(observed_rows) is not int or observed_rows <= 0:
        raise ContractViolation(
            "trend_breadth: as-of observation count must be positive"
        )
    a, b = finite_marks(full), finite_marks(truncated)
    expected = observed_rows + 1  # initial cash row plus decision marks
    if (
        a.ndim != 1
        or b.ndim != 1
        or len(a) < expected
        or len(b) != expected
        or not np.allclose(a[:expected], b, rtol=0, atol=1e-12)
    ):
        raise ContractViolation(
            "trend_breadth: noncausal no-notice write-off changes as-of equity "
            "before the 24-hour delisting confirmation"
        )


def check_retired_studies(root: Path) -> None:
    """The retirement inventory remains authoritative; never execute archived studies."""
    import json
    from pathlib import PurePosixPath

    try:
        record = json.loads(
            _read(root, "report/legacy-research-retirement-20261126.json")
        )
        if record["schema"] != "legacy_research_bulk_retirement_v1":
            raise ContractViolation("unsupported legacy retirement manifest")
        rows = record["records"]
        paths = [row["path"] for row in rows]
        if len(paths) != len(set(paths)) or not paths:
            raise ContractViolation("invalid retirement path roster")
        for name in paths:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or p.as_posix() != name:
                raise ContractViolation("unsafe retirement path")
            target = root / name
            if target.exists() or target.is_symlink():
                raise ContractViolation(f"retired research path returned: {name}")
        inventory = record["inventory"]
        states = {row["path"]: row["status"] for row in inventory}
        if len(inventory) != 170 or len(states) != 170:
            raise ContractViolation("incomplete retirement inventory")
        counts = {
            status: sum(value == status for value in states.values())
            for status in ("retired_before", "retired_by_batch", "shared")
        }
        if counts != record["inventory_counts"]:
            raise ContractViolation("retirement classifications disagree")
        shared = set(record["retained_inventory"])
        if shared != {name for name, status in states.items() if status == "shared"}:
            raise ContractViolation("shared inventory roster differs")
        if not set(states) - shared <= set(paths):
            raise ContractViolation("retired inventory missing archive identity")
        for name in shared:
            if not (root / name).is_file():
                raise ContractViolation(f"shared inventory path missing: {name}")
        if record["inventory_total"] != 170 or record["inventory_counts"] != {
            "retired_before": 23,
            "retired_by_batch": 137,
            "shared": 10,
        }:
            raise ContractViolation("retirement inventory count mismatch")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ContractViolation):
            raise
        raise ContractViolation("invalid retirement manifest") from error


def check_all(root: Path) -> None:
    check_registered_protocol_coverage(root)
    check_retired_studies(root)
    for binding in PROTOCOLS:
        check_protocol_digest(root, binding)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        check_all(root)
    except (ContractViolation, ValueError) as error:
        parser.exit(
            status=1, message=f"Source/docs semantic contract FAILED: {error}\n"
        )
    print("Source/docs semantic contract OK (registered invariants only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
