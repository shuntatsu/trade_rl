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
from datetime import UTC, datetime, timedelta
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
PROTOCOLS = (
    ProtocolBinding(
        "trend_breadth_v1", "tools/trend_breadth.py", "docs/specs/trend-breadth.md"
    ),
    ProtocolBinding(
        "trend_exit_arms_v1",
        "tools/trend_exit_arms_forward.py",
        "docs/specs/trend-exit-arms-forward.md",
    ),
    ProtocolBinding(
        "mtf_combo_v1", "tools/mtf_combo_forward.py", "docs/specs/mtf-combo-forward.md"
    ),
    ProtocolBinding(
        "btc_trend_ensemble_v1",
        "tools/btc_trend_ensemble.py",
        "docs/specs/btc-trend-ensemble.md",
    ),
)

_DIGEST_LINE = re.compile(r"^Protocol digest: `([0-9a-f]{64})`[ \t]*$", re.MULTILINE)


def _read(root: Path, path: str) -> str:
    try:
        return (root / path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ContractViolation(
            f"{path}: cannot read canonical source/document"
        ) from error


def check_historical_agy_status(root: Path) -> None:
    """Prevent an older PR index from erasing the actual consumed INVALID stage."""
    index = _read(root, "docs/README.md")
    spec = _read(root, "docs/specs/agy-anchor-realistic.md")
    status = _read(root, "docs/research/current-status.md")
    source = _read(root, "tools/agy_anchor_oos.py")

    entries = [
        line for line in index.splitlines() if "(specs/agy-anchor-realistic.md)" in line
    ]
    if (
        len(entries) != 1
        or "BACKCAST_2020 INVALID" not in entries[0]
        or "HOLD" not in entries[0]
        or "FORWARD not authorized" not in entries[0]
        or "not yet run" in entries[0].lower()
    ):
        raise ContractViolation(
            "README agy index must retain BACKCAST_2020 INVALID, HOLD and "
            "FORWARD not authorized instead of a stale not-yet-run claim"
        )
    if (
        "Status: **Historical v1 / economic execution HOLD" not in spec
        or "INVALID BACKCAST_2020" not in spec
        or "AGY v1 is a separate historical INVALID result" not in status
    ):
        raise ContractViolation(
            "historical agy spec/status must keep the observed INVALID evidence"
        )

    # Source is authoritative: the ordinary run and coverage CLI must refuse
    # before even reading the clock or injected market-data factory.
    signature = re.search(r"(?m)^def main\(", source)
    tail = source[signature.end() :] if signature else ""
    pre_clock = tail.partition("    moment = now()")[0]
    guard = 'if args.command in {"run", "coverage"} and rows_factory is None:'
    if not tail or not pre_clock or guard not in pre_clock:
        raise ContractViolation(
            "historical agy source requires the run/coverage pre-data refusal gate"
        )
    block = pre_clock.partition(guard)[2]
    if "return EXIT_REFUSED" not in block:
        raise ContractViolation("historical agy source must fail closed before data")


def check_learning_lab_links(root: Path) -> None:
    """Ensure both proposed Lab surfaces remain indexed when introduced."""
    src = root / "tools/learning_lab.py"
    specs = ("learning-lab-registry.md", "anchor-rl-execution-dev.md")
    paths = tuple(root / "docs/specs" / name for name in specs)
    if not src.is_file() and not any(path.is_file() for path in paths):
        return
    if not src.is_file() or not all(path.is_file() for path in paths):
        raise ContractViolation("Learning Lab index/source introduction is incomplete")
    index = _read(root, "docs/README.md")
    for name in specs:
        if f"(specs/{name})" not in index:
            raise ContractViolation(f"Learning Lab index missing specs/{name}")


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
    if f"`{binding.protocol_id}`" not in doc.splitlines()[0]:
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

    a, b = np.asarray(full, dtype=float), np.asarray(truncated, dtype=float)
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


def check_breadth_asof(root: Path) -> bool:
    """Assert no past settlement can be inferred from later absent trading hours.

    This synthetic E2E uses actual panel-building and actual-unit accounting,
    not a source-text keyword or a copied spec formula.
    """
    source, spec = root / "tools/trend_breadth.py", root / "docs/specs/trend-breadth.md"
    if not source.is_file() and not spec.is_file():
        return False
    if not source.is_file() or not spec.is_file():
        raise ContractViolation("trend_breadth causal source/spec pair is incomplete")
    import numpy as np

    module = importlib.import_module("tools.trend_breadth")
    if Path(getattr(module, "__file__", "")).resolve() != source.resolve():
        raise ContractViolation("trend_breadth synthetic oracle imported wrong source")

    start = datetime(2021, 1, 1, tzinfo=UTC)
    # Same observed 48-hour prefix. Only the longer history knows that
    # a continuous 24-hour dead run follows the last traded bar.
    hours, observed = 72, 48
    bars = {
        "open": np.r_[np.full(observed, 100.0), np.full(hours - observed, np.nan)],
        "high": np.r_[np.full(observed, 100.0), np.full(hours - observed, np.nan)],
        "low": np.r_[np.full(observed, 100.0), np.full(hours - observed, np.nan)],
        "close": np.r_[np.full(observed, 100.0), np.full(hours - observed, np.nan)],
        "qv": np.r_[np.full(observed, 1e9), np.zeros(hours - observed)],
        "vol": np.r_[np.full(observed, 1e6), np.zeros(hours - observed)],
    }
    symbol = ("ZZZUSDT",)
    full = module.Stage("SYNTH_FULL", start, start + timedelta(hours=hours))
    asof = module.Stage("SYNTH_ASOF", start, start + timedelta(hours=observed))
    full_panel, info = module.build_panel({symbol[0]: bars}, full, {}, {}, symbol)
    prior_panel, _ = module.build_panel(
        {symbol[0]: {k: v[:observed].copy() for k, v in bars.items()}},
        asof,
        {},
        {},
        symbol,
    )
    # If a revised source explicitly refuses an un-noticed delisting as a
    # stage error, it must do so before a replay or any receipt publication.
    # Otherwise, the two as-of-equivalent prefixes must reconcile.
    problems = module.panel_problems(full_panel, info)
    if any("no notice" in problem.lower() for problem in problems):
        return True

    period = len(prior_panel.times)
    for field in ("close", "mark_px", "fill_qv", "regime", "vol", "fund", "forced"):
        if not np.array_equal(
            getattr(full_panel, field)[:period],
            getattr(prior_panel, field),
            equal_nan=True,
        ):
            raise ContractViolation(
                f"trend_breadth: as-of input prefix drift in {field}"
            )

    full_result = module.simulate_weights(
        full_panel, np.full((full_panel.bars, 1), 0.5), 0.0, 0
    )
    prior_result = module.simulate_weights(
        prior_panel, np.full((prior_panel.bars, 1), 0.5), 0.0, 0
    )
    if not full_result["valid"] or not prior_result["valid"]:
        raise ContractViolation("trend_breadth: synthetic book became invalid")
    _verify_asof_equity(
        full_result["equity"], prior_result["equity"], observed_rows=period
    )
    return True


def check_all(root: Path) -> None:
    check_registered_protocol_coverage(root)
    check_historical_agy_status(root)
    check_learning_lab_links(root)
    for binding in PROTOCOLS:
        check_protocol_digest(root, binding)
    check_breadth_asof(root)


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
