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
    ProtocolBinding(
        "agy_clean_all_forward_v1",
        "tools/agy_clean_all_forward.py",
        "docs/specs/agy-clean-all-forward.md",
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

    # Validate the executable AST rather than comments, docstrings or untrusted
    # source text: both run and coverage must return directly before the clock.
    try:
        tree = ast.parse(source, filename="tools/agy_anchor_oos.py")
    except SyntaxError as error:
        raise ContractViolation("historical agy source cannot be parsed") from error
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    if len(functions) != 1:
        raise ContractViolation("historical agy source requires one main entrypoint")
    body = functions[0].body
    clock = [
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "moment" for t in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "now"
    ]
    if len(clock) != 1:
        raise ContractViolation(
            "historical agy source requires a single clock boundary"
        )
    expected = ast.parse(
        'if args.command in {"run", "coverage"} and rows_factory is None:\n'
        "    return EXIT_REFUSED\n"
    ).body[0]
    assert isinstance(expected, ast.If)
    gate = [
        (i, node)
        for i, node in enumerate(body[: clock[0]])
        if isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(expected.test)
    ]
    if len(gate) != 1:
        raise ContractViolation(
            "historical agy source requires the run/coverage pre-data refusal gate"
        )
    gate_idx, gate_node = gate[0]
    # Only passive output stream assignments and pure argument parser setup
    # may precede refusal; no arbitrary I/O or evaluation.
    allowed_preflight = {"out", "err", "args"}
    seen: set[str] = set()
    for node in body[:gate_idx]:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or node.targets[0].id not in allowed_preflight
            or node.targets[0].id in seen
        ):
            raise ContractViolation(
                "historical agy pre-data refusal must precede all source reads"
            )
        name = node.targets[0].id
        seen.add(name)
        if name != "args" and any(
            isinstance(expr, ast.Call) for expr in ast.walk(node.value)
        ):
            raise ContractViolation(
                "historical agy pre-data refusal must precede all source reads"
            )
        if name == "args":
            expr = node.value
            if not (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Attribute)
                and expr.func.attr == "parse_args"
                and isinstance(expr.func.value, ast.Call)
                and isinstance(expr.func.value.func, ast.Name)
                and expr.func.value.func.id == "build_parser"
            ):
                raise ContractViolation(
                    "historical agy pre-data refusal must precede all source reads"
                )
    if "args" not in seen:
        raise ContractViolation("historical agy pre-data refusal requires parsed args")
    # A matching condition is insufficient if the body can access market data
    # before returning. Permit only one literal, passive error diagnostic and
    # an unconditional direct refusal as the last statement.
    branch = list(gate_node.body)
    first_stmt = branch[0] if branch else None
    if isinstance(first_stmt, ast.Expr):
        expression = first_stmt.value
        branch.pop(0)
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and isinstance(expression.func.value, ast.Name)
            and expression.func.value.id == "err"
            and expression.func.attr == "write"
            and len(expression.args) == 1
            and isinstance(expression.args[0], ast.Constant)
            and isinstance(expression.args[0].value, str)
            and not expression.keywords
        ):
            raise ContractViolation(
                "historical agy pre-data refusal must not read inside its guard"
            )
    if (
        gate_node.orelse
        or len(branch) != 1
        or not isinstance(branch[0], ast.Return)
        or not isinstance(branch[0].value, ast.Name)
        or branch[0].value.id != "EXIT_REFUSED"
    ):
        raise ContractViolation(
            "historical agy pre-data refusal must immediately return EXIT_REFUSED"
        )


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
            while isinstance(owner, (np.ndarray, memoryview)):
                if np.ma.isMaskedArray(owner):
                    raise ContractViolation(error)
                owner = (
                    np.ndarray.base.__get__(owner)
                    if isinstance(owner, np.ndarray)
                    else owner.obj
                )
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
        if marks.dtype.kind in "iu" and (
            np.any(marks > exact_integer_limit)
            or (marks.dtype.kind == "i" and np.any(marks < -exact_integer_limit))
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


def _verified_no_notice_refusal(module: object, problems: list[str]) -> bool:
    """Check the no-notice label; it never exempts the as-of equity oracle."""
    if not any("no notice" in reason.lower() for reason in problems):
        return False
    from typing import Any

    model: Any = module
    decision = model.decide({}, problems)
    if decision.get("label") != model.LABEL_INVALID:
        raise ContractViolation(
            "trend_breadth: a no-notice stage problem must yield INVALID"
        )
    return True


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
    # An INVALID decision for empty cells does not prove the actual stage
    # refuses before replay or capital release. Always check the as-of account.
    _verified_no_notice_refusal(module, problems)

    period = len(prior_panel.times)
    # The oracle is defined on the *entire* literal 48h observation window.
    # A shorter/nonconsecutive panel could hide earlier economic leakage by
    # comparing only its available prefix.
    if period != observed // 4 or len(full_panel.times) != hours // 4:
        raise ContractViolation(
            "trend_breadth: as-of observation prefix must be nonempty "
            "and contain the complete expected 4h clock rows"
        )
    expected_times = int(start.timestamp()) + 4 * 3_600 * np.arange(1, hours // 4 + 1)
    if not (
        np.array_equal(full_panel.times, expected_times)
        and np.array_equal(prior_panel.times, expected_times[:period])
    ):
        raise ContractViolation(
            "trend_breadth: as-of clock times differ from declared UTC stage"
        )
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
