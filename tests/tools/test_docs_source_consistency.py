"""Source-bound documentation consistency probes (synthetic, no market data)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.docs_source_consistency import (
    ContractViolation,
    ProtocolBinding,
    check_all,
    check_protocol_digest,
)

ROOT = Path(__file__).resolve().parents[2]


def test_current_repo_is_source_doc_consistent() -> None:
    check_all(ROOT)


def test_optional_protocol_digest_and_spec_are_bound(tmp_path: Path) -> None:
    binding = ProtocolBinding(
        protocol_id="trend_breadth_v1",
        source="tools/trend_breadth.py",
        spec="docs/specs/trend-breadth.md",
    )
    assert (
        check_protocol_digest(
            tmp_path, binding, lambda _p: ("trend_breadth_v1", "a" * 64)
        )
        is False
    )
    p = tmp_path / binding.spec
    p.parent.mkdir(parents=True)
    p.write_text(
        "# Trend breadth (`trend_breadth_v1`)\nProtocol digest: `" + "a" * 64 + "`\n"
    )
    with pytest.raises(ContractViolation, match="missing source"):
        check_protocol_digest(
            tmp_path, binding, lambda _p: ("trend_breadth_v1", "a" * 64)
        )
    s = tmp_path / binding.source
    s.parent.mkdir(parents=True)
    s.write_text("PROTOCOL_ID = 'trend_breadth_v1'\n")
    assert check_protocol_digest(
        tmp_path, binding, lambda _p: ("trend_breadth_v1", "a" * 64)
    )
    with pytest.raises(ContractViolation, match="digest mismatch"):
        check_protocol_digest(
            tmp_path, binding, lambda _p: ("trend_breadth_v1", "b" * 64)
        )
    with pytest.raises(ContractViolation, match="protocol_id"):
        check_protocol_digest(tmp_path, binding, lambda _p: ("unrelated_v1", "a" * 64))


def test_ci_checks_semantics_on_fast_and_full() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert ci.count("Docs/source semantic contract") == 2
    assert ci.count("python -m tools.docs_source_consistency --check") == 2
    assert ci.count("tools/docs_source_consistency.py") == 2
    assert "tests/tools/test_docs_source_consistency.py" in ci


def test_project_documentation_describes_checked_scope_and_limits() -> None:
    root_agents = (ROOT / "AGENTS.md").read_text()
    docs_agents = (ROOT / "docs/AGENTS.md").read_text()
    docs_index = (ROOT / "docs/README.md").read_text()
    assert "tools.docs_source_consistency --check" in root_agents
    assert "tools.docs_source_consistency --check" in docs_agents
    assert "tools.docs_source_consistency --check" in docs_index
    assert "semantic" in docs_agents
    assert "部分" in docs_index


def test_unregistered_new_harness_is_never_silently_unchecked(tmp_path: Path) -> None:
    from tools.docs_source_consistency import check_registered_protocol_coverage

    path = tmp_path / "tools/new_research.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "PROTOCOL_ID = 'new_v1'\ndef protocol_digest():\n    return 'a' * 64\n"
    )
    with pytest.raises(ContractViolation, match="unregistered protocol_digest harness"):
        check_registered_protocol_coverage(tmp_path)
    path.unlink()
    check_registered_protocol_coverage(tmp_path)


def test_asof_equity_oracle_rejects_only_future_dependent_marks() -> None:
    from tools.docs_source_consistency import _verify_asof_equity

    _verify_asof_equity([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], observed_rows=2)
    with pytest.raises(ContractViolation, match="noncausal no-notice"):
        _verify_asof_equity([1.0, 1.0, 0.975], [1.0, 1.0, 1.0], 2)
    with pytest.raises(ContractViolation, match="noncausal no-notice"):
        _verify_asof_equity([1.0, 1.0], [1.0, 1.0, 1.0], 2)


@pytest.mark.parametrize(
    "case",
    [
        "positive_inf",
        "negative_inf",
        "nan",
        "future_inf",
        "masked_full",
        "masked_prior",
        "masked_all_visible",
        "masked_scalar",
        "masked_base",
        "masked_memoryview",
        "complex_array",
        "object_array",
        "numeric_text",
        "boolean",
    ],
)
def test_asof_equity_oracle_refuses_invalid_marks(case: str) -> None:
    import numpy as np

    from tools.docs_source_consistency import _verify_asof_equity

    full: object = np.array([1.0, 1.0, 1.0, 1.0])
    prior: object = np.array([1.0, 1.0, 1.0])
    if case in ("positive_inf", "negative_inf", "nan"):
        bad = {"positive_inf": np.inf, "negative_inf": -np.inf, "nan": np.nan}[case]
        full = [1.0, bad, 1.0, 1.0]
        prior = [1.0, bad, 1.0]
    elif case == "future_inf":
        full = [1.0, 1.0, 1.0, np.inf]
    elif case == "masked_full":
        full = np.ma.array(full, mask=[False, True, False, False])
    elif case == "masked_prior":
        prior = np.ma.array(prior, mask=[False, True, False])
    elif case == "masked_all_visible":
        full = np.ma.array(full, mask=False)
    elif case == "masked_scalar":
        prior = [1.0, np.ma.array(1.0, mask=False), 1.0]
    elif case == "masked_base":
        masked = np.ma.array([1.0, 1.0, 1.0], mask=[False, True, False])
        prior = masked.astype(np.float64).view(np.ndarray)
        assert np.ma.isMaskedArray(prior.base)
    elif case == "masked_memoryview":
        prior = memoryview(np.ma.array([1.0, 1.0, 1.0], mask=[False, True, False]))
    elif case == "complex_array":
        prior = np.array([1.0, 1.0 + 3j, 1.0])
    elif case == "object_array":
        prior = np.array([1.0, 1.0, 1.0], dtype=object)
    elif case == "numeric_text":
        prior = ["1.0", "1.0", "1.0"]
    elif case == "boolean":
        prior = [True, True, True]
    else:
        pytest.fail(f"unknown case: {case}")
    with pytest.raises(ContractViolation):
        _verify_asof_equity(full, prior, observed_rows=2)


@pytest.mark.parametrize("observed_rows", [-1, 0, True])
def test_asof_equity_oracle_rejects_vacuous_observation(observed_rows: int) -> None:
    from tools.docs_source_consistency import _verify_asof_equity

    with pytest.raises(ContractViolation):
        _verify_asof_equity([1.0, 1.0], [1.0] * (observed_rows + 1), observed_rows)


@pytest.mark.parametrize("representation", ["integer_array", "mixed_sequence"])
def test_asof_equity_oracle_refuses_lossy_integer_marks(representation: str) -> None:
    import numpy as np

    from tools.docs_source_consistency import _verify_asof_equity

    if representation == "integer_array":
        full: object = np.array([2**53, 2**53 + 1], dtype=np.int64)
        prior: object = np.array([2**53, 2**53], dtype=np.int64)
    else:
        full = [2**53 + 1, 1.0]
        prior = [2**53, 1.0]
    with pytest.raises(ContractViolation):
        _verify_asof_equity(full, prior, 1)


@pytest.mark.parametrize(
    "dtype",
    ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"],
)
def test_asof_equity_oracle_accepts_integer_dtype_boundaries(dtype: str) -> None:
    import numpy as np

    from tools.docs_source_consistency import _verify_asof_equity

    info = np.iinfo(dtype)
    marks = np.array(
        [max(int(info.min), -(2**53)), 1, min(int(info.max), 2**53)], dtype=dtype
    )
    # The exact representability limit is inclusive; narrow integer types
    # have no marks outside the binary64 exact integer interval.
    _verify_asof_equity(marks, marks.copy(), 2)
    _verify_asof_equity(marks, [int(mark) for mark in marks], 2)
    different = marks.copy()
    different[1] = 2
    with pytest.raises(ContractViolation, match="noncausal no-notice"):
        _verify_asof_equity(marks, different, 2)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        ("int64", 2**53 + 1),
        ("int64", -(2**53) - 1),
        ("int64", -(2**63)),
        ("uint64", 2**53 + 1),
        ("uint64", 2**64 - 1),
    ],
)
def test_asof_equity_oracle_rejects_wide_integer_dtype_boundaries(
    dtype: str, value: int
) -> None:
    import numpy as np

    from tools.docs_source_consistency import _verify_asof_equity

    marks = np.array([1, value], dtype=dtype)
    with pytest.raises(ContractViolation, match="finite binary64-compatible"):
        _verify_asof_equity(marks, marks.copy(), 1)
    with pytest.raises(ContractViolation, match="finite binary64-compatible"):
        _verify_asof_equity([np.dtype(dtype).type(value), 1], [1, 1], 1)


@pytest.mark.parametrize("side", ["full", "prior"])
@pytest.mark.parametrize("wrapper", ["masked_stride", "opaque_stride", "subclass"])
def test_asof_equity_oracle_refuses_opaque_array_owners(
    side: str, wrapper: str
) -> None:
    import numpy as np

    from tools.docs_source_consistency import _verify_asof_equity

    if wrapper == "masked_stride":
        masked = np.ma.array([1.0, 1.0, 1.0], mask=[False, True, False])
        marks = np.lib.stride_tricks.as_strided(np.asarray(memoryview(masked)))
    elif wrapper == "opaque_stride":
        # A benign opaque owner is outside the closed native vector contract
        # too: arbitrary base getters cannot establish missingness provenance.
        marks = np.lib.stride_tricks.as_strided(np.ones(3))
    else:

        class ForeignArray(np.ndarray):
            pass

        owner = np.ones(3).view(ForeignArray)
        marks = np.asarray(memoryview(owner))
        assert isinstance(marks.base, memoryview)
        assert type(marks.base.obj) is ForeignArray
    assert type(marks) is np.ndarray
    good = np.ones(3)
    full, prior = (marks, good) if side == "full" else (good, marks)
    with pytest.raises(ContractViolation, match="finite binary64-compatible"):
        _verify_asof_equity(full, prior, 2)


def test_asof_equity_oracle_never_calls_opaque_owner_attributes() -> None:
    import numpy as np

    from tools.docs_source_consistency import _verify_asof_equity

    class OpaqueOwner:
        def __init__(self) -> None:
            self.array = np.ones(3)
            self.__array_interface__ = self.array.__array_interface__
            self.guard = False

        @property
        def __class__(self) -> type:
            if self.guard:
                pytest.fail("opaque owner class getter called")
            return OpaqueOwner

        @property
        def base(self) -> object:
            pytest.fail("opaque owner base getter called")

    owner = OpaqueOwner()
    marks = np.asarray(owner)
    assert marks.base is owner
    owner.guard = True
    with pytest.raises(ContractViolation, match="finite binary64-compatible"):
        _verify_asof_equity(marks, np.ones(3), 2)


@pytest.mark.parametrize("backing", ["owned", "strided", "bytes", "bytearray"])
def test_asof_equity_oracle_accepts_closed_native_array_owners(backing: str) -> None:
    import numpy as np

    from tools.docs_source_consistency import _verify_asof_equity

    if backing == "owned":
        marks = np.ones(3)
    elif backing == "strided":
        marks = np.ones(6)[::2]
    elif backing == "bytes":
        marks = np.frombuffer(np.ones(3).tobytes(), dtype=np.float64)
    else:
        marks = np.frombuffer(bytearray(np.ones(3).tobytes()), dtype=np.float64)
    _verify_asof_equity(marks, np.ones(3), 2)
    _verify_asof_equity(np.ones(3), marks, 2)


def test_asof_equity_oracle_accepts_finite_native_vectors() -> None:
    import numpy as np

    from tools.docs_source_consistency import _verify_asof_equity

    _verify_asof_equity(np.array([1.0, 0.9, 1.1, 0.8]), (1.0, 0.9, 1.1), 2)
    _verify_asof_equity([1, np.float64(0.9), 1.1], np.array([1.0, 0.9, 1.1]), 2)


def test_empty_registered_spec_is_rejected_as_contract_violation(
    tmp_path: Path,
) -> None:
    binding = ProtocolBinding(
        protocol_id="trend_breadth_v1",
        source="tools/trend_breadth.py",
        spec="docs/specs/trend-breadth.md",
    )
    src = tmp_path / binding.source
    src.parent.mkdir(parents=True)
    src.write_text("PROTOCOL_ID = 'trend_breadth_v1'\n")
    spec = tmp_path / binding.spec
    spec.parent.mkdir(parents=True)
    spec.write_bytes(b"")
    with pytest.raises(ContractViolation, match="protocol_id heading mismatch"):
        check_protocol_digest(
            tmp_path, binding, lambda _b: ("trend_breadth_v1", "a" * 64)
        )


def test_installed_digest_rejects_other_checkout_or_spec_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import ModuleType

    from tools.docs_source_consistency import _installed_digest

    binding = ProtocolBinding(
        "trend_breadth_v1",
        "tools/trend_breadth.py",
        "docs/specs/trend-breadth.md",
    )
    src = tmp_path / binding.source
    src.parent.mkdir(parents=True)
    src.write_text("PROTOCOL_ID = 'trend_breadth_v1'\n")
    fake = ModuleType("tools.trend_breadth")
    fake.__file__ = str(tmp_path / "other-checkout.py")
    fake.PROTOCOL_ID = binding.protocol_id  # type: ignore[attr-defined]
    fake.SPEC_RELPATH = binding.spec  # type: ignore[attr-defined]
    fake.protocol_digest = lambda: "a" * 64  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools.trend_breadth", fake)
    with pytest.raises(ContractViolation, match="outside checkout"):
        _installed_digest(tmp_path, binding)
    fake.__file__ = str(src)
    fake.SPEC_RELPATH = "docs/specs/other.md"  # type: ignore[attr-defined]
    with pytest.raises(ContractViolation, match="spec path"):
        _installed_digest(tmp_path, binding)
    fake.SPEC_RELPATH = binding.spec  # type: ignore[attr-defined]
    assert _installed_digest(tmp_path, binding) == (binding.protocol_id, "a" * 64)
