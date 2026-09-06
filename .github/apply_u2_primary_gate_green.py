from __future__ import annotations

import argparse
from pathlib import Path

SOURCE_PATH = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
TEST_PATH = Path("tests/workflows/test_universal_trade_rl_u2_selection_gate.py")


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one patch anchor, found {count}")
    return text.replace(old, new, 1)


def apply_test_patch() -> None:
    text = TEST_PATH.read_text(encoding="utf-8")
    marker = "def test_u2_primary_cell_gate_uses_preregistered_threshold_payload("
    if marker in text:
        raise SystemExit("threshold-binding test already present; refusing duplicate patch")

    anchor = "def test_u2_primary_cell_gate_evidence_rejects_pass_state_tampering() -> None:\n"
    test_block = '''def test_u2_primary_cell_gate_uses_preregistered_threshold_payload(
    monkeypatch,
) -> None:
    from trade_rl.workflows import universal_trade_rl_u2_contract

    module = _module()
    summary = _inclusive_boundary_summary()
    thresholds = dict(universal_trade_rl_u2_contract._selection_thresholds_payload())
    thresholds["turnover_p95_per_day_max_inclusive"] = 0.999999
    monkeypatch.setattr(
        universal_trade_rl_u2_contract,
        "_selection_thresholds_payload",
        lambda: thresholds,
    )

    result = module.evaluate_universal_trade_rl_u2_primary_cell_gate(summary=summary)

    assert result.selection_thresholds_digest == content_digest(thresholds)
    assert result.rejection_reasons == (_TURNOVER_REASON,)
    assert result.passed is False


'''
    text = _replace_once(
        text,
        anchor,
        test_block + anchor,
        label="threshold-binding test insertion",
    )
    TEST_PATH.write_text(text, encoding="utf-8")


def apply_source_patch() -> None:
    text = SOURCE_PATH.read_text(encoding="utf-8")
    if "def evaluate_universal_trade_rl_u2_primary_cell_gate(" in text:
        raise SystemExit("primary gate already implemented; refusing duplicate patch")

    text = _replace_once(
        text,
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass, field\n",
        label="dataclasses import",
    )
    text = _replace_once(
        text,
        "from trade_rl.domain.common import require_sha256\n"
        "from trade_rl.workflows.universal_trade_rl_u2_contract import U2_TRAINING_SEEDS\n",
        "from trade_rl.domain.common import require_sha256\n"
        "from trade_rl.workflows import universal_trade_rl_u2_contract as u2_contract\n"
        "from trade_rl.workflows.universal_trade_rl_u2_contract import U2_TRAINING_SEEDS\n",
        label="U2 contract import",
    )

    gate_block = '''U2_PRIMARY_SELECTION_CELL_GATE_SCHEMA: Final = (
    "universal_trade_rl_u2_primary_selection_cell_gate_v1"
)
_U2_PRIMARY_SELECTION_MANDATORY_CELLS: Final = ("B", "C1", "C2", "D1", "D2")


def _u2_primary_selection_rejection_reasons(
    *,
    summary: UniversalTradeRLU2SelectionMetricSummary,
    thresholds: dict[str, object],
) -> tuple[str, ...]:
    gross_min = _finite(
        thresholds["symbol_balanced_gross_wealth_min_exclusive"],
        field="U2 primary gate balanced gross wealth threshold",
    )
    net_min = _finite(
        thresholds["symbol_balanced_net_wealth_min_exclusive"],
        field="U2 primary gate balanced net wealth threshold",
    )
    median_min = _finite(
        thresholds["median_symbol_net_wealth_min_inclusive"],
        field="U2 primary gate median symbol wealth threshold",
    )
    minimum_min = _finite(
        thresholds["minimum_symbol_net_wealth_min_inclusive"],
        field="U2 primary gate minimum symbol wealth threshold",
    )
    positive_fraction_min = _finite(
        thresholds["positive_net_scope_fraction_min_inclusive"],
        field="U2 primary gate positive fraction threshold",
    )
    cvar_min = _finite(
        thresholds["scope_net_return_cvar10_min_inclusive"],
        field="U2 primary gate CVaR10 threshold",
    )
    turnover_max = _finite(
        thresholds["turnover_p95_per_day_max_inclusive"],
        field="U2 primary gate turnover threshold",
    )
    execution_fraction_required = _finite(
        thresholds["meaningful_execution_symbol_fraction_required"],
        field="U2 primary gate execution fraction threshold",
    )
    hard_risk_required = _non_negative_int(
        thresholds["hard_risk_violation_count_required"],
        field="U2 primary gate hard-risk threshold",
    )
    unexplained_rejection_required = _non_negative_int(
        thresholds["unexplained_execution_reject_count_required"],
        field="U2 primary gate unexplained rejection threshold",
    )
    retention_min = _finite(
        thresholds["positive_gross_log_growth_net_retention_min_inclusive"],
        field="U2 primary gate positive-gross retention threshold",
    )

    reasons: list[str] = []
    if summary.symbol_balanced_gross_wealth <= gross_min:
        reasons.append("symbol_balanced_gross_wealth_not_above_cash")
    if summary.symbol_balanced_net_wealth <= net_min:
        reasons.append("symbol_balanced_net_wealth_not_above_cash")
    if summary.median_symbol_net_wealth < median_min:
        reasons.append("median_symbol_net_wealth_below_cash")
    if summary.minimum_symbol_net_wealth < minimum_min:
        reasons.append("minimum_symbol_net_wealth_below_cash")
    if summary.positive_net_scope_fraction < positive_fraction_min:
        reasons.append("positive_net_scope_fraction_below_0_50")
    if summary.scope_net_return_cvar10 < cvar_min:
        reasons.append("scope_net_return_cvar10_below_minus_0_01")
    if summary.turnover_per_day_p95 > turnover_max:
        reasons.append("turnover_p95_per_day_above_1_0")
    if summary.meaningful_execution_symbol_fraction != execution_fraction_required:
        reasons.append("meaningful_execution_symbol_fraction_not_complete")
    if summary.hard_risk_violation_count != hard_risk_required:
        reasons.append("hard_risk_violation_count_nonzero")
    if (
        summary.unexplained_execution_rejection_count
        != unexplained_rejection_required
    ):
        reasons.append("unexplained_execution_rejection_count_nonzero")
    if summary.symbol_balanced_gross_log_growth > 0.0 and (
        summary.positive_gross_log_growth_retention is None
        or summary.positive_gross_log_growth_retention < retention_min
    ):
        reasons.append("positive_gross_log_growth_retention_below_0_50")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2PrimarySelectionCellGateEvidence:
    """Digest-bound primary-seed core-gate evidence for one mandatory U2 cell."""

    summary: UniversalTradeRLU2SelectionMetricSummary
    selection_thresholds_digest: str = field(init=False)
    rejection_reasons: tuple[str, ...] = field(init=False)
    passed: bool = field(init=False)
    schema_version: str = U2_PRIMARY_SELECTION_CELL_GATE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.summary, UniversalTradeRLU2SelectionMetricSummary):
            raise TypeError("U2 primary Selection gate requires a metric summary")
        if self.summary.training_seed != u2_contract.U2_PRIMARY_CANDIDATE_SEED:
            raise ValueError("U2 primary Selection gate requires primary training seed 0")
        if self.summary.cell not in _U2_PRIMARY_SELECTION_MANDATORY_CELLS:
            raise ValueError(
                "U2 primary Selection gate requires mandatory cell B/C1/C2/D1/D2"
            )
        if self.schema_version != U2_PRIMARY_SELECTION_CELL_GATE_SCHEMA:
            raise ValueError("unsupported U2 primary Selection cell gate schema")

        thresholds = u2_contract._selection_thresholds_payload()
        threshold_digest = content_digest(thresholds)
        reasons = _u2_primary_selection_rejection_reasons(
            summary=self.summary,
            thresholds=thresholds,
        )
        object.__setattr__(
            self,
            "selection_thresholds_digest",
            threshold_digest,
        )
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "passed", not reasons)

        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 primary Selection cell gate digest")
            if self.digest != expected_digest:
                raise ValueError("U2 primary Selection cell gate digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @property
    def cell(self) -> str:
        return self.summary.cell

    @property
    def training_seed(self) -> int:
        return self.summary.training_seed

    @property
    def summary_digest(self) -> str:
        return self.summary.digest

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "cell": self.cell,
            "training_seed": self.training_seed,
            "summary_digest": self.summary_digest,
            "selection_thresholds_digest": self.selection_thresholds_digest,
            "rejection_reasons": self.rejection_reasons,
            "passed": self.passed,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_universal_trade_rl_u2_primary_cell_gate(
    *,
    summary: UniversalTradeRLU2SelectionMetricSummary,
) -> UniversalTradeRLU2PrimarySelectionCellGateEvidence:
    """Evaluate the preregistered 11-condition primary core gate for one cell."""

    return UniversalTradeRLU2PrimarySelectionCellGateEvidence(summary=summary)


'''
    text = _replace_once(
        text,
        "@dataclass(frozen=True, slots=True)\nclass UniversalTradeRLU2PairedExcessPoint:\n",
        gate_block
        + "@dataclass(frozen=True, slots=True)\nclass UniversalTradeRLU2PairedExcessPoint:\n",
        label="primary gate insertion",
    )
    text = _replace_once(
        text,
        '    "U2_SELECTION_LEAF_METRICS_SCHEMA",\n',
        '    "U2_PRIMARY_SELECTION_CELL_GATE_SCHEMA",\n'
        '    "U2_SELECTION_LEAF_METRICS_SCHEMA",\n',
        label="schema export",
    )
    text = _replace_once(
        text,
        '    "UniversalTradeRLU2PairedExcessPoint",\n',
        '    "UniversalTradeRLU2PairedExcessPoint",\n'
        '    "UniversalTradeRLU2PrimarySelectionCellGateEvidence",\n',
        label="evidence export",
    )
    text = _replace_once(
        text,
        '    "build_universal_trade_rl_u2_selection_leaf_metrics",\n',
        '    "build_universal_trade_rl_u2_selection_leaf_metrics",\n'
        '    "evaluate_universal_trade_rl_u2_primary_cell_gate",\n',
        label="evaluator export",
    )
    SOURCE_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("tests", "source"))
    args = parser.parse_args()
    if args.phase == "tests":
        apply_test_patch()
    else:
        apply_source_patch()


if __name__ == "__main__":
    main()
