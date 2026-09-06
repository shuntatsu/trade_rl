from __future__ import annotations

from pathlib import Path


SELECTION = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
FINAL_TEST = Path("tests/workflows/test_universal_trade_rl_u2_selection_final.py")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def replace_in_region(
    text: str,
    *,
    start: str,
    end: str,
    old: str,
    new: str,
    label: str,
) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise SystemExit(f"{label}: start marker missing")
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        raise SystemExit(f"{label}: end marker missing")
    region = text[start_index:end_index]
    count = region.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one regional anchor, found {count}")
    region = region.replace(old, new, 1)
    return text[:start_index] + region + text[end_index:]


def patch_selection() -> None:
    text = SELECTION.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '''from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (\n    UniversalTradeRLU2DevelopmentLock,\n)\nfrom trade_rl.workflows.universal_trade_rl_u2_replay import (\n''',
        '''from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (\n    UniversalTradeRLU2DevelopmentLock,\n)\nfrom trade_rl.workflows.universal_trade_rl_u2_selection_pairing import (\n    U2_PAIRED_REPLAY_SCOPE_EVIDENCE_SCHEMA,\n    UniversalTradeRLU2PairedReplayScopeEvidence,\n    build_universal_trade_rl_u2_paired_replay_scope_evidence,\n)\nfrom trade_rl.workflows.universal_trade_rl_u2_replay import (\n''',
        label="selection pairing import",
    )

    segment_start = "@dataclass(frozen=True, slots=True)\nclass UniversalTradeRLU2ReducedBootstrapSegment:"
    segment_end = "\n\n@dataclass(frozen=True, slots=True)\nclass UniversalTradeRLU2DevelopmentBootstrapResult:"
    text = replace_in_region(
        text,
        start=segment_start,
        end=segment_end,
        old='''    source_window: str\n    decision_timestamps_ns: tuple[int, ...]\n    net_log_excess: tuple[float, ...]\n''',
        new='''    source_window: str\n    decision_timestamps_ns: tuple[int, ...]\n    net_log_excess: tuple[float, ...]\n    paired_scope_evidence_digests: tuple[str, ...] = ()\n''',
        label="bootstrap segment fields",
    )
    text = replace_in_region(
        text,
        start=segment_start,
        end=segment_end,
        old='''        values = tuple(\n            _finite(value, field="U2 bootstrap segment net-log excess")\n            for value in self.net_log_excess\n        )\n''',
        new='''        values = tuple(\n            _finite(value, field="U2 bootstrap segment net-log excess")\n            for value in self.net_log_excess\n        )\n        pairing_digests = tuple(self.paired_scope_evidence_digests)\n        for digest in pairing_digests:\n            require_sha256(\n                digest,\n                field="U2 bootstrap paired-scope evidence digest",\n            )\n        if len(set(pairing_digests)) != len(pairing_digests):\n            raise ValueError(\n                "U2 bootstrap paired-scope evidence digests must be unique"\n            )\n''',
        label="bootstrap segment provenance validation",
    )
    text = replace_in_region(
        text,
        start=segment_start,
        end=segment_end,
        old='''        object.__setattr__(self, "decision_timestamps_ns", timestamps)\n        object.__setattr__(self, "net_log_excess", values)\n''',
        new='''        object.__setattr__(self, "decision_timestamps_ns", timestamps)\n        object.__setattr__(self, "net_log_excess", values)\n        object.__setattr__(\n            self,\n            "paired_scope_evidence_digests",\n            pairing_digests,\n        )\n''',
        label="bootstrap segment provenance assignment",
    )
    text = replace_in_region(
        text,
        start=segment_start,
        end=segment_end,
        old='''                "source_window": self.source_window,\n                "decision_timestamps_ns": self.decision_timestamps_ns,\n                "net_log_excess": self.net_log_excess,\n''',
        new='''                "source_window": self.source_window,\n                "decision_timestamps_ns": self.decision_timestamps_ns,\n                "net_log_excess": self.net_log_excess,\n                "paired_scope_evidence_digests": (\n                    self.paired_scope_evidence_digests\n                ),\n''',
        label="bootstrap segment provenance digest",
    )

    reducer_anchor = "\ndef bootstrap_universal_trade_rl_u2_development_panel(\n"
    reducer_code = r'''

def reduce_universal_trade_rl_u2_paired_replay_evidence(
    *,
    pairs: tuple[UniversalTradeRLU2PairedReplayScopeEvidence, ...],
    expected_symbols: tuple[str, ...],
) -> tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...]:
    """Reduce exact D1/D2 candidate/cash pair artifacts for robustness."""

    resolved = tuple(pairs)
    symbols = tuple(expected_symbols)
    if not resolved:
        raise ValueError("U2 paired replay reducer requires pairing evidence")
    if any(
        not isinstance(pair, UniversalTradeRLU2PairedReplayScopeEvidence)
        for pair in resolved
    ):
        raise TypeError("U2 paired replay reducer contains invalid evidence")
    if not symbols or symbols != tuple(sorted(set(symbols))):
        raise ValueError("U2 paired replay expected symbols must be sorted and unique")
    if {pair.concrete_symbol for pair in resolved} - set(symbols):
        raise ValueError("U2 paired replay reducer contains an unexpected symbol")
    if {pair.source_window for pair in resolved} != set(U2_DEVELOPMENT_WINDOWS):
        raise ValueError("U2 paired replay reducer requires complete D1/D2 window closure")

    expected_cells = {
        "development_future_1": "D1",
        "development_future_2": "D2",
    }
    if any(
        pair.cell != expected_cells[pair.source_window]
        for pair in resolved
    ):
        raise ValueError("U2 paired replay reducer window/cell identity drifted")

    expected_keys = {
        (seed, source_window, symbol)
        for seed in U2_TRAINING_SEEDS
        for source_window in U2_DEVELOPMENT_WINDOWS
        for symbol in symbols
    }
    observed_keys = {
        (pair.training_seed, pair.source_window, pair.concrete_symbol)
        for pair in resolved
    }
    if observed_keys != expected_keys or len(resolved) != len(expected_keys):
        raise ValueError(
            "U2 paired replay reducer seed/symbol/window closure is incomplete"
        )

    identities = tuple(
        (
            pair.training_seed,
            pair.source_window,
            pair.concrete_symbol,
            pair.scope_digest,
        )
        for pair in resolved
    )
    if len(set(identities)) != len(identities):
        raise ValueError("U2 paired replay reducer contains duplicate scope identity")

    closure_by_seed = {
        seed: {
            (
                pair.source_window,
                pair.cell,
                pair.concrete_symbol,
                pair.scope_digest,
                pair.evaluation_dataset_digest,
                pair.decision_timestamps_ns,
            )
            for pair in resolved
            if pair.training_seed == seed
        }
        for seed in U2_TRAINING_SEEDS
    }
    canonical_closure = closure_by_seed[U2_TRAINING_SEEDS[0]]
    if not canonical_closure or any(
        closure_by_seed[seed] != canonical_closure
        for seed in U2_TRAINING_SEEDS[1:]
    ):
        raise ValueError("U2 paired replay scope closure drifted across training seeds")

    if any(
        len(
            {
                pair.paired_candidate_checkpoint_digest
                for pair in resolved
                if pair.training_seed == seed
            }
        )
        != 1
        for seed in U2_TRAINING_SEEDS
    ):
        raise ValueError("U2 paired replay checkpoint closure drifted within a seed")

    ordered = tuple(
        sorted(
            resolved,
            key=lambda pair: (
                pair.source_window,
                pair.training_seed,
                pair.concrete_symbol,
                pair.scope_digest,
            ),
        )
    )
    points = tuple(
        UniversalTradeRLU2PairedExcessPoint(
            training_seed=pair.training_seed,
            source_window=pair.source_window,
            concrete_symbol=pair.concrete_symbol,
            decision_timestamp_ns=timestamp,
            candidate_minus_cash_net_log_excess=value,
        )
        for pair in ordered
        for timestamp, value in zip(
            pair.decision_timestamps_ns,
            pair.candidate_minus_cash_net_log_excess,
            strict=True,
        )
    )
    reduced = reduce_universal_trade_rl_u2_development_panel(
        points=points,
        expected_symbols=symbols,
    )
    by_window_pair_digests = {
        source_window: tuple(
            pair.digest
            for pair in ordered
            if pair.source_window == source_window
        )
        for source_window in U2_DEVELOPMENT_WINDOWS
    }
    return tuple(
        UniversalTradeRLU2ReducedBootstrapSegment(
            source_window=segment.source_window,
            decision_timestamps_ns=segment.decision_timestamps_ns,
            net_log_excess=segment.net_log_excess,
            paired_scope_evidence_digests=by_window_pair_digests[
                segment.source_window
            ],
        )
        for segment in reduced
    )
'''
    text = replace_once(
        text,
        reducer_anchor,
        reducer_code + reducer_anchor,
        label="paired replay reducer insertion",
    )

    seed_bootstrap_start = (
        "@dataclass(frozen=True, slots=True)\n"
        "class UniversalTradeRLU2SeedRobustnessBootstrapResult:"
    )
    seed_bootstrap_end = "\n\ndef _bootstrap_universal_trade_rl_u2_seed_robustness(\n"
    text = replace_in_region(
        text,
        start=seed_bootstrap_start,
        end=seed_bootstrap_end,
        old='''    confidence_level: float\n    passed: bool\n    quantile_method: str = _U2_BOOTSTRAP_QUANTILE_METHOD\n''',
        new='''    confidence_level: float\n    passed: bool\n    paired_scope_evidence_digests: tuple[str, ...] = ()\n    quantile_method: str = _U2_BOOTSTRAP_QUANTILE_METHOD\n''',
        label="seed bootstrap provenance field",
    )
    text = replace_in_region(
        text,
        start=seed_bootstrap_start,
        end=seed_bootstrap_end,
        old='''        object.__setattr__(self, "source_windows", source_windows)\n        object.__setattr__(self, "segment_digests", segment_digests)\n        object.__setattr__(self, "block_lengths", block_lengths)\n''',
        new='''        object.__setattr__(self, "source_windows", source_windows)\n        object.__setattr__(self, "segment_digests", segment_digests)\n        object.__setattr__(self, "block_lengths", block_lengths)\n        pairing_digests = tuple(self.paired_scope_evidence_digests)\n        for digest in pairing_digests:\n            require_sha256(\n                digest,\n                field="U2 seed-robustness paired-scope evidence digest",\n            )\n        if len(set(pairing_digests)) != len(pairing_digests):\n            raise ValueError(\n                "U2 seed-robustness paired-scope evidence digests must be unique"\n            )\n        object.__setattr__(\n            self,\n            "paired_scope_evidence_digests",\n            pairing_digests,\n        )\n''',
        label="seed bootstrap provenance validation",
    )
    text = replace_in_region(
        text,
        start=seed_bootstrap_start,
        end=seed_bootstrap_end,
        old='''            "source_windows": self.source_windows,\n            "segment_digests": self.segment_digests,\n            "block_lengths": self.block_lengths,\n            "observed_mean": self.observed_mean,\n''',
        new='''            "source_windows": self.source_windows,\n            "segment_digests": self.segment_digests,\n            "block_lengths": self.block_lengths,\n            "paired_scope_evidence_digests": (\n                self.paired_scope_evidence_digests\n            ),\n            "observed_mean": self.observed_mean,\n''',
        label="seed bootstrap provenance payload",
    )

    seed_bootstrap_function_start = "def _bootstrap_universal_trade_rl_u2_seed_robustness(\n"
    seed_bootstrap_function_end = "\n\ndef _u2_seed_robustness_rejection_reasons(\n"
    text = replace_in_region(
        text,
        start=seed_bootstrap_function_start,
        end=seed_bootstrap_function_end,
        old='''        source_windows=source_windows,\n        segment_digests=tuple(segment.digest for segment in resolved),\n        block_lengths=block_lengths,\n        observed_mean=observed_mean,\n''',
        new='''        source_windows=source_windows,\n        segment_digests=tuple(segment.digest for segment in resolved),\n        block_lengths=block_lengths,\n        paired_scope_evidence_digests=tuple(\n            digest\n            for segment in resolved\n            for digest in segment.paired_scope_evidence_digests\n        ),\n        observed_mean=observed_mean,\n''',
        label="seed bootstrap provenance propagation",
    )

    robustness_start = (
        "@dataclass(frozen=True, slots=True)\n"
        "class UniversalTradeRLU2SeedRobustnessEvidence:"
    )
    robustness_end = "\n\ndef evaluate_universal_trade_rl_u2_seed_robustness(\n"
    text = replace_in_region(
        text,
        start=robustness_start,
        end=robustness_end,
        old='''    @property\n    def bootstrap_lower_ci(self) -> float:\n        return self.bootstrap_result.lower_ci\n''',
        new='''    @property\n    def cash_pairing_provenance_digest(self) -> str | None:\n        digests = self.bootstrap_result.paired_scope_evidence_digests\n        if not digests:\n            return None\n        return content_digest(\n            {\n                "schema_version": (\n                    "universal_trade_rl_u2_cash_pairing_provenance_v1"\n                ),\n                "scope": self.scope,\n                "paired_scope_evidence_digests": digests,\n            }\n        )\n\n    @property\n    def bootstrap_lower_ci(self) -> float:\n        return self.bootstrap_result.lower_ci\n''',
        label="robustness provenance property",
    )
    text = replace_in_region(
        text,
        start=robustness_start,
        end=robustness_end,
        old='''            "robustness_thresholds_digest": self.robustness_thresholds_digest,\n            "bootstrap_result_digest": self.bootstrap_result.digest,\n            "seed_symbol_balanced_net_wealth": self.seed_symbol_balanced_net_wealth,\n''',
        new='''            "robustness_thresholds_digest": self.robustness_thresholds_digest,\n            "bootstrap_result_digest": self.bootstrap_result.digest,\n            "cash_pairing_provenance_digest": (\n                self.cash_pairing_provenance_digest\n            ),\n            "seed_symbol_balanced_net_wealth": self.seed_symbol_balanced_net_wealth,\n''',
        label="robustness provenance payload",
    )

    final_start = (
        "@dataclass(frozen=True, slots=True)\n"
        "class UniversalTradeRLU2DevelopmentSelectionEvidence:"
    )
    final_end = "\n\ndef build_universal_trade_rl_u2_development_selection_evidence(\n"
    final_bootstrap_block = '''        if (\n            aggregate.bootstrap_result.source_windows != expected_windows\n            or aggregate.bootstrap_result.segment_digests != expected_segment_digests\n            or aggregate.bootstrap_result.block_lengths != expected_block_lengths\n        ):\n            raise ValueError(\n                "U2 final Selection D1+D2 aggregate bootstrap identity mismatch"\n            )\n'''
    text = replace_in_region(
        text,
        start=final_start,
        end=final_end,
        old=final_bootstrap_block,
        new=final_bootstrap_block
        + '''        if any(\n            gate.cash_pairing_provenance_digest is None\n            for gate in robustness\n        ):\n            raise ValueError(\n                "U2 final Selection requires cash pairing provenance for "\n                "every robustness bootstrap"\n            )\n        expected_pairing_provenance = (\n            d1.bootstrap_result.paired_scope_evidence_digests\n            + d2.bootstrap_result.paired_scope_evidence_digests\n        )\n        if (\n            aggregate.bootstrap_result.paired_scope_evidence_digests\n            != expected_pairing_provenance\n        ):\n            raise ValueError(\n                "U2 final Selection D1+D2 cash pairing provenance mismatch"\n            )\n''',
        label="final pairing provenance gate",
    )
    text = replace_in_region(
        text,
        start=final_start,
        end=final_end,
        old='''            "robustness_gate_digests": tuple(\n                gate.digest for gate in self.seed_robustness_gates\n            ),\n            "selected_checkpoint_digest": self.selected_checkpoint_digest,\n''',
        new='''            "robustness_gate_digests": tuple(\n                gate.digest for gate in self.seed_robustness_gates\n            ),\n            "cash_pairing_provenance_digests": tuple(\n                gate.cash_pairing_provenance_digest\n                for gate in self.seed_robustness_gates\n            ),\n            "selected_checkpoint_digest": self.selected_checkpoint_digest,\n''',
        label="final pairing provenance payload",
    )

    text = replace_once(
        text,
        '''    "UniversalTradeRLU2PairedExcessPoint",\n    "UniversalTradeRLU2ReducedBootstrapSegment",\n''',
        '''    "U2_PAIRED_REPLAY_SCOPE_EVIDENCE_SCHEMA",\n    "UniversalTradeRLU2PairedReplayScopeEvidence",\n    "build_universal_trade_rl_u2_paired_replay_scope_evidence",\n    "reduce_universal_trade_rl_u2_paired_replay_evidence",\n    "UniversalTradeRLU2PairedExcessPoint",\n    "UniversalTradeRLU2ReducedBootstrapSegment",\n''',
        label="selection pairing public API",
    )

    SELECTION.write_text(text, encoding="utf-8")


def patch_final_test() -> None:
    text = FINAL_TEST.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''def _segments():\n    module = _module()\n    return (\n        module.UniversalTradeRLU2ReducedBootstrapSegment(\n            source_window="development_future_1",\n            decision_timestamps_ns=(100, 200, 300, 400),\n            net_log_excess=(0.02, 0.02, 0.02, 0.02),\n        ),\n        module.UniversalTradeRLU2ReducedBootstrapSegment(\n            source_window="development_future_2",\n            decision_timestamps_ns=(500, 600, 700, 800),\n            net_log_excess=(0.02, 0.02, 0.02, 0.02),\n        ),\n    )\n''',
        '''def _segments(*, with_pairing_provenance: bool = True):\n    module = _module()\n\n    def pairing_digests(source_window: str) -> tuple[str, ...]:\n        if not with_pairing_provenance:\n            return ()\n        return tuple(\n            content_digest(\n                {\n                    "fixture": "u2-final-pairing-provenance",\n                    "source_window": source_window,\n                    "seed": seed,\n                    "symbol": symbol,\n                }\n            )\n            for seed in (0, 1, 2)\n            for symbol in ("DEV_A", "DEV_B")\n        )\n\n    return (\n        module.UniversalTradeRLU2ReducedBootstrapSegment(\n            source_window="development_future_1",\n            decision_timestamps_ns=(100, 200, 300, 400),\n            net_log_excess=(0.02, 0.02, 0.02, 0.02),\n            paired_scope_evidence_digests=pairing_digests(\n                "development_future_1"\n            ),\n        ),\n        module.UniversalTradeRLU2ReducedBootstrapSegment(\n            source_window="development_future_2",\n            decision_timestamps_ns=(500, 600, 700, 800),\n            net_log_excess=(0.02, 0.02, 0.02, 0.02),\n            paired_scope_evidence_digests=pairing_digests(\n                "development_future_2"\n            ),\n        ),\n    )\n''',
        label="final test segment provenance fixture",
    )
    text = replace_once(
        text,
        '''def _passing_selection_children():\n    module = _module()\n    d1_leaves = _cell_leaves("D1")\n    d2_leaves = _cell_leaves("D2")\n    segments = _segments()\n''',
        '''def _passing_selection_children(*, with_pairing_provenance: bool = True):\n    module = _module()\n    d1_leaves = _cell_leaves("D1")\n    d2_leaves = _cell_leaves("D2")\n    segments = _segments(with_pairing_provenance=with_pairing_provenance)\n''',
        label="final test passing child fixture",
    )
    text = replace_once(
        text,
        '''    (\n        _u2_contract,\n        _base_lock,\n        _development_lock,\n        _checkpoint_closure,\n        primary,\n        robustness,\n    ) = _final_fixture()\n\n    with pytest.raises(ValueError, match="cash|pair|provenance|bootstrap"):\n        _build_final(primary=primary, robustness=robustness)\n''',
        '''    primary, robustness = _passing_selection_children(\n        with_pairing_provenance=False\n    )\n\n    with pytest.raises(ValueError, match="cash|pair|provenance|bootstrap"):\n        _build_final(primary=primary, robustness=robustness)\n''',
        label="final test missing provenance negative",
    )
    FINAL_TEST.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch_selection()
    patch_final_test()
