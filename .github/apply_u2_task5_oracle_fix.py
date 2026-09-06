from pathlib import Path

path = Path("tests/workflows/test_universal_trade_rl_u2_selection_final.py")
text = path.read_text(encoding="utf-8")

segments_anchor = '''def _summary_for_seed(*, leaves, training_seed: int):
'''
paired_helper = '''def _paired_segments():
    module = _module()
    symbols = ("DEV_A", "DEV_B")
    pairs = tuple(
        module.UniversalTradeRLU2PairedReplayScopeEvidence(
            training_seed=seed,
            source_window=window,
            cell=cell,
            concrete_symbol=symbol,
            scope_digest=content_digest(
                {
                    "fixture": "u2-final-pairing-scope",
                    "seed": seed,
                    "window": window,
                    "symbol": symbol,
                }
            ),
            evaluation_dataset_digest=content_digest(
                {"fixture": "u2-final-pairing-dataset", "symbol": symbol}
            ),
            paired_candidate_checkpoint_digest=content_digest(
                {"fixture": "u2-final-pairing-checkpoint", "seed": seed}
            ),
            candidate_replay_evidence_digest=content_digest(
                {
                    "fixture": "u2-final-pairing-candidate",
                    "seed": seed,
                    "window": window,
                    "symbol": symbol,
                }
            ),
            cash_replay_evidence_digest=content_digest(
                {
                    "fixture": "u2-final-pairing-cash",
                    "seed": seed,
                    "window": window,
                    "symbol": symbol,
                }
            ),
            decision_timestamps_ns=timestamps,
            candidate_minus_cash_net_log_excess=(0.02, 0.02, 0.02, 0.02),
        )
        for seed in (0, 1, 2)
        for window, cell, timestamps in (
            ("development_future_1", "D1", (100, 200, 300, 400)),
            ("development_future_2", "D2", (500, 600, 700, 800)),
        )
        for symbol in symbols
    )
    return module.reduce_universal_trade_rl_u2_paired_replay_evidence(
        pairs=pairs,
        expected_symbols=symbols,
    )


'''
if segments_anchor not in text:
    raise SystemExit("Task5 oracle summary anchor missing")
if "def _paired_segments():" not in text:
    text = text.replace(segments_anchor, paired_helper + segments_anchor, 1)

passing_old = '''    d1_leaves = _cell_leaves("D1")
    d2_leaves = _cell_leaves("D2")
    segments = _segments()
'''
passing_new = '''    d1_leaves = _cell_leaves("D1")
    d2_leaves = _cell_leaves("D2")
    segments = _paired_segments()
'''
if passing_old not in text:
    raise SystemExit("Task5 oracle passing fixture anchor missing")
text = text.replace(passing_old, passing_new, 1)

aggregate_old = '''    altered_d12 = module.evaluate_universal_trade_rl_u2_seed_robustness(
        scope="D1+D2",
        leaves=_cell_leaves("D1") + altered_d2_leaves,
        segments=_segments(),
    )
'''
aggregate_new = '''    altered_d12 = module.evaluate_universal_trade_rl_u2_seed_robustness(
        scope="D1+D2",
        leaves=_cell_leaves("D1") + altered_d2_leaves,
        segments=_paired_segments(),
    )
'''
if aggregate_old not in text:
    raise SystemExit("Task5 oracle aggregate substitution anchor missing")
text = text.replace(aggregate_old, aggregate_new, 1)

negative_old = '''def test_u2_final_selection_rejects_robustness_without_cash_pairing_provenance() -> (
    None
):
    (
        _u2_contract,
        _base_lock,
        _development_lock,
        _checkpoint_closure,
        primary,
        robustness,
    ) = _final_fixture()

    with pytest.raises(ValueError, match="cash|pair|provenance|bootstrap"):
        _build_final(primary=primary, robustness=robustness)
'''
negative_new = '''def test_u2_final_selection_rejects_robustness_without_cash_pairing_provenance() -> (
    None
):
    module = _module()
    (
        _u2_contract,
        _base_lock,
        _development_lock,
        _checkpoint_closure,
        primary,
        _robustness,
    ) = _final_fixture()
    d1_leaves = _cell_leaves("D1")
    d2_leaves = _cell_leaves("D2")
    legacy_segments = _segments()
    legacy_robustness = (
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D1",
            leaves=d1_leaves,
            segments=legacy_segments,
        ),
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D2",
            leaves=d2_leaves,
            segments=legacy_segments,
        ),
        module.evaluate_universal_trade_rl_u2_seed_robustness(
            scope="D1+D2",
            leaves=d1_leaves + d2_leaves,
            segments=legacy_segments,
        ),
    )
    assert all(gate.passed for gate in legacy_robustness)
    assert all(
        not gate.bootstrap_result.paired_scope_evidence_digests
        for gate in legacy_robustness
    )

    with pytest.raises(ValueError, match="cash|pair|provenance|bootstrap"):
        _build_final(primary=primary, robustness=legacy_robustness)
'''
if negative_old not in text:
    raise SystemExit("Task5 oracle negative anchor missing")
text = text.replace(negative_old, negative_new, 1)

path.write_text(text, encoding="utf-8")
