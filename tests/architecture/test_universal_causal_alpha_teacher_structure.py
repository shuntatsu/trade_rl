from __future__ import annotations


def test_causal_alpha_facade_reexports_responsibility_modules() -> None:
    from trade_rl.workflows import universal_causal_alpha_contracts as contracts
    from trade_rl.workflows import universal_causal_alpha_fitting as fitting
    from trade_rl.workflows import universal_causal_alpha_selection as selection
    from trade_rl.workflows import universal_causal_alpha_teacher as facade

    assert facade.CausalAlphaEpisodePartition is contracts.CausalAlphaEpisodePartition
    assert facade.CausalAlphaSymbolSamples is contracts.CausalAlphaSymbolSamples
    assert facade.CausalAlphaExpandingFit is contracts.CausalAlphaExpandingFit
    assert facade.CausalAlphaCandidateConfig is contracts.CausalAlphaCandidateConfig
    assert facade.CausalAlphaSelectionEvidence is contracts.CausalAlphaSelectionEvidence
    assert (
        facade.UniversalCausalAlphaTeacherPackage
        is contracts.UniversalCausalAlphaTeacherPackage
    )

    assert (
        facade.build_chronological_episode_partition
        is fitting.build_chronological_episode_partition
    )
    assert facade.build_causal_alpha_symbol_samples is fitting.build_causal_alpha_symbol_samples
    assert (
        facade.fit_expanding_causal_alpha_models
        is fitting.fit_expanding_causal_alpha_models
    )
    assert facade.build_causal_alpha_episode_batch is fitting.build_causal_alpha_episode_batch

    assert (
        facade.default_causal_alpha_candidate_grid
        is selection.default_causal_alpha_candidate_grid
    )
    assert facade.rank_causal_alpha_candidates is selection.rank_causal_alpha_candidates
