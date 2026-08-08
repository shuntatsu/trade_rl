from __future__ import annotations

from trade_rl.integrations.nautilus.event_projection import MarketPhase
from trade_rl.workflows.stage_a_nautilus_historical_replay import (
    StageANautilusHistoricalReplayInterval,
    project_stage_a_nautilus_historical_interval_events,
)

from tests.workflows.test_stage_a_nautilus_historical_replay import (
    _replay_for_market,
    _single_symbol_market,
)


def test_historical_interval_activates_target_once_after_first_open_quote() -> None:
    market = _single_symbol_market()
    replay = _replay_for_market(market)
    from trade_rl.workflows.stage_a_nautilus_historical_replay import (
        build_stage_a_nautilus_historical_replay_intervals,
    )

    interval = build_stage_a_nautilus_historical_replay_intervals(replay, market)[1]
    assert isinstance(interval, StageANautilusHistoricalReplayInterval)
    assert len(interval.source_bars) >= 2

    events = project_stage_a_nautilus_historical_interval_events(interval)

    phases = tuple(event.phase for event in events)
    activation_positions = tuple(
        index for index, phase in enumerate(phases) if phase is MarketPhase.TARGET_ACTIVATION
    )
    assert len(activation_positions) == 1
    activation = activation_positions[0]
    assert phases[activation - 1] is MarketPhase.OPEN_QUOTE

    policy_decisions = tuple(
        event for event in events if event.phase is MarketPhase.POLICY_DECISION
    )
    assert len(policy_decisions) == len(interval.source_bars)
    assert all(
        left.timestamp_ns < right.timestamp_ns
        for left, right in zip(events, events[1:])
    )
