from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing anchor in {path}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def add_tests() -> None:
    append_once(
        "tests/learning/test_oracle_teacher.py",
        "test_oracle_partial_fill_matches_executor_instead_of_invalidating_transition",
        '''

def test_oracle_partial_fill_matches_executor_instead_of_invalidating_transition() -> None:
    from dataclasses import replace

    from trade_rl.learning.oracle_teacher import _open_state_matrix, _transition_matrices
    from trade_rl.simulation.accounting import BookState
    from trade_rl.simulation.execution import MarketExecutor

    market = _market(np.array([100.0, 110.0, 110.0]))
    constrained = replace(
        market,
        volume=np.full_like(market.volume, 10.0),
        max_participation_rate=np.full_like(market.close, 0.01),
        minimum_notional=np.zeros_like(market.close),
    )
    cost = ExecutionCostConfig(
        fee_rate=0.0,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=0.01,
        maintenance_margin_rate=0.0,
    )
    config = OracleTeacherConfig(
        execution_cost=cost,
        reference_portfolio_value=1_000.0,
    )
    target = np.array([[0.45]], dtype=np.float64)
    _, open_weights, open_equity, _ = _open_state_matrix(
        constrained,
        close_index=0,
        prior_close_weights=np.zeros((1, 1), dtype=np.float64),
        prior_scores=np.zeros(1, dtype=np.float64),
        reference_portfolio_value=config.reference_portfolio_value,
    )
    valid, _, _, effective_targets = _transition_matrices(
        constrained,
        config,
        close_index=0,
        current_weights=open_weights,
        open_equity=open_equity,
        targets=target,
    )
    result = MarketExecutor(constrained, cost).execute_interval(
        BookState.zero(
            1,
            config.reference_portfolio_value,
            constrained.close[0],
            contract_multipliers=constrained.resolved_array("contract_multipliers"),
        ),
        target[0],
        start_index=0,
        bars=1,
    )

    assert valid[0, 0]
    expected_open_weight = (
        result.filled_notional_by_symbol[0] / config.reference_portfolio_value
    )
    assert 0.0 < effective_targets[0, 0, 0] < target[0, 0]
    assert effective_targets[0, 0, 0] == pytest.approx(expected_open_weight)


def test_oracle_below_minimum_notional_is_an_executable_noop() -> None:
    from dataclasses import replace

    from trade_rl.learning.oracle_teacher import _open_state_matrix, _transition_matrices

    market = _market(np.array([100.0, 101.0, 101.0]))
    constrained = replace(
        market,
        minimum_notional=np.full_like(market.close, 500.0),
    )
    config = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        reference_portfolio_value=1_000.0,
    )
    _, open_weights, open_equity, _ = _open_state_matrix(
        constrained,
        close_index=0,
        prior_close_weights=np.zeros((1, 1), dtype=np.float64),
        prior_scores=np.zeros(1, dtype=np.float64),
        reference_portfolio_value=config.reference_portfolio_value,
    )
    valid, _, _, effective_targets = _transition_matrices(
        constrained,
        config,
        close_index=0,
        current_weights=open_weights,
        open_equity=open_equity,
        targets=np.array([[0.45]], dtype=np.float64),
    )

    assert valid[0, 0]
    np.testing.assert_array_equal(effective_targets[0, 0], open_weights[0])


def test_delayed_oracle_returns_submitted_actions_and_discards_terminal_pending_action() -> None:
    market = _market(100.0 * np.exp(np.arange(8) * 0.04))
    config = OracleTeacherConfig(
        execution_cost=ExecutionCostConfig.zero(),
        signal_delay_decisions=1,
    )

    targets = oracle_target_path(market, (0, 8), config)

    assert targets.shape == (7, 1)
    assert np.any(targets[:-1, 0] > 0.0)
    np.testing.assert_array_equal(targets[-1], np.zeros(1, dtype=np.float32))
    assert "approximate" in config.schema_version
''',
    )
    replace_once(
        "tests/learning/test_oracle_teacher.py",
        '''def test_oracle_accounts_for_weight_drift_and_direction_permissions() -> None:
    """A profitable entry is invalid when its drifted weight cannot be reduced later."""
''',
        '''def test_oracle_accounts_for_weight_drift_and_direction_permissions() -> None:
    """A blocked reduction is a no-fill hold, matching the real executor."""
''',
    )
    replace_once(
        "tests/learning/test_oracle_teacher.py",
        "    np.testing.assert_array_equal(targets, np.zeros_like(targets))\n\n\ndef test_oracle_one_bar_transition_matches_deterministic_executor",
        "    assert targets[0, 0] > 0.0\n    assert np.all(targets[:, 0] >= 0.0)\n\n\ndef test_oracle_one_bar_transition_matches_deterministic_executor",
    )


def add_implementation() -> None:
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        '"""Train-range-only executable portfolio dynamic-programming oracle targets."""',
        '"""Train-range-only bounded approximate portfolio teacher targets."""',
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        'ORACLE_TEACHER_SCHEMA: Final = "portfolio_dp_oracle_teacher_v2"',
        'ORACLE_TEACHER_SCHEMA: Final = "approximate_portfolio_teacher_v3"',
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        '    """Deterministic portfolio state, risk, and execution contract."""\n',
        '    """Deterministic bounded-state approximation of the execution contract."""\n',
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "    maximum_states: int = 512\n    schema_version: str = ORACLE_TEACHER_SCHEMA\n",
        "    maximum_states: int = 512\n    signal_delay_decisions: int = 0\n    approximation_contract: str = \"bounded_state_partial_fill_v1\"\n    schema_version: str = ORACLE_TEACHER_SCHEMA\n",
    )
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "        cost = self.execution_cost\n",
        "        if (\n            isinstance(self.signal_delay_decisions, bool)\n            or not isinstance(self.signal_delay_decisions, int)\n            or self.signal_delay_decisions not in {0, 1}\n        ):\n            raise ValueError(\n                \"oracle signal_delay_decisions must be exactly zero or one\"\n            )\n        if self.approximation_contract != \"bounded_state_partial_fill_v1\":\n            raise ValueError(\"unsupported oracle approximation contract\")\n        cost = self.execution_cost\n",
    )
    old_transition = '''    execution_index = close_index + 1
    effective_targets = _effective_target_matrix(config, current_weights, targets)
    delta = effective_targets - current_weights[:, None, :]
    absolute_delta = np.abs(delta)
    trade = absolute_delta > _EPSILON
    valid = np.ones(delta.shape[:2], dtype=np.bool_)

    active = dataset.resolved_array("asset_active")[execution_index]
    tradable = dataset.tradable[execution_index]
    buy_allowed = dataset.resolved_array("buy_allowed")[execution_index]
    sell_allowed = dataset.resolved_array("sell_allowed")[execution_index]
    borrow_available = dataset.resolved_array("borrow_available")[execution_index]
    tradable_direction = np.where(
        delta > _EPSILON,
        buy_allowed[None, None, :],
        np.where(delta < -_EPSILON, sell_allowed[None, None, :], True),
    )
    valid &= np.all(
        ~trade | (active[None, None, :] & tradable[None, None, :] & tradable_direction),
        axis=2,
    )
    increasing_short = (delta < -_EPSILON) & (effective_targets < -_EPSILON)
    valid &= np.all(~increasing_short | borrow_available[None, None, :], axis=2)
    if not config.execution_cost.allow_short:
        valid &= ~np.any(effective_targets < -_EPSILON, axis=2)

    requested = absolute_delta * open_equity[:, None, None]
    prices = dataset.open[execution_index]
    market_notional = dataset.market_notional(
        execution_index,
        prices,
        volume=dataset.volume[close_index],
    )
    participation_limit = np.minimum(
        dataset.resolved_array("max_participation_rate")[execution_index],
        config.execution_cost.max_participation_rate,
    )
    capacity = participation_limit * market_notional
    minimum_notional = np.maximum(
        dataset.resolved_array("minimum_notional")[execution_index],
        config.execution_cost.minimum_notional,
    )
    valid &= np.all(
        ~trade | (requested >= minimum_notional[None, None, :] - 1e-9),
        axis=2,
    )
    valid &= np.all(
        ~trade | (requested <= capacity[None, None, :] + 1e-9),
        axis=2,
    )

    participation = np.zeros_like(requested)
    positive_liquidity = market_notional > _EPSILON
    participation[:, :, positive_liquidity] = (
        requested[:, :, positive_liquidity]
        / market_notional[None, None, positive_liquidity]
    )
'''
    new_transition = '''    execution_index = close_index + 1
    requested_targets = _effective_target_matrix(config, current_weights, targets)
    desired_delta = requested_targets - current_weights[:, None, :]
    requested_trade = np.abs(desired_delta) > _EPSILON
    valid_prior = np.isfinite(open_equity) & (open_equity > _EPSILON)
    valid = np.broadcast_to(valid_prior[:, None], desired_delta.shape[:2]).copy()

    active = dataset.resolved_array("asset_active")[execution_index]
    tradable = dataset.tradable[execution_index]
    buy_allowed = dataset.resolved_array("buy_allowed")[execution_index]
    sell_allowed = dataset.resolved_array("sell_allowed")[execution_index]
    borrow_available = dataset.resolved_array("borrow_available")[execution_index]
    direction_allowed = np.where(
        desired_delta > _EPSILON,
        buy_allowed[None, None, :],
        np.where(desired_delta < -_EPSILON, sell_allowed[None, None, :], True),
    )
    executable = (
        active[None, None, :]
        & tradable[None, None, :]
        & direction_allowed
    )
    increasing_short = (desired_delta < -_EPSILON) & (requested_targets < -_EPSILON)
    executable &= ~increasing_short | borrow_available[None, None, :]
    if not config.execution_cost.allow_short:
        executable &= requested_targets >= -_EPSILON

    requested = np.abs(desired_delta) * open_equity[:, None, None]
    prices = dataset.open[execution_index]
    market_notional = dataset.market_notional(
        execution_index,
        prices,
        volume=dataset.volume[close_index],
    )
    participation_limit = np.minimum(
        dataset.resolved_array("max_participation_rate")[execution_index],
        config.execution_cost.max_participation_rate,
    )
    capacity = participation_limit * market_notional
    minimum_notional = np.maximum(
        dataset.resolved_array("minimum_notional")[execution_index],
        config.execution_cost.minimum_notional,
    )
    eligible = (
        requested_trade
        & executable
        & (requested >= minimum_notional[None, None, :] - 1e-9)
    )
    filled_notional = np.where(
        eligible,
        np.minimum(requested, capacity[None, None, :]),
        0.0,
    )
    safe_equity = np.maximum(open_equity[:, None, None], _EPSILON)
    filled_delta = np.sign(desired_delta) * filled_notional / safe_equity
    effective_targets = current_weights[:, None, :] + filled_delta
    absolute_delta = np.abs(filled_delta)

    participation = np.zeros_like(filled_notional)
    positive_liquidity = market_notional > _EPSILON
    participation[:, :, positive_liquidity] = (
        filled_notional[:, :, positive_liquidity]
        / market_notional[None, None, positive_liquidity]
    )
'''
    replace_once("trade_rl/learning/oracle_teacher.py", old_transition, new_transition)
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        '    """Return fully executable portfolio-level target labels inside train range."""\n',
        '    """Return bounded approximate submitted target labels inside train range."""\n',
    )
    old_path = '''    scores = np.full((steps, state_count), -np.inf, dtype=np.float64)
    pointers = np.full((steps, state_count), -1, dtype=np.int64)
    close_weights = np.zeros((steps, state_count, dataset.n_symbols), dtype=np.float64)
    selected_targets = np.zeros_like(close_weights)
    cash_index = int(np.flatnonzero(np.all(np.isclose(states, 0.0), axis=1))[0])

    for step in range(steps):
        close_index = start + step
        if step == 0:
            prior_scores = np.full(state_count, -np.inf, dtype=np.float64)
            prior_scores[cash_index] = 0.0
            prior_close_weights = np.zeros_like(close_weights[0])
        else:
            prior_scores = scores[step - 1]
            prior_close_weights = close_weights[step - 1]
        gap_factor, open_weights, open_equity, valid_prior = _open_state_matrix(
            dataset,
            close_index=close_index,
            prior_close_weights=prior_close_weights,
            prior_scores=prior_scores,
            reference_portfolio_value=config.reference_portfolio_value,
        )
        (
            transition_valid,
            close_factor,
            candidate_close_weights,
            candidate_effective_targets,
        ) = _transition_matrices(
            dataset,
            config,
            close_index=close_index,
            current_weights=open_weights,
            open_equity=open_equity,
            targets=states,
        )
        transition_valid &= valid_prior[:, None]
        candidate_scores = (
            prior_scores[:, None]
            + np.log(np.where(valid_prior, gap_factor, 1.0))[:, None]
            + np.log(np.where(transition_valid, close_factor, 1.0))
        )
        candidate_scores = np.where(transition_valid, candidate_scores, -np.inf)
        best_prior = np.argmax(candidate_scores, axis=0)
        best_scores = candidate_scores[best_prior, np.arange(state_count)]
        scores[step] = best_scores
        pointers[step] = np.where(np.isfinite(best_scores), best_prior, -1)
        close_weights[step] = candidate_close_weights[
            best_prior, np.arange(state_count)
        ]
        selected_targets[step] = candidate_effective_targets[
            best_prior, np.arange(state_count)
        ]
        invalid = ~np.isfinite(best_scores)
        close_weights[step, invalid] = 0.0
        selected_targets[step, invalid] = 0.0

    final_state = int(np.argmax(scores[-1]))
'''
    new_path = '''    scores = np.full((steps, state_count), -np.inf, dtype=np.float64)
    pointers = np.full((steps, state_count), -1, dtype=np.int64)
    close_weights = np.zeros((steps, state_count, dataset.n_symbols), dtype=np.float64)
    cash_index = int(np.flatnonzero(np.all(np.isclose(states, 0.0), axis=1))[0])

    for step in range(steps):
        close_index = start + step
        if step == 0:
            prior_scores = np.full(state_count, -np.inf, dtype=np.float64)
            prior_scores[cash_index] = 0.0
            prior_close_weights = np.zeros_like(close_weights[0])
        else:
            prior_scores = scores[step - 1]
            prior_close_weights = close_weights[step - 1]
        gap_factor, open_weights, open_equity, valid_prior = _open_state_matrix(
            dataset,
            close_index=close_index,
            prior_close_weights=prior_close_weights,
            prior_scores=prior_scores,
            reference_portfolio_value=config.reference_portfolio_value,
        )
        if config.signal_delay_decisions == 0:
            transition_valid, close_factor, candidate_close_weights, _ = (
                _transition_matrices(
                    dataset,
                    config,
                    close_index=close_index,
                    current_weights=open_weights,
                    open_equity=open_equity,
                    targets=states,
                )
            )
            transition_valid &= valid_prior[:, None]
            candidate_scores = (
                prior_scores[:, None]
                + np.log(np.where(valid_prior, gap_factor, 1.0))[:, None]
                + np.log(np.where(transition_valid, close_factor, 1.0))
            )
            candidate_scores = np.where(transition_valid, candidate_scores, -np.inf)
            best_prior = np.argmax(candidate_scores, axis=0)
            best_scores = candidate_scores[best_prior, np.arange(state_count)]
            scores[step] = best_scores
            pointers[step] = np.where(np.isfinite(best_scores), best_prior, -1)
            close_weights[step] = candidate_close_weights[
                best_prior, np.arange(state_count)
            ]
        elif step == 0:
            hold = states[cash_index : cash_index + 1]
            transition_valid, close_factor, candidate_close_weights, _ = (
                _transition_matrices(
                    dataset,
                    config,
                    close_index=close_index,
                    current_weights=open_weights,
                    open_equity=open_equity,
                    targets=hold,
                )
            )
            transition_valid &= valid_prior[:, None]
            candidate_scores = (
                prior_scores[:, None]
                + np.log(np.where(valid_prior, gap_factor, 1.0))[:, None]
                + np.log(np.where(transition_valid, close_factor, 1.0))
            )
            candidate_scores = np.where(transition_valid, candidate_scores, -np.inf)
            best_prior = int(np.argmax(candidate_scores[:, 0]))
            best_score = float(candidate_scores[best_prior, 0])
            scores[step] = best_score
            pointers[step] = best_prior
            close_weights[step] = candidate_close_weights[best_prior, 0]
        else:
            transition_valid, close_factor, candidate_close_weights, _ = (
                _transition_matrices(
                    dataset,
                    config,
                    close_index=close_index,
                    current_weights=open_weights,
                    open_equity=open_equity,
                    targets=states,
                )
            )
            diagonal = np.arange(state_count)
            diagonal_valid = transition_valid[diagonal, diagonal] & valid_prior
            diagonal_scores = (
                prior_scores
                + np.log(np.where(valid_prior, gap_factor, 1.0))
                + np.log(
                    np.where(
                        diagonal_valid,
                        close_factor[diagonal, diagonal],
                        1.0,
                    )
                )
            )
            diagonal_scores = np.where(diagonal_valid, diagonal_scores, -np.inf)
            best_prior = int(np.argmax(diagonal_scores))
            best_score = float(diagonal_scores[best_prior])
            scores[step] = best_score
            pointers[step] = best_prior
            close_weights[step] = candidate_close_weights[best_prior, best_prior]

        invalid = ~np.isfinite(scores[step])
        close_weights[step, invalid] = 0.0

    final_state = (
        cash_index
        if config.signal_delay_decisions == 1
        else int(np.argmax(scores[-1]))
    )
'''
    replace_once("trade_rl/learning/oracle_teacher.py", old_path, new_path)
    replace_once(
        "trade_rl/learning/oracle_teacher.py",
        "    targets = selected_targets[np.arange(steps), state_path]\n",
        "    targets = states[state_path]\n",
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        "                        reference_portfolio_value=unwrapped_teacher.initial_capital,\n                    )\n",
        "                        reference_portfolio_value=unwrapped_teacher.initial_capital,\n                        signal_delay_decisions=(\n                            unwrapped_teacher.config.signal_delay_decisions\n                        ),\n                    )\n",
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task2_oracle.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
