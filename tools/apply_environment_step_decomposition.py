from __future__ import annotations

import re
from pathlib import Path


PATH = Path("trade_rl/rl/environment.py")


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"{label} anchor count is {source.count(old)}, expected one")
    return source.replace(old, new, 1)


def _sub_once(source: str, pattern: str, replacement: str, *, label: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{label} match count is {count}, expected one")
    return updated


def main() -> None:
    source = PATH.read_text(encoding="utf-8")

    import_anchor = (
        "from trade_rl.rl.environment_episode import EpisodeContractSampler\n"
        "from trade_rl.rl.environment_execution import (\n"
    )
    import_replacement = (
        "from trade_rl.rl.environment_decision import (\n"
        "    EnvironmentDecisionPlanner,\n"
        "    EnvironmentDecisionRequest,\n"
        ")\n"
        "from trade_rl.rl.environment_episode import EpisodeContractSampler\n"
        "from trade_rl.rl.environment_execution import (\n"
    )
    source = _replace_once(
        source,
        import_anchor,
        import_replacement,
        label="decision imports",
    )
    observation_anchor = (
        "from trade_rl.rl.environment_observation import (\n"
        "    EnvironmentObservationAssembler,\n"
        "    EnvironmentObservationRuntime,\n"
        ")\n"
        "from trade_rl.rl.environment_transition import EnvironmentTerminationCoordinator\n"
    )
    observation_replacement = (
        "from trade_rl.rl.environment_info import (\n"
        "    EnvironmentInfoBuilder,\n"
        "    EnvironmentStepInfoRequest,\n"
        "    EnvironmentTerminalInfoRequest,\n"
        ")\n"
        "from trade_rl.rl.environment_observation import (\n"
        "    EnvironmentObservationAssembler,\n"
        "    EnvironmentObservationRuntime,\n"
        ")\n"
        "from trade_rl.rl.environment_reward import (\n"
        "    EnvironmentRewardCoordinator,\n"
        "    EnvironmentRewardRequest,\n"
        ")\n"
        "from trade_rl.rl.environment_risk import (\n"
        "    EnvironmentRiskProjector,\n"
        "    EnvironmentRiskRequest,\n"
        ")\n"
        "from trade_rl.rl.environment_transition import EnvironmentTerminationCoordinator\n"
    )
    source = _replace_once(
        source,
        observation_anchor,
        observation_replacement,
        label="step service imports",
    )

    construction_anchor = (
        "        self._termination_coordinator = EnvironmentTerminationCoordinator(\n"
    )
    construction_replacement = (
        "        self._decision_planner = EnvironmentDecisionPlanner(\n"
        "            dataset,\n"
        "            action_spec=self.action_spec,\n"
        "            composer=self.composer,\n"
        "            max_gross=self.pre_trade_risk.config.max_gross,\n"
        "            alpha_enabled=self.alpha_enabled,\n"
        "            accept_legacy_actions=self.config.accept_legacy_actions,\n"
        "            signal_delay_decisions=self.config.signal_delay_decisions,\n"
        "            decision_every=self.config.decision_every,\n"
        "            decision_hours=self.config.decision_hours,\n"
        "        )\n"
        "        self._risk_projector = EnvironmentRiskProjector(\n"
        "            dataset,\n"
        "            emergency_risk_monitor=self.emergency_risk_monitor,\n"
        "            pre_trade_risk=self.pre_trade_risk,\n"
        "            portfolio_risk=self.portfolio_risk,\n"
        "            portfolio_risk_inputs_provider=(\n"
        "                self.portfolio_risk_inputs_provider\n"
        "            ),\n"
        "        )\n"
        "        self._reward_coordinator = EnvironmentRewardCoordinator(\n"
        "            self.reward_tracker,\n"
        "            initial_capital=self.config.initial_capital,\n"
        "        )\n"
        "        self._info_builder = EnvironmentInfoBuilder(\n"
        "            dataset,\n"
        "            self.reward_tracker,\n"
        "        )\n"
        "        self._termination_coordinator = EnvironmentTerminationCoordinator(\n"
    )
    source = _replace_once(
        source,
        construction_anchor,
        construction_replacement,
        label="service construction",
    )

    delegates = '''    def _market_notional(self, index: int) -> np.ndarray:
        return self._risk_projector.market_notional(index)

    def _constrain_target(
        self,
        proposal: np.ndarray,
        book: BookState,
    ) -> RiskConstrainedTarget:
        return self._risk_projector.project(
            EnvironmentRiskRequest(
                proposal=proposal,
                book=book,
                current_index=self.current_index,
            )
        )

    def _parse_action(
        self,
        value: np.ndarray,
    ) -> tuple[
        ResidualAction | ResidualActionV2 | TargetWeightAction,
        np.ndarray,
        int,
        float,
    ]:
        return self._decision_planner.parse_action(value)

    def _decision_bar_count(self) -> int:
        return self._decision_planner.decision_bar_count(
            current_index=self.current_index,
            end_index=self.end_index,
        )

'''
    source = _sub_once(
        source,
        r"    def _market_notional\(.*?(?=    @staticmethod\n    def _merge_liquidation_return)",
        delegates,
        label="compatibility delegates",
    )

    step = '''    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.current_index >= self.end_index:
            raise RuntimeError("step called after the episode ended")
        trends, alpha, factor_basis = self._market_inputs()
        decision = self._decision_planner.plan(
            EnvironmentDecisionRequest(
                action=action,
                trends=trends,
                alpha=alpha,
                factor_basis=factor_basis,
                hybrid_weights=self.hybrid.weights,
                shadow_weights=self.shadow.weights,
                pending_hybrid_target=self._pending_hybrid_target,
                pending_shadow_target=self._pending_shadow_target,
                current_index=self.current_index,
                end_index=self.end_index,
            )
        )
        self._pending_hybrid_target = decision.next_pending_hybrid_target
        self._pending_shadow_target = decision.next_pending_shadow_target
        hybrid_risk = self._risk_projector.project(
            EnvironmentRiskRequest(
                proposal=decision.executed_hybrid_target,
                book=self.hybrid,
                current_index=self.current_index,
            )
        )
        shadow_risk = self._risk_projector.project(
            EnvironmentRiskRequest(
                proposal=decision.executed_shadow_target,
                book=self.shadow,
                current_index=self.current_index,
            )
        )
        previous_hybrid_weights = self.hybrid.weights.copy()
        hybrid_execution = self._execute_stateful_target(
            executor=self.hybrid_executor,
            book=self.hybrid,
            order_book=self._hybrid_order_book,
            target=hybrid_risk.weights,
            bars=decision.bars,
            book_kind="hybrid",
        )
        shadow_execution = self._execute_stateful_target(
            executor=self.shadow_executor,
            book=self.shadow,
            order_book=self._shadow_order_book,
            target=shadow_risk.weights,
            bars=decision.bars,
            book_kind="shadow",
        )
        if hybrid_execution.bars_advanced != shadow_execution.bars_advanced:
            raise RuntimeError("hybrid and shadow books advanced different bar counts")

        self.hybrid = hybrid_execution.book
        self.shadow = shadow_execution.book
        self._hybrid_order_book = hybrid_execution.order_book
        self._shadow_order_book = shadow_execution.order_book
        self.current_index = hybrid_execution.next_index
        self._decision_step_index += 1
        time_limit_reached = self.current_index >= self.end_index
        pending_hybrid_target = self._pending_hybrid_target
        pending_target_discarded = bool(
            time_limit_reached
            and self.config.signal_delay_decisions == 1
            and pending_hybrid_target is not None
        )
        discarded_pending_target = (
            pending_hybrid_target.copy()
            if pending_target_discarded and pending_hybrid_target is not None
            else None
        )
        if time_limit_reached:
            self._pending_hybrid_target = None
            self._pending_shadow_target = None
        transition = self._termination_coordinator.resolve(
            hybrid=self.hybrid,
            shadow=self.shadow,
            current_index=self.current_index,
            time_limit_reached=time_limit_reached,
            hybrid_log_return=hybrid_execution.interval_log_return,
            shadow_log_return=shadow_execution.interval_log_return,
        )
        self.hybrid = transition.hybrid
        self.shadow = transition.shadow
        hybrid_log_return = transition.hybrid_log_return
        shadow_log_return = transition.shadow_log_return
        hybrid_liquidation = transition.hybrid_liquidation
        shadow_liquidation = transition.shadow_liquidation
        liquidation_complete = transition.liquidation_complete
        liquidation_terminal = transition.liquidation_terminal
        emergency_deleverage = transition.emergency_deleverage
        hybrid_terminated = transition.hybrid_terminated
        shadow_terminated = transition.shadow_terminated
        economic_transition = transition.economic_transition
        terminated = economic_transition.terminated
        truncated = economic_transition.truncated
        action_delta_l1 = float(
            np.abs(decision.maintained_action - self._previous_action).sum()
        )
        projection_distance = hybrid_risk.projection_l1
        reward_breakdown = self._reward_coordinator.step(
            EnvironmentRewardRequest(
                hybrid_log_return=hybrid_log_return,
                shadow_log_return=shadow_log_return,
                hybrid=self.hybrid,
                shadow=self.shadow,
                projection_distance=projection_distance,
                hybrid_terminated=hybrid_terminated,
                shadow_terminated=shadow_terminated,
            )
        )
        self._execution_state = self._execution_observation_state(
            requested_weights=hybrid_risk.weights,
            result=hybrid_execution,
            previous_weights=previous_hybrid_weights,
        )
        self._previous_action = decision.maintained_action.copy()
        self._action_diagnostics.update(
            action=decision.maintained_action,
            saturated_count=decision.saturated_count,
            action_delta_l1=action_delta_l1,
            projection_l1=projection_distance,
            constrained=hybrid_risk.was_constrained,
            turnover_overridden=hybrid_risk.turnover_overridden,
        )
        termination_reason = economic_transition.reason
        terminal_accounting_mode = (
            "liquidate_at_close"
            if liquidation_terminal
            else "mark_to_market"
            if time_limit_reached
            else "economic_termination"
        )
        terminal_liquidation_cost = (
            float(hybrid_liquidation.interval_cost)
            if liquidation_terminal and hybrid_liquidation is not None
            else 0.0
        )
        info = self._info_builder.step_info(
            EnvironmentStepInfoRequest(
                action_delta_l1=action_delta_l1,
                raw_max_abs=decision.raw_max_abs,
                saturated_count=decision.saturated_count,
                composition=decision.composition,
                decision_step_index=self._decision_step_index,
                hybrid_log_return=hybrid_log_return,
                shadow_log_return=shadow_log_return,
                emergency_deleverage=emergency_deleverage,
                execution_delay_warmup=decision.execution_delay_warmup,
                submitted_target=decision.submitted_hybrid_target,
                executed_target=decision.executed_hybrid_target,
                hybrid=self.hybrid,
                reward_breakdown=reward_breakdown,
                hybrid_execution=hybrid_execution,
                hybrid_risk=hybrid_risk,
                hybrid_terminated=hybrid_terminated,
                shadow_execution=shadow_execution,
                shadow_risk=shadow_risk,
                shadow_terminated=shadow_terminated,
                liquidation_complete=liquidation_complete,
                liquidation_terminal=liquidation_terminal,
                termination_reason=termination_reason,
                terminal_accounting_mode=terminal_accounting_mode,
                terminal_liquidation_cost=terminal_liquidation_cost,
                pending_target_discarded=pending_target_discarded,
                discarded_pending_target=discarded_pending_target,
                hybrid_liquidation=hybrid_liquidation,
                shadow_liquidation=shadow_liquidation,
            )
        )
        if terminated or truncated:
            info.update(self._terminal_info())
        return (
            self._observation(),
            reward_breakdown.scaled_total,
            terminated,
            truncated,
            info,
        )

'''
    source = _sub_once(
        source,
        r"    def step\(\n.*?(?=    def _book_metrics\()",
        step,
        label="step orchestration",
    )

    terminal = '''    def _terminal_info(self) -> dict[str, object]:
        return self._info_builder.terminal_info(
            EnvironmentTerminalInfoRequest(
                episode_hours=self._episode_hours,
                episode_seed=self._episode_seed,
                action_diagnostics=self._action_diagnostics.snapshot(),
                hybrid=self.hybrid,
                shadow=self.shadow,
                initial_state_mode=self._initial_state_mode,
            )
        )
'''
    source = _sub_once(
        source,
        r"    def _book_metrics\(.*\Z",
        terminal,
        label="terminal information",
    )

    PATH.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
