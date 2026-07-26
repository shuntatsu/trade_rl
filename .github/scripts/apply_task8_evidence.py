from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    content = path.read_text()
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement point, found {count}")
    path.write_text(content.replace(old, new))


model_path = Path("trade_rl/integrations/lagrangian_ppo.py")
content = model_path.read_text()

if "from collections.abc import Callable, Iterator, Mapping" not in content:
    replace_once(
        model_path,
        "from collections.abc import Callable, Iterator\n",
        "from collections.abc import Callable, Iterator, Mapping\n",
    )

content = model_path.read_text()
if "normalize_cost_advantages" not in content:
    replace_once(
        model_path,
        "from trade_rl.rl.lagrangian_advantages import combine_lagrangian_advantages\n",
        "from trade_rl.rl.lagrangian_advantages import (\n"
        "    combine_lagrangian_advantages,\n"
        "    normalize_cost_advantages,\n"
        ")\n"
        "from trade_rl.rl.lagrangian_diagnostics import (\n"
        "    ConstraintCorrelationDiagnostics,\n"
        "    build_constraint_correlation_diagnostics,\n"
        "    build_dual_stability_diagnostics,\n"
        ")\n"
        "from trade_rl.rl.lagrangian_evidence import (\n"
        "    LagrangianRolloutEvidence,\n"
        "    build_lagrangian_rollout_evidence,\n"
        ")\n",
    )

content = model_path.read_text()
if "self.last_constraint_correlation_diagnostics" not in content:
    replace_once(
        model_path,
        "        self.last_dual_update_reports: dict[str, DualUpdateReport] = {}\n",
        "        self.last_dual_update_reports: dict[str, DualUpdateReport] = {}\n"
        "        self.last_constraint_correlation_diagnostics: (\n"
        "            ConstraintCorrelationDiagnostics | None\n"
        "        ) = None\n"
        "        self.dual_report_history: list[dict[str, DualUpdateReport]] = []\n"
        "        self.last_lagrangian_rollout_evidence: (\n"
        "            LagrangianRolloutEvidence | None\n"
        "        ) = None\n",
    )

content = model_path.read_text()
if "dual report history has an invalid type" not in content:
    replace_once(
        model_path,
        "        probe_evidence = getattr(self, \"canonical_action_probe_evidence\", None)\n"
        "        if probe_evidence is not None and not isinstance(\n"
        "            probe_evidence, CanonicalActionProbeEvidence\n"
        "        ):\n"
        "            raise TypeError(\"canonical_action_probe_evidence has an invalid type\")\n",
        "        probe_evidence = getattr(self, \"canonical_action_probe_evidence\", None)\n"
        "        if probe_evidence is not None and not isinstance(\n"
        "            probe_evidence, CanonicalActionProbeEvidence\n"
        "        ):\n"
        "            raise TypeError(\"canonical_action_probe_evidence has an invalid type\")\n"
        "        correlation = getattr(\n"
        "            self, \"last_constraint_correlation_diagnostics\", None\n"
        "        )\n"
        "        if correlation is not None and not isinstance(\n"
        "            correlation, ConstraintCorrelationDiagnostics\n"
        "        ):\n"
        "            raise TypeError(\n"
        "                \"last_constraint_correlation_diagnostics has an invalid type\"\n"
        "            )\n"
        "        history = getattr(self, \"dual_report_history\", None)\n"
        "        if history is None:\n"
        "            self.dual_report_history = []\n"
        "        elif not isinstance(history, list):\n"
        "            raise TypeError(\"dual report history has an invalid type\")\n"
        "        else:\n"
        "            for report_set in history:\n"
        "                if not isinstance(report_set, Mapping) or tuple(report_set) != (\n"
        "                    self.lagrangian_schema.names\n"
        "                ):\n"
        "                    raise ValueError(\"dual report history identity mismatch\")\n"
        "                for name in self.lagrangian_schema.names:\n"
        "                    report = report_set[name]\n"
        "                    if not isinstance(report, DualUpdateReport) or report.name != name:\n"
        "                        raise ValueError(\"dual report history identity mismatch\")\n"
        "        rollout_evidence = getattr(\n"
        "            self, \"last_lagrangian_rollout_evidence\", None\n"
        "        )\n"
        "        if rollout_evidence is not None and not isinstance(\n"
        "            rollout_evidence, LagrangianRolloutEvidence\n"
        "        ):\n"
        "            raise TypeError(\n"
        "                \"last_lagrangian_rollout_evidence has an invalid type\"\n"
        "            )\n",
    )

content = model_path.read_text()
if "def _record_lagrangian_rollout_evidence" not in content:
    marker = "    def checkpoint_identity_payload(self) -> dict[str, object]:\n"
    if content.count(marker) != 1:
        raise RuntimeError("checkpoint identity insertion point mismatch")
    methods = '''    def _flatten_rollout_diagnostics(\n        self,\n    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:\n        \"\"\"Return canonical transition-order copies for observational diagnostics.\"\"\"\n\n        cost_count = len(self.lagrangian_schema.names)\n        raw_cost_tensor = np.asarray(\n            self.cost_rollout_storage.costs,\n            dtype=np.float64,\n        )\n        raw_cost_advantage_tensor = np.asarray(\n            self.cost_rollout_storage.advantages,\n            dtype=np.float64,\n        )\n        expected_shape = (\n            self.n_steps,\n            self.n_envs,\n            cost_count,\n        )\n        if raw_cost_tensor.shape != expected_shape:\n            raise ValueError(\"raw cost rollout shape mismatch\")\n        if raw_cost_advantage_tensor.shape != expected_shape:\n            raise ValueError(\"raw cost advantage rollout shape mismatch\")\n        raw_costs = raw_cost_tensor.swapaxes(0, 1).reshape(-1, cost_count).copy()\n        raw_cost_advantages = (\n            raw_cost_advantage_tensor.swapaxes(0, 1)\n            .reshape(-1, cost_count)\n            .copy()\n        )\n\n        reward_tensor = np.asarray(\n            self.rollout_buffer.advantages,\n            dtype=np.float64,\n        )\n        if reward_tensor.ndim == 2:\n            reward_advantages = reward_tensor.swapaxes(0, 1).reshape(-1).copy()\n        elif reward_tensor.ndim == 1:\n            reward_advantages = reward_tensor.reshape(-1).copy()\n        else:\n            raise ValueError(\"reward advantage rollout shape mismatch\")\n        if reward_advantages.shape[0] != raw_cost_advantages.shape[0]:\n            raise ValueError(\"reward and cost diagnostic transition counts differ\")\n        return reward_advantages, raw_costs, raw_cost_advantages\n\n    def _record_lagrangian_rollout_evidence(self) -> None:\n        \"\"\"Record raw actor penalties and dual evidence without mutating training data.\"\"\"\n\n        reward_advantages, raw_costs, raw_cost_advantages = (\n            self._flatten_rollout_diagnostics()\n        )\n        diagnostics = build_constraint_correlation_diagnostics(\n            cost_names=self.lagrangian_schema.names,\n            raw_costs=raw_costs,\n            raw_cost_advantages=raw_cost_advantages,\n            normalized_cost_advantages=normalize_cost_advantages(\n                raw_cost_advantages\n            ),\n            multipliers=self.frozen_lagrange_multipliers,\n            reward_advantages=reward_advantages,\n        )\n        self.last_constraint_correlation_diagnostics = diagnostics\n\n        reports = {\n            name: self.last_dual_update_reports[name]\n            for name in self.lagrangian_schema.names\n        }\n        self.dual_report_history.append(reports)\n        stability = build_dual_stability_diagnostics(\n            cost_names=self.lagrangian_schema.names,\n            report_history=tuple(self.dual_report_history),\n        )\n        batch = self.last_completed_episode_batch\n        if not isinstance(batch, CompletedEpisodeBatch):\n            raise RuntimeError(\"completed episode batch is unavailable\")\n        probe_evidence = self.canonical_action_probe_evidence\n        if probe_evidence is None:\n            self.last_lagrangian_rollout_evidence = None\n        else:\n            self.last_lagrangian_rollout_evidence = (\n                build_lagrangian_rollout_evidence(\n                    actor_composition_mode=self.actor_composition_mode,\n                    schema=self.lagrangian_schema,\n                    correlation_diagnostics=diagnostics,\n                    stability_diagnostics=stability,\n                    dual_reports=reports,\n                    probe_evidence=probe_evidence,\n                    completed_episode_count=batch.completed_episode_count,\n                    censored_episode_count=batch.censored_episode_count,\n                )\n            )\n        self.logger.record(\n            \"lagrangian/penalty_to_reward_l2_ratio\",\n            diagnostics.penalty_to_reward_l2_ratio,\n        )\n\n'''
    model_path.write_text(content.replace(marker, methods + marker))

content = model_path.read_text()
if "self._record_lagrangian_rollout_evidence()" not in content:
    replace_once(
        model_path,
        "        self._train_cost_critic()\n"
        "        self._update_dual_controller()\n",
        "        self._train_cost_critic()\n"
        "        self._update_dual_controller()\n"
        "        self._record_lagrangian_rollout_evidence()\n",
    )

# Keep the warning=False evidence variant internally consistent.
evidence_test = Path("tests/rl/test_lagrangian_evidence.py")
content = evidence_test.read_text()
old_probe = '''        estimates={"drawdown_excess": 0.3, "drawdown_stop_event": 0.0},\n'''
new_probe = '''        estimates={\n            "drawdown_excess": 0.3 if warning else 0.0,\n            "drawdown_stop_event": 0.0,\n        },\n'''
if old_probe in content:
    evidence_test.write_text(content.replace(old_probe, new_probe))
