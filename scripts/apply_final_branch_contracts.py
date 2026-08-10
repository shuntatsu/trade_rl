#!/usr/bin/env python3
"""Apply final BC and execution contracts after the verified branch merge."""

from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str, *, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"{label} marker changed")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


BEHAVIOR_PATH = Path("trade_rl/integrations/behavior_cloning.py")

OLD_EVALUATION = """    gate_batches: list[np.ndarray] = []
    proposal_batches: list[np.ndarray] = []
    composed_batches: list[np.ndarray] = []
    gate_total = 0.0
    target_total = 0.0
    composed_total = 0.0
    weighted_total = 0.0
    batch_weight = 0
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            observations = _tensor_observations(
                _observation_batch(dataset, batch_indices, provider=provider),
                device=device,
            )
            outputs, gate, target, composed, weighted = _hierarchical_batch_losses(
                policy,
                observations,
                labels,
                batch_indices,
                config=config,
                positive_class_weight=positive_class_weight,
                device=device,
            )
            size = len(batch_indices)
            gate_total += float(gate.detach().cpu()) * size
            target_total += float(target.detach().cpu()) * size
            composed_total += float(composed.detach().cpu()) * size
            weighted_total += float(weighted.detach().cpu()) * size
            batch_weight += size
            gate_batches.append(outputs.gate_probabilities.detach().cpu().numpy())
            proposal_batches.append(outputs.target_actions.detach().cpu().numpy())
            composed_batches.append(outputs.composed_actions.detach().cpu().numpy())
    if batch_weight <= 0:
        raise ValueError("hierarchical BC evaluation batch is empty")
    metrics = hierarchical_bc_metrics(
        gate_probabilities=np.concatenate(gate_batches, axis=0),
        proposal_actions=np.concatenate(proposal_batches, axis=0),
        composed_actions=np.concatenate(composed_batches, axis=0),
        labels=labels,
        gate_threshold=config.gate_prediction_threshold,
        indices=indices,
    )
    return _HierarchicalEvaluation(
        losses=HierarchicalBehaviorCloningLosses(
            gate=gate_total / batch_weight,
            target=target_total / batch_weight,
            composed=composed_total / batch_weight,
            weighted=weighted_total / batch_weight,
        ),
        metrics=metrics,
    )
"""

NEW_EVALUATION = """    gate_batches: list[np.ndarray] = []
    proposal_batches: list[np.ndarray] = []
    composed_batches: list[np.ndarray] = []
    gate_total = 0.0
    target_total = 0.0
    composed_total = 0.0
    active_support = 0
    event_support = 0
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            observations = _tensor_observations(
                _observation_batch(dataset, batch_indices, provider=provider),
                device=device,
            )
            outputs, gate, target, composed, _ = _hierarchical_batch_losses(
                policy,
                observations,
                labels,
                batch_indices,
                config=config,
                positive_class_weight=positive_class_weight,
                device=device,
            )
            active = labels.active_mask[batch_indices]
            events = active & labels.gate_labels[batch_indices]
            batch_active_support = int(np.count_nonzero(active))
            batch_event_support = int(np.count_nonzero(events))
            gate_total += float(gate.detach().cpu()) * batch_active_support
            target_total += float(target.detach().cpu()) * batch_event_support
            composed_total += float(composed.detach().cpu()) * batch_active_support
            active_support += batch_active_support
            event_support += batch_event_support
            gate_batches.append(outputs.gate_probabilities.detach().cpu().numpy())
            proposal_batches.append(outputs.target_actions.detach().cpu().numpy())
            composed_batches.append(outputs.composed_actions.detach().cpu().numpy())
    if active_support <= 0:
        raise ValueError("hierarchical BC evaluation batch is empty")
    gate_mean = gate_total / active_support
    target_mean = 0.0 if event_support == 0 else target_total / event_support
    composed_mean = composed_total / active_support
    weighted_mean = (
        config.gate_loss_weight * gate_mean
        + config.target_loss_weight * target_mean
        + config.composed_loss_weight * composed_mean
    )
    metrics = hierarchical_bc_metrics(
        gate_probabilities=np.concatenate(gate_batches, axis=0),
        proposal_actions=np.concatenate(proposal_batches, axis=0),
        composed_actions=np.concatenate(composed_batches, axis=0),
        labels=labels,
        gate_threshold=config.gate_prediction_threshold,
        indices=indices,
    )
    return _HierarchicalEvaluation(
        losses=HierarchicalBehaviorCloningLosses(
            gate=gate_mean,
            target=target_mean,
            composed=composed_mean,
            weighted=weighted_mean,
        ),
        metrics=metrics,
    )
"""

OLD_SPLIT = """    expected_validation_count = (
        0
        if config.validation_fraction == 0.0
        else max(
            1,
            int(math.floor(sample_count * config.validation_fraction)),
        )
    )
    if validation_indices.size != expected_validation_count:
        raise ValueError(
            "explicit behavior-cloning split disagrees with validation_fraction"
        )
    return train_indices, validation_indices
"""

NEW_SPLIT = """    if hasattr(dataset, "episode_ids"):
        expected_split = behavior_cloning_split(
            dataset,
            validation_fraction=config.validation_fraction,
        )
        for name, actual, expected in (
            ("training", train_indices, expected_split.train_indices),
            ("validation", validation_indices, expected_split.validation_indices),
            ("purged", purged_indices, expected_split.purged_indices),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(
                    "explicit behavior-cloning split disagrees with "
                    f"episode-aware {name} split"
                )
        return train_indices, validation_indices

    expected_validation_count = (
        0
        if config.validation_fraction == 0.0
        else max(
            1,
            int(math.floor(sample_count * config.validation_fraction)),
        )
    )
    if validation_indices.size != expected_validation_count:
        raise ValueError(
            "explicit behavior-cloning split disagrees with validation_fraction"
        )
    return train_indices, validation_indices
"""

EXECUTION_PATH = Path("trade_rl/simulation/execution_stress.py")

OLD_EXECUTION = """        object.__setattr__(
            self,
            "tail_slippage_multiplier_floor",
            multiplier_floor,
        )

    def apply(self, base: ExecutionCostConfig) -> ExecutionCostConfig:
        \"\"\"Return a stressed immutable cost configuration without mutating base.\"\"\"

        if not isinstance(base, ExecutionCostConfig):
            raise TypeError("base must be an ExecutionCostConfig")
        tail_multiplier = base.tail_slippage_multiplier
"""

NEW_EXECUTION = """        object.__setattr__(
            self,
            "tail_slippage_multiplier_floor",
            multiplier_floor,
        )

    @property
    def environment_enabled(self) -> bool:
        return (
            self.fee_multiplier > 1.0
            or self.spread_multiplier > 1.0
            or self.impact_multiplier > 1.0
            or self.slippage_std_multiplier > 1.0
            or self.slippage_std_floor > 0.0
            or self.participation_fraction < 1.0
            or self.minimum_order_latency_bars > 0
            or self.tail_slippage_probability_floor > 0.0
            or self.tail_slippage_multiplier_floor > 0.0
            or self.borrow_rate_multiplier > 1.0
        )

    @property
    def enabled(self) -> bool:
        return (
            self.tick_size_factor != 1.0
            or self.lot_size_factor != 1.0
            or self.minimum_notional_factor != 1.0
            or self.adverse_tick_rounding
            or self.environment_enabled
        )

    def apply(self, base: ExecutionCostConfig) -> ExecutionCostConfig:
        \"\"\"Return a stressed immutable cost configuration without mutating base.\"\"\"

        if not isinstance(base, ExecutionCostConfig):
            raise TypeError("base must be an ExecutionCostConfig")
        if not self.environment_enabled:
            return base
        tail_multiplier = base.tail_slippage_multiplier
"""


def main() -> None:
    replace_once(
        BEHAVIOR_PATH,
        OLD_EVALUATION,
        NEW_EVALUATION,
        label="hierarchical evaluation support normalization",
    )
    replace_once(
        BEHAVIOR_PATH,
        OLD_SPLIT,
        NEW_SPLIT,
        label="behavior-cloning explicit split",
    )
    replace_once(
        EXECUTION_PATH,
        OLD_EXECUTION,
        NEW_EXECUTION,
        label="execution stress identity",
    )


if __name__ == "__main__":
    main()
