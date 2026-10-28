"""One fixed Ridge fit from admitted synthetic paired-action labels only."""

from __future__ import annotations

import math

import numpy as np

from trade_rl.evaluation.paired_portfolio_label import (
    SyntheticPairedLabelBatch,
)
from trade_rl.strategies.decision_value import (
    LinearDecisionValueModel,
    encode_decision_value_state,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    decode_portfolio_action_codes,
)

_FIXED_RIDGE_PENALTY = 1e-4


def fit_synthetic_linear_decision_value(
    batch: SyntheticPairedLabelBatch,
    *,
    candidate_actions: tuple[tuple[int, ...], ...],
) -> LinearDecisionValueModel:
    """Fit one fixed state-action Ridge head for each non-HOLD joint candidate.

    This does not tune a hyperparameter, reweight overlapping observations as
    independent markets, or authorize external/real-data admission.
    """
    if not isinstance(batch, SyntheticPairedLabelBatch):
        raise TypeError("batch must be a synthetic paired-label batch")
    if type(candidate_actions) is not tuple or len(candidate_actions) < 2:
        raise ValueError("candidate roster requires HOLD and non-HOLD actions")
    if not batch.samples or len(batch.group_ids) != len(batch.samples):
        raise ValueError("batch training rows or overlap group identity are invalid")
    first = batch.samples[0]
    n_symbols = len(first.observation.symbols)
    codes = tuple(
        decode_portfolio_action_codes(action, n_symbols=n_symbols)
        for action in candidate_actions
    )
    if codes[0] != (ACTION_HOLD,) * n_symbols or len(set(codes)) != len(codes):
        raise ValueError("candidate roster must begin with unique joint HOLD")
    symbols = first.observation.symbols
    feature_names = first.observation.feature_names
    global_names = first.observation.global_feature_names
    joint_weights = first.target_abs_weights
    ref_digest = first.reference_policy_digest
    cost_digest = first.execution_policy_digest
    feature_digest = first.feature_config_digest
    normalizer_digest = first.normalization_digest
    index = {action: i for i, action in enumerate(codes)}
    rows: list[list[np.ndarray]] = [[] for _ in codes[1:]]
    values: list[list[float]] = [[] for _ in codes[1:]]
    used: set[tuple[str, int, tuple[int, ...]]] = set()
    group = -1
    group_end = -(2**63)
    for i, sample in enumerate(batch.samples):
        if sample.source_scope != "synthetic_only":
            raise ValueError("only synthetic training samples are supported")
        if (
            sample.observation.symbols != symbols
            or sample.observation.feature_names != feature_names
            or sample.observation.global_feature_names != global_names
            or sample.target_abs_weights != joint_weights
            or sample.reference_policy_digest != ref_digest
            or sample.execution_policy_digest != cost_digest
            or sample.feature_config_digest != feature_digest
            or sample.normalization_digest != normalizer_digest
        ):
            raise ValueError(
                "synthetic batch source, feature, action or digest mismatch"
            )
        if (
            sample.fit_cutoff_ns != batch.fit_cutoff_ns
            or sample.decision_time_ns >= batch.fit_cutoff_ns
            or sample.label_first_available_ns >= batch.fit_cutoff_ns
            or sample.decision_time_ns != sample.observation.timestamp_ns
            or sample.terminal_time_ns > sample.label_first_available_ns
        ):
            raise ValueError("sample maturity or fit cutoff is invalid")
        if not math.isfinite(sample.label_delta):
            raise ValueError("sample label must be finite")
        if sample.baseline_action != codes[0] or sample.candidate_action not in index:
            raise ValueError("sample candidate action not in fixed HOLD roster")
        if sample.candidate_action == codes[0]:
            raise ValueError("HOLD labels are structural zero, not training targets")
        key = (sample.dataset_id, sample.decision_time_ns, sample.candidate_action)
        if key in used:
            raise ValueError("duplicate synthetic training sample")
        used.add(key)
        if sample.decision_time_ns > group_end:
            group += 1
            group_end = sample.terminal_time_ns
        else:
            group_end = max(group_end, sample.terminal_time_ns)
        if batch.group_ids[i] != group:
            raise ValueError("overlap group contract differs from actual label windows")
        vector = encode_decision_value_state(
            sample.observation,
            symbols=symbols,
            feature_names=feature_names,
            global_feature_names=global_names,
        )
        head = index[sample.candidate_action] - 1
        rows[head].append(vector)
        values[head].append(float(sample.label_delta))
    if batch.evaluation_start_ns <= batch.fit_cutoff_ns:
        raise ValueError("evaluation must begin after the fit cutoff")
    fitted: list[tuple[float, ...]] = []
    for design, labels in zip(rows, values, strict=True):
        if len(design) < 3:
            raise ValueError("insufficient fixed candidate coverage for Ridge")
        matrix = np.stack(design)
        target = np.asarray(labels, dtype=np.float64)
        gram = matrix.T @ matrix + _FIXED_RIDGE_PENALTY * np.eye(matrix.shape[1])
        coefficients = np.linalg.solve(gram, matrix.T @ target)
        if not np.isfinite(coefficients).all():
            raise ValueError("fit produced non-finite action values")
        fitted.append(tuple(float(value) for value in coefficients))
    return LinearDecisionValueModel(
        schema_version="synthetic_linear_decision_value_v1",
        source_scope="synthetic_only",
        symbols=symbols,
        feature_names=feature_names,
        global_feature_names=global_names,
        feature_config_digest=feature_digest,
        normalization_digest=normalizer_digest,
        execution_policy_digest=cost_digest,
        reference_policy_digest=ref_digest,
        target_abs_weights=joint_weights,
        candidate_actions=codes,
        fit_cutoff_ns=batch.fit_cutoff_ns,
        evaluation_start_ns=batch.evaluation_start_ns,
        ridge_penalty=_FIXED_RIDGE_PENALTY,
        training_rows=len(batch.samples),
        overlap_groups=group + 1,
        coefficients=tuple(fitted),
    )


__all__ = ["fit_synthetic_linear_decision_value"]
