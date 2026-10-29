"""One fixed Ridge fit from admitted synthetic paired-action labels only."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from trade_rl.evaluation.paired_portfolio_label import (
    SyntheticPairedLabelBatch,
)
from trade_rl.strategies.decision_value import (
    ConservativeLinearDecisionValueModel,
    LinearDecisionValueModel,
    encode_decision_value_state,
    encode_decision_value_state_v2,
)
from trade_rl.strategies.decision_value_shallow import (
    RIDGE_PENALTY as SHALLOW_RIDGE_PENALTY,
)
from trade_rl.strategies.decision_value_shallow import (
    ShallowDecisionValueModel,
    expand_state,
    fixed_hidden_weights,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    decode_portfolio_action_codes,
)

_FIXED_RIDGE_PENALTY = 1e-4


@dataclass(frozen=True, slots=True)
class _TrainingSet:
    """Validated B1/B2 shared population: identical rows, labels and masks."""

    source_scope: str
    codes: tuple[tuple[int, ...], ...]
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    global_names: tuple[str, ...]
    joint_weights: tuple[float, ...]
    ref_digest: str
    cost_digest: str
    feature_digest: str
    normalizer_digest: str
    groups: int
    rows: list[list[np.ndarray]]
    values: list[list[float]]
    intervals: list[list[tuple[int, int]]]


def _collect_training(
    batch: SyntheticPairedLabelBatch,
    *,
    candidate_actions: tuple[tuple[int, ...], ...],
    encoder: Callable[..., np.ndarray] = encode_decision_value_state,
) -> _TrainingSet:
    """Shared validation for every value head; no model-specific row dropping."""
    if not isinstance(batch, SyntheticPairedLabelBatch):
        raise TypeError("batch must be a synthetic paired-label batch")
    if type(candidate_actions) is not tuple or len(candidate_actions) < 2:
        raise ValueError("candidate roster requires HOLD and non-HOLD actions")
    if not batch.samples or len(batch.group_ids) != len(batch.samples):
        raise ValueError("batch training rows or overlap group identity are invalid")
    first = batch.samples[0]
    if first.source_scope not in ("synthetic_only", "real_development"):
        raise ValueError(
            "only synthetic or real_development training samples with matching source_scope are supported"
        )
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
    intervals: list[list[tuple[int, int]]] = [[] for _ in codes[1:]]
    used: set[tuple[str, int, tuple[int, ...]]] = set()
    group = -1
    group_end = -(2**63)
    for i, sample in enumerate(batch.samples):
        if sample.source_scope != first.source_scope:
            raise ValueError(
                "only synthetic or real_development training samples with matching source_scope are supported"
            )
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
        obs_to_encode = (
            replace(sample.observation, active_orders=())
            if sample.observation.active_orders
            and encoder is encode_decision_value_state
            else sample.observation
        )
        vector = encoder(
            obs_to_encode,
            symbols=symbols,
            feature_names=feature_names,
            global_feature_names=global_names,
        )
        head = index[sample.candidate_action] - 1
        rows[head].append(vector)
        values[head].append(float(sample.label_delta))
        intervals[head].append((sample.decision_time_ns, sample.terminal_time_ns))
    if batch.evaluation_start_ns <= batch.fit_cutoff_ns:
        raise ValueError("evaluation must begin after the fit cutoff")
    return _TrainingSet(
        source_scope=first.source_scope,
        codes=codes,
        symbols=symbols,
        feature_names=feature_names,
        global_names=global_names,
        joint_weights=joint_weights,
        ref_digest=ref_digest,
        cost_digest=cost_digest,
        feature_digest=feature_digest,
        normalizer_digest=normalizer_digest,
        groups=group + 1,
        rows=rows,
        values=values,
        intervals=intervals,
    )


def _fit_standardized_ridge(
    design: list[np.ndarray],
    labels: list[float],
    penalty: float,
    intervals: list[tuple[int, int]] | None = None,
    *,
    compute_covariance: bool = False,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...] | None]:
    if len(design) < 3:
        raise ValueError("insufficient fixed candidate coverage for Ridge")
    matrix = np.stack(design)
    target = np.asarray(labels, dtype=np.float64)
    n_samples, p_total = matrix.shape

    # Non-bias feature standardization and constant collinearity removal
    feat_matrix = matrix[:, 1:]
    means = np.mean(feat_matrix, axis=0)
    stds = np.std(feat_matrix, axis=0)
    active_mask = stds >= 1e-8
    active_indices = np.where(active_mask)[0]
    k_active = len(active_indices)
    p_eff = 1 + k_active

    z_matrix = np.empty((n_samples, p_eff), dtype=np.float64)
    z_matrix[:, 0] = 1.0
    if k_active > 0:
        z_matrix[:, 1:] = (
            feat_matrix[:, active_indices] - means[active_indices]
        ) / stds[active_indices]

    gram = z_matrix.T @ z_matrix + penalty * np.eye(p_eff)
    beta = np.linalg.solve(gram, z_matrix.T @ target)
    if not np.isfinite(beta).all():
        raise ValueError("fit produced non-finite action values")

    # Affine folding matrix A: z_eff = A @ x
    a_matrix = np.zeros((p_eff, p_total), dtype=np.float64)
    a_matrix[0, 0] = 1.0
    for m in range(k_active):
        col = 1 + active_indices[m]
        mu = means[active_indices[m]]
        s = stds[active_indices[m]]
        a_matrix[1 + m, 0] = -mu / s
        a_matrix[1 + m, col] = 1.0 / s

    coefficients = a_matrix.T @ beta
    if not np.isfinite(coefficients).all():
        raise ValueError("fit produced non-finite action values")

    cov_matrix: tuple[tuple[float, ...], ...] | None = None
    if compute_covariance:
        residuals = target - z_matrix @ beta
        ssr = float(residuals @ residuals)

        # Average overlap multiplicity m_bar across intervals
        if intervals is not None and len(intervals) == n_samples:
            multiplicities: list[int] = []
            for i in range(n_samples):
                d_i, t_i = intervals[i]
                m_i = sum(
                    1
                    for j in range(n_samples)
                    if max(d_i, intervals[j][0]) < min(t_i, intervals[j][1])
                )
                multiplicities.append(m_i)
            m_bar = float(np.mean(multiplicities))
        else:
            m_bar = 1.0

        m_bar = max(1.0, m_bar)
        n_eff = n_samples / m_bar
        dof = max(1.0, n_eff - p_eff)
        sigma2 = max(1e-12, (ssr / dof) * m_bar)

        inv_gram = np.linalg.inv(gram)
        cov_beta = sigma2 * inv_gram
        cov_beta = 0.5 * (cov_beta + cov_beta.T)
        cov_w = a_matrix.T @ cov_beta @ a_matrix
        cov_w = 0.5 * (cov_w + cov_w.T)
        if not np.isfinite(cov_w).all():
            raise ValueError("fit produced non-finite covariance")
        cov_matrix = tuple(tuple(float(val) for val in row) for row in cov_w)

    return tuple(float(val) for val in coefficients), cov_matrix


def _ridge(
    design: list[np.ndarray], labels: list[float], penalty: float
) -> tuple[float, ...]:
    if len(design) < 3:
        raise ValueError("insufficient fixed candidate coverage for Ridge")
    matrix = np.stack(design)
    target = np.asarray(labels, dtype=np.float64)
    gram = matrix.T @ matrix + penalty * np.eye(matrix.shape[1])
    coefficients = np.linalg.solve(gram, matrix.T @ target)
    if not np.isfinite(coefficients).all():
        raise ValueError("fit produced non-finite action values")
    return tuple(float(value) for value in coefficients)


def fit_synthetic_linear_decision_value(
    batch: SyntheticPairedLabelBatch,
    *,
    candidate_actions: tuple[tuple[int, ...], ...],
) -> LinearDecisionValueModel:
    """Fit one fixed state-action Ridge head for each non-HOLD joint candidate.

    This does not tune a hyperparameter, reweight overlapping observations as
    independent markets, or authorize external/real-data admission.
    """
    t = _collect_training(batch, candidate_actions=candidate_actions)
    fitted = tuple(
        _fit_standardized_ridge(d, v, _FIXED_RIDGE_PENALTY, iv)[0]
        for d, v, iv in zip(t.rows, t.values, t.intervals, strict=True)
    )
    return LinearDecisionValueModel(
        schema_version="synthetic_linear_decision_value_v1",
        source_scope=t.source_scope,
        symbols=t.symbols,
        feature_names=t.feature_names,
        global_feature_names=t.global_names,
        feature_config_digest=t.feature_digest,
        normalization_digest=t.normalizer_digest,
        execution_policy_digest=t.cost_digest,
        reference_policy_digest=t.ref_digest,
        target_abs_weights=t.joint_weights,
        candidate_actions=t.codes,
        fit_cutoff_ns=batch.fit_cutoff_ns,
        evaluation_start_ns=batch.evaluation_start_ns,
        ridge_penalty=_FIXED_RIDGE_PENALTY,
        training_rows=len(batch.samples),
        overlap_groups=t.groups,
        coefficients=fitted,
    )


def fit_synthetic_shallow_decision_value(
    batch: SyntheticPairedLabelBatch,
    *,
    candidate_actions: tuple[tuple[int, ...], ...],
) -> ShallowDecisionValueModel:
    """B2: same population as B1, fixed hidden layer, one fixed Ridge per head."""
    t = _collect_training(batch, candidate_actions=candidate_actions)
    width = t.rows[0][0].shape[0] if t.rows[0] else 0
    if width == 0:
        raise ValueError("insufficient fixed candidate coverage for Ridge")
    hidden = fixed_hidden_weights(width)
    fitted = tuple(
        _ridge([expand_state(x, hidden) for x in d], v, SHALLOW_RIDGE_PENALTY)
        for d, v in zip(t.rows, t.values, strict=True)
    )
    return ShallowDecisionValueModel(
        schema_version="synthetic_shallow_decision_value_v1",
        source_scope=t.source_scope,
        symbols=t.symbols,
        feature_names=t.feature_names,
        global_feature_names=t.global_names,
        feature_config_digest=t.feature_digest,
        normalization_digest=t.normalizer_digest,
        execution_policy_digest=t.cost_digest,
        reference_policy_digest=t.ref_digest,
        target_abs_weights=t.joint_weights,
        candidate_actions=t.codes,
        fit_cutoff_ns=batch.fit_cutoff_ns,
        evaluation_start_ns=batch.evaluation_start_ns,
        ridge_penalty=SHALLOW_RIDGE_PENALTY,
        training_rows=len(batch.samples),
        overlap_groups=t.groups,
        hidden_weights=hidden,
        coefficients=fitted,
    )


def fit_conservative_linear_decision_value(
    batch: SyntheticPairedLabelBatch,
    *,
    candidate_actions: tuple[tuple[int, ...], ...],
    uncertainty_penalty_weight: float = 1.0,
    one_way_switch_cost: float = 0.0,
) -> ConservativeLinearDecisionValueModel:
    """Fit conservative linear action-value head with uncertainty covariance.

    Uses design matrix from encode_decision_value_state_v2 (including in-flight state)
    and estimates group-degrees-of-freedom adjusted residual covariance per candidate head.
    """
    if isinstance(uncertainty_penalty_weight, bool) or not isinstance(
        uncertainty_penalty_weight, (int, float)
    ):
        raise TypeError("uncertainty_penalty_weight must be a finite real")
    if (
        not math.isfinite(uncertainty_penalty_weight)
        or uncertainty_penalty_weight < 0.0
    ):
        raise ValueError("uncertainty_penalty_weight must be non-negative and finite")
    if isinstance(one_way_switch_cost, bool) or not isinstance(
        one_way_switch_cost, (int, float)
    ):
        raise TypeError("one_way_switch_cost must be a finite real")
    if not math.isfinite(one_way_switch_cost) or one_way_switch_cost < 0.0:
        raise ValueError("one_way_switch_cost must be non-negative and finite")

    t = _collect_training(
        batch,
        candidate_actions=candidate_actions,
        encoder=encode_decision_value_state_v2,
    )
    coefficients_list: list[tuple[float, ...]] = []
    covariances_list: list[tuple[tuple[float, ...], ...]] = []

    for d, v, iv in zip(t.rows, t.values, t.intervals, strict=True):
        coefs, cov = _fit_standardized_ridge(
            d, v, _FIXED_RIDGE_PENALTY, iv, compute_covariance=True
        )
        assert cov is not None
        coefficients_list.append(coefs)
        covariances_list.append(cov)

    return ConservativeLinearDecisionValueModel(
        schema_version="conservative_linear_decision_value_v2",
        source_scope=t.source_scope,
        symbols=t.symbols,
        feature_names=t.feature_names,
        global_feature_names=t.global_names,
        feature_config_digest=t.feature_digest,
        normalization_digest=t.normalizer_digest,
        execution_policy_digest=t.cost_digest,
        reference_policy_digest=t.ref_digest,
        target_abs_weights=t.joint_weights,
        candidate_actions=t.codes,
        fit_cutoff_ns=batch.fit_cutoff_ns,
        evaluation_start_ns=batch.evaluation_start_ns,
        ridge_penalty=_FIXED_RIDGE_PENALTY,
        uncertainty_penalty_weight=float(uncertainty_penalty_weight),
        one_way_switch_cost=float(one_way_switch_cost),
        training_rows=len(batch.samples),
        overlap_groups=t.groups,
        coefficients=tuple(coefficients_list),
        covariance_matrices=tuple(covariances_list),
    )


__all__ = [
    "fit_conservative_linear_decision_value",
    "fit_synthetic_linear_decision_value",
    "fit_synthetic_shallow_decision_value",
]
