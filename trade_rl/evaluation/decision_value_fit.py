"""One fixed Ridge fit from admitted synthetic paired-action labels only."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from trade_rl.evaluation.paired_portfolio_label import (
    SyntheticPairedLabel,
    SyntheticPairedLabelBatch,
)
from trade_rl.strategies.decision_value import (
    ConservativeLinearDecisionValueModel,
    LinearDecisionValueModel,
    _finite,
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
    dataset_ids: list[list[str]]


def _collect_training(
    batch: SyntheticPairedLabelBatch,
    *,
    candidate_actions: tuple[tuple[int, ...], ...],
    encoder: Callable[..., np.ndarray] = encode_decision_value_state,
    require_complete_candidate_roster: bool = False,
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
    dataset_ids: list[list[str]] = [[] for _ in codes[1:]]
    used: set[tuple[str, int, tuple[int, ...]]] = set()
    action_groups: dict[tuple[str, int], dict[tuple[int, ...], int]] = {}
    first_state_rows: dict[tuple[str, int], SyntheticPairedLabel] = {}
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
        if require_complete_candidate_roster:
            state_key = (sample.dataset_id, sample.decision_time_ns)
            first_state = first_state_rows.setdefault(state_key, sample)
            if first_state.observation != sample.observation:
                raise ValueError(
                    "TD8 candidate population contains different state observations"
                )
            action_groups.setdefault(state_key, {})[sample.candidate_action] = (
                sample.terminal_time_ns
            )
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
        dataset_ids[head].append(sample.dataset_id)
    if require_complete_candidate_roster:
        required = set(codes[1:])
        for state, present in action_groups.items():
            if set(present) != required or len(set(present.values())) != 1:
                raise ValueError(
                    f"incomplete candidate coverage or unequal horizons for {state}"
                )
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
        dataset_ids=dataset_ids,
    )


# Keep the pre-TD8 ridge penalty/covariance math byte-for-byte equivalent.
# An unpenalized intercept or changed rank handling requires a new model schema.
def _fit_standardized_ridge_legacy_v1(
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


def _ridge_legacy_v1(
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


def _fit_standardized_ridge(
    design: list[np.ndarray],
    labels: list[float],
    penalty: float,
    intervals: list[tuple[int, int]] | None = None,
    *,
    expected_width: int | None = None,
    action: tuple[int, ...] | None = None,
    compute_covariance: bool = False,
    unexcited_state_cols: tuple[int, ...] = (),
    unexcited_prior_std: float = 1.0,
    dataset_ids: list[str] | None = None,
    dof_out: list[float] | None = None,
    solver_version: str = "legacy_v1",
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...] | None]:
    if solver_version not in ("legacy_v1", "td8_v2"):
        raise ValueError("unrecognized ridge solver_version")
    if solver_version == "legacy_v1":
        # V1 original B1c did not include epistemic-prior or residual-dof
        # calibration. Keep its exact solver/covariance for frozen v1 models.
        if dof_out is not None or unexcited_prior_std != 1.0:
            raise ValueError(
                "TD8-only uncertainty parameters require solver_version=td8_v2"
            )
        return _fit_standardized_ridge_legacy_v1(
            design, labels, penalty, intervals, compute_covariance=compute_covariance
        )
    p_total = (
        expected_width
        if expected_width is not None
        else (design[0].shape[0] if design else 0)
    )
    if len(design) < 3:
        raise ValueError("insufficient fixed candidate coverage for Ridge")
    matrix = np.stack(design)
    target = np.asarray(labels, dtype=np.float64)
    n_samples, p_actual = matrix.shape
    if p_actual != p_total:
        raise ValueError(
            f"design matrix width {p_actual} does not match expected {p_total}"
        )

    feat_matrix = matrix[:, 1:]
    means = np.mean(feat_matrix, axis=0)
    centered = feat_matrix - means
    stds = np.std(centered, axis=0)

    active_cols = np.where(stds > 1e-8)[0]
    p_act = len(active_cols)

    if p_act == 0:
        z_aug = np.ones((n_samples, 1), dtype=np.float64)
        pen = np.zeros(1, dtype=np.float64)
        gram = z_aug.T @ z_aug + np.diag(pen)
        inv_gram = np.linalg.inv(gram)
        gamma = inv_gram @ (z_aug.T @ target)
        resid = target - (z_aug @ gamma)
        ssr = float(resid @ resid)
        a_mat = np.zeros((1, p_total), dtype=np.float64)
        a_mat[0, 0] = 1.0
        w_aug = a_mat.T @ gamma
        r_dim = 0
    else:
        z_raw = centered[:, active_cols] / stds[active_cols]
        u_svd, s_svd, vt_svd = np.linalg.svd(z_raw, full_matrices=False)
        keep = s_svd > (1e-4 * np.sqrt(float(n_samples)))
        v_keep = vt_svd[keep]
        z_orth = z_raw @ v_keep.T
        r_dim = z_orth.shape[1]

        z_aug = np.ones((n_samples, 1 + r_dim), dtype=np.float64)
        z_aug[:, 1:] = z_orth
        pen = np.zeros(1 + r_dim, dtype=np.float64)
        pen[1:] = penalty
        gram = z_aug.T @ z_aug + np.diag(pen)
        inv_gram = np.linalg.inv(gram)
        gamma = inv_gram @ (z_aug.T @ target)

        resid = target - (z_aug @ gamma)
        ssr = float(resid @ resid)

        a_mat = np.zeros((1 + r_dim, p_total), dtype=np.float64)
        a_mat[0, 0] = 1.0
        if r_dim > 0:
            scaled_v = v_keep / stds[active_cols][None, :]
            a_mat[1:, 0] = -(scaled_v @ means[active_cols])
            for k_idx, col in enumerate(active_cols):
                a_mat[1:, 1 + col] = scaled_v[:, k_idx]
        w_aug = a_mat.T @ gamma

    if not np.isfinite(w_aug).all():
        raise ValueError("fit produced non-finite action values")

    cov_matrix: tuple[tuple[float, ...], ...] | None = None
    if compute_covariance:
        if intervals is not None and len(intervals) == n_samples:
            dec_t = np.array([iv[0] for iv in intervals], dtype=np.float64)
            term_t = np.array([iv[1] for iv in intervals], dtype=np.float64)
            durations = np.maximum(1.0, term_t - dec_t)
            ov = np.maximum(
                0.0,
                np.minimum(term_t[:, None], term_t[None, :])
                - np.maximum(dec_t[:, None], dec_t[None, :]),
            )
            omega = ov / np.sqrt(durations[:, None] * durations[None, :])
            if dataset_ids is not None and len(dataset_ids) == n_samples:
                same_d = np.equal.outer(dataset_ids, dataset_ids)
                omega = np.where(same_d, omega, 0.0)
        else:
            omega = np.eye(n_samples, dtype=np.float64)

        h_mat = z_aug @ inv_gram @ z_aug.T
        i_minus_h = np.eye(n_samples, dtype=np.float64) - h_mat
        denom = max(1.0, float(np.sum((i_minus_h @ omega) * i_minus_h)))
        if dof_out is not None:
            dof_out.append(denom)
        sigma2_u = ssr / denom

        meat = z_aug.T @ omega @ z_aug
        cov_gamma = sigma2_u * (inv_gram @ meat @ inv_gram)
        cov_gamma = 0.5 * (cov_gamma + cov_gamma.T)

        eigvals, eigvecs = np.linalg.eigh(cov_gamma)
        eigvals = np.maximum(eigvals, 0.0)
        l_gamma = eigvecs * np.sqrt(eigvals)
        l_active = a_mat.T @ l_gamma

        extra_cols: list[np.ndarray] = []
        active_set = set(int(c) for c in active_cols)
        for col in sorted(unexcited_state_cols):
            if col not in active_set and 1 + col < p_total:
                vec = np.zeros(p_total, dtype=np.float64)
                vec[0] = -float(means[col]) * unexcited_prior_std
                vec[1 + col] = unexcited_prior_std
                extra_cols.append(vec[:, None])

        cov_factor = np.hstack([l_active, *extra_cols]) if extra_cols else l_active
        cov_w = cov_factor @ cov_factor.T
        cov_w = 0.5 * (cov_w + cov_w.T)
        if not np.isfinite(cov_w).all():
            raise ValueError("fit produced non-finite covariance")
        cov_matrix = tuple(tuple(float(v) for v in row) for row in cov_w)

    return tuple(float(val) for val in w_aug), cov_matrix


def fit_synthetic_linear_decision_value(
    batch: SyntheticPairedLabelBatch,
    *,
    candidate_actions: tuple[tuple[int, ...], ...],
    ridge_penalty: float = _FIXED_RIDGE_PENALTY,
    solver_version: str = "legacy_v1",
) -> LinearDecisionValueModel:
    """Fit one fixed state-action Ridge head for each non-HOLD joint candidate.

    This does not tune a hyperparameter, reweight overlapping observations as
    independent markets, or authorize external/real-data admission.
    """
    pen = _finite(ridge_penalty, field="ridge_penalty")
    if pen <= 0.0:
        raise ValueError("ridge_penalty must be positive and finite")
    t = _collect_training(
        batch,
        candidate_actions=candidate_actions,
        require_complete_candidate_roster=(solver_version == "td8_v2"),
    )
    n_symbols = len(t.symbols)
    width = 1 + n_symbols * len(t.feature_names) + len(t.global_names) + n_symbols
    d_feat = width - 1
    unexcited = tuple(range(d_feat - n_symbols, d_feat))
    fitted = tuple(
        _fit_standardized_ridge(
            d,
            v,
            pen,
            iv,
            expected_width=width,
            action=action,
            unexcited_state_cols=unexcited,
            dataset_ids=d_ids,
            solver_version=solver_version,
        )[0]
        for d, v, iv, action, d_ids in zip(
            t.rows, t.values, t.intervals, t.codes[1:], t.dataset_ids, strict=True
        )
    )
    return LinearDecisionValueModel(
        schema_version=(
            "synthetic_linear_decision_value_v2"
            if solver_version == "td8_v2"
            else "synthetic_linear_decision_value_v1"
        ),
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
        ridge_penalty=pen,
        training_rows=len(batch.samples),
        overlap_groups=t.groups,
        coefficients=fitted,
    )


def fit_synthetic_shallow_decision_value(
    batch: SyntheticPairedLabelBatch,
    *,
    candidate_actions: tuple[tuple[int, ...], ...],
    ridge_penalty: float = SHALLOW_RIDGE_PENALTY,
    solver_version: str = "legacy_v1",
) -> ShallowDecisionValueModel:
    """B2: same population as B1, fixed hidden layer, one fixed Ridge per head."""
    pen = _finite(ridge_penalty, field="ridge_penalty")
    if pen <= 0.0:
        raise ValueError("ridge_penalty must be positive and finite")
    t = _collect_training(
        batch,
        candidate_actions=candidate_actions,
        require_complete_candidate_roster=(solver_version == "td8_v2"),
    )
    n_symbols = len(t.symbols)
    state_width = 1 + n_symbols * len(t.feature_names) + len(t.global_names) + n_symbols
    hidden = fixed_hidden_weights(state_width)
    if solver_version not in ("legacy_v1", "td8_v2"):
        raise ValueError("unrecognized shallow solver_version")
    if solver_version == "legacy_v1":
        legacy_coefficients = tuple(
            _ridge_legacy_v1([expand_state(x, hidden) for x in d], v, pen)
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
            ridge_penalty=pen,
            training_rows=len(batch.samples),
            overlap_groups=t.groups,
            hidden_weights=hidden,
            coefficients=legacy_coefficients,
        )

    non_empty_rows = [r for sublist in t.rows for r in sublist]
    if not non_empty_rows:
        raise ValueError("insufficient fixed candidate coverage for Ridge")
    all_x = np.stack(non_empty_rows)[:, 1:]
    input_means = tuple(float(m) for m in np.mean(all_x, axis=0))
    stds = np.std(all_x - np.asarray(input_means), axis=0)
    input_scales = tuple(float(1.0 / s) if s > 1e-8 else 0.0 for s in stds)

    fitted: list[tuple[float, ...]] = []
    hidden_w = np.asarray(hidden, dtype=np.float64)
    means_arr = np.asarray(input_means, dtype=np.float64)
    scales_arr = np.asarray(input_scales, dtype=np.float64)

    for d, v, action in zip(t.rows, t.values, t.codes[1:], strict=True):
        if len(d) < 3:
            raise ValueError("insufficient fixed candidate coverage for Ridge")
        x_raw = np.stack(d)[:, 1:]
        x_norm = (x_raw - means_arr) * scales_arr
        h_act = np.tanh(x_norm @ hidden_w.T)
        design = np.column_stack((np.ones(len(d), dtype=np.float64), x_norm, h_act))

        pen_diag = np.zeros(design.shape[1], dtype=np.float64)
        pen_diag[1:] = pen
        gram = design.T @ design + np.diag(pen_diag)
        target = np.asarray(v, dtype=np.float64)
        coefs = np.linalg.solve(gram, design.T @ target)
        if not np.isfinite(coefs).all():
            raise ValueError("fit produced non-finite action values")
        fitted.append(tuple(float(x) for x in coefs))

    return ShallowDecisionValueModel(
        schema_version="synthetic_shallow_decision_value_v2",
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
        ridge_penalty=pen,
        training_rows=len(batch.samples),
        overlap_groups=t.groups,
        hidden_weights=hidden,
        coefficients=tuple(fitted),
        input_means=input_means,
        input_scales=input_scales,
    )


def fit_conservative_linear_decision_value(
    batch: SyntheticPairedLabelBatch,
    *,
    candidate_actions: tuple[tuple[int, ...], ...],
    ridge_penalty: float = _FIXED_RIDGE_PENALTY,
    uncertainty_penalty_weight: float = 1.0,
    one_way_switch_cost: float = 0.0,
    unexcited_prior_std: float = 1.0,
    residual_dof_out: list[float] | None = None,
    solver_version: str = "legacy_v1",
) -> ConservativeLinearDecisionValueModel:
    """Fit conservative linear action-value head with uncertainty covariance.

    Uses design matrix from encode_decision_value_state_v2 (including in-flight state)
    and estimates group-degrees-of-freedom adjusted residual covariance per candidate head.
    ``unexcited_prior_std`` is the epistemic prior on state coordinates the training
    labels never varied; the default is the uninformative 1.0 (the post-hoc 1e-3
    calibration of the TD7 re-run is withdrawn). When ``residual_dof_out`` is a list,
    the effective residual degrees of freedom of every head is appended to it.
    """
    pen = _finite(ridge_penalty, field="ridge_penalty")
    if pen <= 0.0:
        raise ValueError("ridge_penalty must be positive and finite")
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
        require_complete_candidate_roster=(solver_version == "td8_v2"),
    )
    n_symbols = len(t.symbols)
    width = 1 + n_symbols * len(t.feature_names) + len(t.global_names) + 2 * n_symbols
    d_feat = width - 1
    unexcited = tuple(range(d_feat - 2 * n_symbols, d_feat))
    coefficients_list: list[tuple[float, ...]] = []
    covariances_list: list[tuple[tuple[float, ...], ...]] = []

    for d, v, iv, action, d_ids in zip(
        t.rows, t.values, t.intervals, t.codes[1:], t.dataset_ids, strict=True
    ):
        coefs, cov = _fit_standardized_ridge(
            d,
            v,
            pen,
            iv,
            expected_width=width,
            action=action,
            compute_covariance=True,
            unexcited_state_cols=unexcited,
            unexcited_prior_std=unexcited_prior_std,
            dataset_ids=d_ids,
            dof_out=residual_dof_out,
            solver_version=solver_version,
        )
        assert cov is not None
        coefficients_list.append(coefs)
        covariances_list.append(cov)

    return ConservativeLinearDecisionValueModel(
        schema_version=(
            "conservative_linear_decision_value_v3"
            if solver_version == "td8_v2"
            else "conservative_linear_decision_value_v2"
        ),
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
        ridge_penalty=pen,
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
