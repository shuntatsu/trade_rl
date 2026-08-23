from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/learning/causal_alpha_v4.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one match, got {count}: {old[:120]!r}")
    text = text.replace(old, new, 1)


replace_once(
    '_V4_MINIMUM_STATE_ESS: Final = 30.0\n',
    '_V4_MINIMUM_STATE_ESS: Final = 30.0\n'
    'CAUSAL_ALPHA_V4_TARGET_SCHEMA: Final = "causal_alpha_v4_target_v1"\n'
    '_V4_TARGET_EPSILON: Final = 1e-12\n',
)

addition = r'''

@dataclass(frozen=True, slots=True)
class CausalAlphaV4TargetConfig:
    """The frozen first V4 slow-anchor/fast-impulse target hypothesis."""

    slow_target_magnitudes: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.25)
    fast_deviation_magnitudes: tuple[float, ...] = (0.0, 0.025, 0.05)
    uncertainty_multiplier: float = 1.0
    execution_cost_multiplier: float = 1.5
    edge_margin: float = 0.001
    slow_rebalance_decisions: int = 16
    fast_rebalance_decisions: int = 4
    maximum_final_target_delta: float = 0.125
    maximum_fast_absolute_deviation: float = 0.05
    schema_version: str = CAUSAL_ALPHA_V4_TARGET_SCHEMA

    def __post_init__(self) -> None:
        slow = tuple(float(value) for value in self.slow_target_magnitudes)
        fast = tuple(float(value) for value in self.fast_deviation_magnitudes)
        if slow != (0.0, 0.025, 0.05, 0.10, 0.25):
            raise ValueError("V4 slow target magnitudes must remain frozen")
        if fast != (0.0, 0.025, 0.05):
            raise ValueError("V4 fast deviation magnitudes must remain frozen")
        if self.uncertainty_multiplier != 1.0:
            raise ValueError("V4 uncertainty multiplier must remain 1.0")
        if self.execution_cost_multiplier != 1.5:
            raise ValueError("V4 execution cost multiplier must remain 1.5")
        if self.edge_margin != 0.001:
            raise ValueError("V4 edge margin must remain 0.001")
        if self.slow_rebalance_decisions != 16:
            raise ValueError("V4 slow cadence must remain 16 decisions")
        if self.fast_rebalance_decisions != 4:
            raise ValueError("V4 fast cadence must remain 4 decisions")
        if self.maximum_final_target_delta != 0.125:
            raise ValueError("V4 maximum final target delta must remain 0.125")
        if self.maximum_fast_absolute_deviation != 0.05:
            raise ValueError("V4 maximum fast deviation must remain 0.05")
        if self.schema_version != CAUSAL_ALPHA_V4_TARGET_SCHEMA:
            raise ValueError("unsupported V4 target config schema")
        object.__setattr__(self, "slow_target_magnitudes", slow)
        object.__setattr__(self, "fast_deviation_magnitudes", fast)

    @property
    def digest(self) -> str:
        return content_digest(self)


def _v4_direct_objective(
    *,
    target: float,
    previous: float,
    expected_return: float,
    uncertainty: float,
    one_way_cost_rate: float,
    config: CausalAlphaV4TargetConfig,
) -> float:
    delta = target - previous
    turnover = abs(delta)
    return (
        delta * expected_return
        - config.uncertainty_multiplier * turnover * uncertainty
        - turnover
        * (one_way_cost_rate * config.execution_cost_multiplier + config.edge_margin)
    )


def _v4_staged_objective(
    *,
    previous: float,
    anchor: float,
    final: float,
    slow_expected_return: float,
    slow_uncertainty: float,
    fast_expected_return: float,
    fast_uncertainty: float,
    one_way_cost_rate: float,
    config: CausalAlphaV4TargetConfig,
) -> tuple[float, float, float]:
    slow = _v4_direct_objective(
        target=anchor,
        previous=previous,
        expected_return=slow_expected_return,
        uncertainty=slow_uncertainty,
        one_way_cost_rate=one_way_cost_rate,
        config=config,
    )
    fast_final = _v4_direct_objective(
        target=final,
        previous=previous,
        expected_return=fast_expected_return,
        uncertainty=fast_uncertainty,
        one_way_cost_rate=one_way_cost_rate,
        config=config,
    )
    fast_anchor = _v4_direct_objective(
        target=anchor,
        previous=previous,
        expected_return=fast_expected_return,
        uncertainty=fast_uncertainty,
        one_way_cost_rate=one_way_cost_rate,
        config=config,
    )
    fast_improvement = fast_final - fast_anchor
    return slow, fast_improvement, slow + fast_improvement


def _v4_is_risk_reduction(previous: float, target: float) -> bool:
    if abs(target - previous) <= _V4_TARGET_EPSILON:
        return True
    if abs(previous) <= _V4_TARGET_EPSILON:
        return abs(target) <= _V4_TARGET_EPSILON
    return (
        previous * target >= -_V4_TARGET_EPSILON
        and abs(target) <= abs(previous) + _V4_TARGET_EPSILON
    )


def _v4_consensus_allows(
    *, previous: float, target: float, fast_expected_return: float, direction_score: float
) -> bool:
    if _v4_is_risk_reduction(previous, target):
        return True
    if (
        abs(fast_expected_return) <= _V4_TARGET_EPSILON
        or abs(direction_score) <= _V4_TARGET_EPSILON
        or fast_expected_return * direction_score <= 0.0
    ):
        return False
    return target * fast_expected_return > 0.0


def _v4_slow_candidates(
    *, previous: float, current_anchor: float, cap: float, config: CausalAlphaV4TargetConfig
) -> tuple[float, ...]:
    values = {
        0.0,
        float(np.clip(previous, -cap, cap)),
        float(np.clip(current_anchor, -cap, cap)),
        -cap,
        cap,
    }
    for magnitude in config.slow_target_magnitudes:
        bounded = min(magnitude, cap)
        values.add(bounded)
        values.add(-bounded)
    return tuple(
        value
        for value in sorted(values)
        if abs(value - previous) <= config.maximum_final_target_delta + _V4_TARGET_EPSILON
    )


def _v4_fast_candidates(
    *, previous: float, anchor: float, cap: float, config: CausalAlphaV4TargetConfig
) -> tuple[float, ...]:
    values = {float(np.clip(anchor, -cap, cap))}
    for magnitude in config.fast_deviation_magnitudes:
        values.add(float(np.clip(anchor + magnitude, -cap, cap)))
        values.add(float(np.clip(anchor - magnitude, -cap, cap)))
    return tuple(
        value
        for value in sorted(values)
        if abs(value - anchor)
        <= config.maximum_fast_absolute_deviation + _V4_TARGET_EPSILON
        and abs(value - previous)
        <= config.maximum_final_target_delta + _V4_TARGET_EPSILON
    )


def _v4_choose_best(
    candidates: tuple[float, ...],
    scores: tuple[float, ...],
    *,
    previous: float,
) -> tuple[float, float]:
    if not candidates or len(candidates) != len(scores):
        raise ValueError("V4 target candidate scores are invalid")
    maximum = max(scores)
    tied = tuple(
        (value, score)
        for value, score in zip(candidates, scores, strict=True)
        if score >= maximum - 1e-15
    )
    return min(
        tied,
        key=lambda item: (
            abs(item[0] - previous),
            abs(item[0]),
            item[0],
        ),
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaV4TargetPath:
    initial_weight: float
    slow_anchors: np.ndarray
    fast_deviations: np.ndarray
    targets: np.ndarray
    slow_expected_returns: np.ndarray
    fast_expected_returns: np.ndarray
    slow_uncertainties: np.ndarray
    fast_uncertainties: np.ndarray
    liquidity_weight_caps: np.ndarray
    slow_objectives: np.ndarray
    fast_objective_improvements: np.ndarray
    final_objectives: np.ndarray
    reasons: tuple[str, ...]
    slow_anchor_change_count: int
    fast_impulse_change_count: int
    submitted_change_count: int
    liquidity_deleveraging_count: int
    sign_flip_count: int
    config_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_weight):
            raise ValueError("V4 target initial weight must be finite")
        arrays: dict[str, np.ndarray] = {}
        shape: tuple[int, ...] | None = None
        for field_name in (
            "slow_anchors",
            "fast_deviations",
            "targets",
            "slow_expected_returns",
            "fast_expected_returns",
            "slow_uncertainties",
            "fast_uncertainties",
            "liquidity_weight_caps",
            "slow_objectives",
            "fast_objective_improvements",
            "final_objectives",
        ):
            array = np.asarray(getattr(self, field_name), dtype=np.float64).reshape(-1).copy()
            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError(f"V4 target path {field_name} must be finite and non-empty")
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise ValueError("V4 target path arrays must align")
            array.setflags(write=False)
            arrays[field_name] = array
        if np.any(arrays["slow_uncertainties"] < 0.0) or np.any(
            arrays["fast_uncertainties"] < 0.0
        ):
            raise ValueError("V4 target uncertainty must be non-negative")
        if np.any(arrays["liquidity_weight_caps"] < 0.0):
            raise ValueError("V4 target liquidity caps must be non-negative")
        reasons = tuple(self.reasons)
        if len(reasons) != len(arrays["targets"]) or any(not reason for reason in reasons):
            raise ValueError("V4 target reasons must cover every decision")
        for field_name in (
            "slow_anchor_change_count",
            "fast_impulse_change_count",
            "submitted_change_count",
            "liquidity_deleveraging_count",
            "sign_flip_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V4 target path {field_name} is invalid")
        require_sha256(self.config_digest, field="V4 target config_digest")
        expected = content_and_arrays_digest(
            {
                "config_digest": self.config_digest,
                "initial_weight": self.initial_weight,
                "reasons": reasons,
                "schema_version": CAUSAL_ALPHA_V4_TARGET_SCHEMA,
                "slow_anchor_change_count": self.slow_anchor_change_count,
                "fast_impulse_change_count": self.fast_impulse_change_count,
                "submitted_change_count": self.submitted_change_count,
                "liquidity_deleveraging_count": self.liquidity_deleveraging_count,
                "sign_flip_count": self.sign_flip_count,
            },
            tuple((field_name, array) for field_name, array in arrays.items()),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 target path digest mismatch")
        for field_name, array in arrays.items():
            object.__setattr__(self, field_name, array)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "digest", expected)


def causal_alpha_v4_target_path(
    prediction_4h: object,
    prediction_24h: object,
    prediction_72h: object,
    *,
    direction_score_4h: object,
    uncertainty_4h: object,
    uncertainty_24h: object,
    uncertainty_72h: object,
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    config: CausalAlphaV4TargetConfig,
    initial_weight: float,
    actionable_mask: object | None = None,
) -> CausalAlphaV4TargetPath:
    """Compile one slow anchor plus one bounded 4h impulse without double cost."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (
            prediction_4h,
            prediction_24h,
            prediction_72h,
            direction_score_4h,
            uncertainty_4h,
            uncertainty_24h,
            uncertainty_72h,
            one_way_cost_rates,
            liquidity_weight_caps,
        )
    )
    rows = int(arrays[0].size)
    if rows == 0 or any(array.shape != (rows,) for array in arrays):
        raise ValueError("V4 target inputs must be non-empty and aligned")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("V4 target inputs must be finite")
    (
        fast_mu,
        prediction_24,
        prediction_72,
        direction,
        fast_sigma,
        uncertainty_24_values,
        uncertainty_72_values,
        costs,
        caps,
    ) = arrays
    if (
        np.any(fast_sigma < 0.0)
        or np.any(uncertainty_24_values < 0.0)
        or np.any(uncertainty_72_values < 0.0)
        or np.any(costs < 0.0)
        or np.any(caps < 0.0)
    ):
        raise ValueError("V4 target uncertainty/cost/cap inputs must be non-negative")
    if not isinstance(config, CausalAlphaV4TargetConfig):
        raise TypeError("V4 target compiler requires CausalAlphaV4TargetConfig")
    if not math.isfinite(initial_weight):
        raise ValueError("V4 target initial weight must be finite")
    actionable = (
        np.ones(rows, dtype=np.bool_)
        if actionable_mask is None
        else np.asarray(actionable_mask, dtype=np.bool_).reshape(-1)
    )
    if actionable.shape != (rows,):
        raise ValueError("V4 target actionable mask must align")

    prediction_72_equivalent = prediction_72 / 3.0
    slow_mu = 0.5 * (prediction_24 + prediction_72_equivalent)
    slow_disagreement = 0.5 * np.abs(prediction_24 - prediction_72_equivalent)
    slow_sigma = np.sqrt(
        0.25 * (
            np.square(uncertainty_24_values)
            + np.square(uncertainty_72_values / 3.0)
        )
        + np.square(slow_disagreement)
    )

    slow_anchors = np.empty(rows, dtype=np.float64)
    fast_deviations = np.empty(rows, dtype=np.float64)
    targets = np.empty(rows, dtype=np.float64)
    slow_objectives = np.empty(rows, dtype=np.float64)
    fast_improvements = np.empty(rows, dtype=np.float64)
    final_objectives = np.empty(rows, dtype=np.float64)
    reasons: list[str] = []
    previous = float(initial_weight)
    current_anchor = float(initial_weight)
    slow_changes = 0
    fast_changes = 0
    submitted = 0
    liquidity_deleveraging = 0
    sign_flips = 0

    for index in range(rows):
        cap = min(float(caps[index]), 1.0)
        old_anchor = current_anchor
        selected_anchor = float(np.clip(current_anchor, -cap, cap))
        selected = previous
        direction_blocked = False

        if abs(previous) > cap + _V4_TARGET_EPSILON:
            selected = float(np.clip(previous, -cap, cap))
            selected_anchor = selected
            reason = "liquidity_deleverage"
            liquidity_deleveraging += 1
            slow_score, fast_improvement, final_score = _v4_staged_objective(
                previous=previous,
                anchor=selected_anchor,
                final=selected,
                slow_expected_return=float(slow_mu[index]),
                slow_uncertainty=float(slow_sigma[index]),
                fast_expected_return=float(fast_mu[index]),
                fast_uncertainty=float(fast_sigma[index]),
                one_way_cost_rate=float(costs[index]),
                config=config,
            )
        elif not bool(actionable[index]):
            selected_anchor = current_anchor
            selected = previous
            reason = "unactionable_hold"
            slow_score, fast_improvement, final_score = _v4_staged_objective(
                previous=previous,
                anchor=selected_anchor,
                final=selected,
                slow_expected_return=float(slow_mu[index]),
                slow_uncertainty=float(slow_sigma[index]),
                fast_expected_return=float(fast_mu[index]),
                fast_uncertainty=float(fast_sigma[index]),
                one_way_cost_rate=float(costs[index]),
                config=config,
            )
        else:
            if index % config.slow_rebalance_decisions == 0:
                candidates = _v4_slow_candidates(
                    previous=previous,
                    current_anchor=current_anchor,
                    cap=cap,
                    config=config,
                )
                unrestricted_scores = tuple(
                    _v4_direct_objective(
                        target=value,
                        previous=previous,
                        expected_return=float(slow_mu[index]),
                        uncertainty=float(slow_sigma[index]),
                        one_way_cost_rate=float(costs[index]),
                        config=config,
                    )
                    for value in candidates
                )
                unrestricted_anchor, _ = _v4_choose_best(
                    candidates, unrestricted_scores, previous=previous
                )
                allowed_pairs = tuple(
                    (value, score)
                    for value, score in zip(candidates, unrestricted_scores, strict=True)
                    if _v4_consensus_allows(
                        previous=previous,
                        target=value,
                        fast_expected_return=float(fast_mu[index]),
                        direction_score=float(direction[index]),
                    )
                )
                if not allowed_pairs:
                    allowed_pairs = ((previous, 0.0),)
                selected_anchor, _ = _v4_choose_best(
                    tuple(value for value, _ in allowed_pairs),
                    tuple(score for _, score in allowed_pairs),
                    previous=previous,
                )
                direction_blocked = abs(unrestricted_anchor - selected_anchor) > _V4_TARGET_EPSILON
            else:
                selected_anchor = float(np.clip(current_anchor, -cap, cap))

            if index % config.fast_rebalance_decisions != 0:
                selected = previous
                slow_score, fast_improvement, final_score = _v4_staged_objective(
                    previous=previous,
                    anchor=selected_anchor,
                    final=selected,
                    slow_expected_return=float(slow_mu[index]),
                    slow_uncertainty=float(slow_sigma[index]),
                    fast_expected_return=float(fast_mu[index]),
                    fast_uncertainty=float(fast_sigma[index]),
                    one_way_cost_rate=float(costs[index]),
                    config=config,
                )
                reason = "cadence_hold"
            else:
                candidates = _v4_fast_candidates(
                    previous=previous,
                    anchor=selected_anchor,
                    cap=cap,
                    config=config,
                )
                scored = tuple(
                    (
                        value,
                        _v4_staged_objective(
                            previous=previous,
                            anchor=selected_anchor,
                            final=value,
                            slow_expected_return=float(slow_mu[index]),
                            slow_uncertainty=float(slow_sigma[index]),
                            fast_expected_return=float(fast_mu[index]),
                            fast_uncertainty=float(fast_sigma[index]),
                            one_way_cost_rate=float(costs[index]),
                            config=config,
                        ),
                    )
                    for value in candidates
                    if _v4_consensus_allows(
                        previous=previous,
                        target=value,
                        fast_expected_return=float(fast_mu[index]),
                        direction_score=float(direction[index]),
                    )
                )
                if not scored:
                    selected = previous
                    selected_anchor = previous
                    slow_score, fast_improvement, final_score = _v4_staged_objective(
                        previous=previous,
                        anchor=selected_anchor,
                        final=selected,
                        slow_expected_return=float(slow_mu[index]),
                        slow_uncertainty=float(slow_sigma[index]),
                        fast_expected_return=float(fast_mu[index]),
                        fast_uncertainty=float(fast_sigma[index]),
                        one_way_cost_rate=float(costs[index]),
                        config=config,
                    )
                    direction_blocked = True
                else:
                    selected, final_score = _v4_choose_best(
                        tuple(value for value, _ in scored),
                        tuple(values[2] for _, values in scored),
                        previous=previous,
                    )
                    selected_values = next(
                        values for value, values in scored if value == selected
                    )
                    slow_score, fast_improvement, final_score = selected_values
                if abs(selected - previous) <= _V4_TARGET_EPSILON:
                    reason = (
                        "direction_disagreement_hold" if direction_blocked else "hold"
                    )
                elif abs(selected - selected_anchor) > _V4_TARGET_EPSILON:
                    reason = "fast_impulse"
                elif abs(selected_anchor - old_anchor) > _V4_TARGET_EPSILON:
                    reason = "slow_rebalance"
                else:
                    reason = "rebalance"

        if abs(selected_anchor - old_anchor) > _V4_TARGET_EPSILON:
            slow_changes += 1
        current_anchor = selected_anchor
        deviation = selected - selected_anchor
        if abs(deviation) > config.maximum_fast_absolute_deviation + _V4_TARGET_EPSILON:
            if reason not in {"cadence_hold", "unactionable_hold"}:
                raise RuntimeError("V4 fast deviation exceeded authored bound")
        if abs(deviation) > _V4_TARGET_EPSILON and reason == "fast_impulse":
            fast_changes += 1
        if abs(selected - previous) > _V4_TARGET_EPSILON:
            submitted += 1
        if previous * selected < 0.0:
            sign_flips += 1

        slow_anchors[index] = selected_anchor
        fast_deviations[index] = deviation
        targets[index] = selected
        slow_objectives[index] = slow_score
        fast_improvements[index] = fast_improvement
        final_objectives[index] = final_score
        reasons.append(reason)
        previous = float(selected)

    return CausalAlphaV4TargetPath(
        initial_weight=float(initial_weight),
        slow_anchors=slow_anchors,
        fast_deviations=fast_deviations,
        targets=targets,
        slow_expected_returns=slow_mu,
        fast_expected_returns=fast_mu,
        slow_uncertainties=slow_sigma,
        fast_uncertainties=fast_sigma,
        liquidity_weight_caps=caps,
        slow_objectives=slow_objectives,
        fast_objective_improvements=fast_improvements,
        final_objectives=final_objectives,
        reasons=tuple(reasons),
        slow_anchor_change_count=slow_changes,
        fast_impulse_change_count=fast_changes,
        submitted_change_count=submitted,
        liquidity_deleveraging_count=liquidity_deleveraging,
        sign_flip_count=sign_flips,
        config_digest=config.digest,
    )
'''

replace_once("\n\n__all__ = [\n", addition + "\n\n__all__ = [\n")
replace_once(
    '    "CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA",\n',
    '    "CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA",\n    "CAUSAL_ALPHA_V4_TARGET_SCHEMA",\n',
)
replace_once(
    '    "CausalAlphaV4SymbolSamples",\n',
    '    "CausalAlphaV4SymbolSamples",\n    "CausalAlphaV4TargetConfig",\n    "CausalAlphaV4TargetPath",\n',
)
replace_once(
    '    "fit_causal_alpha_v4_uncertainty",\n',
    '    "causal_alpha_v4_target_path",\n    "fit_causal_alpha_v4_uncertainty",\n',
)
path.write_text(text, encoding="utf-8")
