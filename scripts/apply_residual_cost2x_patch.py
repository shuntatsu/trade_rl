from pathlib import Path


path = Path(__file__).resolve().parents[1] / "mars_lite/pipeline/residual_pipeline.py"
text = path.read_text(encoding="utf-8")

old_select = '''def select_residual_configuration(
    development_results: dict[str, dict[str, Any]],
    *,
    drawdown_slack: float = 0.05,
) -> dict[str, Any]:
    """Apply the preregistered A/B/C/D selection rule on development data only.

    A is pure base trend. B is PPO trend mixing. C is a fixed +15% alpha diagnostic.
    D is PPO trend mixing plus alpha. C is never itself release-selected; it establishes
    the hurdle that D must beat before the combined RL design is adopted.
    """

    if "A" not in development_results:
        raise ValueError("development matrix requires configuration A")
    scores = {
        name: float(result["paired"]["excess_log_return"])
        for name, result in development_results.items()
    }
    eligible = {
        name: _eligible_relative_result(result, drawdown_slack)
        for name, result in development_results.items()
    }
    selected = "A"
    reasons = ["A is the identity baseline"]

    if eligible.get("B", False):
        selected = "B"
        reasons.append("B adds positive development excess within drawdown slack")

    if eligible.get("D", False):
        hurdle = max(scores.get("B", 0.0), scores.get("C", 0.0))
        if scores["D"] > hurdle:
            selected = "D"
            reasons.append("D strictly beats both B and fixed-alpha diagnostic C")
        else:
            reasons.append("D did not beat the stronger of B and C")

    return {
        "selected": selected,
        "policy_mode": (
            "baseline_only" if selected == "A" else "ppo_residual_ensemble"
        ),
        "scores": scores,
        "eligible": eligible,
        "reasons": reasons,
        "drawdown_slack": drawdown_slack,
    }
'''
new_select = '''def select_residual_configuration(
    development_results: dict[str, dict[str, Any]],
    *,
    cost2x_results: dict[str, dict[str, Any]],
    drawdown_slack: float = 0.05,
) -> dict[str, Any]:
    """Apply the preregistered A/B/C/D rule using 1x and 2x development costs.

    A is pure base trend. B is PPO trend mixing. C is a fixed +15% alpha diagnostic.
    D is PPO trend mixing plus alpha. C is never itself release-selected; it establishes
    the hurdle that D must beat before the combined RL design is adopted.
    """

    if "A" not in development_results or "A" not in cost2x_results:
        raise ValueError("development matrices require configuration A")
    if set(development_results) != set(cost2x_results):
        raise ValueError("1x and 2x development matrices must contain identical configs")
    scores = {
        name: float(result["paired"]["excess_log_return"])
        for name, result in development_results.items()
    }
    cost2x_scores = {
        name: float(result["paired"]["excess_log_return"])
        for name, result in cost2x_results.items()
    }
    base_eligible = {
        name: _eligible_relative_result(result, drawdown_slack)
        for name, result in development_results.items()
    }
    cost2x_eligible = {
        name: score >= 0.0 for name, score in cost2x_scores.items()
    }
    eligible = {
        name: base_eligible[name] and cost2x_eligible[name]
        for name in development_results
    }
    selected = "A"
    reasons = ["A is the identity baseline"]

    if eligible.get("B", False):
        selected = "B"
        reasons.append(
            "B adds positive development excess within drawdown slack and survives 2x costs"
        )

    if eligible.get("D", False):
        hurdle = max(scores.get("B", 0.0), scores.get("C", 0.0))
        if scores["D"] > hurdle:
            selected = "D"
            reasons.append(
                "D strictly beats both B and fixed-alpha diagnostic C and survives 2x costs"
            )
        else:
            reasons.append("D did not beat the stronger of B and C")

    return {
        "selected": selected,
        "policy_mode": (
            "baseline_only" if selected == "A" else "ppo_residual_ensemble"
        ),
        "scores": scores,
        "cost2x_scores": cost2x_scores,
        "base_eligible": base_eligible,
        "cost2x_eligible": cost2x_eligible,
        "eligible": eligible,
        "reasons": reasons,
        "drawdown_slack": drawdown_slack,
    }
'''
if text.count(old_select) != 1:
    raise RuntimeError("unexpected residual selection layout")
text = text.replace(old_select, new_select)

old_init = """    development_results: dict[str, dict[str, Any]] = {}
    development_results["A"] = evaluate_relative_agent(
        identity,
        val_fs,
        env_kwargs=_evaluation_kwargs(
            env_kwargs, trend_family, alpha, alpha_enabled=False
        ),
        bootstrap_seed=args.seed,
    )
"""
new_init = """    development_results: dict[str, dict[str, Any]] = {}
    development_cost2x_results: dict[str, dict[str, Any]] = {}
    a_kwargs = _evaluation_kwargs(
        env_kwargs, trend_family, alpha, alpha_enabled=False
    )
    development_results["A"] = evaluate_relative_agent(
        identity,
        val_fs,
        env_kwargs=a_kwargs,
        bootstrap_seed=args.seed,
    )
    development_cost2x_results["A"] = evaluate_relative_agent(
        identity,
        val_fs,
        env_kwargs={**a_kwargs, "cost_multiplier": 2.0},
        bootstrap_seed=args.seed,
    )
"""
if text.count(old_init) != 1:
    raise RuntimeError("unexpected development matrix initialization")
text = text.replace(old_init, new_init)

old_b = """    development_results["B"] = evaluate_relative_agent(
        b_agent,
        val_fs,
        env_kwargs=_evaluation_kwargs(
            env_kwargs, trend_family, alpha, alpha_enabled=False
        ),
        bootstrap_seed=args.seed,
    )
"""
new_b = """    b_kwargs = _evaluation_kwargs(
        env_kwargs, trend_family, alpha, alpha_enabled=False
    )
    development_results["B"] = evaluate_relative_agent(
        b_agent,
        val_fs,
        env_kwargs=b_kwargs,
        bootstrap_seed=args.seed,
    )
    development_cost2x_results["B"] = evaluate_relative_agent(
        b_agent,
        val_fs,
        env_kwargs={**b_kwargs, "cost_multiplier": 2.0},
        bootstrap_seed=args.seed,
    )
"""
if text.count(old_b) != 1:
    raise RuntimeError("unexpected B matrix evaluation")
text = text.replace(old_b, new_b)

old_c = """        development_results["C"] = evaluate_relative_agent(
            fixed_alpha,
            val_fs,
            env_kwargs=_evaluation_kwargs(
                env_kwargs, trend_family, alpha, alpha_enabled=True
            ),
            bootstrap_seed=args.seed,
        )
"""
new_c = """        c_kwargs = _evaluation_kwargs(
            env_kwargs, trend_family, alpha, alpha_enabled=True
        )
        development_results["C"] = evaluate_relative_agent(
            fixed_alpha,
            val_fs,
            env_kwargs=c_kwargs,
            bootstrap_seed=args.seed,
        )
        development_cost2x_results["C"] = evaluate_relative_agent(
            fixed_alpha,
            val_fs,
            env_kwargs={**c_kwargs, "cost_multiplier": 2.0},
            bootstrap_seed=args.seed,
        )
"""
if text.count(old_c) != 1:
    raise RuntimeError("unexpected C matrix evaluation")
text = text.replace(old_c, new_c)

old_d = """        development_results["D"] = evaluate_relative_agent(
            d_agent,
            val_fs,
            env_kwargs=_evaluation_kwargs(
                env_kwargs, trend_family, alpha, alpha_enabled=True
            ),
            bootstrap_seed=args.seed,
        )
"""
new_d = """        d_kwargs = _evaluation_kwargs(
            env_kwargs, trend_family, alpha, alpha_enabled=True
        )
        development_results["D"] = evaluate_relative_agent(
            d_agent,
            val_fs,
            env_kwargs=d_kwargs,
            bootstrap_seed=args.seed,
        )
        development_cost2x_results["D"] = evaluate_relative_agent(
            d_agent,
            val_fs,
            env_kwargs={**d_kwargs, "cost_multiplier": 2.0},
            bootstrap_seed=args.seed,
        )
"""
if text.count(old_d) != 1:
    raise RuntimeError("unexpected D matrix evaluation")
text = text.replace(old_d, new_d)

old_selection = """    selection = select_residual_configuration(development_results)
"""
new_selection = """    selection = select_residual_configuration(
        development_results,
        cost2x_results=development_cost2x_results,
    )
"""
if text.count(old_selection) != 1:
    raise RuntimeError("unexpected selection call")
text = text.replace(old_selection, new_selection)

old_gate = """        gate = evaluate_residual_gate2(
            hybrid=relative["hybrid"],
            shadow=relative["shadow"],
            flat={"total_return": 0.0, "max_drawdown": 0.0},
            paired_p_value=float(relative["paired"]["p_value"]),
            diagnostic_results=baseline_payload,
        )
"""
new_gate = """        gate = evaluate_residual_gate2(
            hybrid=relative["hybrid"],
            shadow=relative["shadow"],
            flat={"total_return": 0.0, "max_drawdown": 0.0},
            cost2x_hybrid=cost2x["hybrid"],
            cost2x_shadow=cost2x["shadow"],
            paired_p_value=float(relative["paired"]["p_value"]),
            diagnostic_results=baseline_payload,
        )
"""
if text.count(old_gate) != 1:
    raise RuntimeError("unexpected residual Gate 2 call")
text = text.replace(old_gate, new_gate)

old_report = """        "development_matrix": development_results,
        "selection": selection,
"""
new_report = """        "development_matrix": development_results,
        "development_matrix_cost2x": development_cost2x_results,
        "selection": selection,
"""
if text.count(old_report) != 1:
    raise RuntimeError("unexpected development report layout")
text = text.replace(old_report, new_report)

path.write_text(text, encoding="utf-8")
