from pathlib import Path

runtime_path = Path("trade_rl/integrations/sb3_runtime.py")
runtime = runtime_path.read_text()
if "def oracle_teacher_config_for_environment(" not in runtime:
    import_anchor = "from trade_rl.learning.oracle_solver import OracleBatchBackend\n"
    if import_anchor not in runtime:
        raise SystemExit("sb3_runtime Oracle import anchor not found")
    runtime = runtime.replace(
        import_anchor,
        import_anchor + "from trade_rl.learning.oracle_teacher import OracleTeacherConfig\n",
        1,
    )
    marker = "\n\ndef _oracle_solver_config() -> OracleSolverConfig:\n"
    if marker not in runtime:
        raise SystemExit("sb3_runtime solver marker not found")
    block = r'''


def oracle_teacher_config_for_environment(environment: Any) -> OracleTeacherConfig:
    """Derive the exact deterministic Oracle contract from a training environment."""

    risk_service = getattr(environment, "pre_trade_risk", None)
    portfolio_service = getattr(environment, "portfolio_risk", None)
    environment_config = getattr(environment, "config", None)
    risk_config = getattr(risk_service, "config", None)
    portfolio_config = getattr(portfolio_service, "config", None)
    execution_cost = getattr(environment_config, "execution_cost", None)
    signal_delay = getattr(environment_config, "signal_delay_decisions", None)
    initial_capital = getattr(environment, "initial_capital", None)
    if risk_config is None or portfolio_config is None or execution_cost is None:
        raise TypeError("Oracle teacher environment is missing risk or execution config")
    if (
        isinstance(initial_capital, bool)
        or not isinstance(initial_capital, (int, float))
        or not np.isfinite(initial_capital)
        or initial_capital <= 0.0
    ):
        raise ValueError("Oracle teacher environment initial_capital must be positive")
    if isinstance(signal_delay, bool) or not isinstance(signal_delay, int):
        raise TypeError("Oracle teacher signal_delay_decisions must be an integer")
    return OracleTeacherConfig(
        execution_cost=execution_cost,
        portfolio_risk=portfolio_config,
        max_gross=risk_config.max_gross,
        max_abs_weight=risk_config.max_abs_weight,
        entry_threshold=risk_config.entry_threshold,
        exit_threshold=risk_config.exit_threshold,
        no_trade_band=risk_config.no_trade_band,
        reference_portfolio_value=float(initial_capital),
        signal_delay_decisions=signal_delay,
    )
'''
    runtime = runtime.replace(marker, block + marker, 1)
    runtime_path.write_text(runtime)
    compile(runtime, str(runtime_path), "exec")

training_path = Path("trade_rl/integrations/sb3_training.py")
training = training_path.read_text()
helper_import = "from trade_rl.integrations.sb3_runtime import (\n    oracle_teacher_config_for_environment,\n)\n"
if helper_import not in training:
    anchor = "from trade_rl.integrations.sb3_runtime import (\n    _teacher_worker_count as _teacher_worker_count,\n)\n"
    if anchor not in training:
        raise SystemExit("sb3_training runtime import anchor not found")
    training = training.replace(anchor, anchor + helper_import, 1)
old = '''                risk_config = unwrapped_probe.pre_trade_risk.config\n                prefetched_oracle_config = OracleTeacherConfig(\n                    execution_cost=unwrapped_probe.config.execution_cost,\n                    portfolio_risk=unwrapped_probe.portfolio_risk.config,\n                    max_gross=risk_config.max_gross,\n                    max_abs_weight=risk_config.max_abs_weight,\n                    entry_threshold=risk_config.entry_threshold,\n                    exit_threshold=risk_config.exit_threshold,\n                    no_trade_band=risk_config.no_trade_band,\n                    reference_portfolio_value=unwrapped_probe.initial_capital,\n                    signal_delay_decisions=(\n                        unwrapped_probe.config.signal_delay_decisions\n                    ),\n                )\n'''
new = '''                prefetched_oracle_config = oracle_teacher_config_for_environment(\n                    unwrapped_probe\n                )\n'''
if old not in training:
    raise SystemExit("sb3_training Oracle config block not found")
training = training.replace(old, new, 1)
training_path.write_text(training)
compile(training, str(training_path), "exec")
