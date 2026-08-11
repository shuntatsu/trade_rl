from pathlib import Path

path = Path("trade_rl/workflows/universal_training_runner.py")
text = path.read_text()
if "def validate_universal_training_config(" in text:
    raise SystemExit(0)
text = text.replace(
    "from dataclasses import dataclass\n",
    "from dataclasses import asdict, dataclass\n",
    1,
)
text = text.replace(
    "from trade_rl.data.contracts import (\n",
    "from trade_rl.data.contracts import (\n",
    1,
)
anchor = "from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding\n"
if anchor not in text:
    raise SystemExit("runner import anchor not found")
text = text.replace(
    anchor,
    "from trade_rl.rl.actions import ACTION_SCHEMA, ActionMode, ActionSpec\n" + anchor,
    1,
)
marker = "\n\ndef build_universal_instrument_contracts(\n"
if marker not in text:
    raise SystemExit("runner function marker not found")
block = r'''


def validate_universal_training_config(run_config: Any) -> None:
    """Reject any run configuration that violates the maintained Universal surface."""

    training = getattr(run_config, "training", None)
    action = getattr(run_config, "action", None)
    environment = getattr(run_config, "environment", None)
    if training is None or environment is None or not isinstance(action, ActionSpec):
        raise TypeError("Universal training requires a complete TrainingRunConfig surface")
    if (
        ActionMode(action.mode) is not ActionMode.TARGET_WEIGHT
        or action.target_weight_count != 1
        or action.alpha_enabled
        or action.risk_tilt_enabled
        or action.n_factors != 0
    ):
        raise ValueError("Universal training requires exactly one scalar target-weight action")
    if getattr(run_config, "alpha_artifact", None) is not None:
        raise ValueError("Universal target-weight training does not accept alpha artifacts")
    if getattr(run_config, "factor_artifact", None) is not None:
        raise ValueError("Universal target-weight training does not accept factor artifacts")
    if getattr(training, "observation_encoder", None) != "hierarchical_sequence_v2":
        raise ValueError("Universal training requires hierarchical_sequence_v2")
    if not bool(getattr(environment, "structured_sequence_observation", False)):
        raise ValueError("Universal training requires structured sequence observations")
    if not bool(getattr(environment, "finite_horizon_observation", False)):
        raise ValueError("Universal training requires finite-horizon observations")


def concrete_action_spec_digest(action: ActionSpec, symbol: str) -> str:
    """Bind the generic one-action specification to one concrete child environment."""

    if not isinstance(action, ActionSpec):
        raise TypeError("action must be an ActionSpec")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("concrete symbol must be non-empty")
    if ActionMode(action.mode) is not ActionMode.TARGET_WEIGHT or action.target_weight_count != 1:
        raise ValueError("Universal concrete action identity requires one target-weight action")
    return content_digest(
        {
            "action_schema": ACTION_SCHEMA,
            "names": action.names_for_symbols((symbol,)),
            "spec": asdict(action),
        }
    )
'''
text = text.replace(marker, block + marker, 1)
text = text.replace(
    '    "build_universal_instrument_contracts",\n',
    '    "build_universal_instrument_contracts",\n'
    '    "concrete_action_spec_digest",\n'
    '    "validate_universal_training_config",\n',
    1,
)
path.write_text(text)
compile(text, str(path), "exec")
