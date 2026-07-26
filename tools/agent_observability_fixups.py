from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def apply() -> None:
    path = ROOT / "tests/integrations/test_sb3_training.py"
    text = path.read_text(encoding="utf-8")
    marker = "\ndef test_backend_wires_learning_rate_schedule_and_tensorboard("
    if marker not in text:
        raise RuntimeError("generated SB3 observability test marker is missing")
    prefix = text.split(marker, maxsplit=1)[0]
    replacement = r'''

def test_backend_wires_learning_rate_schedule_and_tensorboard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    events: list[str] = []
    vector_environment = VectorEnvironment(events)

    class FakeParameter:
        def numel(self) -> int:
            return 2

    class FakePolicy:
        action_distribution_name = "squashed_diag_gaussian"

        def parameters(self) -> tuple[FakeParameter, ...]:
            return (FakeParameter(),)

    class CapturingPPO:
        device = "cpu"
        num_timesteps = 0

        def __init__(self, policy: str, environment: Any, **kwargs: Any) -> None:
            assert environment is vector_environment
            self.policy = FakePolicy()
            captured["constructor"] = {"policy": policy, **kwargs}

        def learn(self, **kwargs: Any) -> "CapturingPPO":
            captured["learn"] = kwargs
            self.num_timesteps = int(kwargs["total_timesteps"])
            return self

        def save(self, target: str) -> None:
            Path(f"{target}.zip").write_bytes(b"policy")

    monkeypatch.setattr("stable_baselines3.PPO", CapturingPPO)
    monkeypatch.setattr(
        sb3_training,
        "_build_training_environment",
        lambda *args, **kwargs: vector_environment,
    )
    backend = StableBaselines3Backend(lambda: TrainingProbe(events))
    config = replace(
        _training_config(),
        learning_rate_schedule="linear",
        tensorboard_enabled=True,
    )

    backend.train(
        seed=0,
        config=config,
        output_path=tmp_path / "member" / "policy.zip",
    )

    constructor = captured["constructor"]
    assert isinstance(constructor, dict)
    assert callable(constructor["learning_rate"])
    assert constructor["tensorboard_log"] == str(tmp_path / "member" / "tensorboard")
    learn = captured["learn"]
    assert isinstance(learn, dict)
    assert learn["tb_log_name"] == "seed-0-ppo"
'''
    path.write_text(prefix.rstrip() + replacement, encoding="utf-8")

    metrics_path = ROOT / "trade_rl/studio/training_metrics.py"
    metrics = metrics_path.read_text(encoding="utf-8")
    old = '_METRICS: dict[str, tuple[str, str, str]] = {'
    new = '''MetricGroup = Literal["optimization", "policy", "value", "trading"]
MetricUnit = Literal["raw", "rate", "percent", "currency"]

_METRICS: dict[str, tuple[str, MetricGroup, MetricUnit]] = {'''
    if old not in metrics:
        raise RuntimeError("generated training metric metadata annotation is missing")
    metrics_path.write_text(metrics.replace(old, new, 1), encoding="utf-8")


if __name__ == "__main__":
    apply()
