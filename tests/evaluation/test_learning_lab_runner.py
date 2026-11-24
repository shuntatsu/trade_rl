"""Synthetic tests of the agent wrappers, the runner and the ``tools.learning_lab`` driver."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.evaluation import learning_lab_support as support
from tools import core7_g4_comment as g4
from tools import learning_lab as tool
from trade_rl.evaluation.learning_lab import agents, baselines, runner, splits
from trade_rl.evaluation.learning_lab.registry import TrialRegistry

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

COMMIT = "c" * 40
STEPS = 96


def _config(algorithm: str, seed: int = 0) -> agents.AgentConfig:
    return agents.AgentConfig(
        algorithm=algorithm,
        total_timesteps=STEPS,
        seed=seed,
        n_steps=32,
        batch_size=16,
        buffer_size=512,
        learning_starts=16,
        net_arch=(16,),
    )


def _algorithms() -> list[Any]:
    marks = {
        "QRDQN": pytest.mark.skipif(
            not _has("sb3_contrib"),
            reason="sb3-contrib (learning-lab extra) not installed",
        )
    }
    return [pytest.param(a, marks=marks.get(a, ())) for a in agents.ALGORITHMS]


def _has(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


# ------------------------------------------------------------------- agents


@pytest.mark.parametrize("algorithm", _algorithms())
def test_every_agent_trains_checkpoints_and_evaluates_on_a_tiny_dataset(
    algorithm: str, tmp_path: Path
) -> None:
    setup = support.synthetic_setup(n_bars=120)
    registry = TrialRegistry(tmp_path / "trials.jsonl")
    method = agents.method_id_for(algorithm)
    trained, record = runner.train_method(
        setup,
        method,
        0,
        agent_config=_config(algorithm),
        output_dir=tmp_path / "ckpt",
        registry=registry,
        code_commit=COMMIT,
    )
    assert trained.checkpoint.is_file()
    digest = hashlib.sha256(trained.checkpoint.read_bytes()).hexdigest()
    assert digest == trained.checkpoint_sha256 == record.metrics["checkpoint_sha256"]
    sidecar = trained.checkpoint.with_suffix(".sha256").read_text(encoding="utf-8")
    assert sidecar.split()[0] == digest

    validated = runner.evaluate_method(
        setup,
        method,
        0,
        splits.VALIDATE,
        registry=registry,
        code_commit=COMMIT,
        agent_config=_config(algorithm),
        checkpoint=trained.checkpoint,
        checkpoint_sha256=trained.checkpoint_sha256,
    )
    assert validated.status == "completed" and validated.split == splits.VALIDATE
    for key in ("sharpe_per_period", "max_drawdown", "half1", "half2", "n_obs"):
        assert key in validated.metrics
    assert [r.split for r in registry.records()] == [splits.TRAIN, splits.VALIDATE]


def test_evaluation_is_deterministic_and_the_checkpoint_digest_is_enforced(
    tmp_path: Path,
) -> None:
    setup = support.synthetic_setup(n_bars=120)
    env = runner.make_env(setup, splits.TRAIN, "PPO")
    trained = agents.train_agent(env, _config("PPO"), tmp_path)
    model = agents.load_agent(
        "PPO", trained.checkpoint, sha256=trained.checkpoint_sha256
    )
    one = agents.evaluate_agent(
        model, runner.make_env(setup, splits.VALIDATE, "PPO"), method="L1-PPO"
    )
    two = agents.evaluate_agent(
        model, runner.make_env(setup, splits.VALIDATE, "PPO"), method="L1-PPO"
    )
    assert one.bar_returns == two.bar_returns and one.final_equity == two.final_equity
    with pytest.raises(agents.AgentError, match="sha256"):
        agents.load_agent("PPO", trained.checkpoint, sha256="0" * 64)
    with pytest.raises(FileExistsError):
        agents.train_agent(env, _config("PPO"), tmp_path)


def test_the_same_seed_trains_the_same_policy_and_seeds_differ(tmp_path: Path) -> None:
    setup = support.synthetic_setup(n_bars=120)
    outcomes: dict[int, list[tuple[float, ...]]] = {}
    for run, seed in enumerate((0, 0, 1)):
        env = runner.make_env(setup, splits.TRAIN, "A2C")
        trained = agents.train_agent(env, _config("A2C", seed), tmp_path / str(run))
        model = agents.load_agent(
            "A2C", trained.checkpoint, sha256=trained.checkpoint_sha256
        )
        result = agents.evaluate_agent(
            model, runner.make_env(setup, splits.VALIDATE, "A2C"), method="L1-A2C"
        )
        outcomes.setdefault(seed, []).append(result.bar_returns)
    assert outcomes[0][0] == outcomes[0][1]
    assert (
        agents.AgentConfig("A2C", 10, 0).digest
        != agents.AgentConfig("A2C", 10, 1).digest
    )


def test_agents_refuse_a_mismatched_action_space(tmp_path: Path) -> None:
    setup = support.synthetic_setup(n_bars=120)
    with pytest.raises(agents.AgentError, match="continuous"):
        agents.train_agent(
            runner.make_env(setup, splits.TRAIN, "PPO"), _config("SAC"), tmp_path
        )
    with pytest.raises(agents.AgentError):
        agents.AgentConfig("DQN", 10, 0)
    with pytest.raises(agents.AgentError):
        agents.algorithm_for_method("BASE-CASH")


# ------------------------------------------------------------------- runner


def test_agents_train_on_train_rows_only_and_test_needs_validation(
    tmp_path: Path,
) -> None:
    setup = support.synthetic_setup(n_bars=120)
    first, last = setup.window(splits.TRAIN)
    env = runner.make_env(setup, splits.TRAIN, "PPO")
    assert (env.start_index, env.stop_index) == (first, last)
    assert last < setup.window(splits.VALIDATE)[0]
    registry = TrialRegistry(tmp_path / "trials.jsonl")
    with pytest.raises(runner.RunnerError, match="validate"):
        runner.evaluate_method(
            setup,
            "BASE-STRIDE-18",
            0,
            splits.TEST,
            registry=registry,
            code_commit=COMMIT,
        )
    runner.evaluate_method(
        setup,
        "BASE-STRIDE-18",
        0,
        splits.VALIDATE,
        registry=registry,
        code_commit=COMMIT,
    )
    test_record = runner.evaluate_method(
        setup, "BASE-STRIDE-18", 0, splits.TEST, registry=registry, code_commit=COMMIT
    )
    assert test_record.split == splits.TEST
    with pytest.raises(runner.RunnerError):
        runner.evaluate_method(
            setup, "BASE-CASH", 0, splits.TRAIN, registry=registry, code_commit=COMMIT
        )


def test_a_dataset_with_rows_at_the_cutoff_cannot_become_a_setup() -> None:
    stage = support.synthetic_setup(n_bars=40)
    from trade_rl.evaluation.learning_lab.synthetic import synthetic_stage

    late = synthetic_stage(n_bars=40, first_stamp="2026-07-31T00:00:00")
    with pytest.raises(ValueError, match="cutoff"):
        runner.LabSetup(
            dataset=late.dataset,
            signals=late.signals,
            execution_cost=stage.execution_cost,
            risk_config=stage.risk_config,
            splits=stage.splits,
        )


def test_a_failed_training_is_registered_and_the_error_propagates(
    tmp_path: Path,
) -> None:
    setup = support.synthetic_setup(n_bars=120)
    registry = TrialRegistry(tmp_path / "trials.jsonl")
    kwargs: dict[str, Any] = {
        "agent_config": _config("PPO"),
        "output_dir": tmp_path / "ckpt",
        "registry": registry,
        "code_commit": COMMIT,
    }
    runner.train_method(setup, "L1-PPO", 0, **kwargs)
    with pytest.raises(FileExistsError):  # the checkpoint of this run already exists
        runner.train_method(setup, "L1-PPO", 0, **kwargs)
    statuses = [r.status for r in registry.records()]
    assert statuses == ["completed", "failed"]
    assert registry.records()[1].stop_reason
    with pytest.raises(runner.RunnerError):
        runner.train_method(
            setup, "L1-PPO", 1, **kwargs
        )  # the config's seed does not match the run's


def test_baselines_and_the_plan_cover_the_methods() -> None:
    assert baselines.baseline_methods() == (
        "BASE-STRIDE-6",
        "BASE-STRIDE-18",
        "BASE-STRIDE-42",
        "BASE-CASH",
    )
    plan = runner.plan(["L1-PPO", "BASE-CASH"], [0, 1])
    assert [(p["method"], p["seed"]) for p in plan] == [
        ("L1-PPO", 0),
        ("L1-PPO", 1),
        ("BASE-CASH", 0),
    ]
    with pytest.raises(runner.RunnerError):
        runner.plan(["L9-NOPE"], [0])
    assert len(runner.protocol_digest()) == 64
    assert runner.protocol_digest() == runner.protocol_digest()


def _metrics(sharpe: float, h1: float, h2: float, dd: float = 0.1) -> dict[str, Any]:
    return {
        "sharpe_per_period": sharpe,
        "max_drawdown": dd,
        "half1": {"sharpe_per_period": h1},
        "half2": {"sharpe_per_period": h2},
    }


def test_the_dev_rule_needs_all_four_conditions() -> None:
    stride = {
        "BASE-STRIDE-6": _metrics(0.01, 0.01, 0.01),
        "BASE-STRIDE-18": _metrics(0.02, 0.02, 0.02),
        "BASE-STRIDE-42": _metrics(0.015, 0.015, 0.015),
        "BASE-CASH": _metrics(0.0, 0.0, 0.0),
    }
    good = {s: _metrics(0.03, 0.03, 0.03) for s in range(5)}
    assert runner.dev_decision(good, stride)["label"] == runner.LABEL_PROMISING
    one_short = {**good, 0: _metrics(0.01, 0.01, 0.01), 1: _metrics(0.01, 0.01, 0.01)}
    verdict = runner.dev_decision(one_short, stride)
    assert verdict["checks"]["seeds_beating"] is False  # only 3 of 5 beat
    deep = {**good, 4: _metrics(0.03, 0.03, 0.03, dd=0.25)}
    assert runner.dev_decision(deep, stride)["checks"]["drawdown_below_limit"] is False
    lopsided = {s: _metrics(0.03, 0.03, 0.01) for s in range(5)}
    assert runner.dev_decision(lopsided, stride)["checks"]["both_halves"] is False
    assert runner.dev_decision(lopsided, stride)["label"] == runner.LABEL_STOP


# --------------------------------------------------------------------- tool


def _run_tool(argv: list[str], **kwargs: Any) -> tuple[int, dict[str, Any] | None, str]:
    out, err = io.StringIO(), io.StringIO()
    code = tool.main(
        argv,
        stdout=out,
        stderr=err,
        state=lambda: (COMMIT, True),
        **kwargs,
    )
    text = out.getvalue()
    return code, json.loads(text) if text else None, err.getvalue()


def test_plan_and_registry_subcommands_read_no_data(tmp_path: Path) -> None:
    code, document, _ = _run_tool(["plan", "--methods", "L1-SAC", "--seeds", "0", "1"])
    assert code == 0 and document is not None
    assert document["protocol_digest"] == runner.protocol_digest()
    assert [r["seed"] for r in document["runs"]] == [0, 1]
    path = tmp_path / "trials.jsonl"
    code, document, _ = _run_tool(["registry", "--registry", str(path)])
    assert code == 0 and document == {
        "chain_verified": True,
        "records": [],
        "trials": 0,
    }


def test_synthetic_train_evaluate_and_registry_round_trip(tmp_path: Path) -> None:
    registry = tmp_path / "trials.jsonl"
    common = [
        "--method",
        "L1-PPO",
        "--seed",
        "0",
        "--registry",
        str(registry),
        "--timesteps",
        "64",
        "--synthetic-bars",
        "120",
    ]
    code, trained, err = _run_tool(
        ["train", *common, "--output-dir", str(tmp_path / "c")]
    )
    assert code == 0, err
    assert trained is not None and trained["data"] == "synthetic"
    code, evaluated, err = _run_tool(
        [
            "evaluate",
            *common,
            "--split",
            "validate",
            "--checkpoint-dir",
            str(tmp_path / "c"),
        ]
    )
    assert code == 0, err
    assert evaluated is not None and evaluated["record"]["split"] == "validate"
    code, document, _ = _run_tool(
        ["registry", "--registry", str(registry), "--deflated-for-last"]
    )
    assert code == 0 and document is not None
    assert document["trials"] == 2 and "deflated_sharpe" in document


def test_a_baseline_evaluates_without_a_checkpoint(tmp_path: Path) -> None:
    code, document, err = _run_tool(
        [
            "evaluate",
            "--method",
            "BASE-STRIDE-18",
            "--registry",
            str(tmp_path / "t.jsonl"),
            "--split",
            "validate",
            "--synthetic-bars",
            "120",
        ]
    )
    assert code == 0, err
    assert document is not None and document["record"]["method_id"] == "BASE-STRIDE-18"


# ---------------------------------------------------- the real-data authorization


class FakeGh:
    """A fake ``gh api`` returning one owner comment; counts calls."""

    def __init__(
        self, body: str, *, login: str = g4.OWNER_LOGIN, comment_id: int = 7
    ) -> None:
        self.body = body
        self.login = login
        self.comment_id = comment_id
        self.calls = 0

    def __call__(self, args: Any) -> str:
        self.calls += 1
        return json.dumps(
            {
                "id": self.comment_id,
                "html_url": f"https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-{self.comment_id}",
                "issue_url": "https://api.github.com/repos/shuntatsu/trade_rl/issues/810",
                "user": {"login": self.login, "type": "User", "id": g4.OWNER_ID},
                "body": self.body,
                "updated_at": "2026-11-20T00:00:00Z",
            }
        )


URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7"


def _real_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "train",
        "--method",
        "L1-PPO",
        "--registry",
        str(tmp_path / "t.jsonl"),
        "--output-dir",
        str(tmp_path / "c"),
        "--data",
        "real",
        *extra,
    ]


def test_real_data_without_an_authorization_url_is_refused(tmp_path: Path) -> None:
    gh = FakeGh("irrelevant")
    code, document, err = _run_tool(_real_argv(tmp_path), gh=gh)
    assert code == tool.EXIT_UNAUTHENTICATED and document is None
    assert "authorization-url" in err
    assert gh.calls == 0 and not (tmp_path / "t.jsonl").exists()


def test_real_data_refuses_a_comment_that_does_not_quote_the_digest_or_the_id(
    tmp_path: Path,
) -> None:
    digest = runner.protocol_digest()
    for body in (
        "G4 authorization",  # quotes nothing
        f"protocol_digest: {digest}",  # no study id
        f"{runner.STUDY_ID} {'0' * 64}",  # wrong digest
    ):
        code, _, err = _run_tool(
            _real_argv(tmp_path, "--authorization-url", URL), gh=FakeGh(body)
        )
        assert code == tool.EXIT_UNAUTHENTICATED, body
        assert "authorization failed" in err
    other = FakeGh(f"{runner.STUDY_ID} {digest}", login="someone-else")
    code, _, _ = _run_tool(_real_argv(tmp_path, "--authorization-url", URL), gh=other)
    assert code == tool.EXIT_UNAUTHENTICATED
    code, _, _ = _run_tool(
        _real_argv(tmp_path, "--authorization-url", "https://example.com/x"),
        gh=FakeGh("x"),
    )
    assert code == tool.EXIT_UNAUTHENTICATED
    assert not (tmp_path / "t.jsonl").exists()


def test_an_authorized_real_data_request_still_runs_nothing_without_a_loader(
    tmp_path: Path,
) -> None:
    digest = runner.protocol_digest()
    gh = FakeGh(f"G4: {runner.STUDY_ID}\nprotocol_digest: {digest}\n")
    code, document, err = _run_tool(
        _real_argv(tmp_path, "--authorization-url", URL), gh=gh
    )
    assert code == tool.EXIT_NOT_WIRED and document is None
    assert "no real-data loader" in err
    assert gh.calls == 1
    assert not (tmp_path / "t.jsonl").exists() and not (tmp_path / "c").exists()


def test_an_authorized_request_with_a_loader_records_the_receipt(
    tmp_path: Path,
) -> None:
    digest = runner.protocol_digest()
    gh = FakeGh(f"{runner.STUDY_ID} {digest}")

    def loader(_args: Any) -> runner.LabSetup:
        return support.synthetic_setup(n_bars=120)

    code, document, err = _run_tool(
        _real_argv(tmp_path, "--authorization-url", URL, "--timesteps", "64"),
        gh=gh,
        setup_factory=loader,
    )
    assert code == 0, err
    assert document is not None and document["data"] == "real"
    receipt = document["authorization_receipt"]
    assert receipt["authorization_url"] == URL
    assert receipt["quoted_protocol_digest"] == digest
    assert gh.calls == 2  # fetched, then re-fetched unedited before the run


def test_an_edited_comment_between_fetches_authorizes_nothing(tmp_path: Path) -> None:
    digest = runner.protocol_digest()

    class Edited(FakeGh):
        def __call__(self, args: Any) -> str:
            payload = json.loads(super().__call__(args))
            if self.calls > 1:
                payload["body"] += " edited"
            return json.dumps(payload)

    code, _, err = _run_tool(
        _real_argv(tmp_path, "--authorization-url", URL),
        gh=Edited(f"{runner.STUDY_ID} {digest}"),
        setup_factory=lambda _a: support.synthetic_setup(n_bars=120),
    )
    assert code == tool.EXIT_UNAUTHENTICATED and "edited" in err
    assert not (tmp_path / "t.jsonl").exists()


def test_the_numpy_split_index_helper_is_sane_on_the_synthetic_windows() -> None:
    setup = support.synthetic_setup(n_bars=120)
    windows = splits.split_indices(setup.dataset.timestamps, dict(setup.splits))
    assert windows[splits.TRAIN][0] == 0 and windows[splits.TEST][1] == 119
    assert np.all(np.diff(setup.dataset.timestamps.astype("int64")) > 0)
