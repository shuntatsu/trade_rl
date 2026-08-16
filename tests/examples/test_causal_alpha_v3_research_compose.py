from __future__ import annotations

import yaml

from tests.architecture.repository_paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT
COMPOSE = ROOT / "docker/compose.causal-alpha-v3-research.yaml"


def _payload() -> dict[str, object]:
    raw = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_causal_alpha_v3_research_compose_uses_fixed_authored_command() -> None:
    payload = _payload()
    services = payload["services"]
    assert isinstance(services, dict)
    research = services["research"]
    assert isinstance(research, dict)

    assert research["image"] == "${TRADE_RL_CAUSAL_ALPHA_V3_IMAGE:?required}"
    assert research["command"] == [
        "python",
        "scripts/run_universal_causal_alpha_v3_research.py",
        "--config",
        "examples/binance/universal-causal-alpha-v3-research.json",
        "--run-config",
        "examples/binance-multitimeframe/universal-u6-ppo.json",
        "--runtime-factory",
        "trade_rl.workflows.binance_universal_runtime:build_runtime",
        "--runtime-manifest",
        "/workspace/var/universal/runtime-manifest.json",
        "--frozen-metadata-root",
        "/workspace/var/cache/frozen-metadata/usds-m",
        "--output-root",
        "/workspace/var/runs/${TRADE_RL_RUN_GENERATION:?required}",
    ]
    assert research["gpus"] == "all"
    assert research["restart"] == "no"


def test_causal_alpha_v3_research_compose_uses_durable_trusted_mounts() -> None:
    payload = _payload()
    research = payload["services"]["research"]
    volumes = research["volumes"]

    assert "trade-rl-training-data:/workspace/var" in volumes
    assert {
        "type": "bind",
        "source": "${TRADE_RL_UNIVERSAL_ARTIFACT_ROOT:?required}",
        "target": "/workspace/var/universal",
        "read_only": True,
    } in volumes
    assert all(".:/workspace" not in str(item) for item in volumes)

    declared = payload["volumes"]["trade-rl-training-data"]
    assert declared == {"external": True, "name": "trade-rl-training-data"}


def test_causal_alpha_v3_research_compose_binds_container_identity_and_network() -> None:
    payload = _payload()
    research = payload["services"]["research"]

    assert research["labels"] == {
        "trade-rl.kind": "causal-alpha-v3",
        "trade-rl.generation": "${TRADE_RL_RUN_GENERATION:?required}",
        "trade-rl.git-commit": "${TRADE_RL_GIT_COMMIT:?required}",
        "trade-rl.runtime-manifest-digest": (
            "${TRADE_RL_RUNTIME_MANIFEST_DIGEST:?required}"
        ),
    }
    assert research["networks"] == ["trade-rl"]
    assert payload["networks"]["trade-rl"] == {
        "external": True,
        "name": "trade_rl_default",
    }


def test_causal_alpha_v3_research_compose_does_not_make_scientific_configs_variable() -> (
    None
):
    text = COMPOSE.read_text(encoding="utf-8")

    assert "TRADE_RL_PPO_CONFIG" not in text
    assert "TRADE_RL_V3_CONFIG" not in text
    assert "TRADE_RL_RESEARCH_CONFIG" not in text
    assert "TRADE_RL_RUN_CONFIG" not in text
