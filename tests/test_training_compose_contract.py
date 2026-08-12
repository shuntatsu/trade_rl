from __future__ import annotations

from pathlib import Path


def test_training_compose_separates_market_data_ownership() -> None:
    compose = (Path(__file__).resolve().parents[1] / "compose.training.yaml").read_text(
        encoding="utf-8"
    )

    assert "market-data-sync:" in compose
    assert "trade-rl-market-archives:/workspace/market-data/binance-vision" in compose
    assert (
        "trade-rl-market-archives:/workspace/market-data/binance-vision:ro" in compose
    )
    assert "trade-rl-training-data:/workspace/legacy-var:ro" in compose
    assert "trade-rl-training-data:/workspace/var" in compose
    assert "postgres:" in compose
    assert "condition: service_healthy" in compose
    assert "TRADE_RL_DATABASE_URL:" in compose
    assert 'TORCHINDUCTOR_COMPILE_THREADS: "4"' in compose
    assert 'TRADE_RL_TEACHER_WORKERS: "1"' in compose
    assert "TRADE_RL_ORACLE_SOLVER: ${TRADE_RL_ORACLE_SOLVER:-numpy}" in compose
    assert (
        "TRADE_RL_ORACLE_EPISODE_BATCH_SIZE: "
        "${TRADE_RL_ORACLE_EPISODE_BATCH_SIZE:-8}" in compose
    )
    assert (
        "TRADE_RL_ORACLE_TARGET_STATE_BLOCK_SIZE: "
        "${TRADE_RL_ORACLE_TARGET_STATE_BLOCK_SIZE:-}" in compose
    )
    assert (
        "TRADE_RL_ORACLE_CUDA_MEMORY_FRACTION: "
        "${TRADE_RL_ORACLE_CUDA_MEMORY_FRACTION:-0.65}" in compose
    )
    assert 'TRADE_RL_ORACLE_COMPILE_MODE: "disabled"' in compose
    assert "TRADE_RL_ORACLE_COMPILE_CHUNK_SIZE:" not in compose
    assert "TRADE_RL_FROZEN_METADATA_CACHE_ROOT:" in compose
    assert "--legacy-cache-root" in compose
    assert "trade-rl-training-runs:" not in compose
    assert "trade-rl-teacher-cache:" not in compose


def test_universal_training_compose_is_gpu_manifest_and_external_db_bound() -> None:
    compose = (
        Path(__file__).resolve().parents[1] / "compose.universal-training.yaml"
    ).read_text(encoding="utf-8")
    assert "gpus: all" in compose
    assert "external: true" in compose
    assert "name: trade_rl_default" in compose
    assert "/workspace/var/universal" in compose
    assert "read_only: true" in compose
    assert "./:/workspace" not in compose
    assert "restart: \"no\"" in compose
    assert "universal-u6-ppo.json" in compose
    assert "universal-u6-lagrangian.json" in compose
    assert "universal-u6-discounted.json" in compose
    assert (
        "TRADE_RL_UNIVERSAL_ORACLE_MAX_EPISODES_PER_SYMBOL: "
        "${TRADE_RL_UNIVERSAL_ORACLE_MAX_EPISODES_PER_SYMBOL:-10}" in compose
    )
