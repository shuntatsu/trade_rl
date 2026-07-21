from __future__ import annotations

from pathlib import Path

import yaml


def test_compose_defines_persistent_healthy_postgres_service() -> None:
    payload = yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8"))

    postgres = payload["services"]["postgres"]
    assert postgres["image"].startswith("postgres:16")
    assert postgres["environment"] == {
        "POSTGRES_DB": "${TRADE_RL_POSTGRES_DB:-trade_rl}",
        "POSTGRES_USER": "${TRADE_RL_POSTGRES_USER:-trade_rl}",
        "POSTGRES_PASSWORD": "${TRADE_RL_POSTGRES_PASSWORD:-trade_rl}",
    }
    assert postgres["ports"] == ["${TRADE_RL_POSTGRES_PORT:-5432}:5432"]
    assert "trade-rl-postgres:/var/lib/postgresql/data" in postgres["volumes"]
    assert postgres["healthcheck"]["test"][0:2] == [
        "CMD-SHELL",
        "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}",
    ]
    assert "trade-rl-postgres" in payload["volumes"]
