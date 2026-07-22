from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    (ROOT / path).write_text(content, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    content = read(path)
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement anchor, found {count}")
    write(path, content.replace(old, new, 1))


write(
    "trade_rl/simulation/__init__.py",
    '''"""Portfolio execution and accounting simulation."""

from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionResult
from trade_rl.simulation.execution_adapter import StatefulCompatibilityMarketExecutor

MarketExecutor = StatefulCompatibilityMarketExecutor

__all__ = [
    "BookState",
    "EconomicTerminationReason",
    "ExecutionCostConfig",
    "ExecutionResult",
    "MarketExecutor",
]
''',
)

write(
    "trade_rl/telemetry/__init__.py",
    '''"""Training telemetry contracts and append-only indexed storage."""

from trade_rl.telemetry import training as _training
from trade_rl.telemetry.indexed_training import (
    IndexedTrainingTelemetryWriter,
    StrictTrainingTelemetryRecord,
    indexed_training_telemetry_status,
    read_indexed_training_telemetry,
)

TELEMETRY_SCHEMA_VERSION = _training.TELEMETRY_SCHEMA_VERSION
TelemetryEventType = _training.TelemetryEventType
TrainingTelemetryPage = _training.TrainingTelemetryPage
TrainingTelemetryRecord = StrictTrainingTelemetryRecord
TrainingTelemetryStatus = _training.TrainingTelemetryStatus
TrainingTelemetryWriter = IndexedTrainingTelemetryWriter
read_training_telemetry = read_indexed_training_telemetry
training_telemetry_status = indexed_training_telemetry_status

__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
    "TelemetryEventType",
    "TrainingTelemetryPage",
    "TrainingTelemetryRecord",
    "TrainingTelemetryStatus",
    "TrainingTelemetryWriter",
    "read_training_telemetry",
    "training_telemetry_status",
]
''',
)

write(
    "trade_rl/studio/__init__.py",
    '''"""Local-only user interface runtime for Trade RL research artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trade_rl.studio.catalog import StudioCatalog
    from trade_rl.studio.settings import StudioSettings


def __getattr__(name: str) -> Any:
    if name == "StudioCatalog":
        from trade_rl.studio.catalog import StudioCatalog

        return StudioCatalog
    if name == "StudioSettings":
        from trade_rl.studio.settings import StudioSettings

        return StudioSettings
    raise AttributeError(name)


__all__ = ["StudioCatalog", "StudioSettings"]
''',
)

write(
    "trade_rl/catalog/__init__.py",
    '''"""Searchable metadata catalog for immutable research artifacts."""

from trade_rl.catalog.contracts import (
    ArtifactCatalog,
    ArtifactKind,
    ArtifactQuery,
    ArtifactRecord,
    ArtifactRegistration,
    ArtifactStatus,
    cache_key_digest,
)
from trade_rl.catalog.postgres_sealed_test import PostgresSealedTestReservationStore

__all__ = [
    "ArtifactCatalog",
    "ArtifactKind",
    "ArtifactQuery",
    "ArtifactRecord",
    "ArtifactRegistration",
    "ArtifactStatus",
    "PostgresSealedTestReservationStore",
    "cache_key_digest",
]
''',
)

replace_once(
    "trade_rl/rl/environment_execution.py",
    '''from trade_rl.simulation.execution import (
    ExecutionCostConfig,
    ExecutionResult,
    MarketExecutor,
)
from trade_rl.simulation.order_reconciliation import reconcile_target
from trade_rl.simulation.orders import OrderBookState, OrderType, TimeInForce
from trade_rl.simulation.stateful_execution import StatefulExecutionResult
''',
    '''from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionResult
from trade_rl.simulation.orders import OrderBookState
from trade_rl.simulation.stateful_execution import StatefulExecutionResult
from trade_rl.simulation.target_execution import execute_target_statefully
''',
)
replace_once(
    "trade_rl/rl/environment_execution.py",
    '''        reconciliation = reconcile_target(
            dataset_id=self.dataset.dataset_id,
            target_identity=target_identity,
            execution_policy_digest=executor.execution_policy_digest,
            target_weights=target_vector,
            book=book,
            order_book=order_book,
            reference_prices=self.dataset.close[request.start_index],
            decision_equity=max(book.portfolio_value, _LIQUIDATION_TOLERANCE),
            submit_index=request.start_index,
            latency_bars=self.execution_cost.order_latency_bars,
            order_type=OrderType(self.execution_cost.order_type),
            time_in_force=TimeInForce.GTC,
            expiry_index=None,
            limit_offset_rate=self.execution_cost.limit_offset_rate,
            maximum_gross=self.execution_cost.max_leverage,
        )
        return executor.execute_orders(
            book,
            reconciliation.order_book,
            reconciliation.new_intents,
            start_index=request.start_index,
            bars=request.bars,
        )
''',
    '''        return execute_target_statefully(
            executor,
            book,
            order_book,
            target_vector,
            start_index=request.start_index,
            bars=request.bars,
            target_identity=target_identity,
        )
''',
)

replace_once(
    "trade_rl/rl/environment_episode.py",
    '''                candidate_starts = valid_starts[available]
                if candidate_starts.size == 0:
                    candidate_starts = valid_starts
''',
    '''                candidate_starts = valid_starts[available]
                if candidate_starts.size == 0:
                    raise ValueError(
                        "episode sampling feature is unavailable for every valid start"
                    )
''',
)

contracts = read("trade_rl/catalog/contracts.py")
contracts = contracts.replace("import json\n", "", 1)
anchor = "from typing import Mapping, Protocol, TypeAlias\n"
if contracts.count(anchor) != 1:
    raise RuntimeError("catalog contracts import anchor changed")
contracts = contracts.replace(
    anchor,
    anchor + "\nfrom trade_rl.domain.canonical_json import canonical_json_bytes\n",
    1,
)
canonical_block = '''def canonical_json_bytes(
    value: Mapping[str, JsonValue] | Mapping[str, object],
) -> bytes:
    frozen = _freeze_json(value, field_name="JSON payload")
    if not isinstance(frozen, Mapping):
        raise ValueError("JSON payload must be an object")
    return json.dumps(
        thaw_json(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


'''
if contracts.count(canonical_block) != 1:
    raise RuntimeError("catalog canonical JSON implementation anchor changed")
contracts = contracts.replace(canonical_block, "", 1)
write("trade_rl/catalog/contracts.py", contracts)

replace_once(
    "trade_rl/catalog/postgres.py",
    '''    def reserve_sealed_test_access(self, record: SealedTestAccessRecord) -> None:
        with self._connect() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO catalog_sealed_test_access (
                            experiment_plan_digest, dataset_id, fold_index,
                            test_start, test_stop, selected_configuration,
                            selected_policy_digest, access_digest
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (
                            experiment_plan_digest, dataset_id, fold_index
                        ) DO NOTHING
                        RETURNING access_digest
                        """,
                        (
                            record.experiment_plan_digest,
                            record.dataset_id,
                            record.fold_index,
                            record.test_range.start,
                            record.test_range.stop,
                            record.selected_configuration,
                            record.selected_policy_digest,
                            record.access_digest,
                        ),
                    )
                    if cursor.fetchone() is None:
                        raise ValueError(
                            "sealed outer test was already opened for this plan"
                        )
''',
    '''    def reserve_sealed_test_access(self, record: SealedTestAccessRecord) -> None:
        from trade_rl.catalog.postgres_sealed_test import (
            PostgresSealedTestReservationStore,
        )

        PostgresSealedTestReservationStore(
            self._database_url,
            connection_factory=self._connection_factory,
        ).reserve_sealed_test_access(record)
''',
)

for root_name in ("trade_rl", "tests", "tools"):
    root = ROOT / root_name
    for path in root.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        if relative in {
            "trade_rl/telemetry/training.py",
            "trade_rl/telemetry/indexed_training.py",
        }:
            continue
        content = path.read_text(encoding="utf-8")
        updated = content.replace(
            "from trade_rl.telemetry.training import (",
            "from trade_rl.telemetry import (",
        )
        if updated != content:
            path.write_text(updated, encoding="utf-8")

for root_name in ("trade_rl", "tests", "tools"):
    root = ROOT / root_name
    for path in root.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("trade_rl/simulation/"):
            continue
        content = path.read_text(encoding="utf-8")
        if "from trade_rl.simulation import MarketExecutor" in content:
            continue
        updated = content
        multiline = "from trade_rl.simulation.execution import (\n"
        if multiline in updated and "    MarketExecutor,\n" in updated:
            updated = updated.replace(
                multiline,
                "from trade_rl.simulation import MarketExecutor\n" + multiline,
                1,
            )
            updated = updated.replace("    MarketExecutor,\n", "", 1)
        single_pattern = re.compile(
            r"^from trade_rl\.simulation\.execution import ([^\n]+)$",
            re.MULTILINE,
        )

        def split_import(match: re.Match[str]) -> str:
            names = [name.strip() for name in match.group(1).split(",")]
            if "MarketExecutor" not in names:
                return match.group(0)
            remaining = [name for name in names if name != "MarketExecutor"]
            result = "from trade_rl.simulation import MarketExecutor"
            if remaining:
                result += "\nfrom trade_rl.simulation.execution import " + ", ".join(
                    remaining
                )
            return result

        updated = single_pattern.sub(split_import, updated)
        if updated != content:
            path.write_text(updated, encoding="utf-8")

replace_once(
    "trade_rl/studio/api.py",
    '''from trade_rl.studio.telemetry import (
    StudioTelemetryReader,
    TelemetryEventsResponse,
    TelemetryStatusResponse,
)
''',
    '''from trade_rl.studio.strict_telemetry import (
    StrictStudioTelemetryReader as StudioTelemetryReader,
)
from trade_rl.studio.telemetry import TelemetryEventsResponse, TelemetryStatusResponse
''',
)

write(
    ".github/workflows/postgres-catalog.yml",
    '''name: PostgreSQL Catalog

on:
  push:
    branches:
      - main
  pull_request:
    paths:
      - compose.yaml
      - .env.example
      - pyproject.toml
      - uv.lock
      - trade_rl/catalog/**
      - trade_rl/data/artifact.py
      - trade_rl/evaluation/walk_forward/**
      - trade_rl/workflows/**
      - trade_rl/cli/**
      - tests/catalog/**
      - tests/evaluation/**
      - tests/workflows/**
      - tests/ops/test_postgres_compose_contract.py
      - .github/workflows/postgres-catalog.yml
  workflow_dispatch:

permissions:
  contents: read

jobs:
  postgres-catalog:
    runs-on: ubuntu-latest
    env:
      TRADE_RL_DATABASE_URL: postgresql://trade_rl:trade_rl@localhost:5432/trade_rl
      TRADE_RL_TEST_DATABASE_URL: postgresql://trade_rl:trade_rl@localhost:5432/trade_rl
    steps:
      - name: Checkout exact head
        uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
          persist-credentials: false
      - name: Set up uv
        uses: astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86
        with:
          python-version: "3.12"
          enable-cache: true
      - name: Validate Compose
        run: docker compose config --quiet
      - name: Start PostgreSQL
        run: docker compose up -d postgres
      - name: Wait for PostgreSQL
        run: |
          for attempt in $(seq 1 30); do
            if docker compose exec -T postgres pg_isready -U trade_rl -d trade_rl; then
              exit 0
            fi
            sleep 2
          done
          docker compose logs postgres
          exit 1
      - name: Install
        run: uv sync --extra dev --extra postgres
      - name: Migrate
        run: uv run trade-rl catalog migrate
      - name: Unit and integration tests
        run: >-
          uv run pytest
          tests/catalog
          tests/ops/test_postgres_compose_contract.py
          tests/data/test_market_artifact_catalog.py
          tests/cli/test_catalog_commands.py
          -q
      - name: Stop PostgreSQL
        if: always()
        run: docker compose down -v
''',
)
