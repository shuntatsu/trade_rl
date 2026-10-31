"""Attempt-2 tooling and the watchdog stay offline, journal-free and order-free.

Static checks cover direct imports only. The transitive and call-level properties
are proved in fresh interpreters: ``watchdog`` must not even load the Binance
stack or the journal modules, while ``attempt2`` does load the collector stack at
import time (it reuses ``seal_paper_study``), so its offline property is proved at
call level by a recorded, refused socket layer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import trade_rl.evaluation as evaluation
from tools.agent_repo.source_index import ImportCollector, within_module
from trade_rl.evaluation.paper.operations import seal_paper_study

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
PAPER = PACKAGE / "evaluation" / "paper"
NETWORK = {"socket", "http", "urllib", "requests", "ssl", "ftplib"}
AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-4242424242"


def _imports(name: str) -> set[str]:
    return ImportCollector(PACKAGE).collect(PAPER / name)


def _run(script: str, *arguments: str) -> Any:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_attempt2_direct_imports_have_no_network_collector_or_order_surface() -> None:
    imported = _imports("attempt2.py")
    assert not {item.split(".", 1)[0] for item in imported} & NETWORK
    for forbidden in (
        "trade_rl.integrations",
        "trade_rl.evaluation.paper.supervisor",
        "trade_rl.evaluation.paper.storage_probe",
        "trade_rl.evaluation.paper.engine",
        "trade_rl.evaluation.paper.store",
        "trade_rl.evaluation.paper.assessment",
        "trade_rl.evaluation.final_test",
        "trade_rl.strategies",
        "trade_rl.simulation",
    ):
        assert not any(within_module(item, forbidden) for item in imported), forbidden


def test_watchdog_direct_imports_have_no_journal_or_market_surface() -> None:
    imported = _imports("watchdog.py")
    assert not {item.split(".", 1)[0] for item in imported} & NETWORK
    # control is allowed only for its read-only ``incomplete_cycle_exists`` probe.
    for forbidden in (
        "trade_rl.integrations",
        "trade_rl.evaluation.paper.store",
        "trade_rl.evaluation.paper.engine",
        "trade_rl.evaluation.paper.supervisor",
        "trade_rl.evaluation.paper.assessment",
        "trade_rl.strategies",
        "trade_rl.simulation",
    ):
        assert not any(within_module(item, forbidden) for item in imported), forbidden


def test_watchdog_transitive_imports_exclude_the_market_stack_and_journal() -> None:
    loaded = _run(
        "import json, sys\n"
        "import trade_rl.evaluation.paper.watchdog\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    modules = set(loaded)
    for forbidden in (
        "trade_rl.integrations",
        "trade_rl.evaluation.paper.store",
        "trade_rl.evaluation.paper.engine",
        "trade_rl.evaluation.paper.supervisor",
        "trade_rl.evaluation.paper.assessment",
        "http.client",
        "ssl",
        "urllib.request",
        "requests",
    ):
        assert not any(within_module(name, forbidden) for name in modules), forbidden


def test_plan_and_check_attempt_no_connection_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    sealed = tmp_path / "sealed"
    start = datetime.now(UTC) + timedelta(days=1)
    digest = seal_paper_study(
        sealed,
        start_at=start,
        attempt_lineage={
            "schema": "carry_paper_attempt_lineage_v2",
            "attempt_number": 1,
            "predecessor": None,
        },
    )
    script = (
        "import json, socket, sys\n"
        "from datetime import UTC, datetime, timedelta\n"
        "from pathlib import Path\n"
        "attempts = []\n"
        "def refuse(*args, **kwargs):\n"
        "    attempts.append(repr(args)[:80])\n"
        "    raise OSError('network is forbidden')\n"
        "socket.socket.connect = refuse\n"
        "socket.socket.connect_ex = refuse\n"
        "socket.getaddrinfo = refuse\n"
        "socket.create_connection = refuse\n"
        "from trade_rl.evaluation.paper.attempt2 import plan_attempt2\n"
        "from trade_rl.evaluation.paper.watchdog import check_collection\n"
        "draft = plan_attempt2(\n"
        "    Path(sys.argv[1]) / 'attempt2',\n"
        "    start_at=datetime.now(UTC) + timedelta(days=1),\n"
        "    authorization=sys.argv[2],\n"
        "    predecessor_root_status='unavailable',\n"
        ")\n"
        "report = check_collection(Path(sys.argv[3]), protocol_sha256=sys.argv[4])\n"
        "print(json.dumps(dict(attempts=attempts, status=draft['status'],\n"
        "    state=report['state'])))\n"
    )
    result = _run(script, str(tmp_path), AUTH, str(sealed), digest)
    assert result["attempts"] == []
    assert result["status"] == "DRAFT_NOT_SEALED"
    assert result["state"] == "not_started"
    assert not (tmp_path / "attempt2").exists()


def test_paper_operations_do_not_widen_the_public_evaluation_api() -> None:
    assert not any(
        name.lower().startswith(("attempt2", "watchdog", "heartbeat"))
        for name in evaluation.__all__
    )
