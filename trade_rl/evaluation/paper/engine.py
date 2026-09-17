"""Replayable paper commands: verified evidence, canonical account, durable events."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.paper.account import PaperAccount, PaperSettings, timestamp
from trade_rl.evaluation.paper.store import PaperJournal
from trade_rl.integrations.binance.forward_evidence import read_forward_snapshot
from trade_rl.integrations.binance.forward_rules import read_forward_rules

_SCHEMA = "funding_carry_paper_engine_v1"


@dataclass(frozen=True)
class EvidenceRef:
    path: str
    sha256: str


class PaperEngine:
    """No network or live orders; every command consumes already saved evidence."""

    @staticmethod
    def initialize(
        root: str | Path, *, settings: PaperSettings, study: dict[str, Any]
    ) -> str:
        if not isinstance(study, dict):
            raise ValueError("paper study must be an object")
        return PaperJournal.initialize(
            root, dict(engine=_SCHEMA, settings=settings.to_dict(), study=study)
        )

    def __init__(self, root: str | Path, *, expected_protocol_sha256: str) -> None:
        self.journal = PaperJournal(
            root, expected_protocol_sha256=expected_protocol_sha256
        )
        protocol = json.loads((self.journal.root / "protocol.json").read_bytes())[
            "protocol"
        ]
        if (
            set(protocol) != {"engine", "settings", "study"}
            or protocol["engine"] != _SCHEMA
            or not isinstance(protocol["study"], dict)
        ):
            raise ValueError("unsupported paper engine protocol")
        config = dict(protocol["settings"])
        for name in ("start_at", "close_at"):
            config[name] = timestamp(config[name])
        self._account = PaperAccount(PaperSettings(**config))
        self._tip = expected_protocol_sha256
        self._records: dict[str, dict[str, Any]] = {}
        for record in self.journal.events():
            event = record["event"]
            payload = event["payload"]
            if set(payload) != {"request", "result"}:
                raise ValueError("paper command payload schema mismatch")
            outcome = self._apply(self._account, event["kind"], payload["request"])
            if canonical_json_bytes(outcome) != canonical_json_bytes(payload["result"]):
                raise ValueError("paper replay result differs from committed result")
            self._records[event["key"]] = record
            self._tip = record["sha256"]

    def _path(self, reference: dict[str, Any]) -> Path:
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError("invalid paper evidence reference")
        name = reference["path"]
        if not isinstance(name, str) or not name or "\\" in name:
            raise ValueError("invalid paper evidence path")
        path = Path(name)
        if path.is_absolute() or any(part in {".", ".."} for part in name.split("/")):
            raise ValueError("paper evidence path must stay within the journal")
        destination = self.journal.root / path
        if not destination.resolve().is_relative_to(self.journal.root.resolve()) or any(
            part.is_symlink()
            for part in [destination, *destination.parents]
            if part != self.journal.root
        ):
            raise ValueError("paper evidence path escapes the journal")
        digest = reference["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("paper evidence reference requires a SHA-256 digest")
        return destination

    def _apply(
        self, account: PaperAccount, kind: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        fields = {
            "decision": {"at", "market", "rules"},
            "execution": {"at", "market"},
            "gap": {"at", "reason"},
        }
        if (
            kind not in fields
            or not isinstance(request, dict)
            or set(request) != fields[kind]
        ):
            raise ValueError("paper command schema mismatch")
        at = timestamp(request["at"])
        if kind == "gap":
            return account.gap(at=at, reason=request["reason"])
        market = request["market"]
        snapshot = read_forward_snapshot(
            self._path(market), expected_sha256=market["sha256"], as_of=at
        )
        if kind == "decision":
            rule = request["rules"]
        else:
            if account.pending is None:
                raise ValueError("no saved pending decision")
            rule = account.pending["rules"]
        metadata = read_forward_rules(
            self._path(rule), expected_sha256=rule["sha256"], as_of=at
        )
        if kind == "decision":
            return account.decide(snapshot, metadata, rule, at)
        return account.execute(snapshot, metadata, at)

    def _append(self, key: str, kind: str, request: dict[str, Any]) -> dict[str, Any]:
        request = json.loads(canonical_json_bytes(request))
        previous = self._records.get(key)
        if previous is not None:
            event = previous["event"]
            if event["kind"] != kind or canonical_json_bytes(
                event["payload"]["request"]
            ) != canonical_json_bytes(request):
                raise ValueError(
                    "paper idempotency key was reused with a different command"
                )
            return copy.deepcopy(previous)
        proposed = copy.deepcopy(self._account)
        outcome = self._apply(proposed, kind, request)
        record = self.journal.append(
            key, kind, dict(request=request, result=outcome), expected_parent=self._tip
        )
        self._account, self._tip = proposed, record["sha256"]
        self._records[key] = record
        return copy.deepcopy(record)

    def status(self) -> dict[str, Any]:
        return dict(**self._account.status(), tip=self._tip, events=len(self._records))

    def unsettled_funding(self) -> list[dict[str, str]]:
        return self._account.unsettled_funding()

    def decide(
        self, key: str, *, market: EvidenceRef, rules: EvidenceRef, at: datetime
    ) -> dict[str, Any]:
        return self._append(
            key,
            "decision",
            dict(at=at.isoformat(), market=asdict(market), rules=asdict(rules)),
        )

    def execute(self, key: str, *, market: EvidenceRef, at: datetime) -> dict[str, Any]:
        return self._append(
            key, "execution", dict(at=at.isoformat(), market=asdict(market))
        )

    def gap(self, key: str, *, at: datetime, reason: str) -> dict[str, Any]:
        return self._append(key, "gap", dict(at=at.isoformat(), reason=reason))
