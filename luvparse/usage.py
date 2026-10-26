"""Cost and latency accounting for every model call.

Every place that calls a model records what it spent, so a parsing run can report
cost per page. This module is the single ledger for that. It is deliberately free
of provider imports: a client records tokens and cost, and the caller decides
where the rows go.

Pass the same ``UsageLedger`` down through a run, then read ``ledger.to_rows()``.
``operation`` should be one of:

    vision.ocr      vision.figure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class UsageRecord:
    """One model call: what it was, what it cost, and how long it took."""

    operation: str
    provider: str
    model: str
    seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float = 0.0
    document_id: str | None = None
    stage: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total_tokens(self) -> int:
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    def as_row(self) -> dict[str, Any]:
        """Map onto a plain dict row."""
        return {
            "document_id": self.document_id,
            "stage": self.stage,
            "operation": self.operation,
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "seconds": round(self.seconds, 4),
            "metadata": dict(self.metadata),
        }


class UsageLedger:
    """An in-memory list of ``UsageRecord`` with totals for a run.

    Cheap to create and safe to pass around. It performs no I/O, so tests can
    assert on costs without a database, and the caller can flush it once per
    document or per query.
    """

    def __init__(self, document_id: str | None = None) -> None:
        self.records: list[UsageRecord] = []
        # One ledger per document, so a call site need not repeat the id.
        self.document_id = document_id

    def record(
        self,
        *,
        operation: str,
        provider: str,
        model: str,
        seconds: float,
        usage: dict[str, Any] | None = None,
        cost_usd: float | None = None,
        document_id: str | None = None,
        stage: str | None = None,
        **metadata: Any,
    ) -> UsageRecord:
        """Record one call.

        ``usage`` is the provider's usage payload when it has one. OpenRouter
        reports ``prompt_tokens``, ``completion_tokens`` and ``cost`` there.
        """
        payload = usage or {}
        resolved_cost = cost_usd
        if resolved_cost is None:
            resolved_cost = _to_float(payload.get("cost")) or 0.0

        record = UsageRecord(
            operation=operation,
            provider=provider,
            model=model,
            seconds=float(seconds),
            prompt_tokens=_to_int(payload.get("prompt_tokens")),
            completion_tokens=_to_int(payload.get("completion_tokens")),
            cost_usd=float(resolved_cost),
            document_id=document_id or self.document_id,
            stage=stage,
            metadata=metadata,
        )
        self.records.append(record)
        return record

    @property
    def total_cost_usd(self) -> float:
        return round(sum(record.cost_usd for record in self.records), 8)

    @property
    def total_seconds(self) -> float:
        return round(sum(record.seconds for record in self.records), 4)

    def by_operation(self) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        for record in self.records:
            bucket = summary.setdefault(
                record.operation,
                {"calls": 0, "cost_usd": 0.0, "seconds": 0.0, "tokens": 0},
            )
            bucket["calls"] += 1
            bucket["cost_usd"] = round(bucket["cost_usd"] + record.cost_usd, 8)
            bucket["seconds"] = round(bucket["seconds"] + record.seconds, 4)
            bucket["tokens"] += record.total_tokens
        return summary

    def summary(self) -> dict[str, Any]:
        return {
            "calls": len(self.records),
            "cost_usd": self.total_cost_usd,
            "seconds": self.total_seconds,
            "by_operation": self.by_operation(),
        }

    def to_rows(self) -> list[dict[str, Any]]:
        return [record.as_row() for record in self.records]

    def extend(self, records: Iterable[UsageRecord]) -> None:
        self.records.extend(records)
