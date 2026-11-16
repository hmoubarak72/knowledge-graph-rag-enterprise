"""Structured logging + an append-only JSONL writer for decision logs."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

_configured = False


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    global _configured
    if not _configured:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(colors=False),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        _configured = True
    return structlog.get_logger(name)  # type: ignore[no-any-return]


class JsonlLog:
    """Thread-safe append-only JSONL file. Used for routing decisions and ER audit trails."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        record = {"ts": datetime.now(UTC).isoformat(), **record}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
