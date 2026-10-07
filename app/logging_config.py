"""JSONL event log (one JSON object per line) plus console logging setup."""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

console = logging.getLogger("triage")


def configure_console_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class JsonlLogger:
    """Append-only JSONL writer. Thread-safe. Adds a UTC timestamp to every record."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        line = json.dumps(record, ensure_ascii=False, default=str)
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            # A broken log must never break a request, but it must be loud.
            console.exception("Failed to write JSONL log record")
