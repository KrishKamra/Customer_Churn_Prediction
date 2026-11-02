"""Structured logging.

Logs are emitted as single-line JSON when ``CHURNAI_LOG_FORMAT=json`` (for log shippers
such as Loki / Datadog) and as a compact human-readable format otherwise. Extra keyword
context passed via ``logger.info("msg", extra={"ctx": {...}})`` is merged into the record.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict):
            payload.update(ctx)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-friendly formatter that appends structured context as ``key=value`` pairs."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict) and ctx:
            base += " | " + " ".join(f"{k}={v}" for k, v in ctx.items())
        return base


def configure_logging(level: str | None = None) -> None:
    """Configure the root ``churnai`` logger once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    fmt = os.getenv("CHURNAI_LOG_FORMAT", "console").lower()
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root = logging.getLogger("churnai")
    root.handlers[:] = [handler]
    root.setLevel((level or os.getenv("CHURNAI_LOG_LEVEL") or "INFO").upper())
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (``churnai.<name>``), configuring logging on first use."""
    configure_logging()
    return logging.getLogger(f"churnai.{name}")
