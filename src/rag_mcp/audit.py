"""Structured audit log for tool calls.

Logged: tool name, keyed hash of the arguments, duration, outcome, backend status code.
Never logged: question or query text, passage text, answers, tokens or keys.

The argument hash is an HMAC with a random per-process key: equal calls can be correlated
within one run, but short questions cannot be recovered offline by hashing guesses.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

LOGGER_NAME = "rag_mcp.audit"
_KEY = secrets.token_bytes(32)
logger = logging.getLogger(LOGGER_NAME)


def args_digest(args: dict[str, Any]) -> str:
    canonical = json.dumps(args, sort_keys=True, ensure_ascii=False).encode()
    return hmac.new(_KEY, canonical, hashlib.sha256).hexdigest()[:16]


@dataclass
class AuditRecord:
    tool: str
    args_hmac: str
    status: str = "ok"
    backend_status: int | None = None


@asynccontextmanager
async def audited(tool: str, args: dict[str, Any]) -> AsyncIterator[AuditRecord]:
    record = AuditRecord(tool=tool, args_hmac=args_digest(args))
    start = time.perf_counter()
    try:
        yield record
    except BaseException:
        if record.status == "ok":
            record.status = "error"
        raise
    finally:
        logger.info(
            "tool_call",
            extra={
                "audit": {
                    "tool": record.tool,
                    "args_hmac": record.args_hmac,
                    "status": record.status,
                    "backend_status": record.backend_status,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                }
            },
        )


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "audit", {}))
        return json.dumps(payload)


def configure_logging() -> None:
    """Logs go to stderr: stdout carries the stdio protocol and must stay clean."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    # The SDK logs tracebacks of unexpected tool crashes; tools here raise only ToolError with
    # fixed messages, so no request text reaches these loggers on the normal error paths.
    logging.getLogger("httpx").setLevel(logging.WARNING)
