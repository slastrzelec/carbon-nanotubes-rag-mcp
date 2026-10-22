"""Server configuration.

The backend URL is fixed at start-up and validated against a host allow-list, so neither a
tool argument nor a model can point the server at another host (SSRF control). Settings are
read from environment variables only; nothing here reads or needs an LLM API key.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost")

# The backend's own limits (carbon-nanotubes-rag api.py); the server must not be wider.
BACKEND_MAX_TEXT_CHARS = 500
BACKEND_MAX_SEARCH_TOP_K = 10


class ConfigError(ValueError):
    """Invalid configuration; raised at start-up, never at request time."""


def validate_backend_url(url: str, allowed_hosts: tuple[str, ...]) -> str:
    """Returns the normalised URL or raises ConfigError."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ConfigError("backend URL must use http or https")
    if parts.username or parts.password:
        raise ConfigError("backend URL must not contain credentials")
    host = (parts.hostname or "").lower()
    if not host or host not in {h.lower() for h in allowed_hosts}:
        raise ConfigError("backend host is not in the allow-list")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ConfigError("backend URL must not contain a path, query or fragment")
    try:
        port = parts.port
    except ValueError as exc:
        raise ConfigError("backend URL has an invalid port") from exc
    netloc = host if port is None else f"{host}:{port}"
    return f"{parts.scheme}://{netloc}"


@dataclass(frozen=True)
class Settings:
    backend_url: str = DEFAULT_BACKEND_URL
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    free_timeout_s: float = 10.0
    ask_timeout_s: float = 60.0
    max_retries: int = 2
    retry_backoff_s: float = 0.5
    ask_per_minute: int = 6
    ask_per_day: int = 100
    free_per_minute: int = 30
    ask_top_k: int = 5
    max_response_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_url", validate_backend_url(self.backend_url, self.allowed_hosts))
        if not 0 < self.free_timeout_s <= 60 or not 0 < self.ask_timeout_s <= 300:
            raise ConfigError("timeouts out of range")
        if not 0 <= self.max_retries <= 5:
            raise ConfigError("max_retries out of range")
        if min(self.ask_per_minute, self.ask_per_day, self.free_per_minute) < 1:
            raise ConfigError("budgets must be positive")
        if not 1 <= self.ask_top_k <= BACKEND_MAX_SEARCH_TOP_K:
            raise ConfigError("ask_top_k out of range")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        hosts_raw = env.get("RAG_MCP_ALLOWED_HOSTS", "")
        hosts = tuple(h.strip() for h in hosts_raw.split(",") if h.strip()) or DEFAULT_ALLOWED_HOSTS
        try:
            return cls(
                backend_url=env.get("RAG_MCP_BACKEND_URL", DEFAULT_BACKEND_URL),
                allowed_hosts=hosts,
                ask_per_minute=int(env.get("RAG_MCP_ASK_PER_MINUTE", "6")),
                ask_per_day=int(env.get("RAG_MCP_ASK_PER_DAY", "100")),
                free_per_minute=int(env.get("RAG_MCP_FREE_PER_MINUTE", "30")),
            )
        except ValueError as exc:
            if isinstance(exc, ConfigError):
                raise
            raise ConfigError("numeric setting is not an integer") from exc
