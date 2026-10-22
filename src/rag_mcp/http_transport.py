"""Streamable HTTP transport with a static bearer token.

This is a deliberate simplification for local or single-user remote use, not the MCP OAuth 2.1
authorization flow: one shared secret from the environment, compared in constant time, never logged.
The SDK's DNS-rebinding protection (Host/Origin allow-lists) stays enabled.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

from rag_mcp.config import ConfigError

MIN_TOKEN_CHARS = 32
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


@dataclass(frozen=True)
class HttpSettings:
    token: str
    host: str = "127.0.0.1"
    port: int = 8765
    allowed_hosts: tuple[str, ...] = ("127.0.0.1:*", "localhost:*")
    allowed_origins: tuple[str, ...] = ("http://127.0.0.1:*", "http://localhost:*")

    def __post_init__(self) -> None:
        if len(self.token) < MIN_TOKEN_CHARS:
            raise ConfigError(f"RAG_MCP_HTTP_TOKEN must be at least {MIN_TOKEN_CHARS} characters")
        if not 1 <= self.port <= 65535:
            raise ConfigError("HTTP port out of range")
        if not self.allowed_hosts:
            raise ConfigError("at least one allowed Host header value is required")

    def __repr__(self) -> str:  # never print the token
        return f"HttpSettings(host={self.host!r}, port={self.port}, token=<redacted>)"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> HttpSettings:
        def split(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = env.get(name, "")
            return tuple(v.strip() for v in raw.split(",") if v.strip()) or default

        try:
            port = int(env.get("RAG_MCP_HTTP_PORT", "8765"))
        except ValueError as exc:
            raise ConfigError("RAG_MCP_HTTP_PORT is not an integer") from exc
        return cls(
            token=env.get("RAG_MCP_HTTP_TOKEN", ""),
            host=env.get("RAG_MCP_HTTP_HOST", "127.0.0.1"),
            port=port,
            allowed_hosts=split("RAG_MCP_HTTP_ALLOWED_HOSTS", cls.allowed_hosts),
            allowed_origins=split("RAG_MCP_HTTP_ALLOWED_ORIGINS", cls.allowed_origins),
        )


class BearerTokenMiddleware:
    """Rejects every HTTP request without the exact bearer token. Lifespan events pass through."""

    def __init__(self, app: ASGIApp, token: str):
        self._app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        provided = b""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                provided = value
                break
        if not hmac.compare_digest(provided, self._expected):
            body = json.dumps({"error": "unauthorized"}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b"Bearer"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self._app(scope, receive, send)


def build_http_app(server: MCPServer, http: HttpSettings) -> ASGIApp:
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(http.allowed_hosts),
        allowed_origins=list(http.allowed_origins),
    )
    app = server.streamable_http_app(transport_security=security, host=http.host)
    return BearerTokenMiddleware(app, http.token)
