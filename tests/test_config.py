import pytest

from rag_mcp.config import ConfigError, Settings, validate_backend_url

ALLOWED = ("127.0.0.1", "localhost")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("http://LOCALHOST:8000/", "http://localhost:8000"),
        ("https://localhost", "https://localhost"),
    ],
)
def test_accepts_allow_listed_hosts(url, expected):
    assert validate_backend_url(url, ALLOWED) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://evil.example.com:8000",
        "http://127.0.0.1.evil.example.com",
        "http://user:pass@127.0.0.1:8000",
        "http://127.0.0.1:8000/admin",
        "http://127.0.0.1:8000?x=1",
        "file:///etc/passwd",
        "ftp://127.0.0.1",
        "http://127.0.0.1:99999",
        "http://:8000",
    ],
)
def test_rejects_unsafe_urls(url):
    with pytest.raises(ConfigError):
        validate_backend_url(url, ALLOWED)


def test_from_env_reads_overrides():
    s = Settings.from_env(
        {"RAG_MCP_BACKEND_URL": "http://api:8000", "RAG_MCP_ALLOWED_HOSTS": "api", "RAG_MCP_ASK_PER_DAY": "7"}
    )
    assert s.backend_url == "http://api:8000"
    assert s.ask_per_day == 7


def test_from_env_defaults_to_loopback():
    assert Settings.from_env({}).backend_url == "http://127.0.0.1:8000"


@pytest.mark.parametrize(
    "env",
    [
        {"RAG_MCP_BACKEND_URL": "http://api:8000"},  # host not allow-listed by default
        {"RAG_MCP_ASK_PER_DAY": "lots"},
        {"RAG_MCP_ASK_PER_MINUTE": "0"},
    ],
)
def test_from_env_rejects_bad_values(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


@pytest.mark.parametrize(
    "kwargs",
    [{"free_timeout_s": 0}, {"ask_timeout_s": 1000}, {"max_retries": 9}, {"ask_top_k": 11}],
)
def test_rejects_out_of_range_settings(kwargs):
    with pytest.raises(ConfigError):
        Settings(**kwargs)
