"""Repository hygiene: no secrets, no corpus data, no LLM key handling in the server."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "_to_delete",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
}
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
        files = [ROOT / f for f in out.split("\0") if f]
        if files:
            return files
    except (OSError, subprocess.CalledProcessError):
        pass
    return [
        p for p in ROOT.rglob("*") if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(ROOT).parts)
    ]


def test_no_env_files_or_corpus_data_tracked():
    bad = [
        p
        for p in tracked_files()
        if (p.name.startswith(".env") and p.name != ".env.example")
        or p.suffix in {".pdf", ".index", ".faiss", ".pkl"}
        or p.name.startswith("chunks")
    ]
    assert bad == []


def test_no_api_keys_in_tracked_files():
    hits = []
    for p in tracked_files():
        try:
            text = p.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if KEY_PATTERN.search(text):
            hits.append(p.relative_to(ROOT))
    assert hits == []


def test_server_package_never_touches_llm_keys():
    src = "\n".join(p.read_text("utf-8") for p in (ROOT / "src" / "rag_mcp").rglob("*.py"))
    assert "OPENAI" not in src
    assert "API_KEY" not in src


def test_env_example_has_no_values_for_secrets():
    for line in (ROOT / ".env.example").read_text("utf-8").splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if "KEY" in name or "TOKEN" in name:
            assert value.strip() == "", name
