"""Mutation check: re-introduces known, security-relevant bugs one at a time and requires the
test suite to fail for each. A surviving mutant means a control is not actually tested.

Run from the repo root: python scripts/mutation_check.py
Works on a temporary copy; the working tree is never modified.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (file, original snippet, mutated snippet, what the mutant breaks)
MUTANTS = [
    (
        "src/rag_mcp/server.py",
        "raise ToolError(exc.public_message)",
        "raise ToolError(exc.debug_detail)",
        "backend error text forwarded to the model",
    ),
    (
        "src/rag_mcp/audit.py",
        '"args_hmac": record.args_hmac,',
        '"args_hmac": str(args),',
        "request text written to the audit log",
    ),
    (
        "src/rag_mcp/server.py",
        "max_length=BACKEND_MAX_TEXT_CHARS",
        "max_length=100_000",
        "input length limit removed",
    ),
    (
        "src/rag_mcp/backend.py",
        "self._settings.ask_timeout_s, retry=False",
        "self._settings.ask_timeout_s, retry=True",
        "paid /query call retried",
    ),
    (
        "src/rag_mcp/formatting.py",
        'return text.replace("<<<", "< < <").replace(">>>", "> > >")',
        "return text",
        "untrusted-content delimiters can be forged",
    ),
    (
        "src/rag_mcp/formatting.py",
        "        if key in seen:\n",
        "        if False:\n",
        "duplicate passages not removed",
    ),
    (
        "src/rag_mcp/config.py",
        "if not host or host not in {h.lower() for h in allowed_hosts}:",
        "if not host:",
        "backend host allow-list disabled",
    ),
    (
        "src/rag_mcp/backend.py",
        "follow_redirects=False,",
        "follow_redirects=True,",
        "redirects to other hosts followed",
    ),
    (
        "src/rag_mcp/server.py",
        '    _charge(ask_budget, "ask_papers")\n',
        "\n",
        "ask_papers budget not enforced",
    ),
    (
        "src/rag_mcp/server.py",
        "    stripped = text.strip()\n    if not stripped:",
        "    stripped = text\n    if False:",
        "whitespace-only input accepted",
    ),
    (
        "src/rag_mcp/http_transport.py",
        "if not hmac.compare_digest(provided, self._expected):",
        'if not provided.startswith(b"Bearer "):',
        "any bearer token accepted",
    ),
    (
        "src/rag_mcp/http_transport.py",
        "enable_dns_rebinding_protection=True",
        "enable_dns_rebinding_protection=False",
        "DNS-rebinding protection disabled",
    ),
    (
        "src/rag_mcp/http_transport.py",
        "if len(self.token) < MIN_TOKEN_CHARS:",
        "if False:",
        "weak HTTP token accepted",
    ),
    (
        "src/rag_agent/runner.py",
        'if k in _SYSTEM_VARS or k.startswith("RAG_MCP_")}',
        "}",
        "OpenAI key passed to the MCP server process",
    ),
    (
        "src/rag_agent/graph.py",
        'if state.get("tool_calls_made", 0) >= max_tool_calls:',
        "if False:",
        "agent tool-call budget removed",
    ),
    (
        "src/rag_agent/cli.py",
        'if split == "test" and out.exists():',
        "if False:",
        "held-out result can be overwritten",
    ),
    (
        "src/rag_agent/cli.py",
        'if split_checksum(items, "test") != frozen:',
        "if False:",
        "modified test split accepted",
    ),
    (
        "src/rag_agent/evaluation.py",
        "return (retrieved and _cites(run.answer, run.documents)),",
        "return retrieved,",
        "answers without citations scored as correct",
    ),
]


def run_suite(cwd: Path) -> int:
    env_path = str(cwd / "src")
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--no-header"]
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, env={**_base_env(), "PYTHONPATH": env_path}, check=False
    ).returncode


def _base_env() -> dict[str, str]:
    import os

    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def main() -> int:
    survivors = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(
            ROOT, work, ignore=shutil.ignore_patterns(".git", ".venv", "venv", "_to_delete", "__pycache__")
        )
        if run_suite(work) != 0:
            print("Baseline suite fails; fix it before running mutation checks.")
            return 2
        for rel, original, mutated, meaning in MUTANTS:
            path = work / rel
            source = path.read_text("utf-8")
            if source.count(original) < 1:
                print(f"STALE  {rel}: snippet for '{meaning}' not found")
                survivors.append(meaning)
                continue
            path.write_text(source.replace(original, mutated), "utf-8")  # every occurrence
            killed = run_suite(work) != 0
            path.write_text(source, "utf-8")
            print(f"{'KILLED ' if killed else 'SURVIVED'} {meaning}")
            if not killed:
                survivors.append(meaning)
    print(f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
