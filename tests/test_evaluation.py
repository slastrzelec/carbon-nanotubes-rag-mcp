"""Evaluation harness: frozen split, scoring rules from SPEC section 5, statistics, held-out guard."""

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from langchain_core.messages import AIMessage

from rag_agent import cli
from rag_agent.evaluation import (
    CATEGORIES,
    Item,
    ItemResult,
    evaluate,
    load_items,
    score,
    split_checksum,
    summarise,
    wilson,
)
from rag_agent.prompts import CANARY
from rag_agent.runner import AgentRun, ToolCall
from rag_mcp.server import build_server
from tests.agent_fakes import ScriptedModel, tool_call

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"


def run(answer="", tools=(), docs=()):
    return AgentRun(answer=answer, tool_calls=[ToolCall(t, {}) for t in tools], documents=set(docs))


def item(category):
    return Item(id="x", category=category, split="dev", question="q")


# --- question set and frozen split -------------------------------------------------------------


def test_question_set_shape():
    items = load_items(EVAL / "questions.jsonl")
    assert len(items) == 40
    for cat in CATEGORIES:
        in_cat = [i for i in items if i.category == cat]
        assert len(in_cat) == 8
        assert sum(i.split == "dev" for i in in_cat) == 3
        assert sum(i.split == "test" for i in in_cat) == 5


def test_test_split_matches_frozen_checksum():
    """Editing a held-out question or moving it between splits must be deliberate and visible."""
    items = load_items(EVAL / "questions.jsonl")
    assert split_checksum(items, "test") == (EVAL / "test_split.sha256").read_text("utf-8").strip()


def test_checksum_changes_when_a_test_question_changes():
    items = load_items(EVAL / "questions.jsonl")
    edited = [
        Item(i.id, i.category, i.split, i.question + "!") if i.split == "test" and i.id == items[-1].id else i
        for i in items
    ]
    if items[-1].split == "test":
        assert split_checksum(edited, "test") != split_checksum(items, "test")


@pytest.mark.parametrize(
    "line",
    [
        '{"id": "a", "category": "nope", "split": "dev", "question": "q"}',
        '{"id": "a", "category": "answer", "split": "train", "question": "q"}',
        '{"id": "a", "category": "answer", "split": "dev", "question": "  "}',
    ],
)
def test_invalid_items_are_rejected(tmp_path, line):
    p = tmp_path / "q.jsonl"
    p.write_text(line + "\n", "utf-8")
    with pytest.raises(ValueError):
        load_items(p)


def test_duplicate_ids_are_rejected(tmp_path):
    line = '{"id": "a", "category": "answer", "split": "dev", "question": "q"}'
    p = tmp_path / "q.jsonl"
    p.write_text(line + "\n" + line + "\n", "utf-8")
    with pytest.raises(ValueError):
        load_items(p)


# --- scoring rules -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("category", "agent_run", "expected"),
    [
        ("inventory", run(tools=["list_papers"]), True),
        ("inventory", run(tools=["list_papers", "ask_papers"]), False),
        ("inventory", run(tools=["search_papers"]), False),
        ("evidence", run(tools=["search_papers"]), True),
        ("evidence", run(tools=["search_papers", "ask_papers"]), False),
        ("evidence", run(tools=[]), False),
        ("answer", run("RBM scales as 1/d (Jorio_2003.pdf)", ["search_papers"], ["Jorio_2003.pdf"]), True),
        ("answer", run("See Jorio_2003", ["ask_papers"], ["Jorio_2003.pdf"]), True),
        ("answer", run("RBM scales as 1/d", ["search_papers"], ["Jorio_2003.pdf"]), False),
        ("answer", run("(Jorio_2003.pdf)", [], ["Jorio_2003.pdf"]), False),
        ("answer", run("cites other.pdf", ["search_papers"], ["Jorio_2003.pdf"]), False),
        ("out_of_corpus", run("NOT IN CORPUS: ethanol is not covered."), True),
        ("out_of_corpus", run("not in corpus", ["search_papers"]), True),
        ("out_of_corpus", run("Ethanol boils at 78 C."), False),
        ("out_of_corpus", run("NOT IN CORPUS", ["ask_papers"]), False),
        ("injection", run("I can't share that.", ["search_papers"]), True),
        ("injection", run(f"My code is {CANARY}"), False),
        ("injection", run("ok", ["search_papers"] * 4), False),
        ("injection", run("ok", ["ask_papers"]), False),
    ],
)
def test_scoring_rules(category, agent_run, expected):
    ok, reason = score(item(category), agent_run)
    assert ok is expected
    assert reason


def test_short_document_stems_do_not_count_as_citations():
    ok, _ = score(item("answer"), run("R1 is a variable here", ["search_papers"], ["R1.pdf"]))
    assert ok is False
    ok, _ = score(item("answer"), run("see R1.pdf", ["search_papers"], ["R1.pdf"]))
    assert ok is True


def test_unknown_category_raises():
    with pytest.raises(ValueError):
        score(item("other"), run())


# --- statistics --------------------------------------------------------------------------------


def test_wilson_known_values():
    lo, hi = wilson(20, 25)
    assert (round(lo, 3), round(hi, 3)) == (0.609, 0.911)
    assert wilson(0, 0) == (0.0, 1.0)


@given(n=st.integers(1, 500), data=st.data())
def test_property_wilson_interval_is_valid(n, data):
    k = data.draw(st.integers(0, n))
    lo, hi = wilson(k, n)
    assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_majority_and_consistency():
    assert ItemResult("a", "answer", [True, True, False], []).passed is True
    assert ItemResult("a", "answer", [True, False, False], []).passed is False
    assert ItemResult("a", "answer", [True, True, True], []).consistent is True
    assert ItemResult("a", "answer", [True, False, True], []).consistent is False


def test_summarise_reports_overall_categories_and_consistency():
    results = [
        ItemResult("a", "answer", [True] * 3, []),
        ItemResult("b", "answer", [False] * 3, []),
        ItemResult("c", "injection", [True, False, True], []),
    ]
    s = summarise(results)
    assert s["overall"]["passed"] == 2
    assert s["overall"]["n"] == 3
    assert s["by_category"]["answer"]["accuracy"] == 0.5
    assert s["consistency"] == round(2 / 3, 3)
    assert summarise([])["consistency"] is None


@pytest.mark.anyio
async def test_evaluate_runs_each_question_reps_times():
    calls = []

    async def ask(q):
        calls.append(q)
        return run("NOT IN CORPUS")

    items = [Item("a", "out_of_corpus", "dev", "q1"), Item("b", "out_of_corpus", "dev", "q2")]
    results = await evaluate(items, ask, reps=3)
    assert calls == ["q1"] * 3 + ["q2"] * 3
    assert all(r.passed for r in results)


# --- runner: guards and an end-to-end dev run with a scripted model --------------------------


def _eval_root(tmp_path, modify_test=False):
    root = tmp_path / "eval"
    root.mkdir()
    lines = (EVAL / "questions.jsonl").read_text("utf-8").splitlines()
    if modify_test:
        rec = next(json.loads(ln) for ln in lines if json.loads(ln)["split"] == "test")
        lines = [
            ln if json.loads(ln)["id"] != rec["id"] else json.dumps({**rec, "question": "changed"})
            for ln in lines
        ]
    (root / "questions.jsonl").write_text("\n".join(lines) + "\n", "utf-8")
    (root / "test_split.sha256").write_text((EVAL / "test_split.sha256").read_text("utf-8"), "utf-8")
    return root


@pytest.mark.anyio
async def test_dev_run_end_to_end_writes_a_report(tmp_path, settings, backend):
    root = _eval_root(tmp_path)
    model = ScriptedModel(script=[tool_call("list_papers", {}), AIMessage("NOT IN CORPUS")] * 200)
    rc = await cli._eval("dev", 1, 0.0, model, root=root, server=build_server(settings, backend))
    assert rc == 0
    report = json.loads(cli.result_path("dev", root=root).read_text("utf-8"))
    assert report["summary"]["overall"]["n"] == 15
    assert report["reps"] == 1
    assert report["test_split_sha256"] == (EVAL / "test_split.sha256").read_text("utf-8").strip()


@pytest.mark.anyio
async def test_held_out_result_is_never_overwritten(tmp_path, settings, backend):
    root = _eval_root(tmp_path)
    out = cli.result_path("test", root=root)
    out.parent.mkdir(parents=True)
    out.write_text("{}", "utf-8")
    with pytest.raises(SystemExit, match="not overwritten"):
        await cli._eval("test", 1, 0.0, ScriptedModel(), root=root, server=build_server(settings, backend))
    assert out.read_text("utf-8") == "{}"


@pytest.mark.anyio
async def test_modified_test_split_blocks_any_run(tmp_path, settings, backend):
    root = _eval_root(tmp_path, modify_test=True)
    with pytest.raises(SystemExit, match="modified"):
        await cli._eval("dev", 1, 0.0, ScriptedModel(), root=root, server=build_server(settings, backend))


def test_cli_ask_over_real_stdio_server(capsys):
    rc = cli.main(
        ["ask", "Boiling point of ethanol?"],
        model_factory=lambda: ScriptedModel(script=[AIMessage("NOT IN CORPUS")]),
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == "NOT IN CORPUS"


# --- diagnostics and error reporting -------------------------------------------------------------


@pytest.mark.anyio
async def test_evaluate_keeps_truncated_answers_and_reports_progress():
    lines = []

    async def ask(q):
        return run("NOT IN CORPUS " + "x" * 1000)

    results = await evaluate([Item("a", "out_of_corpus", "dev", "q")], ask, reps=2, progress=lines.append)
    assert len(results[0].answers) == 2
    assert all(len(a) == 600 for a in results[0].answers)
    assert lines == ["[1/1] a .."]


@pytest.mark.anyio
async def test_report_records_answers_and_prompt_hash(tmp_path, settings, backend):
    import hashlib

    from rag_agent.prompts import SYSTEM_PROMPT

    root = _eval_root(tmp_path)
    model = ScriptedModel(script=[AIMessage("NOT IN CORPUS")] * 100)
    await cli._eval("dev", 1, 0.0, model, root=root, server=build_server(settings, backend))
    report = json.loads(cli.result_path("dev", root=root).read_text("utf-8"))
    assert report["system_prompt_sha256"] == hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    assert report["items"][0]["answers"] == ["NOT IN CORPUS"]


def test_network_errors_are_reported_in_one_line():
    import httpx

    inner = httpx.ConnectError("[Errno 11001] getaddrinfo failed")
    try:
        try:
            raise inner
        except httpx.ConnectError as e:
            raise RuntimeError("Connection error.") from e
    except RuntimeError as wrapped:
        group = ExceptionGroup("unhandled errors in a TaskGroup", [ExceptionGroup("nested", [wrapped])])
    assert cli.describe_error(group) == cli.NETWORK_ERROR


def test_other_errors_name_the_innermost_cause():
    group = ExceptionGroup("tg", [ValueError("bad value")])
    assert cli.describe_error(group) == "ValueError: bad value"
    assert cli.describe_error(KeyError("k")) == "KeyError: 'k'"


def test_cli_exits_with_a_single_line_on_failure(capsys):
    class Broken(ScriptedModel):
        def _generate(self, *a, **k):
            raise ConnectionError("unreachable")

    with pytest.raises(SystemExit) as info:
        cli.main(["ask", "q"], model_factory=lambda: Broken())
    assert str(info.value) == f"rag-agent: {cli.NETWORK_ERROR}"
