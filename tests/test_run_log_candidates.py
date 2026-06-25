"""Candidates audit trail: run_log.emit_candidates + the grounder's per-article dispositions.

The grounder surveys a retrieved pool per claim and credits only some of it. Before this trail the
pool lived only in the runtime prompt and was unrecoverable — you could see what was CITED (in the
artifact sources) but not what was SURVEYED-and-rejected or what CONTRADICTED. These tests pin the
audit sink (.candidates/<run_id>.jsonl) and that each article gets the right disposition.
"""
import json
from pathlib import Path

import pytest

from valagents import run_log
from valagents.agents.grounder import ground_claim
from valagents.web_search import Article
from tests.test_grounding_support_agent import (
    CLAIM, FC, A_SUPPORT, A_SYNTH, A_CONTRA, _Backend, _llm, _cfg,
)


@pytest.fixture(autouse=True)
def _reset_run_logger():
    # contextvars set without a token leak across tests in the same thread/context — isolate explicitly.
    token = run_log._current.set(None)
    yield
    run_log._current.reset(token)


def test_candidate_path_derived_from_logs_bind(tmp_path):
    # bind to <base>/.logs/<id>.jsonl → candidate sink is the sibling <base>/.candidates/<id>.jsonl
    logger = run_log.bind(tmp_path / ".logs" / "run-1.jsonl")
    assert Path(logger.candidate_path) == tmp_path / ".candidates" / "run-1.jsonl"


def test_emit_candidates_writes_jsonl_record(tmp_path):
    run_log.bind(tmp_path / ".logs" / "run-1.jsonl")
    run_log.emit_candidates(
        "c1", tick=2, n_retrieved=2, n_credited=1, contradicted=False,
        candidates=[{"label": "A1", "title": "T", "url": "u", "published": "2025", "disposition": "credited"}],
    )
    sink = tmp_path / ".candidates" / "run-1.jsonl"
    rec = json.loads(sink.read_text().splitlines()[0])
    assert rec["claim"] == "c1" and rec["tick"] == 2
    assert rec["n_retrieved"] == 2 and rec["n_credited"] == 1 and rec["contradicted"] is False
    assert rec["candidates"][0]["disposition"] == "credited"
    assert "time" in rec  # stamped by the logger


def test_emit_candidates_noop_when_unbound():
    # fail-soft: audit capture must never break a run, even with no logger bound
    run_log.emit_candidates("c1", tick=0, n_retrieved=0, n_credited=0, contradicted=False, candidates=[])


async def test_grounder_emits_per_article_dispositions(tmp_path):
    # A1 credited (on-property supports), A2 quote_failed (off-property supports), A3 contradicts,
    # A4 uncited (in the pool but in no citation). The credited supports + a valid contradicts also
    # exercises the pass→uncertain downgrade, so n_credited reflects the capped count.
    run_log.bind(tmp_path / ".logs" / "run-x.jsonl")
    a_uncited = Article(title="z", summary="an unrelated frustrated magnet heat capacity study",
                        url="http://arxiv.org/abs/2501.09999v1", published="2025-09-09")
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: mixed"
    payload = {"citations": [
        {"label": "A1", "direction": "supports",
         "quote": "The measured noise PSD is temperature-independent below 1 K."},
        {"label": "A2", "direction": "supports",
         "quote": "Single crystals of YbZn2GaO5 were grown by the floating-zone method in this study."},
        {"label": "A3", "direction": "contradicts",
         "quote": "In YbZn2GaO5 the noise PSD shows a strong temperature dependence below 1 K."}]}
    await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA, a_uncited]),
                       _llm(tail, payload), _cfg())

    rec = json.loads((tmp_path / ".candidates" / "run-x.jsonl").read_text().splitlines()[0])
    disp = {c["label"]: c["disposition"] for c in rec["candidates"]}
    assert disp == {"A1": "credited", "A2": "quote_failed", "A3": "contradicts", "A4": "uncited"}
    assert rec["claim"] == "c1" and rec["n_retrieved"] == 4
    assert rec["n_credited"] == 1 and rec["contradicted"] is True
    # the audit row carries enough to show the human the survey: every article labelled with title+url
    a4 = next(c for c in rec["candidates"] if c["label"] == "A4")
    assert a4["title"] == "z" and a4["url"] == "http://arxiv.org/abs/2501.09999v1"


async def test_grounder_marks_inadmissible_contradiction_unverified(tmp_path):
    # a CONTRADICTS direction whose quote is not in the abstract is not counted as a contradiction
    # (contradicted stays False) but is still logged — as contradicts_unverified, not dropped.
    run_log.bind(tmp_path / ".logs" / "run-y.jsonl")
    tail = "CLAIM: c1 | SUPPORT: uncertain | INDEPENDENT_SOURCES: 0 | BASIS: claimed"
    payload = {"citations": [
        {"label": "A3", "direction": "contradicts",
         "quote": "A fabricated sentence that appears nowhere in the A3 abstract at all."}]}
    await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())

    rec = json.loads((tmp_path / ".candidates" / "run-y.jsonl").read_text().splitlines()[0])
    disp = {c["label"]: c["disposition"] for c in rec["candidates"]}
    assert disp["A3"] == "contradicts_unverified"
    assert rec["contradicted"] is False
    assert disp["A1"] == "uncited" and disp["A2"] == "uncited"
