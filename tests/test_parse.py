import pytest
from valagents.parse import (StrictTailError, parse_tail, parse_tail_lines,
                             checked, checked_lines)
from tests.fake_llm import FakeLLM

def test_parse_tail_extracts_keys():
    t = "reasoning...\nCLAIM: x rises with y | REGIME: low T | FALSIFIABLE: yes"
    d = parse_tail(t, ["CLAIM", "REGIME", "FALSIFIABLE"])
    assert d["falsifiable"] == "yes" and d["regime"] == "low T"

def test_parse_tail_missing_key_raises():
    with pytest.raises(StrictTailError):
        parse_tail("CLAIM: x | REGIME: y", ["CLAIM", "FALSIFIABLE"])

def test_parse_tail_lines_one_per_line():
    t = ("CLAIM: c1 | TYPE: mathematical | DEPENDS_ON: none\n"
         "CLAIM: c2 | TYPE: empirical | DEPENDS_ON: c1")
    rows = parse_tail_lines(t, ["CLAIM", "TYPE", "DEPENDS_ON"])
    assert len(rows) == 2 and rows[1]["type"] == "empirical"

async def test_checked_reasks_once_then_succeeds():
    bodies = iter(["no tail here", "CLAIM: x | FALSIFIABLE: yes"])
    llm = FakeLLM(lambda a, m: next(bodies))
    out = await checked("formalizer", [{"role": "user", "content": "q"}],
                        ["CLAIM", "FALSIFIABLE"], llm=llm)
    assert out["falsifiable"] == "yes" and len(llm.calls) == 2

async def test_checked_double_failure_returns_none(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    bodies = iter(["bad body one", "bad body two"])
    llm = FakeLLM(lambda a, m: next(bodies))
    out = await checked("formalizer", [{"role": "user", "content": "q"}],
                        ["CLAIM", "FALSIFIABLE"], llm=llm)
    assert out is None and len(llm.calls) == 2
    assert "bad body one" in caplog.text and "bad body two" in caplog.text  # BOTH bodies logged
