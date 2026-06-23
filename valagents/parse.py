"""Verdict parsing: a label parser plus the strict machine-readable tail contract."""
from __future__ import annotations
import logging
import re
from valagents.llm import LLMClient

log = logging.getLogger(__name__)

def parse_label(text: str, *labels: str) -> str | None:
    best = None
    for label in labels:
        for m in re.finditer(rf"\b{re.escape(label)}\s*:\s*<?\s*([A-Za-z0-9 _\-]+?)\s*>?(?:\s|$|[.,;])",
                             text, re.IGNORECASE):
            best = m.group(1).strip().lower()
    return best

class StrictTailError(Exception):
    pass

def _row(line: str, required_keys: list[str]) -> dict[str, str] | None:
    out: dict[str, str] = {}
    for key in required_keys:
        m = re.search(rf"\b{re.escape(key)}\s*:\s*(.+?)\s*(?=\||$)", line, re.IGNORECASE)
        if not m:
            return None
        out[key.lower()] = m.group(1).strip()
    return out

def parse_tail(text: str, required_keys: list[str]) -> dict[str, str]:
    """Last line that carries all required keys."""
    rows = parse_tail_lines(text, required_keys)
    return rows[-1]

def parse_tail_lines(text: str, required_keys: list[str]) -> list[dict[str, str]]:
    rows = [r for line in text.splitlines() if (r := _row(line, required_keys))]
    if not rows:
        raise StrictTailError(f"no line carried all of {required_keys}")
    return rows

def _reask(required_keys: list[str]) -> str:
    return ("Your previous reply was missing the required machine-readable tail. "
            "Reply with ONLY that one line, exactly: "
            + " | ".join(f"{k}: <value>" for k in required_keys))

async def _attempt(agent, messages, required_keys, llm, multi):
    body = await llm.complete(agent, messages)
    parse = parse_tail_lines if multi else parse_tail
    try:
        return parse(body, required_keys), body
    except StrictTailError:
        reask = list(messages) + [{"role": "assistant", "content": body},
                                  {"role": "user", "content": _reask(required_keys)}]
        body2 = await llm.complete(agent, reask)
        try:
            return parse(body2, required_keys), (body, body2)
        except StrictTailError:
            log.warning("strict-tail double failure agent=%s\n--body1--\n%s\n--body2--\n%s",
                        agent, body, body2)
            return None, (body, body2)

async def checked(agent, messages, required_keys, *, llm: LLMClient) -> dict | None:
    out, _ = await _attempt(agent, messages, required_keys, llm, multi=False)
    return out

async def checked_lines(agent, messages, required_keys, *, llm: LLMClient) -> list[dict] | None:
    out, _ = await _attempt(agent, messages, required_keys, llm, multi=True)
    return out
