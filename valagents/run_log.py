"""Lightweight per-run event log.

A run binds a :class:`RunLogger` for the duration of its execution; any module can
then call :func:`emit` to append a structured event to ``<results>/<run_id>/logs.jsonl``
without threading a logger through agent signatures. The binding lives in a
``contextvars.ContextVar``, so concurrent runs (each its own asyncio task) get
isolated loggers and ``emit`` is a no-op when nothing is bound (e.g. in unit tests).

Events are summaries, not full prompts/responses. Raw agent responses go to a sibling
``agent_outputs.jsonl`` and the grounder's surveyed pool to ``candidates.jsonl`` — all
three in the run's own folder ``<results>/<run_id>/``.
"""
from __future__ import annotations

import contextvars
import json
from datetime import datetime
from pathlib import Path

_current: contextvars.ContextVar = contextvars.ContextVar("valagents_run_logger", default=None)


def events_path(results_base: str, run_id: str) -> Path:
    return Path(results_base) / run_id / "logs.jsonl"


def agent_outputs_path(results_base: str, run_id: str) -> Path:
    return Path(results_base) / run_id / "agent_outputs.jsonl"


def candidates_path(results_base: str, run_id: str) -> Path:
    """Per-run grounder candidate-pool audit log: the FULL set of articles retrieved per claim and each
    one's disposition (credited / quote_failed / contradicts / uncited) — the auditable record of what
    the grounder surveyed vs cited, including rejected and contradicting papers."""
    return Path(results_base) / run_id / "candidates.jsonl"


class RunLogger:
    def __init__(self, path) -> None:
        # The bind path is the run's log file (<results>/<run_id>/logs.jsonl); the agent-output and
        # candidate sinks are fixed-name siblings in that same run folder.
        self.path = str(path)
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.agent_output_path = str(p.parent / "agent_outputs.jsonl")
        self.candidate_path = str(p.parent / "candidates.jsonl")

    def emit(self, event: str, **fields) -> None:
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def emit_agent_output(self, agent: str, **fields) -> None:
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "agent": agent, **fields}
        with open(self.agent_output_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def emit_candidates(self, claim_id: str, **fields) -> None:
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "claim": claim_id, **fields}
        with open(self.candidate_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def bind(path) -> RunLogger:
    """Create a logger at ``path`` and bind it for the current run/task context."""
    logger = RunLogger(path)
    _current.set(logger)
    return logger


def emit(event: str, **fields) -> None:
    """Append an event to the bound run log, if any. Never raises into the run."""
    logger = _current.get()
    if logger is None:
        return
    try:
        logger.emit(event, **fields)
    except Exception:  # a logging failure must never break a run
        pass


def emit_agent_output(agent: str, **fields) -> None:
    """Append a raw agent response record, if a run logger is bound."""
    logger = _current.get()
    if logger is None:
        return
    try:
        logger.emit_agent_output(agent, **fields)
    except Exception:  # output capture must never break a run
        pass


def emit_candidates(claim_id: str, **fields) -> None:
    """Append a grounder candidate-pool record (full retrieved set + dispositions), if a logger is bound."""
    logger = _current.get()
    if logger is None:
        return
    try:
        logger.emit_candidates(claim_id, **fields)
    except Exception:  # audit capture must never break a run
        pass


def read_events(path, since: int = 0) -> list[dict]:
    """Return events from ``path`` starting at index ``since`` (for incremental polling)."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out[since:]
