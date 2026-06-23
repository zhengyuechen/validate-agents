"""Lightweight per-run event log.

A run binds a :class:`RunLogger` for the duration of its execution; any module can
then call :func:`emit` to append a structured event to ``<results>/.logs/<run_id>.jsonl``
without threading a logger through agent signatures. The binding lives in a
``contextvars.ContextVar``, so concurrent runs (each its own asyncio task) get
isolated loggers and ``emit`` is a no-op when nothing is bound (e.g. in unit tests).

Events are summaries, not full prompts/responses — those stay in the final
artifacts. The intent is a live "what is happening now" timeline: which task is
running, whether arXiv grounding succeeded, and how the tournament is progressing.
"""
from __future__ import annotations

import contextvars
import json
from datetime import datetime
from pathlib import Path

_current: contextvars.ContextVar = contextvars.ContextVar("cosci_run_logger", default=None)


def events_path(results_base: str, run_id: str) -> Path:
    return Path(results_base) / ".logs" / f"{run_id}.jsonl"


class RunLogger:
    def __init__(self, path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields) -> None:
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
        with open(self.path, "a") as f:
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
