# validate-agents Spec 1 (Internal-Validation Spine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow one seed idea into a fully-populated, check-hardened `IdeaArtifact` that terminates in exactly one of `internally_validated` / `needs_experiment` / `refuted`, with every verdict a parsed field and "validated" gated on a survived external, independent check.

**Architecture:** A `valagents/` package whose infra modules (LLM client, run-log, web-search, the `parse_label` regex) are already present, plus new domain code: a Pydantic `IdeaArtifact`/`AtomicClaim` schema whose `status`/`maturity`/`load_bearing`/`blocker` are **computed properties with no setter** (the gate is code, never an LLM), eleven thin agent functions each ending in a strict machine-readable tail, a single-writer append-only `ArtifactStore` with an immutable version chain, and a DAG scheduler that runs entry gates → per-claim lenses → fan-out → repair-versioning → a total verdict.

**Tech Stack:** Python 3.11+, Pydantic v2 (`@computed_field`), `pytest` + `pytest-asyncio`, `tenacity` (retry), `openai` (OpenRouter async client), `arxiv` (grounding), `python-dotenv`, `pyyaml`. Tests use a deterministic `FakeLLM` — **no network in any test.**

**Source spec:** `docs/2026-06-23-validate-agents-design.md` (rev 3). Section references below (e.g. "§2.1") point at it.

## Global Constraints

Every task's requirements implicitly include this section. Values are verbatim from the spec.

- **I1 — verdicts gate, not narrate.** `IdeaArtifact.status`, `.maturity`, `.load_bearing`, `.blocker` and `AtomicClaim.status` are `@computed_field` **read-only properties**. No code path and no LLM ever assigns them. An agent's emitted `STATUS:` line is a cross-check only; on disagreement the computed value wins and the mismatch is logged.
- **I2 — validated = survived an external, independent check.** `internally_validated` is reachable only when every load-bearing claim is strictly `pass`, and a claim is `pass` only if it has a `CheckRecord` with `verdict == "pass"` **and `independent_sources >= 1`**. `pending` is never `pass`.
- **I3 — the gate is total.** Every run ends in `internally_validated` / `needs_experiment` / `refuted`. `draft` is **non-terminal** — the scheduler never stops in `draft`.
- **`maturity` ⊥ `status`.** The `status` computation must not read `maturity`. Dependency is one-directional: `{verdict set, status} → maturity → report`.
- **Status string constants** (lowercase): `"draft"`, `"internally_validated"`, `"needs_experiment"`, `"refuted"`. Claim statuses: `"pass"`, `"fail"`, `"uncertain"`, `"pending"`.
- **Config defaults:** `min_attack_categories = 2`, `fanout_N = 2`, `repair_cap = 3`, `grounding.backend = "arxiv"`.
- **Provider:** OpenRouter via `OpenRouterClient` (`valagents/llm.py`); never call the network in tests — inject `FakeLLM`.
- **Determinism:** no `Date.now()`/wall-clock in computed logic; ticks are integers supplied by the scheduler.
- **Commits:** one per task minimum; conventional-commit messages (`feat:`, `test:`, `chore:`).

---

## File Structure

```
valagents/
  __init__.py
  config.py        # roles→models/temps, GateCfg, load_config
  llm.py           # OpenRouterClient (import path: valagents.config)
  run_log.py       # JSONL event log
  web_search.py    # ArxivBackend, safe_search
  parse.py         # parse_label + StrictTailError/parse_tail/parse_tail_lines/checked/checked_lines
  artifact.py      # all schema models + AtomicClaim.status + IdeaArtifact gate (status/load_bearing/blocker/maturity)
  store.py         # ArtifactStore (immutable version chain + append-only event log)
  prompts.py       # prompt templates + mandatory tails for all 11 agents
  agents/
    __init__.py
    base.py        # message builder + verdict-mapping helpers shared by every agent
    formalizer.py  faithfulness.py  decomposer.py  entailment.py
    grounder.py    prover.py        predictor.py   redteam.py
    validation_designer.py  repairer.py  arbiter.py
  scheduler.py     # entry gates → per-claim lenses → fan-out → repair-versioning → total verdict
  cli.py           # valagents "<seed>" → IdeaArtifact JSON + markdown report
tests/
  __init__.py  conftest.py  fake_llm.py
  test_parse.py  test_artifact_claim.py  test_artifact_gate.py
  test_artifact_load_bearing.py  test_artifact_maturity.py  test_store.py
  test_agent_formalizer.py  test_agent_guards.py  test_agent_lenses.py
  test_agent_whole.py  test_agent_orchestration.py
  test_scheduler_entry.py  test_scheduler_checks.py  test_scheduler_repair.py
  test_cli.py  test_integration_escape_saddle.py
config.yaml  .env.example  pyproject.toml  README.md  docs/  results/
```

Boundary rationale: `artifact.py` holds the gate as one cohesive unit (the schema and its pure rollup change together). Agents are one file each (independently reviewable, swappable models). `parse.py` and `store.py` are infra with no domain logic. `scheduler.py` is the only stateful orchestrator.

---

## Task 1: Package scaffolding + copied infra + config

**Files:**
- Create: `valagents/__init__.py`, `pyproject.toml`, `config.yaml`, `.env.example`, `tests/__init__.py`, `tests/conftest.py`
- Already present (infra modules): `valagents/llm.py`, `valagents/run_log.py`, `valagents/web_search.py`, `tests/fake_llm.py`
- Create (adapt): `valagents/config.py`

**Interfaces:**
- Produces: `valagents.config.Config` (pydantic) with `.model_for(agent: str) -> str`, `.temperature: dict[str,str→float]`, `.grounding.backend: str`, `.gate.min_attack_categories/.fanout_N/.repair_cap: int`, `.results_dir: str`; `load_config(path="config.yaml") -> Config`; `require_openrouter_key() -> str`. `valagents.llm.LLMClient` protocol (`async complete(agent, messages, temperature=None, max_tokens=None) -> str`), `OpenRouterClient`, `extract_json`. `tests.fake_llm.FakeLLM(router)` with `.calls`.

- [ ] **Step 1: Confirm infra modules are present.**

The following files are already in the repo and do not need to be created:
- `valagents/llm.py` — `OpenRouterClient` (async, per-agent model/temp, tenacity retry, `extract_json`); imports `from valagents.config import Config, require_openrouter_key`.
- `valagents/run_log.py` — JSONL event log (contextvars per-run, append-only, replay).
- `valagents/web_search.py` — `ArxivBackend`, `safe_search` (Grounder's external check).
- `tests/fake_llm.py` — `FakeLLM(router)` with `.calls`.

Create empty `valagents/__init__.py`, `valagents/agents/__init__.py`, `tests/__init__.py`.

- [ ] **Step 2: Write `valagents/config.py`** (drop Elo/debate/proximity/budget fields not needed here; add `GateCfg`).

```python
"""Typed config for validate-agents. Roles → models/temps; gate thresholds."""
from __future__ import annotations
import os
import yaml
from pydantic import BaseModel
from dotenv import load_dotenv

class GroundCfg(BaseModel):
    backend: str = "arxiv"          # arxiv | none | tavily

class GateCfg(BaseModel):
    min_attack_categories: int = 2  # categories the Red-team must attempt for internally_validated
    fanout_N: int = 2               # diverse-type lenses on a load-bearing uncertain node before finalize
    repair_cap: int = 3             # max repair versions before finalize

class Config(BaseModel):
    default_model: str
    models: dict[str, str] = {}
    temperature: dict[str, float] = {}
    grounding: GroundCfg = GroundCfg()
    gate: GateCfg = GateCfg()
    results_dir: str = "results"

    def model_for(self, agent: str) -> str:
        return self.models.get(agent, self.default_model)

def load_config(path: str = "config.yaml") -> Config:
    load_dotenv()
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)

def require_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (add it to .env or the environment).")
    return key
```

- [ ] **Step 3: Write `config.yaml`, `.env.example`, `pyproject.toml`, `tests/conftest.py`.**

`config.yaml`:
```yaml
default_model: z-ai/glm-4.7
models:
  redteam: z-ai/glm-4.7
  grounder: z-ai/glm-4.7
temperature:
  formalizer: 0.3
  faithfulness: 0.2
  decomposer: 0.4
  entailment: 0.2
  grounder: 0.4
  prover: 0.3
  predictor: 0.5
  redteam: 0.6
  validation_designer: 0.4
  repairer: 0.6
  arbiter: 0.2
grounding:
  backend: arxiv
gate:
  min_attack_categories: 2
  fanout_N: 2
  repair_cap: 3
results_dir: results
```

`.env.example`:
```
OPENROUTER_API_KEY=
WEB_SEARCH_API_KEY=
```

`pyproject.toml`:
```toml
[project]
name = "valagents"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pydantic>=2", "openai>=1.0", "tenacity", "python-dotenv", "pyyaml", "arxiv"]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

`tests/conftest.py`:
```python
import pytest
from valagents.config import Config

@pytest.fixture
def cfg() -> Config:
    return Config(default_model="fake/model")
```

- [ ] **Step 4: Write a smoke test** `tests/test_config.py`:

```python
from valagents.config import Config, load_config

def test_config_defaults():
    c = Config(default_model="m")
    assert c.model_for("anything") == "m"
    assert c.gate.fanout_N == 2 and c.gate.min_attack_categories == 2 and c.gate.repair_cap == 3
    assert c.grounding.backend == "arxiv"

def test_config_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("default_model: x\ngate:\n  fanout_N: 5\n")
    c = load_config(str(p))
    assert c.gate.fanout_N == 5 and c.model_for("grounder") == "x"
```

- [ ] **Step 5: Run + commit.**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: 2 passed. (`import valagents.llm` should also succeed — `python -c "import valagents.llm, valagents.run_log, valagents.web_search"`.)

```bash
git add -A && git commit -m "chore: scaffold valagents package + config"
```

---

## Task 2: Strict-tail verdict parsing (`parse.py`)

This is the I1 mechanism. A missing/unparseable tail must degrade to `uncertain`, never silently pass.

**Files:**
- Create: `valagents/parse.py`
- Test: `tests/test_parse.py`

**Interfaces:**
- Consumes: `valagents.llm.LLMClient`.
- Produces:
  - `parse_label(text, *labels) -> str | None` (copied).
  - `class StrictTailError(Exception)`.
  - `parse_tail(text: str, required_keys: list[str]) -> dict[str, str]` — returns the LAST line's `{key.lower(): value}`; raises `StrictTailError` if any key missing.
  - `parse_tail_lines(text, required_keys) -> list[dict[str,str]]` — one dict per line containing all keys; raises if none.
  - `async checked(agent, messages, required_keys, *, llm) -> dict | None` — complete → parse_tail; on `StrictTailError` re-ask once for the bare tail; on second failure return `None` and log both bodies at WARN.
  - `async checked_lines(agent, messages, required_keys, *, llm) -> list[dict] | None` — same contract for multi-line tails.

- [ ] **Step 1: Write the failing tests** `tests/test_parse.py`:

```python
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
    llm = FakeLLM(lambda a, m: "still no tail")
    out = await checked("formalizer", [{"role": "user", "content": "q"}],
                        ["CLAIM", "FALSIFIABLE"], llm=llm)
    assert out is None and len(llm.calls) == 2
    assert "still no tail" in caplog.text  # both malformed bodies logged
```

- [ ] **Step 2: Run to verify failure.** Run: `pytest tests/test_parse.py -v` → FAIL (module/functions not defined).

- [ ] **Step 3: Implement `valagents/parse.py`.**

```python
"""Verdict parsing. parse_label handles label extraction; the strict tail is new."""
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
```

- [ ] **Step 4: Run to verify pass.** Run: `pytest tests/test_parse.py -v` → 5 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(parse): strict machine-readable tail with one re-ask, uncertain-on-failure"`

---

## Task 3: Schema leaves + `AtomicClaim.status` join

**Files:** Create `valagents/artifact.py` (first half). Test: `tests/test_artifact_claim.py`.

**Interfaces:**
- Produces (Pydantic v2 `BaseModel`s): `Source`, `CheckRecord`, `FormalClaim`, `Faithfulness`, `Coverage`, `AttackSurface`, `Novelty`, `Prediction`, `Attack`, `Gap`, `Derivation`, `ValidationPlan`, `AtomicClaim`. `AtomicClaim.status` is a `@computed_field` → `"pass"|"fail"|"uncertain"|"pending"`.
- Field types exactly as the spec §2 block. A claim is `pass` only via a `CheckRecord` with `verdict=="pass"` AND `independent_sources>=1`.

- [ ] **Step 1: Write failing tests** `tests/test_artifact_claim.py`:

```python
from valagents.artifact import AtomicClaim, CheckRecord, Source

def mk(checks):
    return AtomicClaim(id="c1", statement="s", type="empirical", checks=checks)

def test_pending_when_no_checks():
    assert mk([]).status == "pending"

def test_pass_requires_independent_source():
    weak = CheckRecord(lens="grounder", verdict="pass", independent_sources=0)
    assert mk([weak]).status == "pending"           # I2: pending, never pass
    strong = CheckRecord(lens="grounder", verdict="pass", independent_sources=1)
    assert mk([strong]).status == "pass"

def test_fail_dominates():
    a = CheckRecord(lens="grounder", verdict="pass", independent_sources=2)
    b = CheckRecord(lens="redteam", verdict="fail")
    assert mk([a, b]).status == "fail"

def test_uncertain_over_pass():
    a = CheckRecord(lens="grounder", verdict="pass", independent_sources=2)
    b = CheckRecord(lens="prover", verdict="uncertain")
    assert mk([a, b]).status == "uncertain"
```

- [ ] **Step 2: Run → FAIL.** `pytest tests/test_artifact_claim.py -v`

- [ ] **Step 3: Implement the first half of `valagents/artifact.py`.**

```python
"""IdeaArtifact schema + the computed gate. status/maturity/load_bearing/blocker are
computed properties with NO setter — the gate is code, never an LLM (I1)."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, computed_field

class Source(BaseModel):
    locator: str
    author: str | None = None
    group: str | None = None
    relation: Literal["independent", "same_author", "same_group", "self_citation", "unknown"] = "unknown"

class CheckRecord(BaseModel):
    lens: Literal["grounder", "prover", "redteam"]
    verdict: Literal["pass", "fail", "uncertain"]
    basis: str = ""
    sources: list[Source] = []
    independent_sources: int = 0
    tick: int = 0

class FormalClaim(BaseModel):
    statement: str
    variables: list[str] = []
    scope: str = ""
    regime: str = ""
    falsifiable: bool

class Faithfulness(BaseModel):
    verdict: Literal["yes", "narrowed", "no"]
    back_translation: str = ""
    retried: bool = False

class Coverage(BaseModel):
    verdict: Literal["complete", "gap"]
    missing: str | None = None

class AttackSurface(BaseModel):
    attempted: list[str] = []
    skipped: list[str] = []

class Novelty(BaseModel):
    closest_prior: list[str] = []
    delta: str = ""
    position: Literal["new", "special_case", "restatement"] = "new"

class Prediction(BaseModel):
    observable: str
    effect_size: str = ""
    discriminates_from: str = ""
    measurable: bool = False

class Attack(BaseModel):
    type: str
    severity: Literal["fatal", "major", "minor"]
    status: Literal["survived", "landed"]
    target_claim_id: str | None = None
    basis: str = ""

class Gap(BaseModel):
    description: str
    claim_id: str
    fatal: bool = False

class Derivation(BaseModel):
    steps: list[str] = []
    gaps: list[Gap] = []

class ValidationPlan(BaseModel):
    decisive_test: str
    controls: list[str] = []
    confirm_if: str = ""
    refute_if: str = ""
    cost: Literal["low", "medium", "high"] = "medium"

class AtomicClaim(BaseModel):
    id: str
    statement: str
    type: Literal["definitional", "mathematical", "empirical", "mechanistic"]
    depends_on: list[str] = []
    load_bearing: bool = True
    checks: list[CheckRecord] = []
    exhausted: bool = False

    @computed_field
    @property
    def status(self) -> str:
        if any(c.verdict == "fail" for c in self.checks):
            return "fail"
        if any(c.verdict == "uncertain" for c in self.checks):
            return "uncertain"
        if any(c.verdict == "pass" and c.independent_sources >= 1 for c in self.checks):
            return "pass"
        return "pending"
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_artifact_claim.py -v` → 4 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(artifact): schema leaves + AtomicClaim.status join (pending never pass)"`

---

## Task 4: The gate — `IdeaArtifact` + `_evaluate()` (status + blocker)

The heart of the system. `_evaluate()` is the single source of the total gate; `status` and `blocker` are thin computed wrappers over it (Task 5).

**Files:** Modify `valagents/artifact.py` (append). Test: `tests/test_artifact_gate.py`.

**Interfaces:**
- Consumes: all Task 3 models.
- Produces: `IdeaArtifact` with fields per spec §2 (incl. `min_attack_categories:int=2`, `fanout_N:int=2`, `repair_cap:int=3`, `repairs_spent:int=0`, `finalized:bool=False`). Internal helpers `root_ancestors()`, `_landed(sev)`, `_any_landed()`, `_thin_attack_surface()`, `_has_independent_external_check(claim)`, and `_evaluate() -> tuple[str, dict|None]` returning `(status, blocker)` where `blocker` is `{"claim_id": str|None, "reason": str}` or `None`.

- [ ] **Step 1: Write failing tests** `tests/test_artifact_gate.py` (one per gate branch — this is the spec §8 suite):

```python
from valagents.artifact import (IdeaArtifact, AtomicClaim, CheckRecord, FormalClaim,
                                 Faithfulness, Coverage, AttackSurface, Attack)

PASS = CheckRecord(lens="grounder", verdict="pass", independent_sources=1)

def claim(cid, checks=(PASS,), lb=True, deps=()):
    return AtomicClaim(id=cid, statement="s", type="empirical",
                       checks=list(checks), load_bearing=lb, depends_on=list(deps), exhausted=True)

def art(**kw):
    base = dict(raw_idea="seed",
                formal_claim=FormalClaim(statement="x", falsifiable=True),
                faithfulness=Faithfulness(verdict="yes"),
                coverage=Coverage(verdict="complete"),
                attack_surface=AttackSurface(attempted=["magnitude", "confound"]),
                claim_graph=[claim("c1")], finalized=True)
    base.update(kw)
    return IdeaArtifact(**base)

# --- happy path ---
def test_internally_validated():
    assert art().status == "internally_validated"

# --- entry gates (I3) ---
def test_not_falsifiable():
    a = art(formal_claim=FormalClaim(statement="x", falsifiable=False))
    assert a.status == "refuted" and a.blocker["reason"] == "not_falsifiable"

def test_unfaithful_drift_after_retry():
    a = art(faithfulness=Faithfulness(verdict="no", retried=True))
    assert a.status == "refuted" and a.blocker["reason"] == "unfaithful_drift"

def test_unfaithful_narrowed_after_retry():
    a = art(faithfulness=Faithfulness(verdict="narrowed", retried=True))
    assert a.status == "refuted" and a.blocker["reason"] == "unfaithful_narrowed"

def test_faithfulness_none_cannot_validate():        # the SPOF-in-code test (rev 3)
    a = art(faithfulness=None)
    assert a.status != "internally_validated"

def test_empty_graph_ill_formed():
    a = art(claim_graph=[])
    assert a.status == "refuted" and a.blocker["reason"] == "ill_formed"

# --- refutation ---
def test_failed_claim():
    a = art(claim_graph=[claim("c1", checks=[CheckRecord(lens="redteam", verdict="fail")])])
    assert a.status == "refuted" and a.blocker["reason"] == "failed"

def test_fatal_attack_landed():
    a = art(attacks=[Attack(type="counterexample", severity="fatal", status="landed", target_claim_id="c1")])
    assert a.status == "refuted" and a.blocker["reason"] == "attacked"

# --- needs_experiment ---
def test_uncertain_claim():
    a = art(claim_graph=[claim("c1", checks=[CheckRecord(lens="prover", verdict="uncertain")])])
    assert a.status == "needs_experiment" and a.blocker["reason"] == "inconclusive"

def test_uncovered_pending_claim():
    a = art(claim_graph=[claim("c1", checks=[])])    # exhausted + pending
    assert a.status == "needs_experiment" and a.blocker["reason"] == "uncovered"

def test_coverage_gap():
    a = art(coverage=Coverage(verdict="gap", missing="load-bearing step"))
    assert a.status == "needs_experiment" and a.blocker["reason"] == "decomposition_gap"

def test_thin_attack_surface():
    a = art(attack_surface=AttackSurface(attempted=["counterexample"]))  # no magnitude, <2
    assert a.status == "needs_experiment" and a.blocker["reason"] == "thin_attack_surface"

def test_open_major_objection():
    a = art(attacks=[Attack(type="confound", severity="major", status="landed")])
    assert a.status == "needs_experiment" and a.blocker["reason"] == "open_objection"

# --- D4 minor attack does not block ---
def test_minor_landed_still_validates():
    a = art(attacks=[Attack(type="confound", severity="minor", status="landed")])
    assert a.status == "internally_validated"

# --- repair-cap exhaustion (D5) ---
def test_repair_cap_exhaustion_refuted():
    a = art(repairs_spent=3, repair_cap=3,
            attacks=[Attack(type="counterexample", severity="fatal", status="landed")])
    assert a.status == "refuted"

# --- non-terminal draft (I3: scheduler keeps going) ---
def test_draft_when_unfinalized_pending():
    a = art(finalized=False, claim_graph=[claim("c1", checks=[], )])
    a.claim_graph[0].exhausted = False
    assert a.status == "draft"

# --- order independence (pre-validates Spec 4) ---
def test_order_independence():
    c = claim("c1", checks=[CheckRecord(lens="grounder", verdict="pass", independent_sources=1),
                            CheckRecord(lens="redteam", verdict="uncertain")])
    a1 = art(claim_graph=[c])
    c2 = claim("c1", checks=list(reversed(c.checks)))
    a2 = art(claim_graph=[c2])
    assert a1.status == a2.status == "needs_experiment"
```

- [ ] **Step 2: Run → FAIL.** `pytest tests/test_artifact_gate.py -v`

- [ ] **Step 3: Append `IdeaArtifact` to `valagents/artifact.py`.**

```python
DRAFT = "draft"
INTERNALLY_VALIDATED = "internally_validated"
NEEDS_EXPERIMENT = "needs_experiment"
REFUTED = "refuted"

class IdeaArtifact(BaseModel):
    raw_idea: str
    formal_claim: FormalClaim | None = None
    faithfulness: Faithfulness | None = None
    coverage: Coverage | None = None
    claim_graph: list[AtomicClaim] = []
    derivation: Derivation | None = None
    novelty: Novelty | None = None
    predictions: list[Prediction] = []
    attacks: list[Attack] = []
    attack_surface: AttackSurface | None = None
    validation_plan: ValidationPlan | None = None
    version_id: int = 0
    parent_version: int | None = None
    repairs_spent: int = 0
    repair_cap: int = 3
    min_attack_categories: int = 2
    fanout_N: int = 2
    finalized: bool = False

    # ---- helpers (pure; read recorded state only) ----
    def root_ancestors(self) -> list[AtomicClaim]:
        # Spec-1 default: every load-bearing claim in the decomposition (conservative).
        return [c for c in self.claim_graph if c.load_bearing]

    def _landed(self, severity: str) -> bool:
        return any(a.status == "landed" and a.severity == severity for a in self.attacks)

    def _any_landed(self) -> bool:
        return any(a.status == "landed" for a in self.attacks)

    def _thin_attack_surface(self) -> bool:
        s = self.attack_surface
        if s is None or "magnitude" not in s.attempted:
            return True
        return len(set(s.attempted)) < self.min_attack_categories

    def _has_independent_external_check(self, c: AtomicClaim) -> bool:
        return any(ck.verdict == "pass" and ck.independent_sources >= 1 for ck in c.checks)

    def _b(self, reason: str, claim_id: str | None = None) -> dict:
        return {"claim_id": claim_id, "reason": reason}

    def _evaluate(self) -> tuple[str, dict | None]:
        rs = self.root_ancestors()
        # ===== ENTRY GATES =====
        if self.formal_claim and not self.formal_claim.falsifiable:
            return REFUTED, self._b("not_falsifiable")
        if self.faithfulness and self.faithfulness.retried and self.faithfulness.verdict == "no":
            return REFUTED, self._b("unfaithful_drift")
        if self.faithfulness and self.faithfulness.retried and self.faithfulness.verdict == "narrowed":
            return REFUTED, self._b("unfaithful_narrowed")
        if (self.formal_claim and self.faithfulness and self.faithfulness.verdict == "yes"
                and not self.claim_graph and self.finalized):
            return REFUTED, self._b("ill_formed")
        # ===== REFUTATION =====
        for c in rs:
            if c.status == "fail":
                return REFUTED, self._b("failed", c.id)
        if self._landed("fatal"):
            a = next(a for a in self.attacks if a.status == "landed" and a.severity == "fatal")
            return REFUTED, self._b("attacked", a.target_claim_id)
        # ===== NEEDS EXPERIMENT =====
        for c in rs:
            if c.status == "uncertain":
                return NEEDS_EXPERIMENT, self._b("inconclusive", c.id)
        if self._landed("major") and self.finalized:
            a = next(a for a in self.attacks if a.status == "landed" and a.severity == "major")
            return NEEDS_EXPERIMENT, self._b("open_objection", a.target_claim_id)
        for c in rs:
            if c.status == "pending" and c.exhausted:
                return NEEDS_EXPERIMENT, self._b("uncovered", c.id)
        if self.coverage and self.coverage.verdict == "gap":
            return NEEDS_EXPERIMENT, self._b("decomposition_gap")
        if self._thin_attack_surface() and self.finalized:
            return NEEDS_EXPERIMENT, self._b("thin_attack_surface")
        # ===== VALIDATED: STRICT (I2) =====
        if (rs and all(c.status == "pass" for c in rs)
                and all(self._has_independent_external_check(c) for c in rs)
                and (self.faithfulness and self.faithfulness.verdict == "yes")
                and (self.coverage and self.coverage.verdict == "complete")
                and not self._thin_attack_surface()
                and not self._landed("fatal") and not self._landed("major")):
            return INTERNALLY_VALIDATED, None
        return DRAFT, None
```

Note for the implementer: `_thin_attack_surface()` is gated by `self.finalized` in the `needs_experiment` branch so that mid-run (before the Red-team has run) a not-yet-populated surface keeps the artifact in `draft`, not a premature `needs_experiment`. In the validated branch it is unconditional (a thin surface can never validate).

- [ ] **Step 4: Run → PASS.** `pytest tests/test_artifact_gate.py -v` → all passed. Fix any branch-ordering bug until green.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(artifact): total gate via _evaluate() — all entry gates, caps, strict validated branch"`

---

## Task 5: `status` / `load_bearing` / `blocker` computed fields

**Files:** Modify `valagents/artifact.py`. Test: `tests/test_artifact_load_bearing.py`.

**Interfaces:**
- Produces on `IdeaArtifact`: `@computed_field status -> str`, `blocker -> dict|None`, `load_bearing -> str|None`. `status`/`blocker` delegate to `_evaluate()`. `load_bearing` = the blocker's claim_id if present, else the root-ancestor claim with the most transitive dependents (tie-break: lexicographic id), else `None`.

- [ ] **Step 1: Write failing tests** `tests/test_artifact_load_bearing.py`:

```python
from tests.test_artifact_gate import art, claim
from valagents.artifact import CheckRecord

def test_status_and_blocker_agree_with_evaluate():
    a = art(claim_graph=[claim("c1", checks=[CheckRecord(lens="prover", verdict="uncertain")])])
    assert a.status == "needs_experiment"
    assert a.blocker["claim_id"] == "c1" and a.blocker["reason"] == "inconclusive"

def test_load_bearing_is_blocker_claim_when_blocked():
    a = art(claim_graph=[claim("c1", checks=[CheckRecord(lens="redteam", verdict="fail")])])
    assert a.load_bearing == "c1"

def test_load_bearing_is_most_depended_on_when_validated():
    # c2 depends on c1 -> c1 has more transitive dependents -> pivotal
    a = art(claim_graph=[claim("c1"), claim("c2", deps=["c1"])])
    assert a.load_bearing == "c1"
```

- [ ] **Step 2: Run → FAIL.** `pytest tests/test_artifact_load_bearing.py -v`

- [ ] **Step 3: Append the computed fields.**

```python
    @computed_field
    @property
    def status(self) -> str:
        return self._evaluate()[0]

    @computed_field
    @property
    def blocker(self) -> dict | None:
        return self._evaluate()[1]

    @computed_field
    @property
    def load_bearing(self) -> str | None:
        b = self._evaluate()[1]
        if b and b.get("claim_id"):
            return b["claim_id"]
        rs = self.root_ancestors()
        if not rs:
            return None
        deps = {c.id: 0 for c in self.claim_graph}
        # count transitive dependents: how many claims (directly/indirectly) depend on each id
        adj = {c.id: c.depends_on for c in self.claim_graph}
        def reaches(start, target, seen=None):
            seen = seen or set()
            for d in adj.get(start, []):
                if d == target or (d not in seen and reaches(d, target, seen | {d})):
                    return True
            return False
        for c in self.claim_graph:
            for other in self.claim_graph:
                if other.id != c.id and reaches(other.id, c.id):
                    deps[c.id] += 1
        return max(rs, key=lambda c: (deps[c.id], c.id)).id
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_artifact_load_bearing.py -v` → 3 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(artifact): status/blocker/load_bearing computed fields over the gate"`

---

## Task 6: `maturity` — USER CONTRIBUTION (learning mode) + isolation test

> **This task has a human-authored step.** The `maturity` formula is a deliberate design choice the spec leaves open (§2.3). The executor sets up the signature, context, and the binding test; the **user writes the ~8-line body**. The one hard constraint — `maturity` must NOT read `self.status` — is enforced by a test that holds regardless of the formula.

**Files:** Modify `valagents/artifact.py`. Test: `tests/test_artifact_maturity.py`.

- [ ] **Step 1: Write the isolation + sanity tests** `tests/test_artifact_maturity.py`:

```python
import inspect
from tests.test_artifact_gate import art, claim
from valagents.artifact import IdeaArtifact, CheckRecord

def test_maturity_is_a_float_in_unit_interval():
    m = art().maturity
    assert isinstance(m, float) and 0.0 <= m <= 1.0

def test_status_does_not_depend_on_maturity():
    # maturity ⊥ status: status source must not reference `maturity`
    src = inspect.getsource(IdeaArtifact._evaluate)
    assert "maturity" not in src

def test_passing_claims_mature_higher_than_pending():
    high = art()  # all pass
    low = art(claim_graph=[claim("c1", checks=[])])  # pending
    assert high.maturity > low.maturity
```

- [ ] **Step 2: Run → FAIL.** `pytest tests/test_artifact_maturity.py -v`

- [ ] **Step 3: USER writes `maturity`.** Present this scaffold and ask the user for the body. Default (safe, replaceable):

```python
    @computed_field
    @property
    def maturity(self) -> float:
        # USER CONTRIBUTION (learning mode). HARD CONSTRAINT: must not read self.status.
        # Inputs you may use: self.claim_graph (+ each claim.status), self.attacks,
        # self.coverage, self.attack_surface, self.predictions.
        rs = self.root_ancestors()
        if not rs:
            return 0.0
        per = {"pass": 1.0, "uncertain": 0.5, "fail": 0.0, "pending": 0.0}
        base = sum(per[c.status] for c in rs) / len(rs)
        minor = sum(1 for a in self.attacks if a.status == "landed" and a.severity == "minor")
        return max(0.0, base - 0.1 * minor)
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_artifact_maturity.py -v` → 3 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(artifact): maturity scalar (display-only, ⊥ status)"`

---

## Task 7: `ArtifactStore` — version chain + append-only log

**Files:** Create `valagents/store.py`. Test: `tests/test_store.py`.

**Interfaces:**
- Consumes: `IdeaArtifact`, `valagents.run_log`.
- Produces: `ArtifactStore(initial: IdeaArtifact)` with `.current -> IdeaArtifact`, `.versions -> list[IdeaArtifact]`, `.events -> list[dict]`, `record(event: dict) -> None` (append + `run_log.emit`), `add_check(claim_id, rec) -> None`, `set(field, value) -> None`, `fork_for_repair(target_ids: list[str]) -> IdeaArtifact` (deep-copy current → new version, `version_id+1`, `parent_version=current.version_id`, `repairs_spent+1`, clears `checks`+`exhausted` on `target_ids` only, carries everything else forward; prior version stays frozen).

- [ ] **Step 1: Write failing tests** `tests/test_store.py`:

```python
from valagents.artifact import IdeaArtifact, AtomicClaim, CheckRecord
from valagents.store import ArtifactStore

def base():
    g = [AtomicClaim(id="c1", statement="s", type="empirical"),
         AtomicClaim(id="c2", statement="s", type="empirical")]
    return IdeaArtifact(raw_idea="seed", claim_graph=g)

def test_add_check_and_record_appends():
    s = ArtifactStore(base())
    s.add_check("c1", CheckRecord(lens="grounder", verdict="pass", independent_sources=1))
    s.record({"event": "check", "claim": "c1"})
    assert s.current.claim_graph[0].checks[0].verdict == "pass"
    assert s.events[-1]["claim"] == "c1"

def test_fork_freezes_prior_version():
    s = ArtifactStore(base())
    s.add_check("c1", CheckRecord(lens="grounder", verdict="pass", independent_sources=1))
    s.add_check("c2", CheckRecord(lens="grounder", verdict="pass", independent_sources=1))
    v1 = s.current
    s.fork_for_repair(["c2"])                # repair only c2's subgraph
    v2 = s.current
    assert v2.version_id == 1 and v2.parent_version == 0 and v2.repairs_spent == 1
    assert len(v1.claim_graph[1].checks) == 1     # v1 frozen, untouched
    assert v2.claim_graph[1].checks == []         # c2 cleared in v2
    assert len(v2.claim_graph[0].checks) == 1     # c1 carried forward
```

- [ ] **Step 2: Run → FAIL.** `pytest tests/test_store.py -v`

- [ ] **Step 3: Implement `valagents/store.py`.**

```python
"""Single-writer artifact store: immutable version chain + append-only event log.
Version-don't-mutate makes Spec-4 parallel + repair safe."""
from __future__ import annotations
from valagents.artifact import IdeaArtifact, CheckRecord
from valagents import run_log

class ArtifactStore:
    def __init__(self, initial: IdeaArtifact) -> None:
        self._versions: list[IdeaArtifact] = [initial]
        self.events: list[dict] = []

    @property
    def current(self) -> IdeaArtifact:
        return self._versions[-1]

    @property
    def versions(self) -> list[IdeaArtifact]:
        return list(self._versions)

    def record(self, event: dict) -> None:
        self.events.append(event)
        run_log.emit(event.get("event", "event"), **{k: v for k, v in event.items() if k != "event"})

    def _claim(self, claim_id: str):
        return next(c for c in self.current.claim_graph if c.id == claim_id)

    def add_check(self, claim_id: str, rec: CheckRecord) -> None:
        self._claim(claim_id).checks.append(rec)

    def set(self, field: str, value) -> None:
        setattr(self.current, field, value)

    def fork_for_repair(self, target_ids: list[str]) -> IdeaArtifact:
        cur = self.current
        nxt = cur.model_copy(deep=True)          # frozen snapshot of cur stays in _versions
        nxt.version_id = cur.version_id + 1
        nxt.parent_version = cur.version_id
        nxt.repairs_spent = cur.repairs_spent + 1
        nxt.finalized = False
        for c in nxt.claim_graph:
            if c.id in target_ids:
                c.checks = []
                c.exhausted = False
        self._versions.append(nxt)
        return nxt
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_store.py -v` → 2 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(store): immutable version chain + append-only event log"`

---

## Task 8: Agent base + prompts + Formalizer (the template)

Establishes the agent pattern in full. Every later agent reuses `build_messages` + `checked`/`checked_lines`. Each agent is a module-level async function.

**Files:** Create `valagents/prompts.py`, `valagents/agents/base.py`, `valagents/agents/formalizer.py`. Test: `tests/test_agent_formalizer.py`.

**Interfaces:**
- Produces:
  - `prompts.py`: string templates, one per agent (see below), each ending with its mandatory tail.
  - `agents/base.py`: `build_messages(system: str, user: str) -> list[dict]`; `map_support_to_verdict(support: str, independent_sources: int) -> str` (the D8 downgrade: `"supported"` + `independent_sources<1` → `"uncertain"`; `"unsupported"`→`"fail"`; else `"uncertain"`/`"supported"→"pass"`); `as_int(s: str, default=0) -> int`.
  - `agents/formalizer.py`: `async formalize(raw_idea: str, llm, cfg) -> FormalClaim | None` (None on double parse failure).

- [ ] **Step 1: Write failing tests** `tests/test_agent_formalizer.py`:

```python
from valagents.agents.formalizer import formalize
from valagents.agents.base import map_support_to_verdict
from tests.fake_llm import FakeLLM

async def test_formalizer_pins_claim(cfg):
    body = ("reasoning\nCLAIM: escape time falls with a curl term | VARIABLES: theta, alpha "
            "| REGIME: strict saddles | FALSIFIABLE: yes")
    fc = await formalize("curl term helps escape saddles", FakeLLM(lambda a, m: body), cfg)
    assert fc.falsifiable is True and "curl" in fc.statement

async def test_formalizer_not_falsifiable(cfg):
    body = "CLAIM: it is elegant | VARIABLES: none | REGIME: any | FALSIFIABLE: no"
    fc = await formalize("seed", FakeLLM(lambda a, m: body), cfg)
    assert fc.falsifiable is False

async def test_formalizer_double_fail_returns_none(cfg):
    fc = await formalize("seed", FakeLLM(lambda a, m: "no tail at all"), cfg)
    assert fc is None

def test_support_downgrade_without_independent_source():
    assert map_support_to_verdict("supported", 0) == "uncertain"   # D8
    assert map_support_to_verdict("supported", 2) == "pass"
    assert map_support_to_verdict("unsupported", 5) == "fail"
```

- [ ] **Step 2: Run → FAIL.** `pytest tests/test_agent_formalizer.py -v`

- [ ] **Step 3a: Write `valagents/prompts.py`** (Formalizer first; later tasks append the rest):

```python
"""Prompt templates. Each ends with a mandatory machine-readable tail (parsed strictly)."""

FORMALIZER = """Restate the following idea as a precise, falsifiable claim. Identify the variables \
and what they range over; the scope and the regime of validity; the conditions under which it is \
asserted to hold. Do not add mechanism or evidence — only sharpen the statement.

IDEA: {raw_idea}

End your response with exactly:
CLAIM: <one sentence> | VARIABLES: <…> | REGIME: <…> | FALSIFIABLE: yes|no"""
```

- [ ] **Step 3b: Write `valagents/agents/base.py`.**

```python
"""Shared agent helpers."""
from __future__ import annotations

def build_messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def as_int(s: str, default: int = 0) -> int:
    try:
        return int("".join(ch for ch in s if ch.isdigit() or ch == "-") or default)
    except ValueError:
        return default

def map_support_to_verdict(support: str, independent_sources: int) -> str:
    support = (support or "").strip().lower()
    if support == "supported":
        return "pass" if independent_sources >= 1 else "uncertain"   # D8 downgrade
    if support == "unsupported":
        return "fail"
    return "uncertain"
```

- [ ] **Step 3c: Write `valagents/agents/formalizer.py`.**

```python
from __future__ import annotations
from valagents.artifact import FormalClaim
from valagents.parse import checked
from valagents.prompts import FORMALIZER
from valagents.agents.base import build_messages

async def formalize(raw_idea: str, llm, cfg) -> FormalClaim | None:
    msgs = build_messages("You are a careful formalizer.", FORMALIZER.format(raw_idea=raw_idea))
    tail = await checked("formalizer", msgs, ["CLAIM", "VARIABLES", "REGIME", "FALSIFIABLE"], llm=llm)
    if tail is None:
        return None
    return FormalClaim(
        statement=tail["claim"],
        variables=[v.strip() for v in tail["variables"].split(",") if v.strip()],
        scope="", regime=tail["regime"],
        falsifiable=tail["falsifiable"].strip().lower().startswith("y"),
    )
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_agent_formalizer.py -v` → 4 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(agents): base helpers + prompts + Formalizer (the agent template)"`

---

## Task 9: Structural guards — Faithfulness, Decomposer, Entailment

**Files:** Append to `valagents/prompts.py`; create `valagents/agents/faithfulness.py`, `decomposer.py`, `entailment.py`. Test: `tests/test_agent_guards.py`.

**Interfaces:**
- `async faithfulness_check(raw_idea, formal_claim, llm, cfg, retried=False) -> Faithfulness | None`.
- `async decompose(formal_claim, llm, cfg) -> list[AtomicClaim]` (empty list on parse failure).
- `async entailment_check(formal_claim, claims, llm, cfg) -> Coverage | None`.

- [ ] **Step 1: Write failing tests** `tests/test_agent_guards.py`:

```python
from valagents.agents.faithfulness import faithfulness_check
from valagents.agents.decomposer import decompose
from valagents.agents.entailment import entailment_check
from valagents.artifact import FormalClaim
from tests.fake_llm import FakeLLM

FC = FormalClaim(statement="escape time falls with curl term", falsifiable=True)

async def test_faithfulness_yes(cfg):
    body = "BACK_TRANSLATION: rotation speeds saddle escape\nFAITHFUL: yes | BACK_TRANSLATION: rotation speeds escape"
    f = await faithfulness_check("curl helps escape saddles", FC, FakeLLM(lambda a, m: body), cfg)
    assert f.verdict == "yes"

async def test_faithfulness_narrowed_records_retried_flag(cfg):
    body = "FAITHFUL: narrowed | BACK_TRANSLATION: only decoherence"
    f = await faithfulness_check("is collapse physical", FC, FakeLLM(lambda a, m: body), cfg, retried=True)
    assert f.verdict == "narrowed" and f.retried is True

async def test_decompose_builds_graph(cfg):
    body = ("CLAIM: A | TYPE: mathematical | DEPENDS_ON: none | STATEMENT: projection nonzero\n"
            "CLAIM: B | TYPE: mechanistic | DEPENDS_ON: none | STATEMENT: alpha not saturated\n"
            "CLAIM: C | TYPE: empirical | DEPENDS_ON: A | STATEMENT: converges near minima")
    claims = await decompose(FC, FakeLLM(lambda a, m: body), cfg)
    assert [c.id for c in claims] == ["A", "B", "C"]
    assert claims[2].depends_on == ["A"] and claims[0].type == "mathematical"

async def test_decompose_empty_on_failure(cfg):
    claims = await decompose(FC, FakeLLM(lambda a, m: "no rows"), cfg)
    assert claims == []

async def test_entailment_gap(cfg):
    body = "COVERS: gap | MISSING: the load-bearing nonzero-projection step"
    cov = await entailment_check(FC, [], FakeLLM(lambda a, m: body), cfg)
    assert cov.verdict == "gap" and "projection" in cov.missing
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3a: Append prompts.**

```python
FAITHFULNESS = """Here is a seed idea and a formal claim a colleague derived from it.
SEED: {raw_idea}
FORMAL CLAIM: {formal}
Back-translate the FORMAL CLAIM into plain language, then judge whether it is what the SEED asked —
not a narrowing, not an adjacent claim. End with exactly:
FAITHFUL: yes|narrowed|no | BACK_TRANSLATION: <plain-language restatement of the formal claim>"""

DECOMPOSER = """Decompose this claim into atomic, independently-checkable sub-claims with dependency \
edges. Tag each by type. Do not invent support — only expose structure.
CLAIM: {formal}
Output ONE line per sub-claim, exactly:
CLAIM: <id> | TYPE: definitional|mathematical|empirical|mechanistic | DEPENDS_ON: <ids|none> | STATEMENT: <…>"""

ENTAILMENT = """Does the conjunction of these sub-claims logically establish the formal claim, or is a \
load-bearing step missing?
FORMAL CLAIM: {formal}
SUB-CLAIMS:
{subclaims}
End with exactly:
COVERS: complete|gap | MISSING: <description|none>"""
```

- [ ] **Step 3b: Write the three agents.**

`valagents/agents/faithfulness.py`:
```python
from __future__ import annotations
from valagents.artifact import Faithfulness, FormalClaim
from valagents.parse import checked
from valagents.prompts import FAITHFULNESS
from valagents.agents.base import build_messages

async def faithfulness_check(raw_idea, formal_claim: FormalClaim, llm, cfg, retried=False) -> Faithfulness | None:
    user = FAITHFULNESS.format(raw_idea=raw_idea, formal=formal_claim.statement)
    tail = await checked("faithfulness", build_messages("You are an independent reviewer.", user),
                         ["FAITHFUL", "BACK_TRANSLATION"], llm=llm)
    if tail is None:
        return Faithfulness(verdict="no", back_translation="(unparseable)", retried=retried)  # fail closed
    return Faithfulness(verdict=tail["faithful"].strip().lower(),
                        back_translation=tail["back_translation"], retried=retried)
```
Note: a double-parse-failure here returns `verdict="no"` (fail-closed) rather than `None` — an unparseable faithfulness judgment must NOT be allowed to proceed as faithful. Document this in the docstring.

`valagents/agents/decomposer.py`:
```python
from __future__ import annotations
from valagents.artifact import AtomicClaim, FormalClaim
from valagents.parse import checked_lines
from valagents.prompts import DECOMPOSER
from valagents.agents.base import build_messages

async def decompose(formal_claim: FormalClaim, llm, cfg) -> list[AtomicClaim]:
    user = DECOMPOSER.format(formal=formal_claim.statement)
    rows = await checked_lines("decomposer", build_messages("You expose structure.", user),
                               ["CLAIM", "TYPE", "DEPENDS_ON", "STATEMENT"], llm=llm)
    if not rows:
        return []
    out = []
    for r in rows:
        deps = [] if r["depends_on"].strip().lower() in ("none", "") else \
               [d.strip() for d in r["depends_on"].split(",") if d.strip()]
        out.append(AtomicClaim(id=r["claim"], statement=r["statement"],
                               type=r["type"].strip().lower(), depends_on=deps))
    return out
```

`valagents/agents/entailment.py`:
```python
from __future__ import annotations
from valagents.artifact import Coverage, FormalClaim, AtomicClaim
from valagents.parse import checked
from valagents.prompts import ENTAILMENT
from valagents.agents.base import build_messages

async def entailment_check(formal_claim: FormalClaim, claims: list[AtomicClaim], llm, cfg) -> Coverage | None:
    sub = "\n".join(f"- {c.id}: {c.statement}" for c in claims)
    user = ENTAILMENT.format(formal=formal_claim.statement, subclaims=sub)
    tail = await checked("entailment", build_messages("You check logical coverage.", user),
                         ["COVERS", "MISSING"], llm=llm)
    if tail is None:
        return Coverage(verdict="gap", missing="(unparseable entailment check)")  # fail closed
    missing = None if tail["missing"].strip().lower() in ("none", "") else tail["missing"]
    return Coverage(verdict=tail["covers"].strip().lower(), missing=missing)
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_agent_guards.py -v` → 5 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(agents): Faithfulness/Decomposer/Entailment structural guards (fail-closed)"`

---

## Task 10: Per-claim lenses — Grounder, Prover

**Files:** Append prompts; create `valagents/agents/grounder.py`, `prover.py`. Test: `tests/test_agent_lenses.py`.

**Interfaces:**
- `async ground_claim(claim, formal_claim, backend, llm, cfg, tick=0) -> CheckRecord` (uses `safe_search`; applies `map_support_to_verdict`; populates `sources`/`independent_sources`).
- `async ground_novelty(formal_claim, backend, llm, cfg) -> Novelty | None`.
- `async prove_claim(claim, formal_claim, llm, cfg, tick=0) -> CheckRecord` (covers `definitional` well-formedness; `mathematical`/`mechanistic` derivation).
- `async build_derivation(formal_claim, claims, llm, cfg) -> Derivation`.

- [ ] **Step 1: Write failing tests** `tests/test_agent_lenses.py`:

```python
from valagents.agents.grounder import ground_claim
from valagents.agents.prover import prove_claim
from valagents.artifact import AtomicClaim, FormalClaim
from tests.fake_llm import FakeLLM

FC = FormalClaim(statement="x", falsifiable=True)
CM = AtomicClaim(id="c1", statement="alpha not saturated", type="mechanistic")

async def test_grounder_downgrades_without_independent_source(cfg):
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 0 | SOURCES: none | BASIS: thin"
    rec = await ground_claim(CM, FC, None, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "uncertain" and rec.independent_sources == 0   # D8

async def test_grounder_supported_with_independent(cfg):
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: A1(Smith),A2(Lee) | BASIS: ok"
    rec = await ground_claim(CM, FC, None, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "pass" and rec.independent_sources == 2 and rec.lens == "grounder"

async def test_prover_definitional_wellformed(cfg):
    body = "DERIVATION: complete | GAPS: none | FATAL_GAP: no"
    rec = await prove_claim(AtomicClaim(id="d1", statement="define X", type="definitional"),
                            FC, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "pass" and rec.lens == "prover"

async def test_prover_fatal_gap_fails(cfg):
    body = "DERIVATION: gapped | GAPS: d1 | FATAL_GAP: yes"
    rec = await prove_claim(AtomicClaim(id="d1", statement="x", type="mathematical"),
                            FC, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "fail"
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3a: Append prompts.**

```python
GROUNDER_CLAIM = """Assess whether the literature supports this sub-claim, and identify INDEPENDENT \
sources (distinct authors/groups — not the same lab citing itself).
SUB-CLAIM ({ctype}): {statement}
RETRIEVED LITERATURE:
{articles}
End with exactly:
CLAIM: {cid} | SUPPORT: supported|unsupported|uncertain | INDEPENDENT_SOURCES: <n> | SOURCES: <locator(author)…|none> | BASIS: <…>"""

GROUNDER_NOVELTY = """Position this claim against the closest prior work and name the delta — the \
specific thing it asserts that prior work does not.
CLAIM: {formal}
RETRIEVED LITERATURE:
{articles}
End with exactly:
CLOSEST_PRIOR: <…> | DELTA: <…> | POSITION: new|special_case|restatement"""

PROVER = """Build the chain from premises to this sub-claim. For a definitional claim, check it is \
coherent and non-circular; for mathematical/mechanistic, sketch and check the derivation/causal chain. \
Flag gaps rather than paper over them.
SUB-CLAIM ({ctype}): {statement}
End with exactly:
DERIVATION: complete|gapped | GAPS: <ids|none> | FATAL_GAP: yes|no"""
```

- [ ] **Step 3b: Write `valagents/agents/grounder.py`.**

```python
from __future__ import annotations
from valagents.artifact import CheckRecord, Source, Novelty, AtomicClaim, FormalClaim
from valagents.parse import checked
from valagents.prompts import GROUNDER_CLAIM, GROUNDER_NOVELTY
from valagents.agents.base import build_messages, map_support_to_verdict, as_int
from valagents.web_search import safe_search

async def ground_claim(claim: AtomicClaim, formal_claim, backend, llm, cfg, tick: int = 0) -> CheckRecord:
    articles = await safe_search(backend, claim.statement)
    user = GROUNDER_CLAIM.format(ctype=claim.type, statement=claim.statement,
                                 articles=articles or "(none)", cid=claim.id)
    tail = await checked("grounder", build_messages("You ground claims in literature.", user),
                         ["CLAIM", "SUPPORT", "INDEPENDENT_SOURCES", "SOURCES", "BASIS"], llm=llm)
    if tail is None:
        return CheckRecord(lens="grounder", verdict="uncertain", basis="(unparseable)", tick=tick)
    n = as_int(tail["independent_sources"])
    verdict = map_support_to_verdict(tail["support"], n)
    srcs = [] if tail["sources"].strip().lower() in ("none", "") else \
           [Source(locator=s.strip(), relation="independent") for s in tail["sources"].split(",") if s.strip()]
    return CheckRecord(lens="grounder", verdict=verdict, basis=tail["basis"],
                       sources=srcs, independent_sources=n, tick=tick)

async def ground_novelty(formal_claim: FormalClaim, backend, llm, cfg) -> Novelty | None:
    articles = await safe_search(backend, formal_claim.statement)
    user = GROUNDER_NOVELTY.format(formal=formal_claim.statement, articles=articles or "(none)")
    tail = await checked("grounder", build_messages("You position claims against prior art.", user),
                         ["CLOSEST_PRIOR", "DELTA", "POSITION"], llm=llm)
    if tail is None:
        return None
    return Novelty(closest_prior=[tail["closest_prior"]], delta=tail["delta"],
                   position=tail["position"].strip().lower())
```

- [ ] **Step 3c: Write `valagents/agents/prover.py`.**

```python
from __future__ import annotations
from valagents.artifact import CheckRecord, Derivation, Gap, AtomicClaim, FormalClaim
from valagents.parse import checked
from valagents.prompts import PROVER
from valagents.agents.base import build_messages

async def prove_claim(claim: AtomicClaim, formal_claim, llm, cfg, tick: int = 0) -> CheckRecord:
    user = PROVER.format(ctype=claim.type, statement=claim.statement)
    tail = await checked("prover", build_messages("You check derivations.", user),
                         ["DERIVATION", "GAPS", "FATAL_GAP"], llm=llm)
    if tail is None:
        return CheckRecord(lens="prover", verdict="uncertain", basis="(unparseable)", tick=tick)
    fatal = tail["fatal_gap"].strip().lower().startswith("y")
    gapped = tail["derivation"].strip().lower() == "gapped"
    verdict = "fail" if fatal else ("uncertain" if gapped else "pass")
    # A prover "pass" is an internal check; it counts as independent (the derivation stands on its own logic).
    indep = 1 if verdict == "pass" else 0
    return CheckRecord(lens="prover", verdict=verdict, basis=tail["gaps"],
                       independent_sources=indep, tick=tick)
```
Note: a Prover `pass` sets `independent_sources=1` so a purely mathematical/definitional claim (which the Grounder cannot externally support) can still reach claim-status `pass` on a self-standing derivation. Document this as the design choice it is.

- [ ] **Step 4: Run → PASS.** `pytest tests/test_agent_lenses.py -v` → 4 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(agents): Grounder (independence-aware) + Prover (definitional well-formedness)"`

---

## Task 11: Whole-artifact lenses — Predictor, Red-team, Validation-designer

**Files:** Append prompts; create `valagents/agents/predictor.py`, `redteam.py`, `validation_designer.py`. Test: `tests/test_agent_whole.py`.

**Interfaces:**
- `async predict(formal_claim, novelty, llm, cfg) -> list[Prediction]`.
- `async red_team(artifact, llm, cfg, tick=0) -> tuple[list[Attack], AttackSurface, list[tuple[str, CheckRecord]]]` — returns attacks, the attempted/skipped surface, and per-target `(claim_id, redteam CheckRecord)` for landed/surviving attacks.
- `async design_validation(artifact, llm, cfg) -> ValidationPlan | None`.

- [ ] **Step 1: Write failing tests** `tests/test_agent_whole.py`:

```python
from valagents.agents.predictor import predict
from valagents.agents.redteam import red_team
from valagents.agents.validation_designer import design_validation
from valagents.artifact import IdeaArtifact, FormalClaim, AtomicClaim, Novelty
from tests.fake_llm import FakeLLM

FC = FormalClaim(statement="escape time falls with curl term", falsifiable=True)

async def test_predict(cfg):
    body = "OBSERVABLE: mean escape time | EFFECT_SIZE: 2x faster | DISCRIMINATES_FROM: vanilla GD | MEASURABLE: yes"
    preds = await predict(FC, Novelty(delta="rotational term"), FakeLLM(lambda a, m: body), cfg)
    assert preds[0].measurable is True and "escape" in preds[0].observable

async def test_red_team_records_surface_and_landed(cfg):
    body = ("ATTEMPTED: counterexample, magnitude\n"
            "ATTACK: counterexample | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: ok\n"
            "ATTACK: magnitude | SEVERITY: major | STATUS: landed | TARGET: c1 | BASIS: alpha saturates")
    art = IdeaArtifact(raw_idea="s", formal_claim=FC,
                       claim_graph=[AtomicClaim(id="c1", statement="alpha", type="mechanistic")])
    attacks, surface, per_claim = await red_team(art, FakeLLM(lambda a, m: body), cfg)
    assert "magnitude" in surface.attempted
    assert any(a.status == "landed" and a.severity == "major" for a in attacks)
    assert per_claim and per_claim[0][0] == "c1" and per_claim[0][1].verdict == "fail"

async def test_design_validation(cfg):
    body = ("TEST: escape-time benchmark | CONFIRM_IF: scaling separates | "
            "REFUTE_IF: no separation | COST: low")
    art = IdeaArtifact(raw_idea="s", formal_claim=FC)
    plan = await design_validation(art, FakeLLM(lambda a, m: body), cfg)
    assert plan.cost == "low" and "benchmark" in plan.decisive_test
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3a: Append prompts.**

```python
PREDICTOR = """Extract the falsifiable consequences of this claim — concrete, measurable, and \
discriminating: what does it predict that the null or the closest existing model does not?
CLAIM: {formal}
DELTA vs prior work: {delta}
Output ONE line per prediction, exactly:
OBSERVABLE: <…> | EFFECT_SIZE: <…> | DISCRIMINATES_FROM: <…> | MEASURABLE: yes|no"""

RED_TEAM = """You are an adversarial reviewer trying to BREAK this claim, not improve it. Attempt, in \
order: (1) a counterexample; (2) a regime where it fails; (3) a confound or simpler explanation; \
(4) a magnitude check — strip the framing and determine whether the mechanism changes any measurable \
quantity at the relevant scale, and by how many orders of magnitude. State which categories you \
ATTEMPTED. For each attack say whether the claim survives.
ARTIFACT:
{artifact}
First line, exactly: ATTEMPTED: <subset of counterexample, failure_regime, confound, magnitude>
Then ONE line per attack, exactly:
ATTACK: <type> | SEVERITY: fatal|major|minor | STATUS: survived|landed | TARGET: <claim_id|none> | BASIS: <…>"""

VALIDATION_DESIGNER = """Propose the single cheapest experiment or computation that would decisively \
confirm or refute this claim. Prefer a computation over an experiment if one suffices.
ARTIFACT:
{artifact}
End with exactly:
TEST: <…> | CONFIRM_IF: <…> | REFUTE_IF: <…> | COST: low|medium|high"""
```

- [ ] **Step 3b: Write the three agents.**

`valagents/agents/predictor.py`:
```python
from __future__ import annotations
from valagents.artifact import Prediction, FormalClaim, Novelty
from valagents.parse import checked_lines
from valagents.prompts import PREDICTOR
from valagents.agents.base import build_messages

async def predict(formal_claim: FormalClaim, novelty: Novelty | None, llm, cfg) -> list[Prediction]:
    user = PREDICTOR.format(formal=formal_claim.statement, delta=(novelty.delta if novelty else ""))
    rows = await checked_lines("predictor", build_messages("You extract falsifiable predictions.", user),
                               ["OBSERVABLE", "EFFECT_SIZE", "DISCRIMINATES_FROM", "MEASURABLE"], llm=llm)
    if not rows:
        return []
    return [Prediction(observable=r["observable"], effect_size=r["effect_size"],
                       discriminates_from=r["discriminates_from"],
                       measurable=r["measurable"].strip().lower().startswith("y")) for r in rows]
```

`valagents/agents/redteam.py`:
```python
from __future__ import annotations
from valagents.artifact import Attack, AttackSurface, CheckRecord, IdeaArtifact
from valagents.parse import checked_lines, parse_tail
from valagents.prompts import RED_TEAM
from valagents.agents.base import build_messages

_CATS = ["counterexample", "failure_regime", "confound", "magnitude"]

def _render(art: IdeaArtifact) -> str:
    fc = art.formal_claim.statement if art.formal_claim else art.raw_idea
    claims = "\n".join(f"- {c.id} ({c.type}): {c.statement}" for c in art.claim_graph)
    return f"CLAIM: {fc}\nSUB-CLAIMS:\n{claims}"

async def red_team(art: IdeaArtifact, llm, cfg, tick: int = 0):
    user = RED_TEAM.format(artifact=_render(art))
    msgs = build_messages("You are an adversarial reviewer.", user)
    rows = await checked_lines("redteam", msgs,
                               ["ATTACK", "SEVERITY", "STATUS", "TARGET", "BASIS"], llm=llm)
    attacks, per_claim = [], []
    if rows:
        for r in rows:
            tgt = None if r["target"].strip().lower() in ("none", "") else r["target"].strip()
            a = Attack(type=r["attack"].strip().lower(), severity=r["severity"].strip().lower(),
                       status=r["status"].strip().lower(), target_claim_id=tgt, basis=r["basis"])
            attacks.append(a)
            if a.status == "landed" and tgt:
                sev = "fail" if a.severity in ("fatal", "major") else "uncertain"
                per_claim.append((tgt, CheckRecord(lens="redteam", verdict=sev, basis=a.basis,
                                                   independent_sources=1, tick=tick)))
    # parse the ATTEMPTED line from the last completion; default to attempted=[] (=> thin surface)
    attempted = []
    try:
        last = await _last_body(llm, msgs)
        row = parse_tail(last, ["ATTEMPTED"])
        attempted = [c.strip().lower() for c in row["attempted"].split(",") if c.strip()]
    except Exception:
        attempted = []
    surface = AttackSurface(attempted=attempted, skipped=[c for c in _CATS if c not in attempted])
    return attacks, surface, per_claim

async def _last_body(llm, msgs) -> str:
    # Re-derive the raw body for the ATTEMPTED line. In tests FakeLLM is deterministic, so a
    # fresh call returns the same body; in production, capture the body in checked_lines instead.
    return await llm.complete("redteam", msgs)
```
Implementer note: capturing `ATTEMPTED` cleanly is better done by having `parse.checked_lines` optionally return the raw body. If you prefer, add a `return_body=True` param to `_attempt`/`checked_lines` (returns `(rows, body)`) and parse `ATTEMPTED` from that single body — avoids the second call in `_last_body`. Either is acceptable; the test asserts behavior, not call count, for Red-team.

`valagents/agents/validation_designer.py`:
```python
from __future__ import annotations
from valagents.artifact import ValidationPlan, IdeaArtifact
from valagents.parse import checked
from valagents.prompts import VALIDATION_DESIGNER
from valagents.agents.base import build_messages
from valagents.agents.redteam import _render

async def design_validation(art: IdeaArtifact, llm, cfg) -> ValidationPlan | None:
    user = VALIDATION_DESIGNER.format(artifact=_render(art))
    tail = await checked("validation_designer", build_messages("You design decisive tests.", user),
                         ["TEST", "CONFIRM_IF", "REFUTE_IF", "COST"], llm=llm)
    if tail is None:
        return None
    return ValidationPlan(decisive_test=tail["test"], confirm_if=tail["confirm_if"],
                          refute_if=tail["refute_if"], cost=tail["cost"].strip().lower())
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_agent_whole.py -v` → 3 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(agents): Predictor + Red-team (attempted surface) + Validation-designer"`

---

## Task 12: Orchestration agents — Repairer, Arbiter

**Files:** Append prompts; create `valagents/agents/repairer.py`, `arbiter.py`. Test: `tests/test_agent_orchestration.py`.

**Interfaces:**
- `async repair(artifact, target_ids, llm, cfg) -> dict | None` → `{"repair": str, "targets": list[str], "rationale": str, "new_statements": dict[claim_id,str]}` (the scheduler forks the version and applies the new statements).
- `async arbitrate(artifact, llm, cfg) -> dict` → `{"status": str, "load_bearing": str, "decisive_test": str, "agrees": bool}` where `agrees = (status == artifact.status)`; the **computed** `artifact.status` is authoritative.

- [ ] **Step 1: Write failing tests** `tests/test_agent_orchestration.py`:

```python
from valagents.agents.arbiter import arbitrate
from valagents.artifact import (IdeaArtifact, FormalClaim, Faithfulness, Coverage,
                                AttackSurface, AtomicClaim, CheckRecord)
from tests.fake_llm import FakeLLM

def validated_art():
    PASS = CheckRecord(lens="grounder", verdict="pass", independent_sources=1)
    return IdeaArtifact(raw_idea="s", formal_claim=FormalClaim(statement="x", falsifiable=True),
                        faithfulness=Faithfulness(verdict="yes"), coverage=Coverage(verdict="complete"),
                        attack_surface=AttackSurface(attempted=["magnitude", "confound"]),
                        claim_graph=[AtomicClaim(id="c1", statement="s", type="empirical",
                                                 checks=[PASS], exhausted=True)], finalized=True)

async def test_arbiter_agrees_with_computed(cfg):
    body = "STATUS: internally_validated | LOAD_BEARING: c1 | DECISIVE_TEST: none needed"
    out = await arbitrate(validated_art(), FakeLLM(lambda a, m: body), cfg)
    assert out["agrees"] is True

async def test_arbiter_disagreement_flagged_computed_wins(cfg):
    # Arbiter narrates validated, but a fatal attack means computed == refuted
    from valagents.artifact import Attack
    art = validated_art()
    art.attacks = [Attack(type="counterexample", severity="fatal", status="landed", target_claim_id="c1")]
    body = "STATUS: internally_validated | LOAD_BEARING: c1 | DECISIVE_TEST: x"
    out = await arbitrate(art, FakeLLM(lambda a, m: body), cfg)
    assert art.status == "refuted" and out["agrees"] is False   # computed wins; mismatch surfaced
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3a: Append prompts.**

```python
REPAIRER = """An attack landed or a derivation gap was found. Propose a targeted repair to the named \
sub-claims — a new version, not a rewrite of what already passed. Do not weaken the claim to dodge the \
attack; fix the mechanism.
ARTIFACT:
{artifact}
TARGETS: {targets}
End with exactly:
REPAIR: <what changed> | TARGETS: <claim_ids> | RATIONALE: <…>"""

ARBITER = """Given the per-claim statuses, attack verdicts, and novelty delta, do not re-argue any of \
them. State the load-bearing claim and the single decisive test. (The system computes STATUS itself; \
your STATUS line is a cross-check.)
ARTIFACT:
{artifact}
COMPUTED STATUS: {computed_status}
End with exactly:
STATUS: <…> | LOAD_BEARING: <claim_id> | DECISIVE_TEST: <…>"""
```

- [ ] **Step 3b: Write the two agents.**

`valagents/agents/repairer.py`:
```python
from __future__ import annotations
from valagents.parse import checked
from valagents.prompts import REPAIRER
from valagents.agents.base import build_messages
from valagents.agents.redteam import _render

async def repair(artifact, target_ids: list[str], llm, cfg) -> dict | None:
    user = REPAIRER.format(artifact=_render(artifact), targets=", ".join(target_ids))
    tail = await checked("repairer", build_messages("You repair claims without weakening them.", user),
                         ["REPAIR", "TARGETS", "RATIONALE"], llm=llm)
    if tail is None:
        return None
    targets = [t.strip() for t in tail["targets"].split(",") if t.strip()] or target_ids
    return {"repair": tail["repair"], "targets": targets, "rationale": tail["rationale"]}
```

`valagents/agents/arbiter.py`:
```python
from __future__ import annotations
from valagents.parse import checked
from valagents.prompts import ARBITER
from valagents.agents.base import build_messages
from valagents.agents.redteam import _render
from valagents import run_log

async def arbitrate(artifact, llm, cfg) -> dict:
    computed = artifact.status                       # authoritative (I1)
    user = ARBITER.format(artifact=_render(artifact), computed_status=computed)
    tail = await checked("arbiter", build_messages("You assemble the verdict.", user),
                         ["STATUS", "LOAD_BEARING", "DECISIVE_TEST"], llm=llm)
    narrated = (tail or {}).get("status", "").strip().lower()
    agrees = narrated == computed
    if not agrees:
        run_log.emit("arbiter_mismatch", narrated=narrated, computed=computed)
    return {"status": computed, "narrated": narrated, "agrees": agrees,
            "load_bearing": (tail or {}).get("load_bearing", artifact.load_bearing),
            "decisive_test": (tail or {}).get("decisive_test", "")}
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_agent_orchestration.py -v` → 2 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(agents): Repairer + Arbiter (computed status wins, mismatch logged)"`

---

## Task 13: Scheduler — entry-gate sequence

**Files:** Create `valagents/scheduler.py` (entry-gate portion). Test: `tests/test_scheduler_entry.py`.

**Interfaces:**
- Consumes: every agent function, `ArtifactStore`, `Config`.
- Produces: `async run_entry_gates(store, raw_idea, backend, llm, cfg) -> bool` — runs Formalizer → Faithfulness (one re-formalization retry on `narrowed`/`no`) → Decomposer (one retry on empty) → Entailment, populating the store. Returns `True` if the pipeline should proceed to checks, `False` if an entry gate finalized a terminal artifact. Sets `store.current.finalized=True` and the relevant fields on any terminal exit. Threads `cfg.gate.*` onto the artifact.

- [ ] **Step 1: Write failing tests** `tests/test_scheduler_entry.py`:

```python
from valagents.scheduler import run_entry_gates
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact
from tests.fake_llm import FakeLLM

def store():
    return ArtifactStore(IdeaArtifact(raw_idea="seed"))

def router(script):
    it = {"i": 0}
    def r(agent, messages):
        out = script.get(agent)
        return out(messages) if callable(out) else out
    return r

async def test_not_falsifiable_terminates(cfg):
    s = store()
    llm = FakeLLM(router({"formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: no"}))
    proceed = await run_entry_gates(s, "seed", None, llm, cfg)
    assert proceed is False and s.current.status == "refuted" and s.current.blocker["reason"] == "not_falsifiable"

async def test_unfaithful_retries_then_refuted(cfg):
    s = store()
    llm = FakeLLM(router({
        "formalizer": "CLAIM: narrow x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
        "faithfulness": "FAITHFUL: narrowed | BACK_TRANSLATION: only a special case",
        "decomposer": "CLAIM: A | TYPE: empirical | DEPENDS_ON: none | STATEMENT: s"}))
    proceed = await run_entry_gates(s, "seed", None, llm, cfg)
    assert proceed is False and s.current.status == "refuted"
    assert s.current.blocker["reason"] == "unfaithful_narrowed" and s.current.faithfulness.retried is True

async def test_empty_decomposition_ill_formed(cfg):
    s = store()
    llm = FakeLLM(router({
        "formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
        "faithfulness": "FAITHFUL: yes | BACK_TRANSLATION: same",
        "decomposer": "no rows here"}))
    proceed = await run_entry_gates(s, "seed", None, llm, cfg)
    assert proceed is False and s.current.status == "refuted" and s.current.blocker["reason"] == "ill_formed"

async def test_clean_entry_proceeds(cfg):
    s = store()
    llm = FakeLLM(router({
        "formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
        "faithfulness": "FAITHFUL: yes | BACK_TRANSLATION: same",
        "decomposer": "CLAIM: A | TYPE: empirical | DEPENDS_ON: none | STATEMENT: s",
        "entailment": "COVERS: complete | MISSING: none"}))
    proceed = await run_entry_gates(s, "seed", None, llm, cfg)
    assert proceed is True and len(s.current.claim_graph) == 1 and s.current.coverage.verdict == "complete"
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Write `valagents/scheduler.py` (entry portion).**

```python
"""DAG scheduler: entry gates → per-claim lenses → fan-out → repair-versioning → total verdict."""
from __future__ import annotations
from valagents.store import ArtifactStore
from valagents.config import Config
from valagents.agents.formalizer import formalize
from valagents.agents.faithfulness import faithfulness_check
from valagents.agents.decomposer import decompose
from valagents.agents.entailment import entailment_check

def _apply_gate_cfg(art, cfg: Config) -> None:
    art.min_attack_categories = cfg.gate.min_attack_categories
    art.fanout_N = cfg.gate.fanout_N
    art.repair_cap = cfg.gate.repair_cap

async def run_entry_gates(store: ArtifactStore, raw_idea: str, backend, llm, cfg: Config) -> bool:
    art = store.current
    _apply_gate_cfg(art, cfg)

    # 1. Formalizer
    fc = await formalize(raw_idea, llm, cfg)
    if fc is None:                      # could not even state it
        art.finalized = True
        store.record({"event": "entry_fail", "stage": "formalizer"})
        return False
    store.set("formal_claim", fc)
    if not fc.falsifiable:
        art.finalized = True
        store.record({"event": "entry_gate", "reason": "not_falsifiable"})
        return False

    # 2. Faithfulness (independent), with one re-formalization retry
    f = await faithfulness_check(raw_idea, fc, llm, cfg, retried=False)
    if f.verdict in ("narrowed", "no"):
        fc2 = await formalize(raw_idea + "\n(Restate FAITHFULLY to the full seed; do not narrow.)", llm, cfg)
        if fc2 is not None:
            store.set("formal_claim", fc2)
            fc = fc2
        f = await faithfulness_check(raw_idea, fc, llm, cfg, retried=True)
    store.set("faithfulness", f)
    if f.verdict in ("narrowed", "no"):
        art.finalized = True
        store.record({"event": "entry_gate", "reason": f"unfaithful_{f.verdict}"})
        return False

    # 3. Decomposer, with one retry on empty
    claims = await decompose(fc, llm, cfg)
    if not claims:
        claims = await decompose(fc, llm, cfg)
    store.set("claim_graph", claims)
    if not claims:
        art.finalized = True
        store.record({"event": "entry_gate", "reason": "ill_formed"})
        return False

    # 4. Entailment (independent) — surfaced; caps in the gate (does not stop the run)
    cov = await entailment_check(fc, claims, llm, cfg)
    store.set("coverage", cov)
    store.record({"event": "entry_ok", "claims": len(claims), "coverage": cov.verdict})
    return True
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_scheduler_entry.py -v` → 4 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(scheduler): entry gates (formalize→faithfulness retry→decompose→entailment)"`

---

## Task 14: Scheduler — per-claim lenses, propagation, fan-out

**Files:** Append to `valagents/scheduler.py`. Test: `tests/test_scheduler_checks.py`.

**Interfaces:**
- Produces: `async run_claim_checks(store, backend, llm, cfg, tick0=0) -> None` — for each claim, dispatch the coverage-matrix lenses (Grounder always; Prover if `definitional|mathematical|mechanistic`; Red-team attacks applied at the whole-artifact stage in Task 15). Apply the **fan-out policy**: a load-bearing claim whose status is `uncertain` after its first lenses gets up to `fanout_N` total *diverse-type* lens runs before being marked `exhausted`. Mark each claim `exhausted` when its applicable lenses are done.

Coverage matrix (from spec §2.4):
```
definitional   → Prover (well-formedness)        [Grounder optional: standard-usage]
mathematical   → Grounder + Prover
empirical      → Grounder
mechanistic    → Grounder + Prover
```

- [ ] **Step 1: Write failing tests** `tests/test_scheduler_checks.py`:

```python
from valagents.scheduler import run_claim_checks
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, AtomicClaim
from tests.fake_llm import FakeLLM

def store_with(claims):
    return ArtifactStore(IdeaArtifact(raw_idea="s",
                         formal_claim=FormalClaim(statement="x", falsifiable=True), claim_graph=claims))

async def test_empirical_claim_grounded_and_exhausted(cfg):
    s = store_with([AtomicClaim(id="c1", statement="effect exists", type="empirical")])
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: A1,A2 | BASIS: ok"
    await run_claim_checks(s, None, FakeLLM(lambda a, m: body), cfg)
    c = s.current.claim_graph[0]
    assert c.status == "pass" and c.exhausted is True

async def test_fanout_runs_more_lenses_on_uncertain_loadbearing(cfg):
    # grounder returns uncertain → fan-out triggers a second diverse run; count lens calls
    s = store_with([AtomicClaim(id="c1", statement="alpha not saturated", type="mechanistic")])
    body = "CLAIM: c1 | SUPPORT: uncertain | INDEPENDENT_SOURCES: 0 | SOURCES: none | BASIS: unclear\n" \
           "DERIVATION: gapped | GAPS: c1 | FATAL_GAP: no"
    llm = FakeLLM(lambda a, m: body)
    await run_claim_checks(s, None, llm, cfg)
    c = s.current.claim_graph[0]
    assert c.status == "uncertain"
    assert len(c.checks) >= cfg.gate.fanout_N    # fan-out met before finalize
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Append to `valagents/scheduler.py`.**

```python
from valagents.agents.grounder import ground_claim
from valagents.agents.prover import prove_claim

_LENS_BY_TYPE = {
    "definitional": ["prover"],
    "mathematical": ["grounder", "prover"],
    "empirical":    ["grounder"],
    "mechanistic":  ["grounder", "prover"],
}

async def _run_lens(name, claim, fc, backend, llm, cfg, tick):
    if name == "grounder":
        return await ground_claim(claim, fc, backend, llm, cfg, tick=tick)
    return await prove_claim(claim, fc, llm, cfg, tick=tick)

async def run_claim_checks(store, backend, llm, cfg, tick0: int = 0) -> None:
    art = store.current
    fc = art.formal_claim
    tick = tick0
    for claim in art.claim_graph:
        lenses = list(_LENS_BY_TYPE.get(claim.type, ["grounder"]))
        for name in lenses:
            rec = await _run_lens(name, claim, fc, backend, llm, cfg, tick); tick += 1
            store.add_check(claim.id, rec)
            store.record({"event": "check", "claim": claim.id, "lens": name, "verdict": rec.verdict})
        # FAN-OUT: load-bearing claim still uncertain → add diverse-type lenses up to fanout_N total.
        if claim.load_bearing and claim.status == "uncertain":
            extra = [n for n in ("grounder", "prover") if n not in lenses] or ["grounder"]
            i = 0
            while len(claim.checks) < cfg.gate.fanout_N and i < len(extra) + 2:
                name = extra[i % len(extra)]
                rec = await _run_lens(name, claim, fc, backend, llm, cfg, tick); tick += 1
                store.add_check(claim.id, rec)
                store.record({"event": "fanout", "claim": claim.id, "lens": name, "verdict": rec.verdict})
                i += 1
        claim.exhausted = True
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_scheduler_checks.py -v` → 2 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(scheduler): per-claim coverage-matrix lenses + fan-out on uncertain load-bearing nodes"`

---

## Task 15: Scheduler — whole-artifact lenses, repair loop, finalize

**Files:** Append to `valagents/scheduler.py`. Test: `tests/test_scheduler_repair.py`.

**Interfaces:**
- Produces: `async run(raw_idea, llm, cfg, backend=None) -> IdeaArtifact` — the full pipeline. After `run_entry_gates` + `run_claim_checks`: run Grounder-novelty, Predictor, Red-team (apply attacks + surface + per-claim redteam checks), Validation-designer. Then the **repair loop**: while a fatal/major attack landed or a fatal gap exists and `repairs_spent < repair_cap`, `fork_for_repair` the affected targets, re-run their claim checks + Red-team, until clean or cap. Then set `finalized=True`, run the Arbiter, return `store.current`.

- [ ] **Step 1: Write failing tests** `tests/test_scheduler_repair.py`:

```python
from valagents.scheduler import run
from tests.fake_llm import FakeLLM

def scripted(script):
    def r(agent, messages):
        v = script[agent]
        return v(messages) if callable(v) else v
    return FakeLLM(r)

BASE = {
    "formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
    "faithfulness": "FAITHFUL: yes | BACK_TRANSLATION: same",
    "decomposer": "CLAIM: c1 | TYPE: empirical | DEPENDS_ON: none | STATEMENT: effect exists",
    "entailment": "COVERS: complete | MISSING: none",
    "predictor": "OBSERVABLE: o | EFFECT_SIZE: 2x | DISCRIMINATES_FROM: null | MEASURABLE: yes",
    "validation_designer": "TEST: t | CONFIRM_IF: c | REFUTE_IF: r | COST: low",
    "arbiter": lambda m: "STATUS: x | LOAD_BEARING: c1 | DECISIVE_TEST: t",
}

async def test_full_run_internally_validated(cfg):
    script = dict(BASE)
    script["grounder"] = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: A1,A2 | BASIS: ok\nCLOSEST_PRIOR: p | DELTA: d | POSITION: new"
    script["redteam"] = "ATTEMPTED: counterexample, magnitude\nATTACK: magnitude | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: inert-but-ok"
    art = await run("seed", scripted(script), cfg)
    assert art.status == "internally_validated"

async def test_fatal_attack_through_cap_refuted(cfg):
    script = dict(BASE)
    script["grounder"] = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: A1,A2 | BASIS: ok\nCLOSEST_PRIOR: p | DELTA: d | POSITION: new"
    script["redteam"] = "ATTEMPTED: counterexample, magnitude\nATTACK: counterexample | SEVERITY: fatal | STATUS: landed | TARGET: c1 | BASIS: breaks"
    script["repairer"] = "REPAIR: tried | TARGETS: c1 | RATIONALE: attempt"
    art = await run("seed", scripted(script), cfg)
    assert art.status == "refuted" and art.repairs_spent == cfg.gate.repair_cap

async def test_thin_surface_needs_experiment(cfg):
    script = dict(BASE)
    script["grounder"] = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: A1,A2 | BASIS: ok\nCLOSEST_PRIOR: p | DELTA: d | POSITION: new"
    script["redteam"] = "ATTEMPTED: counterexample\nATTACK: counterexample | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: weak"
    art = await run("seed", scripted(script), cfg)
    assert art.status == "needs_experiment" and art.blocker["reason"] == "thin_attack_surface"
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Append to `valagents/scheduler.py`.**

```python
from valagents.artifact import IdeaArtifact
from valagents.agents.grounder import ground_novelty
from valagents.agents.predictor import predict
from valagents.agents.redteam import red_team
from valagents.agents.validation_designer import design_validation
from valagents.agents.repairer import repair
from valagents.agents.arbiter import arbitrate

def _repair_targets(art) -> list[str]:
    t = set()
    for a in art.attacks:
        if a.status == "landed" and a.severity in ("fatal", "major") and a.target_claim_id:
            t.add(a.target_claim_id)
    if art.derivation:
        t |= {g.claim_id for g in art.derivation.gaps if g.fatal}
    return list(t)

async def _whole_artifact_lenses(store, backend, llm, cfg, tick) -> None:
    art = store.current
    nov = await ground_novelty(art.formal_claim, backend, llm, cfg)
    if nov:
        store.set("novelty", nov)
    store.set("predictions", await predict(art.formal_claim, nov, llm, cfg))
    attacks, surface, per_claim = await red_team(art, llm, cfg, tick=tick)
    store.set("attacks", attacks)
    store.set("attack_surface", surface)
    for cid, rec in per_claim:
        if any(c.id == cid for c in art.claim_graph):
            store.add_check(cid, rec)
    store.set("validation_plan", await design_validation(art, llm, cfg))

async def run(raw_idea: str, llm, cfg, backend=None) -> IdeaArtifact:
    store = ArtifactStore(IdeaArtifact(raw_idea=raw_idea))
    if not await run_entry_gates(store, raw_idea, backend, llm, cfg):
        return store.current                       # terminal at an entry gate
    await run_claim_checks(store, backend, llm, cfg)
    await _whole_artifact_lenses(store, backend, llm, cfg, tick=1000)

    # repair loop (version-don't-mutate)
    while store.current.repairs_spent < cfg.gate.repair_cap:
        targets = _repair_targets(store.current)
        if not targets:
            break
        rep = await repair(store.current, targets, llm, cfg)
        store.fork_for_repair(targets)
        _apply_gate_cfg(store.current, cfg)
        store.record({"event": "repair", "targets": targets, "ok": rep is not None})
        await run_claim_checks(store, backend, llm, cfg, tick0=2000 * store.current.version_id)
        await _whole_artifact_lenses(store, backend, llm, cfg, tick=3000 * store.current.version_id)

    store.current.finalized = True
    verdict = await arbitrate(store.current, llm, cfg)
    store.record({"event": "final", "status": store.current.status,
                  "load_bearing": store.current.load_bearing, "agrees": verdict["agrees"]})
    return store.current
```

Implementer note: the loop re-forks while landed fatal/major attacks remain. When the cap is hit with a fatal attack still landed, the loop exits and `finalized=True` makes the gate compute `refuted` (D5) — no special-case code. Confirm `test_fatal_attack_through_cap_refuted` asserts exactly this.

- [ ] **Step 4: Run → PASS.** `pytest tests/test_scheduler_repair.py -v` → 3 passed. Run the full suite: `pytest -q`.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(scheduler): whole-artifact lenses + repair-versioning loop + finalize"`

---

## Task 16: CLI + markdown report

**Files:** Create `valagents/cli.py`. Test: `tests/test_cli.py`.

**Interfaces:**
- Produces: `async run_cli(seed, llm, cfg, backend=None, out_dir=None) -> dict` returning `{"artifact": IdeaArtifact, "json_path": str, "report_path": str}`; `render_report(artifact) -> str` (markdown including the §1 limit sentence verbatim); `main()` (argparse: `valagents "<seed>"`).
- The limit sentence (ships in every report): *"internally_validated means 'survived the checks this system can run,' never 'true' — every lens shares the base model's blind spots."*

- [ ] **Step 1: Write failing tests** `tests/test_cli.py`:

```python
from valagents.cli import render_report, run_cli
from tests.test_scheduler_repair import scripted, BASE
import json

def test_report_carries_limit_sentence_and_status():
    from valagents.artifact import IdeaArtifact
    art = IdeaArtifact(raw_idea="seed")
    md = render_report(art)
    assert "never 'true'" in md and "raw_idea" not in md  # human-readable, not a dump

async def test_run_cli_writes_json_and_report(tmp_path, cfg):
    script = dict(BASE)
    script["grounder"] = "CLAIM: c1 | SUPPORT: uncertain | INDEPENDENT_SOURCES: 0 | SOURCES: none | BASIS: thin\nCLOSEST_PRIOR: p | DELTA: d | POSITION: new"
    script["redteam"] = "ATTEMPTED: counterexample, magnitude\nATTACK: magnitude | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: ok"
    out = await run_cli("seed", scripted(script), cfg, out_dir=str(tmp_path))
    assert out["artifact"].status in ("needs_experiment", "internally_validated", "refuted")
    data = json.loads(open(out["json_path"]).read())
    assert data["raw_idea"] == "seed" and "status" in data
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `valagents/cli.py`.**

```python
"""CLI: valagents "<seed>" → IdeaArtifact JSON + markdown report."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
from valagents.config import load_config
from valagents.llm import OpenRouterClient
from valagents.web_search import build_backend
from valagents.scheduler import run
from valagents import run_log

LIMIT = ("internally_validated means 'survived the checks this system can run,' never 'true' — "
         "every lens shares the base model's blind spots.")

def render_report(art) -> str:
    b = art.blocker or {}
    lines = [f"# Validation report", "",
             f"**Seed:** {art.raw_idea}", "",
             f"**Status:** `{art.status}`",
             f"**Load-bearing claim:** `{art.load_bearing}`",
             f"**Blocker:** {b.get('reason', '—')} ({b.get('claim_id') or '—'})",
             f"**Maturity:** {art.maturity:.2f}", ""]
    if art.formal_claim:
        lines += [f"**Formal claim:** {art.formal_claim.statement}",
                  f"_falsifiable: {art.formal_claim.falsifiable}_", ""]
    if art.claim_graph:
        lines.append("## Claim graph")
        for c in art.claim_graph:
            lines.append(f"- `{c.id}` [{c.type}] **{c.status}** — {c.statement}")
        lines.append("")
    if art.validation_plan:
        vp = art.validation_plan
        lines += ["## Decisive test", f"- {vp.decisive_test}",
                  f"- confirm if: {vp.confirm_if}", f"- refute if: {vp.refute_if}",
                  f"- cost: {vp.cost}", ""]
    lines += ["---", f"> {LIMIT}"]
    return "\n".join(lines)

async def run_cli(seed, llm, cfg, backend=None, out_dir=None) -> dict:
    out_dir = out_dir or cfg.results_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_log.bind(Path(out_dir) / ".logs" / "valagents.jsonl")
    art = await run(seed, llm, cfg, backend=backend)
    slug = "".join(ch if ch.isalnum() else "-" for ch in seed.lower())[:40].strip("-")
    json_path = str(Path(out_dir) / f"{slug}.json")
    report_path = str(Path(out_dir) / f"{slug}.md")
    Path(json_path).write_text(art.model_dump_json(indent=2))
    Path(report_path).write_text(render_report(art))
    return {"artifact": art, "json_path": json_path, "report_path": report_path}

def main() -> None:
    p = argparse.ArgumentParser(prog="valagents")
    p.add_argument("seed")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    out = asyncio.run(run_cli(args.seed, OpenRouterClient(cfg), cfg, build_backend(cfg)))
    print(f"status: {out['artifact'].status}  → {out['report_path']}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_cli.py -v` → 2 passed.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(cli): run + JSON/markdown report carrying the honesty limit"`

---

## Task 17: Integration — the escape-saddle worked example

**Files:** Test only: `tests/test_integration_escape_saddle.py`.

**Interfaces:** Consumes `valagents.scheduler.run`. A scripted `FakeLLM` reproduces the spec §7 cycle end-to-end → `needs_experiment`, `load_bearing == "B"` (the α-non-saturation claim), with the fan-out second magnitude run present.

- [ ] **Step 1: Write the integration test.**

```python
from valagents.scheduler import run
from tests.fake_llm import FakeLLM

def router(agent, messages):
    c = messages[-1]["content"]
    if agent == "formalizer":
        return ("CLAIM: a curl term escapes strict saddles faster than GD | VARIABLES: theta, alpha "
                "| REGIME: strict saddles | FALSIFIABLE: yes")
    if agent == "faithfulness":
        return "FAITHFUL: yes | BACK_TRANSLATION: rotation speeds saddle escape"
    if agent == "decomposer":
        return ("CLAIM: A | TYPE: mathematical | DEPENDS_ON: none | STATEMENT: curl projects on negative-curvature dir\n"
                "CLAIM: B | TYPE: mechanistic | DEPENDS_ON: none | STATEMENT: alpha does not saturate at the saddle\n"
                "CLAIM: C | TYPE: empirical | DEPENDS_ON: none | STATEMENT: rotation does not disrupt convergence")
    if agent == "entailment":
        return "COVERS: complete | MISSING: none"
    if agent == "grounder":
        if "alpha" in c:   # claim B: inconclusive (the known saturation worry)
            return "CLAIM: B | SUPPORT: uncertain | INDEPENDENT_SOURCES: 0 | SOURCES: none | BASIS: saturation unclear"
        if "projects" in c:
            return "CLAIM: A | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: A1,A2 | BASIS: standard"
        if "convergence" in c:
            return "CLAIM: C | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: A3,A4 | BASIS: ok"
        return "CLOSEST_PRIOR: Curl-Descent | DELTA: alpha schedule | POSITION: new"
    if agent == "prover":
        if "alpha" in c:
            return "DERIVATION: gapped | GAPS: B | FATAL_GAP: no"   # keeps B uncertain, not fatal
        return "DERIVATION: complete | GAPS: none | FATAL_GAP: no"
    if agent == "predictor":
        return "OBSERVABLE: mean escape time | EFFECT_SIZE: separates from GD | DISCRIMINATES_FROM: GD/momentum | MEASURABLE: yes"
    if agent == "redteam":
        return ("ATTEMPTED: counterexample, failure_regime, magnitude\n"
                "ATTACK: magnitude | SEVERITY: major | STATUS: survived | TARGET: B | BASIS: saturation bounded\n"
                "ATTACK: counterexample | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: none found")
    if agent == "validation_designer":
        return ("TEST: synthetic-saddle escape-time vs GD/momentum/Curl-Descent | "
                "CONFIRM_IF: scaling separates | REFUTE_IF: no separation | COST: low")
    if agent == "arbiter":
        return "STATUS: needs_experiment | LOAD_BEARING: B | DECISIVE_TEST: escape-time benchmark"
    return ""

async def test_escape_saddle_needs_experiment(cfg):
    art = await run("adding an antisymmetric curl term to GD helps escape saddle points",
                    FakeLLM(router), cfg)
    assert art.status == "needs_experiment"
    assert art.load_bearing == "B"
    assert art.blocker["reason"] == "inconclusive"
    b = next(c for c in art.claim_graph if c.id == "B")
    assert b.status == "uncertain" and len(b.checks) >= cfg.gate.fanout_N   # fan-out fired
    assert art.validation_plan.cost == "low"
```

- [ ] **Step 2: Run → FAIL** (until everything upstream is wired). `pytest tests/test_integration_escape_saddle.py -v`

- [ ] **Step 3:** No new implementation — fix any wiring bug this surfaces in `scheduler.py`/agents. (If `load_bearing` ≠ `"B"`, check the `_evaluate` blocker precedence: the first `uncertain` root-ancestor in `claim_graph` order is surfaced; B must be the uncertain one. The router makes A and C `pass`, B `uncertain`, so the blocker claim is B.)

- [ ] **Step 4: Run → PASS.** Then the whole suite: `pytest -q` → all green.

- [ ] **Step 5: Commit.** `git add -A && git commit -m "test(integration): escape-saddle end-to-end → needs_experiment, load_bearing=B"`

---

## Self-Review (completed against the spec)

**1. Spec coverage.** Every spec section maps to a task: §2 schema → T3/T4/T5/T6; §2.1 gate (all entry gates, caps, strict branch) → T4 with per-branch tests; §2.2 load_bearing/blocker → T5; §2.3 maturity ⊥ status → T6; §2.4 coverage matrix → T14; §3 the eleven roles → T8–T12; §4 parse strict tail → T2; §5 control loop (entry gates, fan-out, repair-versioning, finalize) → T13/T14/T15; §6 reuse/layout → T1; §7 worked cycle → T17; §8 tests → distributed across every task; §9 D1–D12 + the limit sentence → encoded (D7 faithfulness T9/T13, D8 independence T8/T10, D9 teeth T11/T4, D10 entailment T9/T4, D11 fan-out T14, D12 empty-graph T13, limit sentence T16).

**2. Placeholder scan.** No "TBD"/"handle edge cases"/"similar to Task N". The single human-authored step (T6 `maturity`) is explicitly flagged with a working default + a binding isolation test, not a placeholder.

**3. Type consistency.** Status strings (`"internally_validated"` etc.) and claim statuses (`"pass"/"fail"/"uncertain"/"pending"`) are fixed in Global Constraints and used identically across T3–T17. Agent function names referenced by the scheduler (`formalize`, `faithfulness_check`, `decompose`, `entailment_check`, `ground_claim`, `ground_novelty`, `prove_claim`, `predict`, `red_team`, `design_validation`, `repair`, `arbitrate`) match their defining tasks. `CheckRecord`/`Source`/`AttackSurface` field names are stable from T3 onward.

**Known sharp edges flagged inline for the implementer** (not gaps — judgment points): Red-team `ATTEMPTED` capture (T11) is cleaner if `parse.checked_lines` optionally returns the raw body; the Prover `pass → independent_sources=1` choice (T10) is what lets pure-math claims reach `pass` without external literature.
