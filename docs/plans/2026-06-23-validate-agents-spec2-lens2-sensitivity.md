# validate-agents Spec 2 Lens 2 — `sensitivity_ratio` Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the first magnitude/detectability check — `sensitivity_ratio` — as a preregistered, sourced, code-computed comparison that becomes a Red-team **attack** (inert → `challenged`), reusing the symbolic lens's sandbox and discipline.

**Architecture:** Extend `ComputationPlan` with `kind="magnitude"` + the sourced magnitude fields (all three `comparison_kind`s named, only `sensitivity_ratio` wired this slice). The existing subprocess runner dispatches on `kind`; the magnitude branch parses sourced numeric quantities with the *same* restricted parser, then computes `abs(predicted−baseline)/sensitivity ≥ threshold` with numpy and judges **in code**. A new `design_magnitude` agent emits the structured plan only (no verdict); `run_magnitude_checks` in the scheduler turns the executed verdict into an `Attack` (survived/landed), augmenting the reasoned Red-team with fail-closed fallback. `_evaluate()` is unchanged.

**Tech Stack:** Python 3.11+, Pydantic v2, SymPy (restricted parse → float), **numpy** (new pinned dep; float arithmetic only), the Task-2 subprocess sandbox, `pytest`. Tests run **real numpy** (deterministic, offline); only the Magnitude-Designer (LLM) is faked.

**Source spec:** `docs/2026-06-23-validate-agents-spec2-lens2-design.md`. Section refs (e.g. "§3") point there.

## Global Constraints

- **F1 — structured plan, NO arbitrary code.** The Designer emits only structured fields. The runner parses each quantity with `sympy.parsing.sympy_parser.parse_expr` over a restricted `local_dict`/`global_dict` with `__builtins__={}`, rejects `"__"` pre-parse, then `float(expr.evalf())`. numpy does float arithmetic, never executes plan strings. No `sympify`/`eval`/`exec`.
- **F3 — code judges, no LLM after execution.** `run_plan`/`verdict_to_attack` take no `llm`; the verdict is `abs(predicted−baseline)/sensitivity ≥ threshold` in code. Tests assert no `llm` param.
- **Fail-closed on missing source/quantity (the anti-laundering core).** A `sensitivity_ratio` plan missing ANY of `predicted_effect`, `baseline_or_null`, `sensitivity`, `sensitivity_source`, `threshold` (empty/whitespace counts as missing) → `ComputationResult(ok=False)` → `uncertain`. **Never pass, never a landed attack.** `sensitivity_source` is mandatory and surfaced in the attack basis.
- **L2-D9 — `"magnitude"` is added to `AttackSurface.attempted` ONLY on a decisive executed verdict (`survived`/`landed`).** A `uncertain`/missing-source/can't-compute run must NOT mark it — otherwise a fail-closed non-execution would satisfy `_thin_attack_surface()`.
- **F2 — augment + fallback.** `run_magnitude_checks` runs after the reasoned `red_team()`. A `uncertain` magnitude result → no attack, no mark; the reasoned Red-team magnitude stands. A decisive result adds the executed attack.
- **Attack mapping:** `matched=="confirm"` (detectable) → `Attack(type="magnitude", status="survived")`; `matched=="refute"` (inert) → `Attack(type="magnitude", status="landed", severity="fatal" if the prediction is discriminating else "major")`.
- **Gate untouched:** `_evaluate()` must reference neither `"magnitude"` nor `"comparison_kind"` (test asserts).
- **Loud caveat:** the executor verifies the predicted effect clears the *preregistered, sourced* sensitivity — not that the sensitivity is literature-correct. The attack basis shows computed ratio, threshold, sensitivity, and `sensitivity_source`.
- **Tests:** deterministic, no network. Run from the repo root in the deps env: `conda run -n cosci-reproduce python -m pytest …`. numpy must be installed: `conda run -n cosci-reproduce pip install numpy`.
- **Commits:** plain message, **no `Co-Authored-By`/`Claude-Session` trailer**. Stage only files you changed.

---

## File Structure

```
valagents/
  computation.py     # MODIFY: ComputationPlan += magnitude fields (3 comparison_kinds named); default symbolic fields; add verdict_to_attack()
  sandbox/runner.py  # MODIFY: _run dispatches on kind; add _run_magnitude (sensitivity_ratio); reuse restricted parser
  prompts.py         # MODIFY: add MAGNITUDE_DESIGNER (sensitivity_ratio)
  agents/magnitude_designer.py   # NEW: design_magnitude() -> ComputationPlan | None (no verdict)
  scheduler.py       # MODIFY: run_magnitude_checks(); call it in _whole_artifact_lenses after attacks/surface set
requirements.txt     # MODIFY: add numpy
tests/
  test_magnitude_model.py   test_magnitude_executor.py   test_magnitude_integration.py
```

---

## Task 1: Data model — magnitude fields + `verdict_to_attack`

**Files:**
- Modify: `valagents/computation.py`
- Test: `tests/test_magnitude_model.py`

**Interfaces:**
- Consumes: `ComputationVerdict`, `ComputationResult` (existing); `Attack` (artifact.py).
- Produces: `ComputationPlan` with `kind: Literal["symbolic","magnitude"]`, `comparison_kind: Literal["sensitivity_ratio","bound_check","discriminating_margin"] | None`, and the magnitude fields (all defaulted). `verdict_to_attack(v: ComputationVerdict, target_claim_id: str | None, discriminating: bool, tick: int = 0) -> Attack` (no `llm`).

- [ ] **Step 1: Write failing tests** `tests/test_magnitude_model.py`:

```python
import inspect
from valagents.computation import (ComputationPlan, ComputationResult,
                                   ComputationVerdict, verdict_to_attack)

def test_magnitude_plan_omits_symbolic_fields():
    p = ComputationPlan(kind="magnitude", comparison_kind="sensitivity_ratio",
                        predicted_effect="1e-9", baseline_or_null="0", sensitivity="1e-12",
                        sensitivity_source="arXiv:1234", threshold="3")
    assert p.kind == "magnitude" and p.comparison_kind == "sensitivity_ratio"
    assert p.expression == "" and p.expected == ""   # symbolic fields now defaulted

def test_symbolic_plan_still_constructs():            # backward-compat
    p = ComputationPlan(expression="1/x", limit_variable="x", limit_point="oo", expected="0")
    assert p.kind == "symbolic"

def _verdict(matched, discriminating):
    p = ComputationPlan(kind="magnitude", comparison_kind="sensitivity_ratio",
                        predicted_effect="1e-9", baseline_or_null="0", sensitivity="1e-12",
                        sensitivity_source="arXiv:1234", threshold="3", discriminating=discriminating)
    r = ComputationResult(ok=True, computed="ratio=1000", matched=matched)
    v = ComputationVerdict(verdict=("pass" if matched == "confirm" else "fail"), measured="ratio=1000", plan=p, result=r)
    return v

def test_confirm_is_survived_attack():
    a = verdict_to_attack(_verdict("confirm", True), target_claim_id="c1", discriminating=True)
    assert a.type == "magnitude" and a.status == "survived"

def test_refute_discriminating_is_landed_fatal():
    a = verdict_to_attack(_verdict("refute", True), target_claim_id="c1", discriminating=True)
    assert a.status == "landed" and a.severity == "fatal" and a.target_claim_id == "c1"
    assert "sensitivity" in a.basis and "arXiv:1234" in a.basis   # loud source

def test_refute_nondiscriminating_is_landed_major():
    a = verdict_to_attack(_verdict("refute", False), target_claim_id=None, discriminating=False)
    assert a.status == "landed" and a.severity == "major"

def test_verdict_to_attack_takes_no_llm():
    assert "llm" not in inspect.signature(verdict_to_attack).parameters
```

- [ ] **Step 2: Run → FAIL.** `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_model.py -v`

- [ ] **Step 3: Implement.** In `valagents/computation.py`, replace the `ComputationPlan` class body so the symbolic fields are defaulted and the magnitude fields are added:

```python
class ComputationPlan(BaseModel):
    kind: Literal["symbolic", "magnitude"] = "symbolic"
    # --- symbolic (now defaulted so magnitude plans omit them) ---
    expression: str = ""
    variables: list[str] = []
    limit_variable: str = ""
    limit_point: str = ""
    expected: str = ""
    expected_source: str = ""
    # --- magnitude ---
    comparison_kind: Literal["sensitivity_ratio", "bound_check", "discriminating_margin"] | None = None
    predicted_effect: str = ""
    baseline_or_null: str = ""
    sensitivity: str = ""
    sensitivity_source: str = ""
    bound: str = ""
    bound_source: str = ""
    closest_prior_effect: str = ""
    uncertainty: str = ""
    threshold: str = ""
    target_claim_id: str | None = None
    discriminating: bool = False
    # --- criterion / glosses (shared) ---
    criterion: Literal["symbolic_equality", "magnitude"] = "symbolic_equality"
    confirm_if: str = ""
    refute_if: str = ""
```

Append `verdict_to_attack` to `valagents/computation.py` (import `Attack` inside, mirroring `verdict_to_check`):

```python
def verdict_to_attack(v: "ComputationVerdict", target_claim_id, discriminating: bool, tick: int = 0):
    """Map an executed magnitude ComputationVerdict to a Red-team Attack. No LLM (F3).
    Call ONLY on a decisive verdict (matched in {'confirm','refute'})."""
    from valagents.artifact import Attack
    if v.result.matched == "confirm":
        status, severity = "survived", "minor"
    else:  # "refute" — inert / non-discriminating
        status, severity = "landed", ("fatal" if discriminating else "major")
    basis = (f"{v.plan.comparison_kind}: computed = {v.measured or '?'}; "
             f"sensitivity = {v.plan.sensitivity or 'n/a'} "
             f"(source: {v.plan.sensitivity_source or 'n/a'}); threshold = {v.plan.threshold or 'n/a'}")
    return Attack(type="magnitude", severity=severity, status=status,
                  target_claim_id=target_claim_id, basis=basis)
```

- [ ] **Step 4: Run → PASS.** `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_model.py -v`, then full suite `conda run -n cosci-reproduce python -m pytest -q`. (The defaulting of symbolic fields must not break existing symbolic tests.)

- [ ] **Step 5: Commit:**

```bash
git add valagents/computation.py tests/test_magnitude_model.py
git commit -m "feat(computation): magnitude plan fields (3 comparison_kinds) + verdict_to_attack"
```

---

## Task 2: Magnitude runner branch + executor (`sensitivity_ratio`)

**Files:**
- Modify: `valagents/sandbox/runner.py`, `requirements.txt`
- Test: `tests/test_magnitude_executor.py`

**Interfaces:**
- Consumes: `run_plan(plan, cfg, artifacts_dir=None)` (Task-2 of the symbolic slice, unchanged); `ComputationPlan` (Task 1).
- Produces: the runner now handles `kind="magnitude"` / `comparison_kind="sensitivity_ratio"` → `{ok, computed, matched}`; fail-closed on missing quantity/source or compute error.

- [ ] **Step 1: Add numpy.** Append `numpy` to `requirements.txt`. `conda run -n cosci-reproduce pip install numpy`. Verify: `conda run -n cosci-reproduce python -c "import numpy; print(numpy.__version__)"`.

- [ ] **Step 2: Write failing tests** `tests/test_magnitude_executor.py` (real numpy):

```python
import inspect
from valagents.computation import ComputationPlan
from valagents.config import Config
from valagents.sandbox.executor import run_plan

def cfg():
    return Config(default_model="fake")

def mplan(**kw):
    base = dict(kind="magnitude", comparison_kind="sensitivity_ratio",
                predicted_effect="1e-9", baseline_or_null="0", sensitivity="1e-12",
                sensitivity_source="arXiv:1234", threshold="3")
    base.update(kw)
    return ComputationPlan(**base)

def test_detectable_is_confirm():
    v = run_plan(mplan(), cfg())                 # ratio = 1e-9/1e-12 = 1000 >= 3
    assert v.verdict == "pass" and v.result.matched == "confirm" and v.result.ok

def test_inert_is_refute():
    v = run_plan(mplan(predicted_effect="1e-18"), cfg())   # ratio = 1e-6 < 3
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_missing_sensitivity_source_is_uncertain():       # the anti-laundering core
    v = run_plan(mplan(sensitivity_source=""), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_missing_threshold_is_uncertain():
    v = run_plan(mplan(threshold=""), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_zero_sensitivity_is_uncertain():
    v = run_plan(mplan(sensitivity="0"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_dunder_quantity_is_uncertain_not_executed():
    v = run_plan(mplan(predicted_effect="x.__class__"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_run_plan_takes_no_llm():
    assert "llm" not in inspect.signature(run_plan).parameters
```

- [ ] **Step 3: Implement.** In `valagents/sandbox/runner.py`, rename the existing `_run` body to `_run_symbolic` (keep it identical), and add a dispatcher `_run` plus `_run_magnitude`:

```python
def _run(plan: dict) -> dict:
    if plan.get("kind") == "magnitude":
        return _run_magnitude(plan)
    return _run_symbolic(plan)

_MAG_REQUIRED = {
    "sensitivity_ratio": ["predicted_effect", "baseline_or_null", "sensitivity",
                          "sensitivity_source", "threshold"],
}

def _parse_number(s, glob) -> float:
    from sympy.parsing.sympy_parser import parse_expr
    if "__" in str(s):
        raise ValueError("rejected: '__' not allowed")
    return float(parse_expr(str(s), local_dict={}, global_dict=glob, evaluate=True).evalf())

def _run_magnitude(plan: dict) -> dict:
    import sympy
    import numpy as np
    ck = plan.get("comparison_kind")
    required = _MAG_REQUIRED.get(ck)
    if required is None:
        return {"ok": False, "matched": "neither", "error": f"unsupported comparison_kind: {ck}"}
    for field in required:                        # fail-closed: quantity AND source must be present
        if not str(plan.get(field, "")).strip():
            return {"ok": False, "matched": "neither", "error": f"missing required field: {field}"}
    glob = {n: getattr(sympy, n) for n in _ALLOWED}
    glob["__builtins__"] = {}
    try:
        if ck == "sensitivity_ratio":
            predicted = _parse_number(plan["predicted_effect"], glob)
            baseline = _parse_number(plan["baseline_or_null"], glob)
            sensitivity = _parse_number(plan["sensitivity"], glob)
            threshold = _parse_number(plan["threshold"], glob)
            if sensitivity == 0:
                return {"ok": False, "matched": "neither", "error": "sensitivity is zero"}
            ratio = float(np.abs(predicted - baseline) / sensitivity)
            detectable = ratio >= threshold
            return {"ok": True, "computed": f"ratio={ratio:.6g}",
                    "matched": "confirm" if detectable else "refute"}
    except Exception as e:
        return {"ok": False, "matched": "neither", "error": f"{type(e).__name__}: {e}"}
    return {"ok": False, "matched": "neither", "error": "no computation performed"}
```

(Note: `_run_magnitude` does its own `"__"` guard inside `_parse_number`; the symbolic `"__"` guard stays in `_run_symbolic`. `sensitivity_source` is checked for presence but never parsed — it is a string source, not a number.)

- [ ] **Step 4: Run → PASS.** `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_executor.py -v`, then full suite (symbolic executor tests must stay green).

- [ ] **Step 5: Commit:**

```bash
git add valagents/sandbox/runner.py requirements.txt tests/test_magnitude_executor.py
git commit -m "feat(sandbox): magnitude runner branch (sensitivity_ratio) — sourced numpy comparison, fail-closed"
```

---

## Task 3: Magnitude-Designer + scheduler wiring (attack path)

**Files:**
- Create: `valagents/agents/magnitude_designer.py`, `tests/test_magnitude_integration.py`
- Modify: `valagents/prompts.py` (add `MAGNITUDE_DESIGNER`), `valagents/scheduler.py` (`run_magnitude_checks` + call it in `_whole_artifact_lenses`)

**Interfaces:**
- Consumes: `checked` (parse.py), `build_messages` (base.py), `ComputationPlan`/`verdict_to_attack` (Task 1), `run_plan` (Task 2), `Prediction`/`Attack`/`AttackSurface` (artifact.py).
- Produces: `design_magnitude(prediction, art, llm, cfg) -> ComputationPlan | None` (plan only — no verdict); `run_magnitude_checks(store, llm, cfg, tick=0) -> None`.

- [ ] **Step 1: Write failing tests** `tests/test_magnitude_integration.py`:

```python
import inspect
from valagents.scheduler import run_magnitude_checks
from valagents.agents.magnitude_designer import design_magnitude
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, Prediction, AttackSurface
from valagents.config import Config
from tests.fake_llm import FakeLLM

def cfg():
    return Config(default_model="fake")

def store_with_prediction(discriminates=True):
    art = IdeaArtifact(raw_idea="seed", formal_claim=FormalClaim(statement="x", falsifiable=True),
                       predictions=[Prediction(observable="shift", effect_size="1e-9",
                                               discriminates_from=("GR" if discriminates else ""), measurable=True)],
                       attack_surface=AttackSurface(attempted=["counterexample"]))
    return ArtifactStore(art)

INERT = ("COMPARISON_KIND: sensitivity_ratio | PREDICTED_EFFECT: 1e-18 | BASELINE_OR_NULL: 0 "
         "| SENSITIVITY: 1e-12 | SENSITIVITY_SOURCE: arXiv:1234 | THRESHOLD: 3 "
         "| CONFIRM_IF: ratio>=3 | REFUTE_IF: ratio<3")
DETECT = INERT.replace("PREDICTED_EFFECT: 1e-18", "PREDICTED_EFFECT: 1e-9")

def router(body):
    return FakeLLM(lambda a, m: body if a == "magnitude_designer" else "")

async def test_designer_emits_plan_only():
    plan = await design_magnitude(store_with_prediction().current.predictions[0],
                                  store_with_prediction().current, router(INERT), cfg())
    assert plan is not None and plan.kind == "magnitude" and plan.comparison_kind == "sensitivity_ratio"
    assert plan.discriminating is True            # prediction discriminates_from set
    assert "ComputationVerdict" not in inspect.getsource(design_magnitude)   # F1: no verdict

async def test_inert_lands_fatal_magnitude_attack_and_marks_attempted():
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(INERT), cfg())
    mags = [a for a in s.current.attacks if a.type == "magnitude"]
    assert mags and mags[0].status == "landed" and mags[0].severity == "fatal"
    assert "magnitude" in s.current.attack_surface.attempted

async def test_detectable_survives_and_marks_attempted():
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(DETECT), cfg())
    mags = [a for a in s.current.attacks if a.type == "magnitude"]
    assert mags and mags[0].status == "survived"
    assert "magnitude" in s.current.attack_surface.attempted

async def test_uncertain_adds_no_attack_and_does_not_mark_attempted(monkeypatch):
    # L2-D9: a fail-closed magnitude run must NOT mark "magnitude" attempted
    import valagents.sandbox.executor as ex
    from valagents.computation import ComputationVerdict, ComputationResult, ComputationPlan
    def fake(plan, cfg, artifacts_dir=None):
        return ComputationVerdict(verdict="uncertain", measured="", plan=plan,
                                  result=ComputationResult(ok=False, error="missing source"))
    monkeypatch.setattr(ex, "run_plan", fake)
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(INERT), cfg())
    assert not [a for a in s.current.attacks if a.type == "magnitude"]   # no attack
    assert "magnitude" not in s.current.attack_surface.attempted          # NOT marked (teeth not laundered)

async def test_evaluate_ignores_magnitude_fields():
    assert "magnitude" not in inspect.getsource(IdeaArtifact._evaluate)
    assert "comparison_kind" not in inspect.getsource(IdeaArtifact._evaluate)
```

- [ ] **Step 2: Run → FAIL.** `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_integration.py -v`

- [ ] **Step 3: Implement.** Add to `valagents/prompts.py`:

```python
MAGNITUDE_DESIGNER = """You DESIGN a numeric detectability check; you do NOT run or judge it — code does that, \
and you will never see the result. Given a measurable prediction, produce the structured plan to test whether \
the predicted effect is detectable above an established sensitivity.

FORMAL CLAIM: {formal}
PREDICTION: {observable} (effect size: {effect_size}; discriminates from: {discriminates_from})

Give the predicted effect as a NUMBER (SI/natural units), the null/baseline it is measured against, the \
established measurement SENSITIVITY with its SOURCE (where the sensitivity comes from — instrument/paper), \
and the detection THRESHOLD (how many times the sensitivity the effect must clear). Do not output code; do not \
invent a sensitivity without a source.
End with exactly:
COMPARISON_KIND: sensitivity_ratio | PREDICTED_EFFECT: <number> | BASELINE_OR_NULL: <number> | SENSITIVITY: <number> | SENSITIVITY_SOURCE: <where it comes from> | THRESHOLD: <number> | CONFIRM_IF: <…> | REFUTE_IF: <…>"""
```

Create `valagents/agents/magnitude_designer.py`:

```python
"""Magnitude-Designer: emits a structured sensitivity_ratio ComputationPlan for a measurable prediction.
It DESIGNS the check only — returns no verdict and never sees the execution result (F1/F3)."""
from __future__ import annotations
from valagents.computation import ComputationPlan
from valagents.parse import checked
from valagents.prompts import MAGNITUDE_DESIGNER
from valagents.agents.base import build_messages

_KEYS = ["COMPARISON_KIND", "PREDICTED_EFFECT", "BASELINE_OR_NULL", "SENSITIVITY",
         "SENSITIVITY_SOURCE", "THRESHOLD", "CONFIRM_IF", "REFUTE_IF"]

async def design_magnitude(prediction, art, llm, cfg) -> ComputationPlan | None:
    user = MAGNITUDE_DESIGNER.format(
        formal=art.formal_claim.statement if art.formal_claim else "",
        observable=prediction.observable, effect_size=prediction.effect_size,
        discriminates_from=prediction.discriminates_from or "(none)")
    tail = await checked("magnitude_designer", build_messages("You design detectability checks.", user),
                         _KEYS, llm=llm)
    if tail is None or tail["comparison_kind"].strip().lower() != "sensitivity_ratio":
        return None
    try:
        return ComputationPlan(
            kind="magnitude", comparison_kind="sensitivity_ratio",
            predicted_effect=tail["predicted_effect"], baseline_or_null=tail["baseline_or_null"],
            sensitivity=tail["sensitivity"], sensitivity_source=tail["sensitivity_source"],
            threshold=tail["threshold"], confirm_if=tail["confirm_if"], refute_if=tail["refute_if"],
            target_claim_id=art.load_bearing, discriminating=bool(prediction.discriminates_from),
            criterion="magnitude")
    except Exception:
        return None
```

Add `run_magnitude_checks` to `valagents/scheduler.py` (and `from valagents.agents.magnitude_designer import design_magnitude` to the imports):

```python
async def run_magnitude_checks(store, llm, cfg, tick: int = 0) -> None:
    art = store.current
    for p in art.predictions:
        if not p.measurable:
            continue
        plan = await design_magnitude(p, art, llm, cfg)
        if plan is None:
            continue
        from valagents.sandbox.executor import run_plan
        from valagents.computation import verdict_to_attack
        adir = f"{cfg.results_dir}/computations/magnitude" if getattr(cfg, "results_dir", None) else None
        verdict = run_plan(plan, cfg, artifacts_dir=adir)
        store.record({"event": "magnitude_executed", "verdict": verdict.verdict, "computed": verdict.measured})
        if verdict.verdict == "uncertain":
            continue                                  # FAIL-CLOSED: no attack, no attempted-mark (L2-D9), F2
        # attack path (sensitivity_ratio): decisive verdict -> Attack + mark "magnitude" attempted
        attack = verdict_to_attack(verdict, plan.target_claim_id, plan.discriminating, tick=tick)
        art.attacks = art.attacks + [attack]
        if art.attack_surface is not None and "magnitude" not in art.attack_surface.attempted:
            art.attack_surface.attempted = art.attack_surface.attempted + ["magnitude"]
        tick += 1
```

In `_whole_artifact_lenses`, immediately AFTER the existing `store.set("attacks", attacks)` / `store.set("attack_surface", surface)` lines, add:

```python
    await run_magnitude_checks(store, llm, cfg, tick=tick + 500)
```

- [ ] **Step 4: Run → PASS.** `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_integration.py -v`, then full suite. If a pre-existing scheduler/integration test now calls `magnitude_designer` and its scripted `FakeLLM` lacks a response → add `"magnitude_designer": ""` (an empty/unparseable body → `design_magnitude` returns `None` → no-op) to that test's router; this is a necessary router entry, not a weakened assertion.

- [ ] **Step 5: Commit:**

```bash
git add valagents/agents/magnitude_designer.py valagents/prompts.py valagents/scheduler.py tests/test_magnitude_integration.py
git commit -m "feat(magnitude): designer + run_magnitude_checks (sensitivity_ratio attack path, fail-closed, L2-D9)"
```

---

## Self-Review (completed against the spec)

**Spec coverage:** §2 preregistration+loud-source → Task 2 fail-closed required-field check (incl. `sensitivity_source`) + Task 1 `verdict_to_attack` surfaces source in basis; §3 `sensitivity_ratio` attack path (confirm→survived, refute→landed fatal/major by `discriminating`) → Tasks 1+3; F1 no-code → Task 2 restricted `parse_expr`+`__builtins__={}`+`"__"` guard (+ `test_dunder_quantity_is_uncertain`); F3 code-judges → `run_plan`/`verdict_to_attack` no-`llm` tests + the numpy comparison; §5 hook (augment after red_team, F2 fallback) → Task 3 `run_magnitude_checks`; **L2-D9 teeth-not-laundered** → `test_uncertain_adds_no_attack_and_does_not_mark_attempted`; §6 numpy float-only → Task 2; §8 gate purity → `test_evaluate_ignores_magnitude_fields`. (Design slices 4–5: `bound_check` claim path and `discriminating_margin` — deliberately a separate follow-on plan; `comparison_kind` enum already names them so no reshape.)

**Placeholder scan:** none — every step has real code and exact commands.

**Type consistency:** `ComputationPlan` magnitude fields are identical across Tasks 1–3; `verdict_to_attack(v, target_claim_id, discriminating, tick)`, `design_magnitude(prediction, art, llm, cfg)`, `run_magnitude_checks(store, llm, cfg, tick)`, and `run_plan(plan, cfg, artifacts_dir)` signatures match every call site; `Attack(type="magnitude", severity=…, status=…, target_claim_id=…, basis=…)` matches the artifact model; the runner `_run`→`_run_magnitude`/`_run_symbolic` dispatch is internal.

**Flagged for the implementer:** Task 1 changes four `ComputationPlan` fields from required to defaulted — confirm existing symbolic tests still pass (they pass values explicitly, so they're unaffected). In Task 3, `run_magnitude_checks` mutates `art.attacks`/`art.attack_surface.attempted` on `store.current` in place (single-worker, consistent with how `_whole_artifact_lenses` sets them); keep that pattern.
