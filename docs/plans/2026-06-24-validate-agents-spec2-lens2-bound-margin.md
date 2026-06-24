# Spec 2 Lens 2 — bound_check + discriminating_margin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the magnitude/detectability executor by adding the remaining two comparison kinds — `bound_check` (claim/refute path → `refuted`) and `discriminating_margin` (attack path → `challenged`) — on top of the shipped `sensitivity_ratio` slice.

**Architecture:** Two new branches in the existing subprocess magnitude runner (restricted-parse → float → numpy comparison), a two-phase Magnitude-Designer that routes one structured plan per prediction to the right comparison kind, and scheduler wiring that sends a `bound_check` verdict down the *claim* path (inject a load-bearing mathematical claim; violation → `fail` → `refuted`) and a `discriminating_margin` verdict down the existing *attack* path (below-margin → landed attack → `challenged`). The computed gate `_evaluate()` is **not touched** — both paths flow through `claim.status` and `attacks`, which it already reads.

**Tech Stack:** Python, Pydantic v2, SymPy restricted `parse_expr`, numpy, subprocess sandbox, pytest (`conda run -n cosci-reproduce python -m pytest`).

## Global Constraints

Copied verbatim from `docs/2026-06-23-validate-agents-spec2-lens2-design.md` (§2, §3, decision log). Every task's requirements implicitly include these:

- **F1 — no arbitrary code.** Quantities are parsed only with the restricted `parse_expr` (declared-symbol `local_dict`, whitelisted `global_dict`, `__builtins__={}`, `"__"`-rejected) then `evalf()` to a float; numpy does float arithmetic only, never plan strings. No `eval`/`exec`/`sympify` of model text. The designer emits a **structured plan only** — never a verdict, never code, never sees the result.
- **F3 — code judges, never the LLM after execution.** Path is `design_magnitude(llm)` → `run_plan` (no `llm`) → `verdict_to_attack` / `verdict_to_check` (no `llm`).
- **Fail-closed (L2-D2).** A missing required quantity OR a missing source → `uncertain`; never `pass`, never a landed attack, never a refute. Source strings are **presence-checked, never parsed**.
- **Sourced thresholds, loud.** `bound_source` is mandatory for `bound_check`; `closest_prior_source` is mandatory for `discriminating_margin` (**L2-D10**). Both surface in the verdict basis.
- **L2-D9 — anti-laundering of teeth.** `"magnitude"` is added to `AttackSurface.attempted` **only on a decisive executed attack-path verdict** (`survived`/`landed`). Never on a fail-closed `uncertain` run, and **never on the `bound_check` claim path** (a compliance check is not an adversarial detectability probe).
- **L2-D11 — bound-claim idempotence.** `run_magnitude_checks` re-runs each repair iteration; the `bound_check` path clears any prior `origin="bound_check"` claims before re-injecting, so bound claims never accumulate.
- **Gate purity (L2-D6).** `IdeaArtifact._evaluate()` references neither `"magnitude"` nor `"comparison_kind"`. Verdicts stay computed `@computed_field` properties with no setter.

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `valagents/sandbox/runner.py` | `_MAG_REQUIRED` + `_run_magnitude` branches for `bound_check` and `discriminating_margin` | 1, 3 |
| `valagents/computation.py` | `verdict_to_check` kind-aware (bound); `verdict_to_attack` basis kind-aware (margin); new `closest_prior_source` field | 1, 3 |
| `valagents/agents/magnitude_designer.py` | two-phase parse routing one plan per prediction to the right kind | 2, 4 |
| `valagents/prompts.py` | `MAGNITUDE_DESIGNER` explains and routes the comparison kinds | 2, 4 |
| `valagents/scheduler.py` | `run_magnitude_checks` claim path (bound) + idempotence; attack path already handles margin | 2 |
| `tests/test_magnitude_executor.py` | real-numpy executor tests for both new kinds | 1, 3 |
| `tests/test_magnitude_model.py` | `verdict_to_check`/`verdict_to_attack` mapping tests | 1, 3 |
| `tests/test_magnitude_integration.py` | FakeLLM designer + real executor end-to-end | 2, 4 |

**Interfaces that already exist (consume verbatim, do not redefine):**
- `ComputationPlan(kind="magnitude", comparison_kind=..., predicted_effect, baseline_or_null, sensitivity, sensitivity_source, bound, bound_source, closest_prior_effect, uncertainty, threshold, target_claim_id, discriminating, criterion, confirm_if, refute_if)` — `valagents/computation.py`. All fields are `str` (default `""`) except `kind`/`comparison_kind`/`criterion` (Literals), `target_claim_id` (`str|None`), `discriminating` (`bool`).
- `run_plan(plan, cfg, artifacts_dir=None) -> ComputationVerdict` — `valagents/sandbox/executor.py`. Maps `ok=False→uncertain`, `confirm→pass`, `refute→fail`; `ComputationVerdict.measured == result.computed`.
- `verdict_to_attack(v, target_claim_id, discriminating, tick=0) -> Attack` and `verdict_to_check(v, tick=0) -> CheckRecord` — `valagents/computation.py`.
- `AtomicClaim(id, statement, type, role=..., load_bearing=True, checks=[], exhausted=False, origin="decomposed")` and the gate `_evaluate` rule "any load-bearing claim with `status=="fail"` → `REFUTED`" — `valagents/artifact.py`.
- `checked_body(agent, messages, required_keys, *, llm) -> (dict|None, str)`, `parse_tail(text, required_keys) -> dict`, `StrictTailError` — `valagents/parse.py`. `_row` requires **all** `required_keys` on one line.
- `store.add_check(claim_id, record)` and `store.record(dict)` — `valagents/store.py`. Test fixtures: `tests/test_magnitude_integration.py` (`store_with_prediction`, `router`), `tests/fake_llm.py` (`FakeLLM(lambda agent, messages: body)`).

**Test command (all tasks):** `conda run -n cosci-reproduce python -m pytest tests/ -q`

---

### Task 1: `bound_check` executor branch + `verdict_to_check` kind-awareness

**Files:**
- Modify: `valagents/sandbox/runner.py` (`_MAG_REQUIRED`, `_run_magnitude`)
- Modify: `valagents/computation.py` (`verdict_to_check`)
- Test: `tests/test_magnitude_executor.py`, `tests/test_magnitude_model.py`

**Interfaces:**
- Consumes: `run_plan`, `ComputationPlan`, `ComputationResult`, `ComputationVerdict` (existing).
- Produces: a `bound_check` magnitude branch returning `matched="confirm"` when `predicted_effect <= bound` else `"refute"`; `verdict_to_check` that, for a `bound_check` plan, surfaces `bound`/`bound_source` in the basis and as the independent `Source` locator.

- [ ] **Step 1: Write the failing executor tests**

Add to `tests/test_magnitude_executor.py`:

```python
def bplan(**kw):
    base = dict(kind="magnitude", comparison_kind="bound_check",
                predicted_effect="1e-3", bound="1e-2", bound_source="PDG2024")
    base.update(kw)
    return ComputationPlan(**base)

def test_bound_complies_is_confirm():
    v = run_plan(bplan(), cfg())                 # 1e-3 <= 1e-2 -> comply
    assert v.verdict == "pass" and v.result.matched == "confirm" and v.result.ok

def test_bound_violates_is_refute():
    v = run_plan(bplan(predicted_effect="1e-1"), cfg())   # 1e-1 > 1e-2 -> violate
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_bound_missing_source_is_uncertain():    # anti-laundering (L2-D2): no source -> never pass/fail
    v = run_plan(bplan(bound_source=""), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_bound_missing_bound_is_uncertain():
    v = run_plan(bplan(bound=""), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_bound_dunder_is_uncertain_not_executed():
    v = run_plan(bplan(predicted_effect="x.__class__"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_executor.py -q`
Expected: the 5 new tests FAIL (`bound_check` unsupported → `uncertain`, so the confirm/refute ones fail).

- [ ] **Step 3: Add the `bound_check` runner branch**

In `valagents/sandbox/runner.py`, extend `_MAG_REQUIRED`:

```python
_MAG_REQUIRED = {
    "sensitivity_ratio": ["predicted_effect", "baseline_or_null", "sensitivity",
                          "sensitivity_source", "threshold"],
    "bound_check": ["predicted_effect", "bound", "bound_source"],
}
```

In `_run_magnitude`, immediately **after** the `if ck == "sensitivity_ratio":` return block and still inside the `try`, add:

```python
        if ck == "bound_check":
            predicted = _parse_number(plan["predicted_effect"], glob)
            bound = _parse_number(plan["bound"], glob)   # bound_source is presence-checked above, never parsed
            compliant = predicted <= bound
            return {"ok": True, "computed": f"predicted={predicted:.6g}, bound={bound:.6g}",
                    "matched": "confirm" if compliant else "refute"}
```

- [ ] **Step 4: Run the executor tests to verify they pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_executor.py -q`
Expected: PASS (including the existing `sensitivity_ratio` tests).

- [ ] **Step 5: Write the failing `verdict_to_check` tests**

Add to `tests/test_magnitude_model.py` (extend the existing import line to include `verdict_to_check`):

```python
from valagents.computation import (ComputationPlan, ComputationResult,
                                   ComputationVerdict, verdict_to_attack, verdict_to_check)

def _bound_verdict(matched):
    p = ComputationPlan(kind="magnitude", comparison_kind="bound_check",
                        predicted_effect="1e-3", bound="1e-2", bound_source="PDG2024")
    r = ComputationResult(ok=True, computed="predicted=0.001, bound=0.01", matched=matched)
    v = ComputationVerdict(verdict=("pass" if matched == "confirm" else "fail"),
                           measured="predicted=0.001, bound=0.01", plan=p, result=r)
    return v

def test_bound_check_pass_is_independent_sourced_executor_check():
    rec = verdict_to_check(_bound_verdict("confirm"))
    assert rec.lens == "executor" and rec.verdict == "pass"
    assert rec.independent_sources == 1 and rec.sources and rec.sources[0].locator == "PDG2024"
    assert "PDG2024" in rec.basis and "bound" in rec.basis        # loud source

def test_bound_check_fail_maps_to_fail_check():
    rec = verdict_to_check(_bound_verdict("refute"))
    assert rec.verdict == "fail" and rec.independent_sources == 0

def test_verdict_to_check_symbolic_unchanged():
    p = ComputationPlan(expression="1/x", limit_variable="x", limit_point="oo",
                        expected="0", expected_source="textbook")
    r = ComputationResult(ok=True, computed="0", matched="confirm")
    v = ComputationVerdict(verdict="pass", measured="0", plan=p, result=r)
    rec = verdict_to_check(v)
    assert "expected = 0" in rec.basis and rec.sources and rec.sources[0].locator == "textbook"
```

- [ ] **Step 6: Run them to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_model.py -q`
Expected: the bound-check mapping tests FAIL (current `verdict_to_check` surfaces `expected`/`expected_source`, so the basis lacks `PDG2024` and `sources` is empty for a bound plan).

- [ ] **Step 7: Make `verdict_to_check` kind-aware**

In `valagents/computation.py`, replace the body of `verdict_to_check` with:

```python
def verdict_to_check(v: "ComputationVerdict", tick: int = 0):
    """Map an executed ComputationVerdict to a CheckRecord(lens='executor'). No LLM (F3).
    Kind-aware: a bound_check (magnitude) surfaces bound/bound_source; a symbolic limit
    surfaces expected/expected_source."""
    from valagents.artifact import CheckRecord, Source
    indep = 1 if v.verdict == "pass" else 0
    if v.plan.kind == "magnitude" and v.plan.comparison_kind == "bound_check":
        basis = (f"computed {v.measured or '?'}; bound = {v.plan.bound} "
                 f"(source: {v.plan.bound_source or 'n/a'}); matched = {v.result.matched}")
        src = v.plan.bound_source
    else:
        basis = (f"computed limit = {v.measured or '?'}; expected = {v.plan.expected} "
                 f"(source: {v.plan.expected_source or 'n/a'}); matched = {v.result.matched}")
        src = v.plan.expected_source
    sources = ([Source(locator=src, relation="independent")] if src else [])
    return CheckRecord(lens="executor", verdict=v.verdict, basis=basis,
                       independent_sources=indep, sources=sources, tick=tick)
```

- [ ] **Step 8: Run the full magnitude + symbolic suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_model.py tests/test_magnitude_executor.py -q`
Expected: PASS. Then run any symbolic/limit-check test file to confirm no regression:
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add valagents/sandbox/runner.py valagents/computation.py tests/test_magnitude_executor.py tests/test_magnitude_model.py
git commit -m "feat(magnitude): bound_check executor branch + kind-aware verdict_to_check (bound/bound_source)"
```

---

### Task 2: `bound_check` designer routing + scheduler claim-injection (L2-D11 idempotence)

**Files:**
- Modify: `valagents/agents/magnitude_designer.py` (two-phase parse; route `sensitivity_ratio` + `bound_check`)
- Modify: `valagents/prompts.py` (`MAGNITUDE_DESIGNER` — explain both kinds)
- Modify: `valagents/scheduler.py` (`run_magnitude_checks` — claim path + idempotence)
- Test: `tests/test_magnitude_integration.py`

**Interfaces:**
- Consumes: Task 1's `bound_check` executor + `verdict_to_check`; `checked_body`, `parse_tail`, `StrictTailError`; `AtomicClaim`; `store.add_check`.
- Produces: `design_magnitude` that returns a `bound_check` `ComputationPlan` when the LLM picks that kind (else `sensitivity_ratio`, else `None`); `run_magnitude_checks` that injects a load-bearing `mathematical` claim (`origin="bound_check"`) on a decisive bound verdict and clears prior bound claims first.

- [ ] **Step 1: Write the failing integration tests**

Add to `tests/test_magnitude_integration.py`:

```python
BOUND_OK = ("COMPARISON_KIND: bound_check | PREDICTED_EFFECT: 1e-3 | BOUND: 1e-2 "
            "| BOUND_SOURCE: PDG2024 | CONFIRM_IF: p<=bound | REFUTE_IF: p>bound")
BOUND_VIOLATE = BOUND_OK.replace("PREDICTED_EFFECT: 1e-3", "PREDICTED_EFFECT: 1e-1")
BOUND_NO_SOURCE = BOUND_OK.replace("| BOUND_SOURCE: PDG2024 ", "| BOUND_SOURCE:  ")

async def test_bound_violation_injects_failed_claim_and_refutes():
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(BOUND_VIOLATE), cfg())
    bnd = [c for c in s.current.claim_graph if c.origin == "bound_check"]
    assert bnd and bnd[0].status == "fail" and bnd[0].load_bearing
    assert s.current.status == "refuted"

async def test_bound_compliance_injects_passing_sourced_claim():
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(BOUND_OK), cfg())
    bnd = [c for c in s.current.claim_graph if c.origin == "bound_check"]
    assert bnd and bnd[0].status == "pass"
    assert s.current._has_independent_external_check(bnd[0])   # bound_source counts as the external check

async def test_bound_check_does_not_mark_magnitude_attempted():    # L2-D9: claim path, not an attack
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(BOUND_OK), cfg())
    assert "magnitude" not in s.current.attack_surface.attempted
    assert not [a for a in s.current.attacks if a.type == "magnitude"]

async def test_bound_missing_source_injects_no_claim():            # fail-closed: no source -> no claim
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(BOUND_NO_SOURCE), cfg())
    assert not [c for c in s.current.claim_graph if c.origin == "bound_check"]

async def test_bound_idempotent_across_reruns():                   # L2-D11
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(BOUND_OK), cfg())
    await run_magnitude_checks(s, router(BOUND_OK), cfg())          # rerun = a repair iteration
    bnd = [c for c in s.current.claim_graph if c.origin == "bound_check"]
    assert len(bnd) == 1                                           # cleared and re-injected, not duplicated
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_integration.py -q`
Expected: the new tests FAIL (current `design_magnitude` returns `None` for any non-`sensitivity_ratio` kind, so no bound claim is injected).

- [ ] **Step 3: Refactor the designer to a two-phase, kind-routed parse**

Replace the entire body of `valagents/agents/magnitude_designer.py` with:

```python
"""Magnitude-Designer: emits ONE structured magnitude ComputationPlan for a measurable prediction,
routed to the comparison kind the LLM chose. It DESIGNS the check only — returns no verdict and never
sees the execution result (F1/F3)."""
from __future__ import annotations
from valagents.computation import ComputationPlan
from valagents.parse import checked_body, parse_tail, StrictTailError
from valagents.prompts import MAGNITUDE_DESIGNER
from valagents.agents.base import build_messages

# All kinds carry COMPARISON_KIND + PREDICTED_EFFECT + CONFIRM_IF + REFUTE_IF; the rest are kind-specific.
_COMMON = ["COMPARISON_KIND", "PREDICTED_EFFECT", "CONFIRM_IF", "REFUTE_IF"]
_KIND_KEYS = {
    "sensitivity_ratio": ["BASELINE_OR_NULL", "SENSITIVITY", "SENSITIVITY_SOURCE", "THRESHOLD"],
    "bound_check": ["BOUND", "BOUND_SOURCE"],
}

async def design_magnitude(prediction, art, llm, cfg) -> ComputationPlan | None:
    user = MAGNITUDE_DESIGNER.format(
        formal=art.formal_claim.statement if art.formal_claim else "",
        observable=prediction.observable, effect_size=prediction.effect_size,
        discriminates_from=prediction.discriminates_from or "(none)")
    head, body = await checked_body(
        "magnitude_designer", build_messages("You design detectability checks.", user),
        _COMMON, llm=llm)
    if head is None:
        return None
    ck = head["comparison_kind"].strip().lower()
    extra = _KIND_KEYS.get(ck)
    if extra is None:
        return None
    try:
        t = parse_tail(body, _COMMON + extra)        # same body, full key set for this kind
    except StrictTailError:
        return None                                  # missing a kind-specific quantity/source -> no plan (fallback)
    common = dict(kind="magnitude", confirm_if=t["confirm_if"], refute_if=t["refute_if"],
                  target_claim_id=art.load_bearing, discriminating=bool(prediction.discriminates_from),
                  criterion="magnitude")
    try:
        if ck == "sensitivity_ratio":
            return ComputationPlan(comparison_kind="sensitivity_ratio",
                predicted_effect=t["predicted_effect"], baseline_or_null=t["baseline_or_null"],
                sensitivity=t["sensitivity"], sensitivity_source=t["sensitivity_source"],
                threshold=t["threshold"], **common)
        if ck == "bound_check":
            return ComputationPlan(comparison_kind="bound_check",
                predicted_effect=t["predicted_effect"], bound=t["bound"], bound_source=t["bound_source"],
                **common)
    except Exception:
        return None
    return None
```

- [ ] **Step 4: Rewrite the `MAGNITUDE_DESIGNER` prompt to cover both kinds**

In `valagents/prompts.py`, replace the `MAGNITUDE_DESIGNER = """..."""` assignment with:

```python
MAGNITUDE_DESIGNER = """You DESIGN a numeric magnitude check; you do NOT run or judge it — code does that, \
and you will never see the result. Given a measurable prediction, choose ONE comparison and produce its \
structured plan. Output no code; never invent a threshold/sensitivity/bound without naming its SOURCE.

FORMAL CLAIM: {formal}
PREDICTION: {observable} (effect size: {effect_size}; discriminates from: {discriminates_from})

Choose COMPARISON_KIND:
- sensitivity_ratio — is the predicted effect detectable above an established measurement sensitivity? \
Give PREDICTED_EFFECT, BASELINE_OR_NULL, the SENSITIVITY with its SENSITIVITY_SOURCE, and a detection THRESHOLD \
(how many times the sensitivity the effect must clear).
- bound_check — does the predicted effect respect an established upper bound / exclusion limit it must not \
exceed? Give PREDICTED_EFFECT and the BOUND with its BOUND_SOURCE.

All quantities are NUMBERS in SI/natural units. End with EXACTLY ONE line carrying ONLY the fields for your \
chosen kind, e.g.:
COMPARISON_KIND: sensitivity_ratio | PREDICTED_EFFECT: <n> | BASELINE_OR_NULL: <n> | SENSITIVITY: <n> | SENSITIVITY_SOURCE: <where> | THRESHOLD: <n> | CONFIRM_IF: <...> | REFUTE_IF: <...>
COMPARISON_KIND: bound_check | PREDICTED_EFFECT: <n> | BOUND: <n> | BOUND_SOURCE: <where> | CONFIRM_IF: <...> | REFUTE_IF: <...>"""
```

- [ ] **Step 5: Add the `bound_check` claim path + idempotence to `run_magnitude_checks`**

In `valagents/scheduler.py`, replace the whole `run_magnitude_checks` function with:

```python
async def run_magnitude_checks(store, llm, cfg, tick: int = 0) -> None:
    art = store.current
    # L2-D11: drop prior bound_check claims (and their checks) before re-injecting, so repeated runs
    # across repair iterations do not accumulate duplicate bound claims (mirrors red_team overwriting attacks).
    art.claim_graph = [c for c in art.claim_graph if c.origin != "bound_check"]
    bn = 0
    for p in art.predictions:
        if not p.measurable:
            continue
        plan = await design_magnitude(p, art, llm, cfg)
        if plan is None:
            continue
        from valagents.sandbox.executor import run_plan
        from valagents.computation import verdict_to_attack, verdict_to_check
        adir = f"{cfg.results_dir}/computations/magnitude" if getattr(cfg, "results_dir", None) else None
        verdict = run_plan(plan, cfg, artifacts_dir=adir)
        store.record({"event": "magnitude_executed", "kind": plan.comparison_kind,
                      "verdict": verdict.verdict, "computed": verdict.measured})
        if verdict.verdict == "uncertain":
            continue                                  # FAIL-CLOSED: no attack, no claim, no attempted-mark (L2-D9/F2)
        if plan.comparison_kind == "bound_check":
            # CLAIM path: inject a load-bearing mathematical claim; violate -> fail -> REFUTED, comply -> pass.
            bn += 1
            claim_id = f"BND{bn}"
            existing = {c.id for c in art.claim_graph}
            while claim_id in existing:
                claim_id = f"BND{bn}_{len(existing)}"
            claim = AtomicClaim(
                id=claim_id, type="mathematical", load_bearing=True, origin="bound_check",
                statement=(f"The idea's predicted effect respects the established bound "
                           f"({plan.bound}, source: {plan.bound_source})."))
            art.claim_graph.append(claim)
            store.add_check(claim_id, verdict_to_check(verdict, tick=tick))
            claim.exhausted = True
            tick += 1
            continue
        # ATTACK path (sensitivity_ratio): decisive verdict -> Attack + mark "magnitude" attempted.
        attack = verdict_to_attack(verdict, plan.target_claim_id, plan.discriminating, tick=tick)
        art.attacks = art.attacks + [attack]
        if art.attack_surface is not None and "magnitude" not in art.attack_surface.attempted:
            art.attack_surface.attempted = art.attack_surface.attempted + ["magnitude"]
        tick += 1
```

- [ ] **Step 6: Run the integration suite to verify the new and existing tests pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_integration.py -q`
Expected: PASS — the 5 new bound tests AND the existing `sensitivity_ratio` tests (`test_inert_lands_fatal_magnitude_attack_and_marks_attempted`, `test_detectable_survives_and_marks_attempted`, `test_uncertain_adds_no_attack_and_does_not_mark_attempted`, `test_designer_emits_plan_only`, `test_evaluate_ignores_magnitude_fields`).

- [ ] **Step 7: Run the full suite (catch any scheduler/repair regression)**

Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS (in particular `tests/test_scheduler_repair.py` — the empty `magnitude_designer` body still yields `None` → no-op, and the bound-claim clearing is a no-op on a graph of `origin="decomposed"` claims).

- [ ] **Step 8: Commit**

```bash
git add valagents/agents/magnitude_designer.py valagents/prompts.py valagents/scheduler.py tests/test_magnitude_integration.py
git commit -m "feat(magnitude): bound_check claim path — designer routing + inject-bound-claim wiring (refuted on violation, L2-D11 idempotent)"
```

---

### Task 3: `discriminating_margin` executor (model field + runner branch + attack basis)

**Files:**
- Modify: `valagents/computation.py` (`closest_prior_source` field; `verdict_to_attack` basis)
- Modify: `valagents/sandbox/runner.py` (`_MAG_REQUIRED`, `_run_magnitude`)
- Test: `tests/test_magnitude_executor.py`, `tests/test_magnitude_model.py`

**Interfaces:**
- Consumes: existing `run_plan`, `verdict_to_attack`.
- Produces: a `discriminating_margin` branch returning `confirm` when `abs(predicted - closest_prior_effect)/uncertainty >= threshold` else `refute`; the new mandatory `closest_prior_source` field (L2-D10); a `verdict_to_attack` basis that, for `discriminating_margin`, surfaces `closest_prior_effect`/`closest_prior_source`/`uncertainty`/`threshold`.

- [ ] **Step 1: Write the failing executor + model tests**

Add to `tests/test_magnitude_executor.py`:

```python
def dplan(**kw):
    base = dict(kind="magnitude", comparison_kind="discriminating_margin",
                predicted_effect="5e-9", closest_prior_effect="1e-9",
                closest_prior_source="arXiv:5678", uncertainty="1e-9", threshold="3")
    base.update(kw)
    return ComputationPlan(**base)

def test_discriminating_clears_is_confirm():
    v = run_plan(dplan(), cfg())                 # |5e-9-1e-9|/1e-9 = 4 >= 3 -> distinguishable
    assert v.verdict == "pass" and v.result.matched == "confirm"

def test_indistinguishable_is_refute():
    v = run_plan(dplan(predicted_effect="2e-9"), cfg())   # |2e-9-1e-9|/1e-9 = 1 < 3
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_discriminating_missing_source_is_uncertain():    # L2-D10 anti-laundering of the alternative
    v = run_plan(dplan(closest_prior_source=""), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_discriminating_zero_uncertainty_is_uncertain():
    v = run_plan(dplan(uncertainty="0"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_discriminating_dunder_is_uncertain_not_executed():
    v = run_plan(dplan(closest_prior_effect="x.__class__"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok
```

Add to `tests/test_magnitude_model.py`:

```python
def test_closest_prior_source_field_exists():
    p = ComputationPlan(kind="magnitude", comparison_kind="discriminating_margin",
                        closest_prior_source="arXiv:5678")
    assert p.closest_prior_source == "arXiv:5678"

def test_discriminating_margin_basis_is_loud_sourced():
    p = ComputationPlan(kind="magnitude", comparison_kind="discriminating_margin",
                        predicted_effect="5e-9", closest_prior_effect="1e-9",
                        closest_prior_source="arXiv:5678", uncertainty="1e-9", threshold="3",
                        discriminating=True)
    r = ComputationResult(ok=True, computed="margin=4", matched="refute")
    v = ComputationVerdict(verdict="fail", measured="margin=4", plan=p, result=r)
    a = verdict_to_attack(v, target_claim_id="c1", discriminating=True)
    assert a.status == "landed" and a.severity == "fatal"
    assert "closest_prior" in a.basis and "arXiv:5678" in a.basis and "margin=4" in a.basis
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_executor.py tests/test_magnitude_model.py -q`
Expected: FAIL (`closest_prior_source` not a field → `ComputationPlan(...)` raises; `discriminating_margin` unsupported → `uncertain`).

- [ ] **Step 3: Add the `closest_prior_source` field**

In `valagents/computation.py`, in `ComputationPlan`, insert the field right after `closest_prior_effect: str = ""`:

```python
    closest_prior_effect: str = ""
    closest_prior_source: str = ""   # mandatory for discriminating_margin (L2-D10) — sourced alternative
    uncertainty: str = ""
```

- [ ] **Step 4: Add the `discriminating_margin` runner branch**

In `valagents/sandbox/runner.py`, extend `_MAG_REQUIRED` (add the third entry; `closest_prior_source` is presence-checked, never parsed):

```python
    "discriminating_margin": ["predicted_effect", "closest_prior_effect",
                              "closest_prior_source", "uncertainty", "threshold"],
```

In `_run_magnitude`, after the `bound_check` block and still inside the `try`, add:

```python
        if ck == "discriminating_margin":
            predicted = _parse_number(plan["predicted_effect"], glob)
            closest = _parse_number(plan["closest_prior_effect"], glob)
            uncertainty = _parse_number(plan["uncertainty"], glob)
            threshold = _parse_number(plan["threshold"], glob)
            if uncertainty == 0:
                return {"ok": False, "matched": "neither", "error": "uncertainty is zero"}
            margin = float(np.abs(predicted - closest) / uncertainty)
            distinguishable = margin >= threshold
            return {"ok": True, "computed": f"margin={margin:.6g}",
                    "matched": "confirm" if distinguishable else "refute"}
```

- [ ] **Step 5: Make the `verdict_to_attack` basis kind-aware**

In `valagents/computation.py`, in `verdict_to_attack`, replace the single `basis = (...)` assignment with:

```python
    if v.plan.comparison_kind == "discriminating_margin":
        basis = (f"discriminating_margin: computed = {v.measured or '?'}; "
                 f"closest_prior = {v.plan.closest_prior_effect or 'n/a'} "
                 f"(source: {v.plan.closest_prior_source or 'n/a'}); "
                 f"uncertainty = {v.plan.uncertainty or 'n/a'}; threshold = {v.plan.threshold or 'n/a'}")
    else:
        basis = (f"{v.plan.comparison_kind}: computed = {v.measured or '?'}; "
                 f"sensitivity = {v.plan.sensitivity or 'n/a'} "
                 f"(source: {v.plan.sensitivity_source or 'n/a'}); threshold = {v.plan.threshold or 'n/a'}")
```

(The `status, severity` lines above it and the `return Attack(...)` below it are unchanged.)

- [ ] **Step 6: Run the executor + model suites to verify they pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_executor.py tests/test_magnitude_model.py -q`
Expected: PASS (including the existing `sensitivity_ratio` and `bound_check` tests — the `else` basis branch preserves the `sensitivity_ratio` basis verbatim).

- [ ] **Step 7: Commit**

```bash
git add valagents/computation.py valagents/sandbox/runner.py tests/test_magnitude_executor.py tests/test_magnitude_model.py
git commit -m "feat(magnitude): discriminating_margin executor + closest_prior_source (L2-D10) + kind-aware attack basis"
```

---

### Task 4: `discriminating_margin` designer routing + attack-path integration

**Files:**
- Modify: `valagents/agents/magnitude_designer.py` (`_KIND_KEYS` + construction branch)
- Modify: `valagents/prompts.py` (`MAGNITUDE_DESIGNER` — add the third kind)
- Test: `tests/test_magnitude_integration.py`

**Interfaces:**
- Consumes: Task 3's `discriminating_margin` executor + `closest_prior_source` field; the Task-2 two-phase designer and the attack-path branch in `run_magnitude_checks` (unchanged — `discriminating_margin` flows through the existing `else` attack branch).
- Produces: `design_magnitude` returning a `discriminating_margin` plan when the LLM picks it; end-to-end: below-margin → landed attack → `challenged`; missing `closest_prior_source` → `uncertain` → no attack, `"magnitude"` not marked.

- [ ] **Step 1: Write the failing integration tests**

Add to `tests/test_magnitude_integration.py`:

```python
DM_CLEARS = ("COMPARISON_KIND: discriminating_margin | PREDICTED_EFFECT: 5e-9 "
             "| CLOSEST_PRIOR_EFFECT: 1e-9 | CLOSEST_PRIOR_SOURCE: arXiv:5678 "
             "| UNCERTAINTY: 1e-9 | THRESHOLD: 3 | CONFIRM_IF: margin>=3 | REFUTE_IF: margin<3")
DM_INDISTINCT = DM_CLEARS.replace("PREDICTED_EFFECT: 5e-9", "PREDICTED_EFFECT: 2e-9")
DM_NO_SOURCE = DM_CLEARS.replace("| CLOSEST_PRIOR_SOURCE: arXiv:5678 ", "| CLOSEST_PRIOR_SOURCE:  ")

async def test_dm_designer_emits_plan():
    art = store_with_prediction().current
    p = await design_magnitude(art.predictions[0], art, router(DM_CLEARS), cfg())
    assert p is not None and p.comparison_kind == "discriminating_margin"
    assert p.closest_prior_source == "arXiv:5678"

async def test_dm_indistinct_lands_fatal_and_challenges():
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(DM_INDISTINCT), cfg())
    mags = [a for a in s.current.attacks if a.type == "magnitude"]
    assert mags and mags[0].status == "landed" and mags[0].severity == "fatal"
    assert "magnitude" in s.current.attack_surface.attempted
    assert s.current.verdict_class == "challenged"

async def test_dm_clears_survives_and_marks_attempted():
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(DM_CLEARS), cfg())
    mags = [a for a in s.current.attacks if a.type == "magnitude"]
    assert mags and mags[0].status == "survived"
    assert "magnitude" in s.current.attack_surface.attempted

async def test_dm_missing_source_no_attack_not_marked():    # L2-D10 + L2-D9
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(DM_NO_SOURCE), cfg())
    assert not [a for a in s.current.attacks if a.type == "magnitude"]
    assert "magnitude" not in s.current.attack_surface.attempted
```

- [ ] **Step 2: Run them to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_integration.py -q`
Expected: the new DM tests FAIL (`discriminating_margin` not in `_KIND_KEYS` → designer returns `None`).

- [ ] **Step 3: Add the `discriminating_margin` route to the designer**

In `valagents/agents/magnitude_designer.py`, add the entry to `_KIND_KEYS`:

```python
_KIND_KEYS = {
    "sensitivity_ratio": ["BASELINE_OR_NULL", "SENSITIVITY", "SENSITIVITY_SOURCE", "THRESHOLD"],
    "bound_check": ["BOUND", "BOUND_SOURCE"],
    "discriminating_margin": ["CLOSEST_PRIOR_EFFECT", "CLOSEST_PRIOR_SOURCE", "UNCERTAINTY", "THRESHOLD"],
}
```

And add the construction branch inside the `try`, after the `bound_check` branch and before the `except`:

```python
        if ck == "discriminating_margin":
            return ComputationPlan(comparison_kind="discriminating_margin",
                predicted_effect=t["predicted_effect"], closest_prior_effect=t["closest_prior_effect"],
                closest_prior_source=t["closest_prior_source"], uncertainty=t["uncertainty"],
                threshold=t["threshold"], **common)
```

- [ ] **Step 4: Add the third kind to the `MAGNITUDE_DESIGNER` prompt**

In `valagents/prompts.py`, in `MAGNITUDE_DESIGNER`, add a third bullet after the `bound_check` bullet:

```python
- discriminating_margin — does the predicted effect separate from the closest prior/alternative result? \
Give PREDICTED_EFFECT, the CLOSEST_PRIOR_EFFECT with its CLOSEST_PRIOR_SOURCE, the UNCERTAINTY, and a THRESHOLD \
(how many sigma of separation). Use this only when the prediction discriminates from an alternative.
```

And add the third tail example after the `bound_check` example line:

```python
COMPARISON_KIND: discriminating_margin | PREDICTED_EFFECT: <n> | CLOSEST_PRIOR_EFFECT: <n> | CLOSEST_PRIOR_SOURCE: <where> | UNCERTAINTY: <n> | THRESHOLD: <n> | CONFIRM_IF: <...> | REFUTE_IF: <...>"""
```

(Move the closing `"""` to the end of this new last line.)

- [ ] **Step 5: Run the integration suite to verify pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_magnitude_integration.py -q`
Expected: PASS — the 4 new DM tests plus all `sensitivity_ratio` and `bound_check` tests.

- [ ] **Step 6: Run the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS (all tests, no regression).

- [ ] **Step 7: Commit**

```bash
git add valagents/agents/magnitude_designer.py valagents/prompts.py tests/test_magnitude_integration.py
git commit -m "feat(magnitude): discriminating_margin designer routing + attack-path integration (below-margin -> challenged)"
```

---

## Notes for the executor

- **Gate untouched.** No task edits `valagents/artifact.py`. The existing `test_evaluate_ignores_magnitude_fields` (in `tests/test_magnitude_integration.py`) must stay green every task — it is the gate-purity guard.
- **Fail-closed has two layers, both required.** The designer fails closed on a missing source key (the tail parse drops the line → `None`), and the executor fails closed on a constructed plan with an empty source (presence check → `uncertain`). Tests exercise both; do not remove either.
- **Source strings are presence-checked, never parsed.** `bound_source`, `sensitivity_source`, `closest_prior_source` appear in `_MAG_REQUIRED` (presence) and in the basis (loud) — never passed to `_parse_number`.
- **L2-D9 line not to cross:** `"magnitude"` is added to `attack_surface.attempted` only in the attack-path branch (Task 2 keeps it there); the `bound_check` claim path must never touch `attempted`.
