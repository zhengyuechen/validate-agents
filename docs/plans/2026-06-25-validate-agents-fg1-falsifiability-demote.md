# FG-1: Demote the Falsifiability Hard Entry-Block — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Stop a single LLM `falsifiable=False` flag from hard-blocking the pipeline at entry; demote `not_falsifiable` to a code-witnessed last-resort verdict so checkable theory papers (Ran's QM) reach the prover/grounder/executor.

**Architecture:** Two surgical edits — (3a) delete the `scheduler.py` entry short-circuit so the run flows past faithfulness/decompose/lenses regardless of `falsifiable`; (3b) in `artifact.py._evaluate`, remove the `not_falsifiable` entry-gate and re-introduce it as a last-resort gated on "no root claim received any landed check." The existing faithfulness re-formalize retry then catches the existential abstraction for free (FG-1/FG-2 composition).

**Tech Stack:** Python 3, Pydantic v2, pytest (`asyncio_mode=auto`). Test: `conda run -n cosci-reproduce python -m pytest tests/ -q`.

## Global Constraints

- **`artifact.py` is the protected gate file — this plan is an explicitly OWNER-AUTHORIZED exception** (the falsifiability gate lives in `_evaluate`; same authorization class as PC-D6). No other gate semantics may change.
- **Safety backstop (must remain intact):** validation requires, per root, `_has_independent_external_check` AND the non-definitional-root guard (`artifact.py:273-283`). FG-1 removes only the *premature entry block*; it can never *add* a validation. Do not touch the strict gate.
- **Re-pin all line numbers against HEAD before editing** — they are current as of the PC-D6+guard commits but verify with `grep`.
- **Commits:** plain messages, NO attribution trailers.
- **FG-2 / PC-1b / I2 are NOT in this plan.** FG-1 only.

---

### Task 1: Delete the scheduler entry short-circuit (3a)

**Files:**
- Modify: `valagents/scheduler.py:28-32` (the `if not fc.falsifiable:` block in `run_entry_gates`)
- Test: `tests/test_scheduler_entry.py`

**Interfaces:**
- Produces: `run_entry_gates` returns `True` (proceeds) for a `falsifiable=False` formal_claim; records `{"event": "formalizer_falsifiable", "value": <bool>}`.

- [ ] **Step 1: Update the failing scheduler-entry test**

`tests/test_scheduler_entry.py:20-24` currently asserts the short-circuit. Replace `test_not_falsifiable_terminates` with the new behavior (proceeds, records the flag):

```python
async def test_falsifiable_false_proceeds_and_records(cfg):
    s = ArtifactStore(IdeaArtifact(raw_idea="seed"))
    llm = FakeLLM(router({
        "formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: no",
        "faithfulness": "FAITHFUL: yes | BACK_TRANSLATION: same",
        "decomposer": "CLAIM: c1 | TYPE: empirical | ROLE: novel_core | DEPENDS_ON: none | STATEMENT: effect exists",
        "entailment": "COVERS: complete | MISSING: none",
    }))
    proceed = await run_entry_gates(s, "seed", None, llm, cfg)
    assert proceed is True                                   # FG-1: falsifiable=False no longer blocks entry
    assert s.current.formal_claim.falsifiable is False
    assert any(e.get("event") == "formalizer_falsifiable" and e.get("value") is False
               for e in s.events)
```

(`router` is the existing helper in this file; confirm its signature with `grep -n "def router" tests/test_scheduler_entry.py` and match the other tests' usage.)

- [ ] **Step 2: Run it — expect FAIL**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_scheduler_entry.py::test_falsifiable_false_proceeds_and_records -q`
Expected: FAIL (`proceed is False` — the short-circuit still fires).

- [ ] **Step 3: Delete the short-circuit, replace with a non-blocking record**

In `valagents/scheduler.py`, replace (currently lines 28-32):

```python
    store.set("formal_claim", fc)
    if not fc.falsifiable:
        art.finalized = True
        store.record({"event": "entry_gate", "reason": "not_falsifiable"})
        return False
```

with:

```python
    store.set("formal_claim", fc)
    # FG-1: falsifiable is surfaced, NOT a gate. A falsifiable=False claim flows into faithfulness
    # (whose existing narrowed/no re-formalize retry catches existential abstraction — FG-1/FG-2 compose),
    # decompose, and the lenses; `not_falsifiable` is now a code-witnessed last-resort verdict in _evaluate.
    store.record({"event": "formalizer_falsifiable", "value": fc.falsifiable})
```

- [ ] **Step 4: Run the test — expect PASS**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_scheduler_entry.py -q`
Expected: PASS (all entry tests; the falsifiable case now proceeds).

- [ ] **Step 5: Commit**

```bash
git add valagents/scheduler.py tests/test_scheduler_entry.py
git commit -m "FG-1 (3a): falsifiable is surfaced, not an entry-gate short-circuit"
```

---

### Task 2: Demote `not_falsifiable` to a code-witnessed last-resort verdict (3b)

**Files:**
- Modify: `valagents/artifact.py` — remove the entry-gate `not_falsifiable` (currently 242-243); add `_has_any_landed_check` (after `_has_independent_external_check`, ~232); add the last-resort block after REFUTATION (~257), before NEEDS EXPERIMENT (~259)
- Test: `tests/test_artifact_gate.py`, `tests/test_verdict_class.py`

**Interfaces:**
- Consumes: `CheckRecord.lens ∈ {grounder, prover, redteam, executor}`, `CheckRecord.verdict ∈ {pass, fail, uncertain}` (every check has a real verdict; "pending" is a *claim status*, never a check verdict). `root_ancestors()`, `_b()`.
- Produces: `_has_any_landed_check(c: AtomicClaim) -> bool` — True iff `c` has ≥1 check from grounder/prover/executor. `_evaluate` returns `not_falsifiable` only when `formal_claim.falsifiable is False` AND no root has a landed check.

- [ ] **Step 1: Write the failing artifact-level tests**

In `tests/test_artifact_gate.py` (uses the `art()`/`claim()` helpers), replace nothing yet — add:

```python
def test_falsifiable_false_with_witness_validates():
    # FG-1: a falsifiable=False artifact whose root carries a real code-witnessed check is NO LONGER
    # blocked at entry — it flows to its real verdict (here: internally_validated).
    assert art(formal_claim=FormalClaim(statement="x", falsifiable=False)).status == "internally_validated"

def test_falsifiable_false_refuted_is_refuted_not_ill_posed():
    # a landed contradiction on a falsifiable=False root → REFUTED (refutation precedes the last-resort).
    a = art(formal_claim=FormalClaim(statement="x", falsifiable=False),
            claim_graph=[claim("c1", checks=[CheckRecord(lens="redteam", verdict="fail")])])
    assert a.status == "refuted"

def test_falsifiable_false_nothing_landed_is_not_falsifiable():
    # the last-resort: falsifiable=False AND no root received ANY grounder/prover/executor check
    # → not_falsifiable (demonstrably unassessable), verdict_class ill_posed.
    a = art(formal_claim=FormalClaim(statement="x", falsifiable=False),
            claim_graph=[claim("c1", checks=[])])
    assert a.status == "needs_experiment" and a.blocker["reason"] == "not_falsifiable"
```

`FormalClaim` and `CheckRecord` are already imported in `test_artifact_gate.py` (verify with `grep -n "^from valagents.artifact import" tests/test_artifact_gate.py`).

- [ ] **Step 2: Run them — expect FAIL**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_artifact_gate.py -k falsifiable_false -q`
Expected: FAIL — `..._with_witness_validates` gets `needs_experiment` (entry-gate still fires), the others mis-route.

- [ ] **Step 3: Add the `_has_any_landed_check` helper**

In `valagents/artifact.py`, immediately after `_has_independent_external_check` (ends ~line 232), add:

```python
    def _has_any_landed_check(self, c: AtomicClaim) -> bool:
        # FG-1: a "landed" check = a real lens (grounder/prover/executor) actually produced a verdict on
        # this claim. (Every CheckRecord verdict is pass/fail/uncertain — there is no pending check;
        # 'pending' is a claim STATUS.) redteam attacks live on the artifact, not claim.checks, so they
        # are excluded here. Used to decide whether the run could assess anything at all.
        return any(ck.lens in ("grounder", "prover", "executor")
                   and ck.verdict in ("pass", "fail", "uncertain")
                   for ck in c.checks)
```

- [ ] **Step 4: Remove the entry-gate `not_falsifiable`**

In `_evaluate`, delete the entry-gate lines (currently 242-243):

```python
        if self.formal_claim and not self.formal_claim.falsifiable:
            return NEEDS_EXPERIMENT, self._b("not_falsifiable")
```

- [ ] **Step 5: Add the last-resort, after REFUTATION and before NEEDS EXPERIMENT**

In `_evaluate`, immediately after the fatal-attack refutation check (the `if self._landed("fatal"):` block, ~line 255-257) and before the `# ===== NEEDS EXPERIMENT =====` comment (~line 258), insert:

```python
        # ===== LAST-RESORT: demonstrably unassessable (FG-1) =====
        # not_falsifiable is no longer an entry-gate on an LLM flag — it fires only when the claim is
        # flagged unfalsifiable AND no root received any landed check (the run genuinely could not assess
        # anything). Placed BEFORE the graded needs-experiment reasons so "nothing landed" outranks
        # "uncovered"/"inconclusive" — and when it fires there are no checked roots to mis-shadow, since
        # any landed check makes the guard False.
        if (self.formal_claim and not self.formal_claim.falsifiable
                and not any(self._has_any_landed_check(c) for c in rs)):
            return NEEDS_EXPERIMENT, self._b("not_falsifiable")
```

- [ ] **Step 6: Run the new artifact tests — expect PASS**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_artifact_gate.py -k falsifiable_false -q`
Expected: PASS (all three).

- [ ] **Step 7: Update the stale verdict-class test**

`tests/test_verdict_class.py:41-44` (`test_verdict_class_ill_posed_not_falsifiable`) uses `art(formal_claim=falsifiable=False)`, whose default root `c1` now carries a landed grounder PASS → it validates, not ill_posed. Update it to the last-resort (no landed check):

```python
def test_verdict_class_ill_posed_not_falsifiable():
    a = art(formal_claim=FormalClaim(statement="x", falsifiable=False),
            claim_graph=[claim("c1", checks=[])])          # nothing landed -> demonstrably unassessable
    assert a.status == "needs_experiment" and a.blocker["reason"] == "not_falsifiable"
    assert a.verdict_class == "ill_posed"
```

- [ ] **Step 8: Run the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS. If any other test asserted the old entry-gate (`falsifiable=False ⇒ immediate not_falsifiable`), update it to the new semantics — identified by the suite, not assumed. The `verdict_class` map (`artifact.py:334`, `not_falsifiable → ill_posed`) is unchanged.

- [ ] **Step 9: Commit**

```bash
git add valagents/artifact.py tests/test_artifact_gate.py tests/test_verdict_class.py
git commit -m "FG-1 (3b): not_falsifiable demoted to a last-resort verdict gated on no landed check"
```

---

### Task 3: Full-suite + end-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Run the whole suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS (pre-existing numpy `RuntimeWarning`s in the simulation suite are unrelated).

- [ ] **Step 2: Confirm the FG-1/FG-2 composition is live (no new code)**

Confirm by reading `scheduler.py` `run_entry_gates`: after the 3a deletion, a `falsifiable=False` formal_claim falls through to the faithfulness check (~line 35) and its existing narrowed/no re-formalize retry. No parallel path was added. (FG-2's existential-abstraction *detector* is a later deliverable; FG-1 already routes the case through the retry.)

---

## Self-Review

**1. Spec coverage (§3 of the design):**
- §3a delete short-circuit + non-blocking record → Task 1. ✓
- §3b remove entry-gate + last-resort gated on no-landed-check + `_has_any_landed_check` helper → Task 2. ✓
- §9 open question (`_has_any_landed_check` definition) → resolved: grounder/prover/executor checks; placed before needs-experiment to outrank `uncovered` (the design's effect table over its literal "after needs-experiment" wording — documented in Task 2 Step 5). ✓
- Maturity-falls-out (§3) → no code; verified implicitly (lenses now run). Strict gate + guard untouched (§2 backstop). ✓
- FG-1/FG-2 composition → Task 3 Step 2. ✓

**2. Placeholder scan:** none — every step has exact code/commands.

**3. Type consistency:** `_has_any_landed_check(c) -> bool` defined in Task 2 and used in the same task's last-resort block; `CheckRecord` fields match the model (`lens`/`verdict`). `not_falsifiable` reason string matches the `verdict_class` map. ✓

**Line-ref caveat:** all numbers are HEAD-current as of the PC-D6+guard commits; re-`grep` before editing (the entry-gate block: `grep -n "not_falsifiable" valagents/artifact.py`; the short-circuit: `grep -n "fc.falsifiable" valagents/scheduler.py`).
