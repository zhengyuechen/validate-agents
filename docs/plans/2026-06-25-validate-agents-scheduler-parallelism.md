# Scheduler parallelism — implementation plan (Spec-4)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans, task-by-task. Steps use checkbox (`- [ ]`) tracking.

**Goal:** cut wall-clock per run (currently ~2 h at `repair_cap=3`) by running the *independent* fan-out concurrently. The scheduler is today 100% sequential (`for claim … for lens …`, then ~10 awaited whole-artifact lenses, all inside the repair loop). Parallelize the two embarrassingly-independent blocks — **per-claim checks across claims** and the **independent whole-artifact lenses** — under a **bounded** concurrency cap, preserving every dependency and the exact set of checks/records produced.

**Architecture:** A tiny `bounded_gather(coros, limit)` helper; `run_claim_checks` gathers one coroutine per claim (each claim's internal lens→symbolic→fanout sequence stays sequential); `_whole_artifact_lenses` runs `novelty` first (it's a dependency), then gathers the independent assessment lenses, then runs the shared-mutation tail (red_team → magnitude → simulation → validation) sequentially. The store is already single-writer with **synchronous** `record`/`add_check`/`set` (`store.py` line 2: *"Version-don't-mutate is what makes Spec-4 parallel + repair safe"*), so cooperative `asyncio` interleaving cannot race it.

**Tech stack:** Python 3 asyncio, Pydantic v2 (config), `FakeLLM`/monkeypatch, `pytest` (asyncio auto mode).

## Global Constraints

- **Bounded concurrency is mandatory.** `valagents/llm.py` has no rate limiting; an unbounded `gather` of ~10 lenses + N claims will draw 429s from OpenRouter. Every gather goes through `bounded_gather(..., cfg.gate.max_concurrency)`.
- **Preserve dependencies (verified in code):** `predict(formal_claim, novelty)` reads `novelty.delta` → novelty must precede predict. `run_magnitude_checks` reads `art.predictions` → after predict. `red_team`, `run_magnitude_checks`, `run_simulation_checks` all mutate `art.attacks`/`art.attack_surface` (and magnitude mutates `art.claim_graph`) → the tail stays **sequential**.
- **No semantic change to the verdict.** The *set* of checks/records/attacks produced must be identical to the sequential version (verified by FakeLLM equivalence tests). Only wall-clock and **event-append order** change.
- **Event order is no longer source order.** `store.record` appends in completion order; under gather that is nondeterministic on real runs. Any test asserting event *order* (not membership) must be relaxed; the `.candidates`/`logs.jsonl` consumers already key on `tick`/`claim`, not append order. (FakeLLM returns instantly, so equivalence tests stay stable — gather completes instant coros in submission order.)
- **Tick pre-allocation:** parallel claims can't share a single incrementing `tick`. Each claim gets a disjoint tick block (`tick0 + i*BLOCK`). Ticks must stay unique; their exact values may change (tests assert uniqueness, not value).
- **`artifact.py` and the gate are untouched.** This is pure scheduling.
- **Commits:** plain messages — NO attribution trailers.
- **Test command:** `conda run -n cosci-reproduce python -m pytest tests/ -q`.

---

### Task 1: `bounded_gather` helper + `max_concurrency` config

**Files:** Create `valagents/concurrency.py`; modify `valagents/config.py` (`GateCfg`); Test `tests/test_concurrency.py`, `tests/test_config.py` (append).

**Interfaces:**
- `async def bounded_gather(coros: list, limit: int) -> list` — runs `coros` with ≤ `limit` in flight, **order-preserving** (returns results in input order); `limit <= 0` → unbounded `gather`.
- `GateCfg.max_concurrency: int = 8`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_concurrency.py
import asyncio, pytest
from valagents.concurrency import bounded_gather

async def test_bounded_gather_preserves_order():
    async def f(x): 
        await asyncio.sleep(0)
        return x * 2
    assert await bounded_gather([f(i) for i in range(5)], limit=2) == [0,2,4,6,8]

async def test_bounded_gather_respects_limit():
    cur = 0; peak = 0
    async def f():
        nonlocal cur, peak
        cur += 1; peak = max(peak, cur)
        await asyncio.sleep(0.01)
        cur -= 1
        return peak
    await bounded_gather([f() for _ in range(10)], limit=3)
    assert peak <= 3

async def test_bounded_gather_unbounded_when_limit_nonpositive():
    async def f(x): return x
    assert await bounded_gather([f(i) for i in range(4)], limit=0) == [0,1,2,3]
```

```python
# tests/test_config.py  (append)
def test_gate_max_concurrency_default():
    from valagents.config import Config
    assert Config(default_model="m").gate.max_concurrency == 8
```

- [ ] **Step 2: Run → fail** (`ModuleNotFoundError` / attribute).
- [ ] **Step 3: Implement**

```python
# valagents/concurrency.py
"""Spec-4: bounded concurrency for the scheduler's independent fan-out.
Order-preserving; a single semaphore caps in-flight LLM calls so parallel lenses
don't draw provider 429s (valagents/llm.py has no rate limiting of its own)."""
from __future__ import annotations
import asyncio

async def bounded_gather(coros: list, limit: int) -> list:
    if not limit or limit <= 0:
        return await asyncio.gather(*coros)
    sem = asyncio.Semaphore(limit)
    async def _run(c):
        async with sem:
            return await c
    return await asyncio.gather(*[_run(c) for c in coros])
```

In `valagents/config.py`, inside `GateCfg` (after `repair_cap`, line 26):
```python
    max_concurrency: int = 8        # Spec-4: max in-flight agent calls across a gathered fan-out
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `git commit -m "Scheduler parallelism: bounded_gather helper + gate.max_concurrency"`.

---

### Task 2: Parallelize `run_claim_checks` across claims

**Files:** Modify `valagents/scheduler.py`; Test `tests/test_scheduler_parallel.py`.

**Interfaces:** extract `_check_one_claim(store, claim, fc, backend, llm, cfg, tick_base, run_id)` — the existing per-claim body (lenses → PC-1b symbolic → fan-out → `exhausted`), verbatim, with a *local* `tick = tick_base`. `run_claim_checks` becomes a `bounded_gather` over claims with disjoint tick blocks; `build_derivation` stays after the gather.

- [ ] **Step 1: Failing equivalence test** — a 3-claim artifact (FakeLLM) yields the *same* per-claim `checks` (lens + verdict multiset) and the same `check` events (as a **set**) as a sequential baseline, and all check ticks are unique.

```python
# tests/test_scheduler_parallel.py  (sketch — reuse existing scheduler-test fixtures/FakeLLM)
async def test_run_claim_checks_parallel_equiv(seq_baseline_factory):
    art_par = await run_claim_checks_via(parallel=True)
    art_seq = seq_baseline_factory()   # the pre-change behavior, captured as expected multisets
    assert checks_multiset(art_par) == checks_multiset(art_seq)
    assert all_ticks_unique(art_par)
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — move lines 122-157 into `_check_one_claim` (local `tick`), then:

```python
async def run_claim_checks(store, backend, llm, cfg, tick0=0, run_id=None) -> None:
    art = store.current
    fc = art.formal_claim
    BLOCK = 10   # max lenses+symbolic+fanout per claim; keeps tick blocks disjoint
    await bounded_gather(
        [_check_one_claim(store, c, fc, backend, llm, cfg, tick0 + i * BLOCK, run_id)
         for i, c in enumerate(art.claim_graph)],
        cfg.gate.max_concurrency,
    )
    if fc is not None:
        store.set("derivation", await build_derivation(fc, art.claim_graph, llm, cfg))
```

Notes: `art.claim_graph` is **not mutated** during `run_claim_checks` (magnitude's claim_graph mutation is in the later tail), so gathering the loop is safe. `store.add_check` scans `claim_graph` read-only and appends to distinct `claim.checks` — synchronous, race-free. The intra-claim order (lens → symbolic → fanout reading `claim.status`) is preserved *inside* `_check_one_claim`.

- [ ] **Step 4: Run the targeted + scheduler suites → pass.**
- [ ] **Step 5: Commit** — `git commit -m "Scheduler parallelism: gather per-claim checks across claims (bounded, tick-blocked)"`.

---

### Task 3: Parallelize the independent whole-artifact lenses

> **IMPLEMENTATION NOTE (2026-06-25) — Step-1 audit FAILED the independence premise; Task 3 NOT built.**
> The audit (which this plan correctly gates correctness on) found **all six Stage-2 lenses import the
> same `_render` from `redteam.py`** (`completer.py:5`, `theory_bridge.py:5`, `positioning.py:5`,
> `known_limits.py:5`, `convincing_case.py:5`, `steelman.py:5`), and `_render` serializes **sibling
> Stage-2 outputs** (`completion`, `theory_bridge`, `prior_art_positioning`, `known_limits`,
> `convincing_case`, `steelman`). So sequentially each later lens's prompt is *enriched with every
> earlier sibling's output* — a deliberate context-building chain, not independent work. Parallelizing
> them strips that context → changes every real prompt → **violates this plan's own "No semantic change"
> Global Constraint.** Worse, the FakeLLM equivalence tests would **falsely pass** (the router ignores
> prompt content), masking a silent real-run quality regression. Per the plan's Step-1 directive ("if any
> reads a sibling, move it to a sequential step") taken to its true extent — ALL read siblings — the
> chained lenses stay sequential, so Task 3 reduces to a no-op. The only genuinely independent lens is
> `predict` (takes `(formal_claim, novelty)`, not `_render`); overlapping it with the chain saves ~1 lens
> latency and adds a wrinkle — not worth it (YAGNI, same call as Task 5's deferral). **Shipped: Tasks 1-2
> (per-claim parallelism — the N-way fan-out win, genuinely independent: per-claim work shares no
> `_render` context). Stage-2 speedup is a quality-vs-speed product decision, not a free win — deferred.**



**Files:** Modify `valagents/scheduler.py` (`_whole_artifact_lenses`); Test `tests/test_scheduler_parallel.py` (append).

**Structure (replaces lines 191-280):**
- **Stage 1 — `novelty`** (sequential; root dependency of `predict`).
- **Stage 2 — `bounded_gather`** of the independent assessment lenses: `completion, theory_bridge, positioning, known_limits, convincing_case, steelman, predict`. Each is a thin wrapper that awaits its agent and does its own `store.set`/`store.record`. `predict` closes over the Stage-1 `novelty`.
- **Stage 3 — sequential tail** (shared `art.attacks`/`attack_surface`/`claim_graph` mutation, order-dependent): `red_team` → `run_magnitude_checks` → `run_simulation_checks` → `design_validation`.

- [ ] **Step 1: VERIFY independence (no code yet).** Read `completer.py`, `theory_bridge.py`, `positioning.py`, `known_limits.py`, `convincing_case.py`, `steelman.py`. Confirm each reads only `art` fields set **before** Stage 2 (`formal_claim`, `claim_graph`, `novelty`) and **not** a sibling Stage-2 output. If any reads a sibling (e.g. steelman reads `convincing_case`), move that one into a Stage-2b sequential step. Record the finding in the commit message.

- [ ] **Step 2: Failing test** — FakeLLM run sets the same store fields (`novelty, completion, theory_bridge, prior_art_positioning, known_limits, convincing_case, steelman_objection, predictions, attacks, attack_surface, validation_plan`) as the sequential baseline; `predictions` reflects `novelty.delta` (dependency preserved); `attacks` includes red_team + magnitude + simulation contributions (tail order preserved).

- [ ] **Step 3: Implement** — wrappers + staged gather:

```python
async def _whole_artifact_lenses(store, backend, llm, cfg, tick, resolver=None, run_id=None) -> None:
    art = store.current
    if art.formal_claim is None:
        return
    # Stage 1: novelty (dependency root for predict)
    novelty = await ground_novelty(art.formal_claim, backend, llm, cfg)
    if novelty is not None:
        store.set("novelty", novelty)
        store.record({"event": "novelty", "position": novelty.position, "delta": novelty.delta})

    # Stage 2: independent assessment lenses (read art + novelty; none reads a sibling output)
    async def _completion():
        c = await complete_idea(art, llm, cfg)
        if c is not None:
            store.set("completion", c)
            store.record({"event": "completion", "status": c.status, "weakest_link": c.weakest_link})
    async def _predict():
        preds = await predict(art.formal_claim, novelty, llm, cfg)
        store.set("predictions", preds)
        store.record({"event": "predictions", "count": len(preds),
                      "measurable": sum(1 for p in preds if p.measurable)})
    # … _bridge, _positioning, _known_limits, _convincing, _steelman mirror the existing blocks …
    await bounded_gather(
        [_completion(), _bridge(), _positioning(), _known_limits(), _convincing(), _steelman(), _predict()],
        cfg.gate.max_concurrency,
    )

    # Stage 3: shared-mutation tail, SEQUENTIAL (art.attacks / attack_surface / claim_graph; magnitude needs predictions)
    attacks, surface, per_claim = await red_team(art, llm, cfg, tick=tick)
    store.set("attacks", attacks); store.set("attack_surface", surface)
    store.record({"event": "redteam", "attacks": len(attacks),
                  "landed": sum(1 for a in attacks if a.status == "landed"), "attempted": surface.attempted})
    claim_ids = {c.id for c in art.claim_graph}
    for cid, rec in per_claim:
        if cid in claim_ids:
            store.add_check(cid, rec)
            store.record({"event": "redteam_check", "claim": cid, "verdict": rec.verdict})
    await run_magnitude_checks(store, llm, cfg, tick=tick + 500, resolver=resolver, run_id=run_id)
    await run_simulation_checks(store, llm, cfg, tick=tick + 700, run_id=run_id)
    plan = await design_validation(art, llm, cfg)
    store.set("validation_plan", plan)
    if plan is not None:
        store.record({"event": "validation_plan", "cost": plan.cost, "test": plan.decisive_test})
```

- [ ] **Step 4: Run the scheduler suites → pass** (relax any event-*order* assertion to membership; see Task 4).
- [ ] **Step 5: Commit** — `git commit -m "Scheduler parallelism: gather independent whole-artifact lenses (novelty-first, sequential shared-mutation tail)"`.

---

### Task 4: Full-suite verification + event-order test repair

**Files:** none new (test fixups only).

- [ ] **Step 1: Run the whole suite** — `conda run -n cosci-reproduce python -m pytest tests/ -q`.
- [ ] **Step 2: Repair only order-coupled tests.** Failures will be tests asserting a strict **sequence** of `store.events` / emitted events that now arrive in completion order. Relax each to **set/multiset membership** or sort by `tick`/`claim` — do **not** re-serialize the scheduler to make them pass. Any failure that is *not* an ordering artifact (a missing or extra check) is a real regression: fix the parallel code, not the test.
- [ ] **Step 3: Confirm equivalence holds at `max_concurrency=1`** — setting the cap to 1 must reproduce sequential behavior exactly (a built-in regression switch). Add one test: `max_concurrency=1` ⇒ identical check multiset to the parallel default.
- [ ] **Step 4: If green, no code commit; commit test fixups** — `git commit -m "Tests: assert event membership not order under parallel scheduler"`.

---

### Task 5 (DEFERRED — do NOT build until a profile justifies it): within-claim lens gather

**Status:** designed, parked. Tasks 2-3 already overlap every claim's full chain (incl. `plan_query → search → adjudicate`, which is a dependency chain *inside* the grounder and cannot be split out). Task 5 only shortens the *critical path within a single claim*.

**Trigger to build:** a profiled parallel run shows per-claim chains on the critical path **with idle semaphore slots** — i.e. `#claims < gate.max_concurrency`. If `#claims ≥ max_concurrency`, the cap is the throughput ceiling and this yields ~nothing; raise `max_concurrency` instead. Decide from the profile, not assumption.

**Change (when triggered):** inside `_check_one_claim`, the initial checks are mutually independent — `grounder`, `prover`, and the PC-1b `_symbolic_check` (for mathematical claims) read only `claim`/`fc`, never each other. Gather them, then run the existing fan-out **after** (it reads `claim.status`, which is computed from those checks, so it stays sequential):

```python
# inside _check_one_claim, replacing the sequential initial-lens loop + symbolic block:
initial = [_run_lens(name, claim, fc, backend, llm, cfg, tick_base + i) for i, name in enumerate(lenses)]
if claim.type == "mathematical":
    initial.append(_symbolic_check(claim, fc, llm, cfg, tick_base + len(lenses), run_id))  # returns (rec, verdict)
results = await bounded_gather(initial, cfg.gate.max_concurrency)
# add_check / record each result (synchronous, race-free); then the fan-out block unchanged.
```

**Safety:** identical to Tasks 2-3 — independent work, synchronous store appends, `claim.status` read only after the gather. The order of `claim.checks` becomes nondeterministic, which is fine (the gate reads checks as a set).

**Why parked:** at the typical ~6 claims / `cap=8` it's a modest latency trim, and it adds the gather-then-fan-out wrinkle for a gain that may be near zero. Cheap to add later; not worth front-loading.

---

## Self-Review

**1. Coverage:** bounded helper + cap (Task 1); per-claim gather (Task 2); whole-artifact staged gather with verified dependencies (Task 3); suite + order-test repair + `cap=1` regression switch (Task 4). ✓
**2. Safety:** store mutations synchronous → no data race; `bounded_gather` caps in-flight calls → no 429 storm; dependencies (novelty→predict→magnitude) and shared-mutation tail kept sequential; tick blocks disjoint. ✓
**3. No semantic drift:** check/record/attack *sets* unchanged (FakeLLM equivalence + `cap=1` switch); only wall-clock and event-append order move; `artifact.py`/gate untouched. ✓
**4. Honest caveat surfaced:** event order is now nondeterministic on real runs — flagged as a Global Constraint and handled in Task 4, with `tick`/`claim` as the stable re-ordering keys for any downstream consumer.

## Execution notes (model/effort)

- Tasks 1, 4 mechanical → cheapest tier. Task 2 standard. Task 3 needs the Step-1 independence audit done carefully (it gates correctness) → standard tier, do not skip the read.
- Expected win: Stage-2 collapses ~7 lens latencies into ~1; per-claim collapses N claim-chains into ~ceil(N·calls/cap). At `cap=8` a ~6-claim artifact's lens-bound phases drop several-fold; restore `repair_cap=3` afterward without the 2 h penalty.
