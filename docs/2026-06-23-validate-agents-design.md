# validate-agents — Design Spec (Spec 1: the internal-validation spine)

- **Date:** 2026-06-23
- **Status:** Approved design, pending implementation plan
- **Sibling / source of reuse:** `../co-scientist-reproduce/` (canonical base — async queue engine, Pydantic models, `parse_label` verdict parser, `run_log` JSONL event log, OpenRouter LLM client). That repo stays **untouched**; we copy infra into a fresh `valagents/` package.
- **Goal in one line:** Grow a single seed idea from a one-liner into a fully-populated, check-hardened `IdeaArtifact`, terminating in exactly one of three honest verdicts — `internally_validated` (every dependency externally checked), `needs_experiment`, or `refuted` — and never in a fourth, undefined state.

This is the **depth-first complement** to Co-Scientist's breadth-first tournament. Co-Scientist fans one goal into many hypotheses and an Elo tournament selects a winner. validate-agents inverts it: **one seed, progressively specified, checked, and hardened into a single complete artifact.** "Validation" is really *maturation under check* — completing the idea while verifying it survives at every level of completion.

---

## 1. Scope & the invariants

### In scope (Spec 1)
- The `IdeaArtifact` / `AtomicClaim` schema with **computed** `status` / `maturity` / `load_bearing` / `blocker`.
- **Eight roles:** Formalizer, Decomposer, Grounder, Prover, Predictor, Red-team, Validation-designer (the seven agents) + Arbiter, plus Repairer.
- A **DAG control loop**: per-claim checks, verdict propagation along dependency edges, version-don't-mutate repair routed only to the affected subgraph.
- **Strict parsed verdicts** (mandatory machine-readable tail + one re-ask + `uncertain`-on-failure).
- CLI: `valagents "<seed idea>"` → `IdeaArtifact` JSON + a markdown report. Single-worker.

### Out of scope (later specs — named so the seams are designed now)
- **Spec 2** — computation sandbox (Computation-designer / Executor / Result-interpreter; SymPy + numpy behind process isolation). This is where the magnitude/equivalence checks stop being *reasoned* and start being *executed*.
- **Spec 3** — file ingestion (Librarian, `SourceDoc`/`Chunk`/`StatedClaim`, provenance with locators, `SentenceTransformer` retrieval index).
- **Spec 4** — dataflow parallelism over the claim DAG (typed semaphores, token bucket, ready-frontier scheduler).

Spec 1 is single-worker, but every scheduler interface (ready-frontier, verdict application, the immutable version chain) is shaped so Spec 4 swaps in `asyncio` **without touching agent or schema code**.

### The three invariants, as enforced code

| # | Invariant | Enforcement in Spec 1 |
|---|---|---|
| **I1** | **Verdicts gate, not narrate.** | `status` / `maturity` / `load_bearing` / `blocker` are Pydantic `@computed_field` properties with **no setter**. No LLM ever writes them. The Arbiter emits a `STATUS:` line, but that is a *cross-check* compared-and-logged against the computed value — the code is the source of truth. A mismatch is logged as a bug signal; the computed value always wins. |
| **I2** | **"Validated" = survived an external check.** | `internally_validated` is structurally unreachable unless every root-ancestor claim is *strictly* `pass`, and `pass` requires ≥1 external `CheckRecord` (Grounder live web search / Prover derivation / Red-team adversarial attack). `pending` is **never** `pass`. `refuted` and `needs_experiment` are first-class, tested outcomes. |
| **I3** | **The gate is total.** | Every run terminates in exactly one of `{internally_validated, needs_experiment, refuted}`. `draft` is **strictly non-terminal** — the scheduler never stops in `draft`. Every edge case (un-falsifiable entry, coverage gap, repair-cap exhaustion, double parse failure, landed non-fatal attack) maps deterministically to one of the three. No fourth state a reader rounds up to "validated." |

> **Honest Spec-1 caveat (the reasoned-not-executed line).** With no sandbox, the Red-team's mandatory **magnitude check** is *reasoned*, not executed. When that check is the crux, the Validation-designer emits it as the decisive computation and the gate lands at `needs_experiment` — never `internally_validated`. Spec 2 promotes the magnitude check to an executed lens. Spec 1's `internally_validated` therefore means precisely: *survived live web-grounding + adversarial red-team + derivation* — and nothing stronger. We do not overclaim what the label means.

---

## 2. Data model — `valagents/artifact.py` (new; Pydantic, matching cosci style)

```python
# ---- leaf records ---------------------------------------------------------
class CheckRecord:                 # one lens's verdict on one claim
    lens: Literal["grounder", "prover", "redteam"]
    verdict: Literal["pass", "fail", "uncertain"]   # parsed from the strict tail
    basis: str                     # parsed BASIS / evidence string
    provenance: str                # reference / locator the verdict rested on
    tick: int                      # when it was applied (for the event log + replay)

class FormalClaim:
    statement: str
    variables: list[str]
    scope: str
    regime: str
    falsifiable: bool              # FALSIFIABLE: yes|no  (entry gate — see §5)

class Novelty:
    closest_prior: list[str]
    delta: str                     # the specific thing this idea asserts that prior work doesn't
    position: Literal["new", "special_case", "restatement"]

class Prediction:
    observable: str
    effect_size: str
    discriminates_from: str        # what null / closest model this distinguishes from
    measurable: bool

class Attack:
    type: str                      # counterexample | failure_regime | confound | magnitude
    severity: Literal["fatal", "major", "minor"]
    status: Literal["survived", "landed"]
    target_claim_id: str | None    # None => artifact-level attack
    basis: str

class Gap:
    description: str
    claim_id: str
    fatal: bool

class Derivation:
    steps: list[str]
    gaps: list[Gap]

class ValidationPlan:
    decisive_test: str
    controls: list[str]
    confirm_if: str
    refute_if: str
    cost: Literal["low", "medium", "high"]

# ---- claims ---------------------------------------------------------------
class AtomicClaim:
    id: str
    statement: str
    type: Literal["definitional", "mathematical", "empirical", "mechanistic"]
    depends_on: list[str]          # claim ids — the DAG edges
    load_bearing: bool = True      # default True (conservative); Decomposer may mark auxiliary
    checks: list[CheckRecord]      # APPEND-ONLY; lenses never overwrite each other
    exhausted: bool = False        # scheduler-set: all applicable lenses have run (or none apply)

    @computed_field                # PURE join — never written by an LLM
    def status(self) -> Literal["pass", "fail", "uncertain", "pending"]:
        if any(c.verdict == "fail" for c in self.checks):
            return "fail"
        if any(c.verdict == "uncertain" for c in self.checks):
            return "uncertain"
        if any(c.verdict == "pass" for c in self.checks):   # pass requires >=1 external check
            return "pass"
        return "pending"           # NO terminal external check yet — never "pass"

# ---- the artifact ---------------------------------------------------------
class IdeaArtifact:
    raw_idea: str
    formal_claim: FormalClaim | None = None
    claim_graph: list[AtomicClaim] = []
    derivation: Derivation | None = None
    novelty: Novelty | None = None
    predictions: list[Prediction] = []
    attacks: list[Attack] = []
    validation_plan: ValidationPlan | None = None
    version_id: int = 0
    parent_version: int | None = None
    repairs_spent: int = 0
    finalized: bool = False        # scheduler-set: no runnable lens remains (or cap hit)

    # ---- COMPUTED, no setter, pure functions of the recorded verdict set ----
    @computed_field
    def status(self) -> Status: ...        # the gate — §2.1
    @computed_field
    def load_bearing(self) -> str | None: ...   # most pivotal root-ancestor claim — §2.2
    @computed_field
    def blocker(self) -> Blocker | None: ...     # what's keeping it from validating — §2.2
    @computed_field
    def maturity(self) -> float: ...       # display scalar ONLY — §2.3, never feeds status
```

### 2.1 The gate — `artifact.status` (pure, total)

`root_ancestors` = the set of `load_bearing` claims the `formal_claim` transitively rests on (Spec 1 default: every claim in the connected decomposition — conservative, requires *more* checks, not fewer).

```python
def status(self) -> Status:
    # --- entry gate (D1): an un-falsifiable claim is ill-posed -> refuted ---
    if self.formal_claim and not self.formal_claim.falsifiable:
        return REFUTED                          # refuted_reason = "not_falsifiable"

    rs = self.root_ancestors()

    # --- refutation ---
    if any(c.status == "fail" for c in rs):
        return REFUTED                          # "failed"
    if self._landed("fatal"):
        return REFUTED                          # "attacked"  (also the repair-cap path, §5)

    # --- needs experiment ---
    if any(c.status == "uncertain" for c in rs):
        return NEEDS_EXPERIMENT                 # "inconclusive"
    if self._landed("major") and self.finalized:    # (D4) unresolved serious objection
        return NEEDS_EXPERIMENT                 # "open_objection"
    if any(c.status == "pending" and c.exhausted for c in rs):   # (D2) coverage gap
        return NEEDS_EXPERIMENT                 # "uncovered"

    # --- validated: STRICT (I2) ---
    if (rs and all(c.status == "pass" for c in rs)        # pending is never pass
            and all(self._has_external_check(c) for c in rs)
            and not self._landed("fatal")                 # guaranteed by earlier return; belt-and-suspenders
            and not self._landed("major")):               # minor landed attacks only lower maturity (D4)
        return INTERNALLY_VALIDATED

    # --- otherwise still building; scheduler keeps going (I3: draft is non-terminal) ---
    return DRAFT
```

Key totality properties:
- **`pending` never masquerades as `pass`** — the validated branch requires `status == "pass"` *and* `_has_external_check`. A claim no lens covered sits at `pending`; once `exhausted`, it routes to `needs_experiment / uncovered` (D2). It can never reach `internally_validated`.
- **Repair-cap exhaustion needs no special clause.** At cap the scheduler sets `finalized=True` and stops repairing (§5); a still-`landed` fatal attack then computes `refuted` here. "Ran out of repair budget" can never read as "not refuted."
- **Severity-graded attacks (D4)** make landed non-fatal attacks total: `major` (unresolved, finalized) → `needs_experiment`; `minor` → recorded, lowers `maturity`, does not block.

### 2.2 `load_bearing` and `blocker` (computed)

- `load_bearing` = the single most pivotal root-ancestor claim — max transitive dependents; if the artifact is `refuted`/`needs_experiment`, the claim that caused it. Surfaced by the Arbiter as "what everything hinges on."
- `blocker` = `{claim_id | None, reason}` where reason ∈ `{not_falsifiable, failed, attacked, open_objection, uncovered, inconclusive}`. This preserves the information that the three-way `status` collapses (e.g. `uncovered` vs `inconclusive` are both `needs_experiment` but mean different things), without adding a fourth status.

### 2.3 `maturity` (computed, display-only) — the I1 one-directional rule

`maturity` is a `[0,1]` scalar for the report and ranking. **Hard constraint (locked now, formula deferred): `maturity` must not be an input to `status`.** The dependency is strictly one-directional — `{verdict set, status} → maturity → report`. If the `status` property ever reads `maturity`, a continuous fudge factor sneaks into a discrete gate and we are back to a tunable knob deciding "validated." A test asserts `status` is invariant under arbitrary `maturity` values (§8).

> **Open choice (learning-mode contribution at implementation time).** The exact `maturity` formula is a genuine design decision the brief leaves open (coverage-weighted? how hard to penalize landed attacks? does `needs_experiment` rank above a half-checked `draft`?). The function signature and surrounding context will be set up and the ~8-line body written by the user during implementation. The *isolation* invariant above holds regardless of the formula.

### 2.4 Lens coverage matrix (closes the orphan-claim gap structurally)

Every claim type must have ≥1 lens that can produce a terminal verdict, so the `pending` backstop rarely fires:

| claim type | Grounder | Prover | Red-team |
|---|---|---|---|
| definitional | prior-art / standard-usage | **well-formedness (coherent, non-circular)** | — |
| mathematical | prior-art / delta | derivation check | counterexample / magnitude |
| empirical | literature support | — | confound / magnitude |
| mechanistic | prior-art / delta | causal-chain check | failure-regime / magnitude |

The **Prover is broadened to cover `definitional`** well-formedness (previously orphaned). The gate's `pending ≠ pass` rule remains the backstop for anything that still slips through.

---

## 3. The eight roles — `valagents/agents/` (new prose; Grounder/Prover/Repairer/Arbiter adapt cosci bodies)

Every agent ends with a **mandatory machine-readable tail**, parsed strictly (§4). Reuse column notes which cosci agent the prose adapts.

| Role | Adapts | Reads | Writes | Mandatory verdict tail (strict; one re-ask) |
|---|---|---|---|---|
| **Formalizer** | — (new) | `raw_idea` | `formal_claim` | `CLAIM: <one sentence> \| VARIABLES: … \| REGIME: … \| FALSIFIABLE: yes\|no` |
| **Decomposer** | — (new) | `formal_claim` | `claim_graph` (+edges, +types) | one line/claim: `CLAIM: <id> \| TYPE: definitional\|mathematical\|empirical\|mechanistic \| DEPENDS_ON: <ids\|none> \| STATEMENT: …` |
| **Grounder** | Reflection + web search | each claim, whole | `novelty`; per-claim `CheckRecord(grounder)` | `CLOSEST_PRIOR: … \| DELTA: … \| POSITION: new\|special_case\|restatement`; per-claim `CLAIM: <id> \| SUPPORT: supported\|unsupported\|uncertain \| BASIS: …` |
| **Prover** | — (new, light) | `formal_claim`, graph | `derivation`; `CheckRecord(prover)` | `DERIVATION: complete\|gapped \| GAPS: <ids\|none> \| FATAL_GAP: yes\|no` |
| **Predictor** | — (new) | `formal_claim`, `novelty` | `predictions` | per-prediction: `OBSERVABLE: … \| EFFECT_SIZE: … \| DISCRIMINATES_FROM: … \| MEASURABLE: yes\|no` |
| **Red-team** | deep-verification, sharpened | whole | `attacks`; `CheckRecord(redteam)` | per-attack: `ATTACK: <type> \| SEVERITY: fatal\|major\|minor \| STATUS: survived\|landed \| TARGET: <claim_id\|none> \| BASIS: …` |
| **Validation-designer** | — (new) | whole | `validation_plan` | `TEST: … \| CONFIRM_IF: … \| REFUTE_IF: … \| COST: low\|medium\|high` |
| **Repairer** | Evolution | landed attack / fatal gap | **new artifact version** | `REPAIR: … \| TARGETS: <claim_ids> \| RATIONALE: …` |
| **Arbiter** | Meta-review | computed fields | final narrative only | `STATUS: … \| LOAD_BEARING: <claim_id> \| DECISIVE_TEST: …` (cross-checked vs computed; computed wins) |

**Red-team's magnitude check is mandatory** (strip the framing; does the mechanism change any measurable quantity at the relevant scale, by how many orders of magnitude?). In Spec 1 it is reasoned — see the §1 caveat.

**Prompt skeletons** (verbatim from the brief, with the forced tail) live in `valagents/prompts/`. The Formalizer / Red-team / Validation-designer / Arbiter skeletons are given in the brief; Grounder / Prover / Repairer adapt cosci's reflection / evolution prompt bodies with the output retargeted to the schema fields above.

---

## 4. Verdict parsing — `valagents/parse.py` (COPIED `parse_label` + NEW strict tail)

```python
def parse_tail(text: str, required_keys: list[str]) -> dict[str, str]:
    """Parse the mandatory 'KEY: value | KEY: value' tail. Raise StrictTailError
    if any required key is missing/unparseable."""

async def checked(agent, messages, required_keys, *, llm) -> dict | None:
    """Run a lens with the strict-tail contract:
       1. complete -> parse_tail
       2. on StrictTailError: ONE re-ask for the bare tail only
       3. on second StrictTailError: return None  (NEVER raise into the scheduler)
                                     log BOTH malformed bodies at WARN (prompt-bug signal)
    A None result is recorded by the caller as an `uncertain` CheckRecord — a check
    WAS attempted (distinct from `pending`, where no check ran). It never becomes `pass`.
    """
```

This is the parse-4/6 lesson made into a standing rule: the failure mode the happy path hides (a lens that can't produce its tail twice) is **surfaced**, not swallowed, and it can only ever degrade to `uncertain`, never silently `pass`.

---

## 5. Control loop — `valagents/scheduler.py` (single-worker now; parallel-ready seams)

```
1. Formalizer -> Decomposer                         # sequential; builds the DAG
   ENTRY GATE (D1): FALSIFIABLE=no -> finalize, status = REFUTED (not_falsifiable). STOP.
2. Walk claim_graph in dependency order; per claim run its applicable lenses
   (coverage matrix §2.4). Each lens appends a CheckRecord; claim.status recomputes (pure).
   Mark claim.exhausted once all applicable lenses have run (or none apply).
3. Whole-artifact lenses once: Grounder(novelty/delta), Predictor, Validation-designer.
4. Propagate verdicts along edges (pure rollup over the DAG).
5. REPAIR: if Red-team lands a fatal/major attack OR Prover finds a fatal gap ->
   Repairer spawns version v(n+1), re-entering ONLY the affected subgraph.
   Unaffected claim verdicts carry forward by IMMUTABILITY (never mutate v(n)).
   Cap: repairs_spent <= 3. AT CAP: finalize (do NOT continue, do NOT hang) ->
        the gate computes REFUTED if a fatal attack still reads `landed` (no special clause).
6. TERMINATION: no runnable lens remains for any claim (all root-ancestors are
   pass / fail / uncertain, or pending+exhausted) AND no repairable attack/gap is pending.
   Set finalized=True. The gate is now total -> Arbiter narrates; CODE computes the verdict.
```

- **Version-don't-mutate** protects a good result from a bad repair and is what makes Spec 4's parallel + repair safe: concurrent readers of v(n) are never corrupted; v(n+1) supersedes only when its checks complete.
- **Order-independence:** because `status`/`maturity` are pure functions of the recorded verdict set, the computed verdict is identical regardless of the order verdicts land — the property that lets Spec 4 parallelize with zero schema change.
- **Single writer + append-only log:** agents return verdicts; the scheduler/`store.py` applies them; `run_log.py` (copied) is the append-only record enabling replay and "did the gate fire?" auditability.

---

## 6. Reuse map & package layout

```
validate-agents/
  valagents/
    llm.py         <- COPIED  OpenRouterClient (async, per-agent model/temp, tenacity retry, extract_json)
    parse.py       <- COPIED  parse_label  +  NEW parse_tail / checked() strict-tail contract (§4)
    run_log.py     <- COPIED  JSONL event log (contextvars per-run, append-only, replay)
    web_search.py  <- COPIED  ArxivBackend, safe_search  (Grounder's external check)
    config.py      <- ADAPTED 8 agents -> models/temps; budget caps; repair cap (=3)
    artifact.py    <- NEW     schema + computed status/maturity/load_bearing/blocker (§2)
    store.py       <- NEW     single-writer ArtifactStore + append-only verdict log (cosci memory.py pattern)
    agents/        <- NEW     base protocol + 8 roles (§3)
    prompts/       <- NEW     verbatim brief skeletons + adapted cosci reflection/evolution bodies
    scheduler.py   <- NEW     DAG loop, propagation, repair-versioning, total-gate termination (§5)
    cli.py         <- NEW     valagents "<seed>" -> IdeaArtifact JSON + markdown report
  tests/           <- NEW     FakeLLM router (deterministic, no network)
  docs/2026-06-23-validate-agents-design.md
  results/<run_id>.jsonl
```

- **Proximity / SentenceTransformer is deliberately NOT copied** — it was write-only in the original and earns its place only in Spec 3 (retrieval index). YAGNI.
- **Provider stays OpenRouter** (inherited from cosci `llm.py`); point any agent at any model via `config.py`.

---

## 7. Worked cycle (the brief's escape-saddle example, Spec-1 path)

Seed: *"adding an antisymmetric curl term to gradient descent helps escape saddle points."*
- **Formalizer** pins θ̇ = −∇L(θ) + ω(t)·α(θ)·J·(θ−θ_c); `FALSIFIABLE: yes`.
- **Decomposer** emits three atomic claims, no edges among them: (A) curl term has nonzero projection on the negative-curvature direction; (B) α(θ) doesn't vanish *and doesn't saturate* at the saddle; (C) rotation doesn't disrupt convergence near minima.
- **Grounder** positions against Curl-Descent + momentum, isolates the delta.
- **Prover** checks (A)'s projection in closed form.
- **Predictor** commits to an escape-time scaling that separates from GD/momentum/Curl-Descent.
- **Red-team** runs the magnitude/failure check and surfaces the known α-saturation mode → attack targets claim (B). *Reasoned*, not executed (Spec-1 caveat).
- **Validation-designer** specifies the synthetic-saddle escape-time benchmark; `CONFIRM_IF` the scaling separates; `COST: low`.
- **Arbiter** narrates; **code computes** `status = needs_experiment`, `load_bearing = (B)`, `decisive_test = the escape-time benchmark` — exactly the next thing to run (and the Spec-2 sandbox could run it in-loop).

---

## 8. Tests that prove the invariants (the whole point) — `tests/`

Deterministic, via a `FakeLLM` router (no network).

**I1 — gate not narrate**
- *Parse-4/6 regression:* lens body with no tail → re-ask fires exactly once → claim becomes `uncertain`, never `pass`.
- *Double re-ask failure:* re-ask also malformed → `checked()` returns `None`, recorded as `uncertain` `CheckRecord`, no exception into scheduler, **both** bodies logged at WARN.
- *Code wins over narration:* Arbiter narrates `internally_validated` while a claim is `fail` → computed `status == refuted`; disagreement logged.
- *maturity ⊥ status:* `status` is invariant under arbitrary injected `maturity` values (proves one-directional dependency).

**I2 — validated = survived an external check**
- *No validation without a check:* any root-ancestor at `pending` → `internally_validated` impossible, even if all other claims `pass` and no attack landed.
- *Coverage gap (the back-door test):* a `definitional` claim no lens covers → stays `pending`+`exhausted` → artifact resolves `needs_experiment`, `blocker.reason == "uncovered"`, surfaced by claim id; `internally_validated` unreachable.
- *Honest outcomes reachable:* one scripted run each yields `refuted` and `needs_experiment`.

**I3 — the gate is total**
- *Not falsifiable:* `FALSIFIABLE: no` → terminal `refuted`, `refuted_reason == "not_falsifiable"`, never `draft`.
- *Repair-cap exhaustion:* fatal attack persists through 3 repairs → `finalized`, `status == refuted` (not `draft`).
- *Landed non-fatal attacks (D4):* unresolved `major` → `needs_experiment / open_objection`; `minor` → `internally_validated` still reachable but `maturity` lower.
- *Order-independence:* `status` identical across shuffled verdict-application orders (pre-validates Spec 4).
- *Version-don't-mutate:* a repair yields v2; v1's `CheckRecord`s are untouched objects; only the affected subgraph re-ran.

**Integration**
- The §7 escape-saddle seed, scripted end-to-end → `needs_experiment`, `load_bearing` = the α-non-saturation claim.

---

## 9. Build & next step

- `validate-agents` gets its own git repo (`git init`), mirroring the sibling repos which are each independent. Spec doc committed first.
- **Next:** invoke the writing-plans skill to turn this spec into a phased implementation plan (schema + parse first → agents → scheduler → tests), then implement, with the `maturity` formula as the learning-mode contribution point.

### Decision log (derived from the green-light review)
- **D1** `FALSIFIABLE: no` → `refuted` (`not_falsifiable`), not `draft`. Extends the totality principle to the entry gate.
- **D2** coverage gap (`pending`+`exhausted` root-ancestor) → `needs_experiment` (`uncovered`); `blocker` keeps it distinct from `inconclusive`.
- **D3** Prover broadened to cover `definitional` well-formedness so no claim type is orphaned (prevention; `pending ≠ pass` is the backstop).
- **D4** severity-graded landed attacks: `fatal`→`refuted`, unresolved `major`→`needs_experiment`, `minor`→recorded/maturity-only — makes "no attack landed" total.
- **D5** repair-cap exhaustion needs no special status clause: scheduler finalizes, gate computes `refuted` from the persisting fatal attack.
- **D6** double-parse-failure → `uncertain` `CheckRecord` (a check was attempted), distinct from `pending` (no check).
