# validate-agents — Design Spec (Spec 1: the internal-validation spine)

- **Date:** 2026-06-23
- **Status:** Approved design (rev 2 — incorporates the green-light gate review *and* the faithfulness / circularity / teeth / entailment review), pending implementation plan
- **Goal in one line:** Grow a single seed idea from a one-liner into a fully-populated, check-hardened `IdeaArtifact`, terminating in exactly one of three honest verdicts — `internally_validated` (the claim the seed actually asked, every dependency externally and independently checked), `needs_experiment`, or `refuted` — and never in a fourth, undefined state.

This is the **depth-first complement** to breadth-first, tournament-style hypothesis generation. That approach fans one goal into many hypotheses and selects a winner via ranking. validate-agents inverts it: **one seed, progressively specified, checked, and hardened into a single complete artifact.** "Validation" is really *maturation under check* — completing the idea while verifying it survives at every level of completion.

---

## 1. Scope & the invariants

### In scope (Spec 1)
- The `IdeaArtifact` / `AtomicClaim` schema with **computed** `status` / `maturity` / `load_bearing` / `blocker`.
- **Seven core lenses:** Formalizer, Decomposer, Grounder, Prover, Predictor, Red-team, Validation-designer; **two orchestration roles:** Arbiter, Repairer; and **two lightweight, independent entry-gate guards:** the **Faithfulness** check (seed ↔ formal_claim) and the **Entailment** check (sub-claims ⊢ formal_claim).
- A **DAG control loop**: entry gates → per-claim checks → verdict propagation along dependency edges → version-don't-mutate repair routed only to the affected subgraph, with a fan-out policy on load-bearing nodes.
- **Strict parsed verdicts** (mandatory machine-readable tail + one re-ask + `uncertain`-on-failure).
- **Independence-aware evidence** from day one (`CheckRecord.sources` / `independent_sources`), even though the Spec-1 backend is keyword-only.
- CLI: `valagents "<seed idea>"` → `IdeaArtifact` JSON + a markdown report. Single-worker.

### Out of scope (later specs — named so the seams are designed now)
- **Spec 2** — computation sandbox (Computation-designer / Executor / Result-interpreter; SymPy + numpy behind process isolation). Where the magnitude/equivalence checks stop being *reasoned* and start being *executed*.
- **Spec 3** — file ingestion (Librarian, `SourceDoc`/`Chunk`/`StatedClaim`, provenance with locators, citation-aware `SentenceTransformer` retrieval index). **This is the consumer of the `CheckRecord` independence fields carved in §2 — carved now precisely so Spec 3 needs no reshape.**
- **Spec 4** — dataflow parallelism over the claim DAG (typed semaphores, token bucket, ready-frontier scheduler). **Executes** the fan-out policy whose *trigger* is defined in Spec 1 (§5).

Spec 1 is single-worker, but every scheduler interface (ready-frontier, verdict application, the immutable version chain) is shaped so Spec 4 swaps in `asyncio` **without touching agent or schema code**.

### The three invariants, as enforced code

| # | Invariant | Enforcement in Spec 1 |
|---|---|---|
| **I1** | **Verdicts gate, not narrate.** | `status` / `maturity` / `load_bearing` / `blocker` are Pydantic `@computed_field` properties with **no setter**. No LLM ever writes them. The Arbiter emits a `STATUS:` line, but that is a *cross-check* compared-and-logged against the computed value — the code is the source of truth. A mismatch is logged as a bug signal; the computed value always wins. |
| **I2** | **"Validated" = survived an external, independent check.** | `internally_validated` is structurally unreachable unless every root-ancestor claim is *strictly* `pass`, and `pass` requires ≥1 external `CheckRecord` **with `independent_sources ≥ 1`**. `pending` is **never** `pass`. `refuted` and `needs_experiment` are first-class, tested outcomes. |
| **I3** | **The gate is total.** | Every run terminates in exactly one of `{internally_validated, needs_experiment, refuted}`. `draft` is **strictly non-terminal**. Every edge case (un-falsifiable entry, unfaithful formalization, empty/degenerate decomposition, decomposition gap, thin attack surface, coverage gap, repair-cap exhaustion, double parse failure, landed non-fatal attack) maps deterministically to one of the three. No fourth state a reader rounds up to "validated." |

### A total gate is necessary but **not sufficient**

Totality guarantees every run ends honestly. It does **not** guarantee the gate guarded the *right thing*. The gate's correctness silently assumes four upstream conditions; this rev makes each one a structural guard:

| Precondition | Without it, the failure is… | Guard |
|---|---|---|
| **Faithfulness** — validate the claim the seed actually asked | Formalizer narrows/drifts, emits `FALSIFIABLE: yes`, total gate validates the *wrong* claim with a clean `internally_validated` | §3 Faithfulness check → second entry gate (§5) |
| **Entailment** — sub-claims actually establish `formal_claim` | a *valid* DAG (every node checkable) that is *unsound* (omits the load-bearing piece): every node passes, validates a claim the decomposition doesn't establish | §3 Entailment `COVERS` pass (known-partial) |
| **Independence** — support isn't circular | "three sources" are the same author/group; novelty/support inflated on self-citation | `CheckRecord.sources`/`independent_sources` + Grounder ≥1-independent rule (§2, §3) |
| **Teeth** — a real attack surface was tried | "survived one weak attack" indistinguishable from "survived all four categories"; pass things by attacking weakly | Red-team `attempted` set + thin-surface cap (§2.1, §3) |

> **Honest Spec-1 caveat (the reasoned-not-executed line).** With no sandbox, the Red-team's mandatory **magnitude check** is *reasoned*, not executed. When that check is the crux, the Validation-designer emits it as the decisive computation and the gate lands at `needs_experiment` — never `internally_validated`. Spec 2 promotes the magnitude check to an executed lens. Spec 1's `internally_validated` therefore means precisely: *survived live web-grounding + adversarial red-team + derivation, on independent support, for the faithfully-pinned claim* — and nothing stronger.
>
> **The limit you cannot test your way out of.** Every lens shares the base model's blind spots. `internally_validated` means **"survived the checks this system can run," never "true."** A reader who reads the label as "true" misuses the tool in exactly the way this design exists to prevent. This sentence ships in the CLI/report output, not just the spec.

---

## 2. Data model — `valagents/artifact.py` (new; Pydantic v2)

```python
# ---- evidence & independence (the §2 seam carved for Spec 3) ---------------
class Source:
    locator: str                   # arXiv id / URL / (Spec 3) doc-chunk locator
    author: str | None             # best-effort in Spec 1; precise in Spec 3
    group: str | None
    relation: Literal["independent", "same_author", "same_group",
                      "self_citation", "unknown"]

class CheckRecord:                 # one lens's verdict on one claim
    lens: Literal["grounder", "prover", "redteam"]
    verdict: Literal["pass", "fail", "uncertain"]   # parsed from the strict tail
    basis: str
    sources: list[Source]          # provenance, now structured
    independent_sources: int       # count of sources with relation == "independent"
    tick: int                      # for the event log + replay

# ---- formalization & structure guards -------------------------------------
class FormalClaim:
    statement: str
    variables: list[str]
    scope: str
    regime: str
    falsifiable: bool              # FALSIFIABLE: yes|no   (entry gate 1)

class Faithfulness:                # seed <-> formal_claim (independent of Formalizer)
    verdict: Literal["yes", "narrowed", "no"]   # FAITHFUL  (entry gate 2)
    back_translation: str          # plain-language restatement of formal_claim
    retried: bool                  # whether a re-formalization retry was spent

class Coverage:                    # conjunction of sub-claims |- formal_claim
    verdict: Literal["complete", "gap"]         # COVERS  (validation precondition; known-partial)
    missing: str | None

class AttackSurface:               # Red-team teeth
    attempted: list[str]           # subset of {counterexample, failure_regime, confound, magnitude}
    skipped: list[str]

# ---- other leaves (unchanged from rev 1) ----------------------------------
class Novelty: closest_prior: list[str]; delta: str; position: Literal["new","special_case","restatement"]
class Prediction: observable: str; effect_size: str; discriminates_from: str; measurable: bool
class Attack: type: str; severity: Literal["fatal","major","minor"]; status: Literal["survived","landed"]; target_claim_id: str | None; basis: str
class Gap: description: str; claim_id: str; fatal: bool
class Derivation: steps: list[str]; gaps: list[Gap]
class ValidationPlan: decisive_test: str; controls: list[str]; confirm_if: str; refute_if: str; cost: Literal["low","medium","high"]

# ---- claims ---------------------------------------------------------------
class AtomicClaim:
    id: str
    statement: str
    type: Literal["definitional", "mathematical", "empirical", "mechanistic"]
    depends_on: list[str]          # claim ids — the DAG edges
    load_bearing: bool = True      # default True (conservative); Decomposer may mark auxiliary
    checks: list[CheckRecord]      # APPEND-ONLY; lenses never overwrite each other
    exhausted: bool = False        # scheduler-set: all applicable lenses (incl. fan-out) have run

    @computed_field                # PURE join — never written by an LLM
    def status(self) -> Literal["pass", "fail", "uncertain", "pending"]:
        if any(c.verdict == "fail" for c in self.checks):
            return "fail"
        if any(c.verdict == "uncertain" for c in self.checks):
            return "uncertain"
        # pass requires an external check that itself rests on >=1 independent source
        if any(c.verdict == "pass" and c.independent_sources >= 1 for c in self.checks):
            return "pass"
        return "pending"           # never "pass"

# ---- the artifact ---------------------------------------------------------
class IdeaArtifact:
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
    finalized: bool = False        # scheduler-set: no runnable lens remains (or a cap was hit)

    @computed_field
    def status(self) -> Status: ...        # the gate — §2.1
    @computed_field
    def load_bearing(self) -> str | None: ...   # §2.2
    @computed_field
    def blocker(self) -> Blocker | None: ...     # §2.2
    @computed_field
    def maturity(self) -> float: ...       # display scalar ONLY — §2.3, never feeds status
```

### 2.1 The gate — `artifact.status` (pure, total)

`root_ancestors` = the `load_bearing` claims `formal_claim` transitively rests on (Spec-1 default: every claim in the connected decomposition — conservative).

```python
def status(self) -> Status:
    # ===== ENTRY GATES (must pin the right, well-posed claim before anything) =====
    if self.formal_claim and not self.formal_claim.falsifiable:
        return REFUTED                          # not_falsifiable
    if self.faithfulness and self.faithfulness.verdict == "no" and self.faithfulness.retried:
        return REFUTED                          # unfaithful_drift   (after one retry)
    if self.faithfulness and self.faithfulness.verdict == "narrowed" and self.faithfulness.retried:
        return REFUTED                          # unfaithful_narrowed
    if self.formal_claim and self.faithfulness and self.faithfulness.verdict == "yes" \
            and not self.claim_graph and self.finalized:
        return REFUTED                          # ill_formed   (empty/degenerate decomposition, after retry)

    rs = self.root_ancestors()

    # ===== REFUTATION =====
    if any(c.status == "fail" for c in rs):
        return REFUTED                          # failed
    if self._landed("fatal"):
        return REFUTED                          # attacked   (also the repair-cap path — §5)

    # ===== NEEDS EXPERIMENT =====
    if any(c.status == "uncertain" for c in rs):
        return NEEDS_EXPERIMENT                 # inconclusive
    if self._landed("major") and self.finalized:
        return NEEDS_EXPERIMENT                 # open_objection      (D4)
    if any(c.status == "pending" and c.exhausted for c in rs):
        return NEEDS_EXPERIMENT                 # uncovered           (D2)
    if self.coverage and self.coverage.verdict == "gap":
        return NEEDS_EXPERIMENT                 # decomposition_gap   (#4, known-partial)
    if self._thin_attack_surface():
        return NEEDS_EXPERIMENT                 # thin_attack_surface (#3)

    # ===== VALIDATED: STRICT (I2) =====
    if (rs and all(c.status == "pass" for c in rs)        # pending is never pass
            and all(self._has_independent_external_check(c) for c in rs)
            and (self.faithfulness and self.faithfulness.verdict == "yes")   # POSITIVE precond — symmetric w/ coverage;
                                                                            # closes faithfulness=None and narrowed/retried=False
                                                                            # in the GATE, not in scheduler ordering (§1 SPOF rule)
            and (self.coverage and self.coverage.verdict == "complete")
            and not self._thin_attack_surface()
            and not self._landed("fatal") and not self._landed("major")):  # minor → maturity only (D4)
        return INTERNALLY_VALIDATED

    return DRAFT                                # non-terminal; scheduler keeps going (I3)
```

`_thin_attack_surface()` = `magnitude` not in `attack_surface.attempted`, **or** fewer than `config.min_attack_categories` (default 2) attempted. `_has_independent_external_check(c)` = some `CheckRecord` on `c` with `verdict == "pass"` and `independent_sources ≥ 1`.

> **Boundary note — teeth checks *coverage*, not *strength*.** `attempted` is the Red-team's self-report, so a *token* magnitude attack listed in `ATTEMPTED` passes `_thin_attack_surface()`. In Spec 1, teeth closes **"didn't try magnitude," not "tried magnitude weakly"** — the latter is exactly what Spec 2's *executed* magnitude check shuts. A reader must not read "passed teeth" as "was attacked hard."

Totality properties (all paths terminate in one of three):
- **`pending` never masquerades as `pass`** — validated branch requires strict `pass` + an independent external check; an orphaned claim sits `pending` → `uncovered` once `exhausted`.
- **Faithfulness guards the *claim*, in the gate not the schedule** — drift/narrowing → `refuted` via the entry gates, *and* `internally_validated` positively requires `faithfulness.verdict == "yes"` in the validated branch (symmetric with `coverage`). The two slip-cases (`faithfulness is None`; `narrowed` with `retried == False`) are closed by the code, not by §5's ordering.
- **Entailment guards the *decomposition*** — `COVERS: gap` → `needs_experiment`, *before* any "validated" is reachable.
- **Independence guards the *evidence*** — a `supported` verdict with zero independent sources never becomes `pass` (downgraded to `uncertain` at the verdict-mapping layer, §3).
- **Teeth guard the *attack*** — a thin/mostly-skipped surface caps below `internally_validated`.
- **Repair-cap exhaustion** needs no special clause: at cap the scheduler finalizes; a still-`landed` fatal attack computes `refuted`.
- **Empty decomposition** (the rev-1 totality hole) → `ill_formed` instead of hanging in `draft`.

### 2.2 `load_bearing` and `blocker` (computed)

- `load_bearing` = the single most pivotal root-ancestor claim (max transitive dependents; or the claim that caused a `refuted`/`needs_experiment`).
- `blocker` = `{claim_id | None, reason}`, reason ∈ `{not_falsifiable, unfaithful_drift, unfaithful_narrowed, ill_formed, failed, attacked, open_objection, uncovered, inconclusive, decomposition_gap, thin_attack_surface}`. Preserves what the three-way `status` collapses, without a fourth status.

### 2.3 `maturity` (computed, display-only) — the I1 one-directional rule

`maturity ∈ [0,1]`, for report/ranking. **Hard constraint (locked now, formula deferred): `maturity` must not be an input to `status`.** Dependency is strictly one-directional — `{verdict set, status} → maturity → report`. A test asserts `status` is invariant under arbitrary injected `maturity` (§8).

> **Open choice (learning-mode contribution at implementation time).** The exact `maturity` formula (coverage-weighting, attack penalty, where `needs_experiment` ranks) is the user's ~8-line contribution. The isolation invariant holds regardless of the formula.

### 2.4 Lens coverage matrix + structural guards

Per-claim lenses (every claim *type* has ≥1 terminal lens — closes the orphan-claim gap; `pending ≠ pass` is the backstop):

| claim type | Grounder | Prover | Red-team |
|---|---|---|---|
| definitional | prior-art / standard-usage | **well-formedness (coherent, non-circular)** | — |
| mathematical | prior-art / delta | derivation check | counterexample / magnitude |
| empirical | literature support | — | confound / magnitude |
| mechanistic | prior-art / delta | causal-chain check | failure-regime / magnitude |

Structural guards (not per-claim lenses — they guard the *whole*): **Faithfulness** (entry), **Entailment/`COVERS`** (precondition), **attack-surface teeth** (precondition). The **Grounder independent-sources rule**: `SUPPORT: supported` with `independent_sources < 1` is downgraded to `uncertain` in code — so on Spec-1's keyword backend, thin retrieval honestly lands `uncertain` rather than a false `pass`.

---

## 3. The roles — `valagents/agents/`

| Role | Reads | Writes | Mandatory verdict tail (strict; one re-ask) |
|---|---|---|---|
| **Formalizer** | `raw_idea` | `formal_claim` | `CLAIM: <one sentence> \| VARIABLES: … \| REGIME: … \| FALSIFIABLE: yes\|no` |
| **Faithfulness** | `raw_idea`, `formal_claim` | `faithfulness` | `FAITHFUL: yes\|narrowed\|no \| BACK_TRANSLATION: <plain-language restatement of formal_claim>` |
| **Decomposer** | `formal_claim` | `claim_graph` (+edges, +types) | one line/claim: `CLAIM: <id> \| TYPE: definitional\|mathematical\|empirical\|mechanistic \| DEPENDS_ON: <ids\|none> \| STATEMENT: …` |
| **Entailment** | `formal_claim`, `claim_graph` | `coverage` | `COVERS: complete\|gap \| MISSING: <desc\|none>` |
| **Grounder** | each claim, whole | `novelty`; per-claim `CheckRecord(grounder)` | `CLOSEST_PRIOR: … \| DELTA: … \| POSITION: new\|special_case\|restatement`; per-claim `CLAIM: <id> \| SUPPORT: supported\|unsupported\|uncertain \| INDEPENDENT_SOURCES: <n> \| SOURCES: <locator(author)…> \| BASIS: …` |
| **Prover** | `formal_claim`, graph | `derivation`; `CheckRecord(prover)` | `DERIVATION: complete\|gapped \| GAPS: <ids\|none> \| FATAL_GAP: yes\|no` |
| **Predictor** | `formal_claim`, `novelty` | `predictions` | per-prediction: `OBSERVABLE: … \| EFFECT_SIZE: … \| DISCRIMINATES_FROM: … \| MEASURABLE: yes\|no` |
| **Red-team** | whole | `attacks`, `attack_surface`; `CheckRecord(redteam)` | `ATTEMPTED: <subset of counterexample,failure_regime,confound,magnitude>`; per-attack: `ATTACK: <type> \| SEVERITY: fatal\|major\|minor \| STATUS: survived\|landed \| TARGET: <claim_id\|none> \| BASIS: …` |
| **Validation-designer** | whole | `validation_plan` | `TEST: … \| CONFIRM_IF: … \| REFUTE_IF: … \| COST: low\|medium\|high` |
| **Repairer** | landed attack / fatal gap | **new artifact version** | `REPAIR: … \| TARGETS: <claim_ids> \| RATIONALE: …` |
| **Arbiter** | computed fields | final narrative only | `STATUS: … \| LOAD_BEARING: <claim_id> \| DECISIVE_TEST: …` (cross-checked vs computed; computed wins) |

- **Faithfulness** back-translates `formal_claim` to plain language and asks "is this what the seed asked?" Independence from the Formalizer is the point — the author does not grade its own pin. `narrowed`/`no` → one bounded re-formalization retry (§5), then `refuted`.
- **Magnitude is the mandatory Red-team category**; its omission alone trips `_thin_attack_surface()`. The `ATTEMPTED` set is what makes "tried weakly" visible and cappable.
- **Grounder** must populate `SOURCES`/`INDEPENDENT_SOURCES`; the code (not the LLM) downgrades `supported`+0-independent to `uncertain`.

Prompt skeletons (verbatim from the brief, with the forced tail) live in `valagents/prompts/`; Grounder / Prover / Repairer write the prompt body retargeted to these fields.

---

## 4. Verdict parsing — `valagents/parse.py` (`parse_label` + strict tail)

```python
def parse_tail(text, required_keys) -> dict:        # raise StrictTailError on missing/unparseable key
async def checked(agent, messages, required_keys, *, llm) -> dict | None:
    # 1. complete -> parse_tail
    # 2. StrictTailError -> ONE re-ask for the bare tail only
    # 3. second StrictTailError -> return None (NEVER raise into scheduler);
    #    log BOTH malformed bodies at WARN (prompt-bug signal)
    # caller records None as an `uncertain` CheckRecord — a check WAS attempted
    # (distinct from `pending`); it can never become `pass`.
```

The parse-4/6 lesson as a standing rule: the failure the happy path hides (a lens that can't produce its tail twice) is surfaced, and can only ever degrade to `uncertain`.

---

## 5. Control loop — `valagents/scheduler.py` (single-worker now; parallel-ready seams)

```
1. Formalizer -> formal_claim.
   ENTRY GATE 1 (falsifiability): FALSIFIABLE=no -> finalize, REFUTED/not_falsifiable. STOP.
2. Faithfulness (independent) -> faithfulness.
   ENTRY GATE 2 (faithfulness): FAITHFUL in {narrowed,no} ->
        ONE re-formalization retry with targeted feedback (set faithfulness.retried);
        still {narrowed,no} -> finalize, REFUTED/unfaithful_(narrowed|drift). STOP.
3. Decomposer -> claim_graph.
   GUARD (empty/degenerate): no claims -> ONE Decomposer retry;
        still empty at finalize -> REFUTED/ill_formed. STOP.
4. Entailment (independent) -> coverage.  COVERS=gap is surfaced as blocker (known-partial;
   Red-team partly backstops). It caps below internally_validated (-> needs_experiment/decomposition_gap).
5. Walk claim_graph in dependency order; per claim run its applicable lenses (matrix §2.4).
   Each lens appends a CheckRecord (with sources/independent_sources); claim.status recomputes (pure).
   FAN-OUT POLICY (#5): a LOAD-BEARING claim that resolves to `uncertain` is NOT marked
        exhausted until >= config.fanout_N (default 2) DIVERSE-TYPE lenses have run on it
        (e.g. a counterexample search AND a magnitude angle — NOT repeated same-type runs).
        Fan-out's value is DISAGREEMENT-DETECTION, not agreement-as-corroboration: repeated
        same-type runs are correlated draws sharing the base model's blind spots — the §1 limit,
        and D8's same-author correlation one level up. So "met fanout_N" by repetition is NOT
        read as strong support; disagreement among diverse lenses is the signal, kept as
        multiple CheckRecords. [Spec 1: sequential. Spec 4: parallel — same join.]
6. Whole-artifact lenses once: Grounder(novelty/delta), Red-team(attacks + attack_surface),
   Predictor, Validation-designer.
7. Propagate verdicts along edges (pure rollup over the DAG).
8. REPAIR: fatal/major attack OR fatal gap -> Repairer spawns version v(n+1), re-entering ONLY
   the affected subgraph. Unaffected verdicts carry forward by IMMUTABILITY (never mutate v(n)).
   Cap: repairs_spent <= 3. AT CAP: finalize (do NOT continue, do NOT hang).
9. TERMINATION: no runnable lens remains (all root-ancestors pass/fail/uncertain, or pending+exhausted;
   load-bearing uncertain nodes have met fanout_N) AND no repairable attack/gap pending.
   Set finalized=True. Gate is total -> Arbiter narrates; CODE computes the verdict.
```

- **Version-don't-mutate** makes Spec-4 parallel+repair safe: v(n) readers never corrupted; v(n+1) supersedes only when its checks complete.
- **Order-independence:** `status`/`maturity` are pure functions of the recorded verdict set — identical regardless of arrival order (the property that lets Spec 4 parallelize with zero schema change).
- **Single writer + append-only log:** agents return verdicts; the scheduler/`store.py` applies them; `run_log.py` is the append-only replayable record.

---

## 6. Package layout

```
validate-agents/
  valagents/
    llm.py         # OpenRouterClient (async, per-agent model/temp, tenacity retry, extract_json)
    parse.py       # parse_label  +  parse_tail / checked() strict-tail contract (§4)
    run_log.py     # JSONL event log (contextvars per-run, append-only, replay)
    web_search.py  # ArxivBackend, safe_search  (Grounder's external check)
    config.py      # roles -> models/temps; budget caps; repair cap (=3);
                   #   min_attack_categories (=2); fanout_N (=2)
    artifact.py    # schema + computed status/maturity/load_bearing/blocker (§2)
    store.py       # single-writer ArtifactStore + append-only verdict log
    agents/        # base protocol + 7 lenses + Arbiter/Repairer + Faithfulness/Entailment guards (§3)
    prompts/       # verbatim brief skeletons for all roles
    scheduler.py   # entry gates, DAG loop, fan-out policy, repair-versioning, total-gate termination (§5)
    cli.py         # valagents "<seed>" -> IdeaArtifact JSON + markdown report (carries the §1 limit sentence)
  tests/           # FakeLLM router (deterministic, no network)
  docs/2026-06-23-validate-agents-design.md
  results/<run_id>.jsonl
```

- **Proximity / SentenceTransformer not included** — earns its place in Spec 3's citation-aware retriever. YAGNI.
- **Provider:** OpenRouter; any role -> any model via `config.py`.

---

## 7. Worked cycle (the brief's escape-saddle example, Spec-1 path)

Seed: *"adding an antisymmetric curl term to gradient descent helps escape saddle points."*
- **Formalizer** pins θ̇ = −∇L(θ) + ω(t)·α(θ)·J·(θ−θ_c); `FALSIFIABLE: yes`.
- **Faithfulness** back-translates → "a rotational term added to GD escapes saddles faster"; `FAITHFUL: yes` (no narrowing — the pin keeps the seed's claim). Proceed.
- **Decomposer** → three claims, no edges: (A) curl term has nonzero projection on the negative-curvature direction; (B) α(θ) doesn't vanish *and doesn't saturate* at the saddle; (C) rotation doesn't disrupt convergence near minima. **Entailment** `COVERS: complete`.
- **Grounder** positions vs Curl-Descent + momentum, isolates the delta; records sources + independence.
- **Prover** checks (A)'s projection in closed form.
- **Predictor** commits to an escape-time scaling separating from GD/momentum/Curl-Descent.
- **Red-team** `ATTEMPTED: {counterexample, failure_regime, magnitude}`, surfaces the α-saturation mode → attack targets (B). *Reasoned*, not executed (caveat). Because (B) is load-bearing and `uncertain`, the **fan-out policy** runs a second independent magnitude angle before finalizing (B).
- **Validation-designer** → synthetic-saddle escape-time benchmark; `CONFIRM_IF` the scaling separates; `COST: low`.
- **Arbiter** narrates; **code computes** `status = needs_experiment`, `load_bearing = (B)`, `decisive_test = the escape-time benchmark`.

---

## 8. Tests that prove the invariants — `tests/` (deterministic, FakeLLM, no network)

**I1 — gate not narrate**
- Parse-4/6 regression: no tail → re-ask once → `uncertain`, never `pass`.
- Double re-ask failure: re-ask also malformed → `checked()` returns `None` → `uncertain` CheckRecord, no exception into scheduler, **both** bodies logged.
- Code wins over narration: Arbiter narrates `internally_validated` while a claim is `fail` → computed `refuted`; mismatch logged.
- `maturity ⊥ status`: `status` invariant under arbitrary injected `maturity`.

**I2 — validated = survived an external, independent check**
- No validation without a check: any root-ancestor `pending` → `internally_validated` impossible.
- Back-door (coverage gap): a `definitional` claim no lens covers → `pending`+`exhausted` → `needs_experiment`/`uncovered`, surfaced by claim id.
- **Independence (#2):** Grounder returns `supported` with `INDEPENDENT_SOURCES: 0` → downgraded to `uncertain` → claim never `pass` → not `internally_validated`.
- Honest outcomes reachable: scripted runs yield `refuted` and `needs_experiment`.

**I3 — the gate is total**
- **Faithfulness (#1):** seed "is collapse physical" → Formalizer drifts to "decoherence occurs", `FALSIFIABLE: yes` → Faithfulness `narrowed`/`no` → retry → terminal `refuted`/`unfaithful_*`; assert it never reaches `internally_validated`.
- Not falsifiable: `FALSIFIABLE: no` → `refuted`/`not_falsifiable`.
- **Empty decomposition (minor-1):** degenerate Decomposer → retry → `refuted`/`ill_formed`, no hang in `draft`.
- **Decomposition gap (#4):** sub-claims omit the load-bearing piece → `COVERS: gap` → `needs_experiment`/`decomposition_gap`, surfaced (known-partial).
- **Thin attack surface (#3):** Red-team `ATTEMPTED` missing `magnitude` (or <2 categories) → capped at `needs_experiment`/`thin_attack_surface`, even with all claims `pass`.
- **Fan-out policy (#5):** a load-bearing `uncertain` node is not `exhausted`/finalized until `fanout_N` independent lenses have run.
- Repair-cap exhaustion: fatal attack persists through 3 repairs → `finalized`, `refuted`.
- Landed non-fatal (D4): unresolved `major` → `needs_experiment`/`open_objection`; `minor` → `internally_validated` reachable, lower `maturity`.
- Order-independence: `status` identical across shuffled verdict-application orders.
- Version-don't-mutate: repair yields v2; v1 `CheckRecord`s untouched; only affected subgraph re-ran.

**Integration**
- The §7 escape-saddle seed, scripted end-to-end → `needs_experiment`, `load_bearing` = the α-non-saturation claim, with the fan-out second magnitude angle present.

---

## 9. Build & next step

- `validate-agents` is its own git repo. Spec doc committed first.
- **Next:** invoke writing-plans → phased plan (schema + parse → entry guards → lenses → scheduler/fan-out → tests), then implement, with the `maturity` formula as the learning-mode contribution.

### Decision log
- **D1** `FALSIFIABLE: no` → `refuted`/`not_falsifiable` (entry gate; extends totality to the entrance).
- **D2** coverage gap (`pending`+`exhausted` root-ancestor) → `needs_experiment`/`uncovered`; `blocker` keeps it distinct from `inconclusive`.
- **D3** Prover broadened to `definitional` well-formedness so no claim type is orphaned (prevention; `pending ≠ pass` is the backstop).
- **D4** severity-graded landed attacks: `fatal`→`refuted`, unresolved `major`→`needs_experiment`, `minor`→maturity-only.
- **D5** repair-cap exhaustion: scheduler finalizes; gate computes `refuted` from the persisting fatal attack (no special clause).
- **D6** double-parse-failure → `uncertain` CheckRecord (a check was attempted), distinct from `pending`.
- **D7** *(rev 2, #1)* **Faithfulness entry gate** — independent seed↔formal_claim back-translation; `narrowed`/`no` → one bounded re-formalization retry → terminal `refuted`/`unfaithful_*`. The SPOF upstream of the whole gate; a total gate on an unverified pin is the project's own failure mode relocated upstream.
- **D8** *(rev 2, #2)* **`CheckRecord` independence seam** — `sources[]` + `independent_sources`; Grounder `supported`+0-independent downgraded to `uncertain` in code. Carved now so Spec 3's citation-aware retriever needs no reshape.
- **D9** *(rev 2, #3)* **Red-team teeth** — `attack_surface.attempted`; magnitude mandatory; thin/mostly-skipped surface caps below `internally_validated`.
- **D10** *(rev 2, #4, known-partial)* **Entailment `COVERS` pass** — conjunction of sub-claims ⊢ `formal_claim`; `gap` caps below `internally_validated`. Catches obvious cases; Red-team backstops the rest.
- **D11** *(rev 2, #5)* **Fan-out policy in Spec 1, execution in Spec 4** — load-bearing `uncertain` nodes require `fanout_N` **diverse-type** lenses before finalizing; value is disagreement-detection, not agreement-as-corroboration (repetition ≠ corroboration — same correlation D8 guards for sources). The append-only join already supports it; only the trigger is new.
- **D12** *(rev 2, minor-1)* **Empty-decomposition guard** — degenerate Decomposer → retry → `refuted`/`ill_formed`; closes a totality hole.
- **Limit (rev 2, minor-2)** — `internally_validated` = "survived the checks this system can run," never "true"; every lens shares the base model's blind spots. Sentence ships in the report output, not just the spec.
