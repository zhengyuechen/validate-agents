# validate-agents — Spec 2 Design (Executed checks; first slice: symbolic known-limit recovery)

- **Date:** 2026-06-23
- **Status:** Approved design, pending implementation plan
- **Builds on:** Spec 1 (`docs/2026-06-23-validate-agents-design.md`), especially **R3** (`inject_limit_checks`: known limits → `mathematical`, `load_bearing` `AtomicClaim`s, today checked by the reasoned **Prover**).
- **Status line:** *Spec 2 upgrades "reasoned plausibility" into "executed checks" for mathematical limits (then magnitudes, then toy computations) — **without changing the pure gate.***
- **One-line goal:** Make known-limit recovery stop being a persuasive sentence and become a **computed symbolic equality** — a `CheckRecord(lens="executor")` produced by running a preregistered plan in a sandbox, judged by code, that the gate treats exactly like grounder/prover/redteam.

---

## 1. Scope

### In scope (this slice — Symbolic Limit Executor only)
- A **Computation-Designer** (LLM) that, for an R3 `limit_recovery` claim, emits a **structured** `ComputationPlan` (no runnable code — see F1).
- A **sandboxed Executor** (code, not an LLM) that builds and runs the SymPy computation from the frozen plan under resource limits.
- A **code verdict** (no LLM): symbolic equality of the computed limit to the preregistered `expected` → `ComputationVerdict` → `CheckRecord(lens="executor")`.
- Gate integration: add `"executor"` to `CheckRecord.lens`; extend the `claim.status` join so an executor pass is a strongest-pass. **`IdeaArtifact._evaluate()` is otherwise unchanged.**

### Out of scope (named so the seams are clear; later Spec-2 slices)
- **Lens 2 — Magnitude / Detectability Executor** (numpy/scipy; predicted effect vs sensitivity/bounds; catches "real but numerically inert"). Same `ComputationPlan`/`ComputationResult`/`ComputationVerdict` shapes, `kind: magnitude`. Built *after* the symbolic lens is solid.
- **Lens 3 — Toy-Model / Simulation Executor** (small falsifiable dynamics). Broader and easier to overfit, and the **only** lens that runs arbitrary generated code → needs container-grade isolation. Deliberately last.
- Verifying that the `expected` expression is the *correct known result from the literature* — see the **loud caveat** (§7).

---

## 2. The four design decisions (locked)

- **F1 — Structured `ComputationPlan`, NOT LLM-emitted code.** For known-limit recovery the LLM emits *fields* (`expression`, `variables`, `limit_variable`, `limit_point`, `expected`, `criterion`, `confirm_if`/`refute_if`); the **Executor builds the SymPy call itself**. There is no arbitrary code to run. This collapses "sandbox a hostile script" into "parse a few expressions in a restricted namespace + bound resources." Arbitrary generated code is forbidden in this slice (it belongs to lens 3).
- **F2 — Executor *augments* the Prover, does not replace it.** On a `limit_recovery` claim the Executor adds a `CheckRecord(lens="executor")`. When it yields a decisive verdict it **dominates** (computed > reasoned). When it can't (no plan / parse error / timeout / SymPy can't evaluate) it yields `uncertain`, and the claim falls back to R3's Prover-reasoned check.
- **F3 — The symbolic verdict is computed in CODE, not interpreted by an LLM.** `simplify(computed_limit − expected) == 0` against the **preregistered** `expected`. **No LLM ever sees the execution result and decides.** The LLM only *designs* the plan; the Executor *runs* and *judges* it. This is the whole point.
- **F4 — Isolation = subprocess + resource limits + no network + restricted parser.** Sufficient *because* F1 means no arbitrary code runs. Container-grade isolation is deferred to lens 3.

---

## 3. Non-negotiable rules (the preregistration discipline)

1. The Computation-Designer must **preregister, before any execution**: `expression`, `variables`, `limit_variable`, `limit_point`/`regime`, `expected` (+ `expected_source`), `criterion`, `confirm_if`/`refute_if`.
2. The plan is **frozen** (stored immutably) before the Executor runs. The Executor runs **only** the frozen plan.
3. The verdict is a **code comparison** against the frozen `criterion`/`expected`. Nothing — no LLM, no human — may change the criterion after seeing output.
4. The executed result becomes a `CheckRecord(lens="executor")`.
5. Sandbox: **network off, filesystem restricted, CPU/wall + memory limits, no arbitrary imports** beyond pinned safe libs (`sympy` for this slice).
6. **Every execution artifact is saved**: the frozen plan JSON, the exact SymPy invocation built from it, stdout/stderr, the result JSON, resource use — under `results/<run>/computations/<claim_id>/`.

---

## 4. Data model — `valagents/computation.py` (new; Pydantic v2)

```python
class ComputationPlan(BaseModel):          # FROZEN before execution
    kind: Literal["symbolic"]              # this slice; "magnitude"/"simulation" later
    expression: str                        # the idea's quantity, e.g. "G*M/r**2 * (1 + a/c**2)"
    variables: list[str]                   # declared symbols, e.g. ["G","M","r","a","c"]
    limit_variable: str                    # the symbol taken to a limit, e.g. "c"
    limit_point: str                       # "oo" | "0" | a value, e.g. "oo"
    expected: str                          # the preregistered known result, e.g. "G*M/r**2"
    expected_source: str = ""              # where the known result comes from (traceability; not yet verified — §7)
    criterion: Literal["symbolic_equality"] = "symbolic_equality"
    confirm_if: str = ""                   # human gloss; the DECISION is code, not this text
    refute_if: str = ""

class ComputationResult(BaseModel):        # what the sandbox produced
    ok: bool                               # ran to completion without error/timeout
    computed: str = ""                     # str(simplified computed limit)
    matched: Literal["confirm", "refute", "neither"] = "neither"
    stdout: str = ""
    stderr: str = ""
    error: str = ""                        # parse error / timeout / sympy failure
    resource_use: dict = {}                # wall_s, peak_rss_kb
    artifacts_path: str = ""

class ComputationVerdict(BaseModel):       # the gate-facing judgment
    verdict: Literal["pass", "fail", "uncertain"]   # equality holds / fails / couldn't run
    measured: str = ""                     # the computed value
    plan: ComputationPlan
    result: ComputationResult
```

`ComputationVerdict` → `CheckRecord(lens="executor", verdict=…, independent_sources=(1 if pass else 0), basis=<computed vs expected>, sources=[Source(locator=expected_source, relation="independent")])`. (A computed equality is an *independent* check — independent of the LLM — so on `pass` it carries `independent_sources=1`, letting a `mathematical` claim reach `pass`.)

---

## 5. Flow — wired into R3's `inject_limit_checks`

```
For each known-limit-recovery AtomicClaim L (type=mathematical, load_bearing, origin=limit_recovery):
  1. Prover (reasoned, as today) -> CheckRecord(lens="prover")            # R3 fallback, kept (F2)
  2. Computation-Designer(L, formal_claim, llm) -> ComputationPlan | None # structured fields (F1)
       if None / unparseable -> skip executor (claim relies on the Prover check)
  3. FREEZE plan -> Executor.run(plan)                                    # subprocess, limits (F4)
       sandbox: parse_expr(expression / expected) in a restricted namespace (declared symbols +
                whitelisted sympy funcs only); computed = sympy.limit(expr, sym(limit_variable), limit_point);
                diff = sympy.simplify(computed - expected_expr)
       verdict (CODE, F3):  diff == 0  -> pass (matched=confirm)
                            diff != 0  -> fail (matched=refute)
                            error/timeout/parse-fail -> uncertain (ok=False)
  4. ComputationVerdict -> CheckRecord(lens="executor") -> store.add_check(L.id, rec)
  5. save artifacts (plan, invocation, stdout/stderr, result JSON)
```

Gate consequence (via the **unchanged** `_evaluate`, which only reads `claim.status`):
- executor `pass` → L `pass` → contributes to `internally_validated` (the reduction is *demonstrated*);
- executor `fail` → L `fail` → **`refuted`** (the idea provably does not reduce to the known result — a computed contradiction);
- executor `uncertain` → falls back to the Prover's reasoned verdict on L.

---

## 6. Sandbox — `valagents/sandbox/` (subprocess runner)

- **`runner.py`** — a tiny, dependency-minimal script: reads a frozen `ComputationPlan` JSON on stdin; imports **only** `json` + `sympy`; builds symbols from `variables`; parses `expression`/`expected` with `sympy.parsing.sympy_parser.parse_expr(local_dict=<declared symbols>, global_dict=<whitelisted sympy functions only>, evaluate=True)` (NOT `sympify`, which can `eval`); computes the limit and the simplified difference; prints a `ComputationResult` JSON. No network code is imported; it touches no filesystem except stdout.
- **`executor.py`** — the parent: `subprocess.run([sys.executable, runner.py], input=plan_json, capture_output=True, timeout=cfg.sandbox.wall_s, preexec_fn=<set RLIMIT_CPU + RLIMIT_AS on POSIX>, env=<empty/minimal, no proxies>)`. Captures stdout/stderr/returncode/wall time; on timeout/non-zero/parse failure → `ComputationResult(ok=False, error=…)`. Writes all artifacts.
- **Config** (`config.py`): `sandbox.wall_s` (default 10), `sandbox.cpu_s` (10), `sandbox.mem_mb` (512), `sandbox.enabled` (true). If `sympy` import fails or `sandbox.enabled` is false, the Executor returns `uncertain` (graceful degradation to the Prover path).
- **Why this is safe enough here:** no arbitrary code is executed — only `parse_expr` over a restricted namespace plus fixed SymPy operations. The subprocess + rlimits bound a pathological symbolic computation (which can hang or blow memory); the empty env + pure-compute runner means no network/filesystem reach. Hostile-script isolation (containers) is a lens-3 concern.

---

## 7. The loud caveat (state it prominently in the doc AND the report)

> **The Executor verifies that the proposed `expression` reduces to the preregistered `expected` expression. It does NOT verify that `expected` is the correct known result from the literature.** The `expected` (and `expected_source`) are *LLM-asserted* — frozen and traceable, but chosen by the Computation-Designer. So this slice catches **"claims to recover X but doesn't"** (the common failure); it does **not** catch **"asserted the wrong X."** Validating `expected` against cited sources is a Spec-3 (grounding) follow-on. This is the honest successor to Spec 1's *reasoned-not-executed* line — the reduction is now executed; the target of the reduction is not yet grounded.

The report's executor `CheckRecord` basis must show: the computed limit, the `expected`, and `expected_source`, so a reader can see what was (and wasn't) verified.

> **Second residual — a computed inequality refutes only as well as the plan was transcribed.** A `fail` (→ `refuted`) means the *frozen `expression`* did not reduce to the *frozen `expected`*. If the Computation-Designer mis-transcribed the idea's expression (or the limit it must satisfy), the inequality is a plan artifact, not a true contradiction — a *false refute*. Mitigations, all already in the system: the plan + invocation + result are saved as artifacts for audit; the **Prover cross-check is retained** (F2); and a refuted limit-recovery claim is a **Repairer target** (a bad transcription → repair → new frozen plan). The design accepts that a computed inequality on a preregistered must-recover limit is the strongest refutation signal available *and* that its trustworthiness is bounded by the designer's transcription fidelity — which is why F2 keeps the reasoned path and the artifacts are auditable.

---

## 8. Gate integration (minimal, additive)

- `CheckRecord.lens`: `Literal["grounder", "prover", "redteam"]` → add `"executor"`.
- `AtomicClaim.status` join: an executor `pass` (with `independent_sources≥1`) is a strongest-pass — extend `has_proof_pass = any(c.lens in ("prover", "executor") for c in passes)` so a `mathematical` claim with an executor `pass` passes even over a non-blocking grounder uncertainty. An executor `fail` is caught by the existing `any(c.verdict == "fail")` → claim `fail`.
- `IdeaArtifact._evaluate()`: **unchanged** — it reads `claim.status`, never `lens`. The new lens flows through the join only. (Confirmed by a test that `_evaluate` source does not reference `"executor"`.)

---

## 9. Testing — `tests/` (deterministic, no network)

The Executor runs **real, pinned SymPy** (deterministic, offline) — so these tests exercise *actual computation*; only the Computation-Designer (LLM) is faked.

- **Executor (real SymPy):**
  - *recovers limit → pass*: plan `expression="G*M/r**2*(1+a/c**2)", limit_variable="c", limit_point="oo", expected="G*M/r**2"` → `simplify(diff)==0` → `pass`.
  - *wrong limit → fail*: same expression, `expected="G*M/r"` → `diff != 0` → `fail`.
  - *timeout/error → uncertain*: a pathological/unparseable plan → `ok=False` → `uncertain` (no exception escapes).
  - *restricted parse*: an `expression` containing a non-whitelisted name (e.g. `__import__("os")`) → parse rejects / errors → `uncertain`, **never executes**.
- **Computation-Designer (FakeLLM):** emits the structured tail → `ComputationPlan` parsed; unparseable → `None`.
- **Integration (FakeLLM designer + real Executor, via `inject_limit_checks`):**
  - recovers → executor `pass` → claim `pass`;
  - wrong → executor `fail` → claim `fail` → artifact **`refuted`**;
  - executor `uncertain` (designer returns no plan) → claim falls back to the Prover verdict.
- **Gate purity:** `inspect.getsource(IdeaArtifact._evaluate)` does not contain `"executor"`.

---

## 10. Build slices (the implementation order)

1. **Data model** — `computation.py` (`ComputationPlan`, `ComputationResult`, `ComputationVerdict`) + add `"executor"` to `CheckRecord.lens` + the `claim.status` join extension; unit tests.
2. **Sandbox runner** — `sandbox/runner.py` + `sandbox/executor.py` with a **toy safe command first** (e.g. compute `limit(1/x, x, oo) == 0`), proving subprocess + rlimits + timeout + artifact capture work; tests.
3. **Symbolic known-limit execution** — the real `parse_expr`/`limit`/`simplify` path + the restricted namespace; the Executor unit tests above.
4. **`ComputationVerdict → CheckRecord`** + the **Computation-Designer** agent/prompt; wire into `inject_limit_checks` (augment-with-fallback, F2).
5. **Negative tests** — recovers→pass, wrong→fail/refuted, timeout/error→uncertain; gate-purity test.
6. **Then** lens 2 (magnitude/detectability) — separate slice, same shapes.

---

## 11. Decision log
- **S2-D1** Structured `ComputationPlan`, no arbitrary generated code in this slice (F1). Arbitrary code → lens 3 only.
- **S2-D2** Executor augments the Prover with graceful fallback (F2): decisive executor verdict dominates; else fall back to the reasoned Prover check.
- **S2-D3** Symbolic verdict computed in code (`simplify(diff)==0`), never LLM-interpreted (F3).
- **S2-D4** Isolation = subprocess + rlimits + no-network + restricted `parse_expr` (F4); sufficient because no arbitrary code runs.
- **S2-D5** `_evaluate()` unchanged; only `CheckRecord.lens` and the `claim.status` join learn the `"executor"` source.
- **S2-D6 (loud caveat)** The Executor proves *reduction to the preregistered `expected`*, not that `expected` is the literature-correct result. Grounding `expected` is a Spec-3 follow-on.
- **S2-D7** Artifacts (plan, invocation, stdout/stderr, result JSON, resource use) saved per claim for audit/replay.
