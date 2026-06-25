# validate-agents — NLI Faithfulness-Gate Design (frozen bidirectional entailment, fail-closed)

- **Date:** 2026-06-25
- **Status:** Draft for review (no user Q&A — design calls made by the author, flagged for scrutiny in §11)
- **Builds on:** `valagents/agents/faithfulness.py` (`faithfulness_check`, `Faithfulness`), `valagents/scheduler.py` (`run_entry_gates`, Gate 2), `valagents/artifact.py` (`Faithfulness`, the `_evaluate` entry-gate reads), the injected-resolver pattern (`value_grounder.LiveFetcher`, `None`=off).
- **Source:** the Faithfulness card + "build first (three)" in `docs/2026-06-25_papers_for_validate_agents_report.md`. Third of the three cheap pure-code wins (CiteAudit ✓ → VeriGuard → **NLI**), ahead of Popper.
- **One-line goal:** Replace the LLM that back-translates-and-self-judges the faithfulness entry gate — which the paper measures as **near-random** (Pearson ≈ 0.02–0.10, ROC-AUC ≈ 53–55) — with a **frozen bidirectional NLI** check (e.g. AlignScore) used as a **fail-closed filter**: it can only *block* a run (mark it ill-posed), never *grant* anything. Cardinal-rule-safe precisely because the entry gate is block-only.

---

## 1. What the faithfulness gate does today

`run_entry_gates` Gate 2 (`scheduler.py:35–49`): `faithfulness_check(raw_idea, formal_claim, …)` asks an LLM to back-translate `formal_claim.statement` and self-judge `FAITHFUL: yes|narrowed|no` (prompt `prompts.py:26–45`). On `narrowed`/`no` it re-formalizes once and re-checks; if still `narrowed`/`no`, the run aborts: `art.finalized=True`, `verdict_class → "ill_posed"` (`artifact.py:231–233, 316`). It is already **fail-closed** (`tail is None → verdict="no"`, `faithfulness.py:18`).

The LLM self-judging whether its own formalization matches the seed is exactly the near-random task the paper indicts.

---

## 2. The design (frozen NLI, bidirectional, fail-closed)

Inject a frozen **NLI scorer** that returns an entailment probability `score(premise, hypothesis) ∈ [0,1]`. Compute **both directions** between the seed and the formal claim:
- `s_fwd = score(premise=raw_idea, hypothesis=formal_claim.statement)`
- `s_bwd = score(premise=formal_claim.statement, hypothesis=raw_idea)`

Verdict by **counting passing directions** against a threshold `t` (NF-D2 — sidesteps the entailment-direction philosophy):
- both `≥ t` (mutual entailment ⇒ same claim) → **`yes`**
- exactly one `≥ t` (one is a proper subset/special case ⇒ partial) → **`narrowed`**
- neither → **`no`**

This verdict feeds the **existing** Gate-2 flow unchanged (narrowed/no → re-formalize once → re-check → abort to ill-posed). `Faithfulness` keeps its fields (`verdict`, `back_translation`, `retried`); `artifact.py` is untouched (it reads `faithfulness.verdict`/`retried`).

---

## 3. Why this is cardinal-rule-safe (block-only)

The faithfulness gate **only ever blocks a run** — a `yes` does not validate anything; it merely *permits* the pipeline to proceed, after which every claim still faces the full gate. So even though an NLI score is "a model's word," its only power here is to **take a run away (ill-posed)**, never to grant credit. That is the opposite of the grounder, where a model's word would *grant* `independent_sources`.

**This is NOT a grounder credit-gate (NF-D3, the standing decision the paper confirms).** Tier-2 deliberately rejected NLI as a grounder *credit* gate ("an NLI score is still a model's word; it relocates say-so; unreliable on dense physics text"), and the Faithfulness paper independently shows NLI **collapses on the partial-support boundary** (AUC ~74–79) — the hard case. So NLI is adopted *only* at the block-only entry gate, never to grant `independent_sources`. (A future "NLI contradicts → downgrade a grounder claim, surfaced loud, never granting" is a *separate* slice and out of scope here.)

---

## 4. The residual & why imperfect NLI is tolerable here

NLI's weak spot is partial support — exactly the `narrowed` middle state. Two error directions:
- **False `narrowed`/`no` on a faithful idea** → wrongly blocks (ill-posed). Annoying, fail-closed, recoverable (the re-formalization retry; the user can rephrase the seed). Safe.
- **False `yes` on a drifted idea** → the run *proceeds*, but a `yes` grants no validation: the drifted idea's claims still face the full gate, which will challenge/refute weak claims. So a false `yes` admits an idea to the pipeline; it never validates it.

Net: NLI imperfection on partial support costs **recall at the entry gate** (and a borderline-drift admission), never a false validation. A **conservative threshold** (NF-D6) biases toward blocking when uncertain — the fail-closed direction. The paper's finding (frozen NLI ROC-AUC ~92 on the clear split, dominating the near-random LLM) is the reason to switch despite the partial-support collapse.

---

## 5. Injection & the dependency (net-new, optional)

No NLI/torch/transformers dependency exists today (`requirements.txt` has none; grep is empty). Forcing a heavy model (AlignScore/RoBERTa or AutoAIS-T5-11B + torch) as a hard dependency is not warranted for a cheap win. So (NF-D4):

- Define a Protocol `NLIScorer` with `score(premise: str, hypothesis: str) -> float` (entailment prob in `[0,1]`).
- `faithfulness_check(raw_idea, formal_claim, llm, cfg, retried=False, nli_scorer=None)` — **`nli_scorer is None` → off → the current LLM self-judge runs unchanged** (today's behavior, no forced dep). When present, NLI **replaces** the LLM verdict.
- A concrete `MNLIScorer` (in `valagents/nli.py`) wrapping **`roberta-large-mnli`** via `transformers` (NF-D4, model choice): it yields a clean `P(entailment)` in `[0,1]` (the natural `score` semantics), and is lighter/more standard than AlignScore (AlignScore is the paper's top performer but a heavier specific package — offered as an optional alternative, not the default). **Lazy-imports** the model on first `score`; the CLI constructs it only when configured (mirrors `LiveFetcher`). Tests inject a fake `NLIScorer` returning canned floats — no model, no network.
- Wire through `run_entry_gates(…, nli_scorer=None)` from `run(…)`; the CLI builds the scorer from config and passes it.
- Fail-soft: an NLI scorer that raises → **`no`** (fail-closed block), recorded, never crashes. (Stricter than grounding's fail-*soft*-to-skip, because this gate's safe direction is to block.)

Config: `FaithfulnessCfg{ nli_backend: str = "none" (none|mnli|alignscore), nli_threshold: float = <set by the §5b probe> }` on `Config.faithfulness`; the CLI builds the scorer iff `nli_backend != "none"`.

### 5b. PREREQUISITE — concordance probe before trusting/enabling NLI (was flag #6, now a gate)

The AUC~92 that justifies the switch is on the paper's general/citation-text benchmark; physics formalizations ("the PSD of YbZn₂GaO₅ is temperature-independent") are **out-of-distribution** for an MNLI model — it may be near-random here too, in which case the switch trades one near-random judge for another *plus* a heavy dependency. So, mirroring the discipline applied to Popper's nulls: **before enabling NLI in any real run, run a small concordance probe** — a handful of real `(seed, formal)` pairs hand-labeled faithful / narrowed / drifted — and confirm the NLI verdict beats the LLM self-judge on them. That probe **also sets `nli_threshold` (NF-D6)**, which is otherwise an unvalidated guess. Ship the *capability* (this spec), but treat NLI-beats-LLM-on-physics as **unproven until the probe passes**; the threshold default is "set by the probe," not a fixed number.

**Enablement note (NF-D5b):** off-by-default keeps the heavy dep optional, but the consequence is **off = the near-random gate the paper indicts stays in force.** The fix only lands when the user sets `nli_backend="mnli"` (after the probe) for real runs / the qual. The spec ships the capability; landing the improvement requires enabling it.

---

## 6. The `back_translation` field under NLI (NF-D5)

With NLI replacing the LLM, there is no generative back-translation. To avoid a second (LLM) call, set `back_translation` to a **structured note** carrying the scores: `f"(NLI gate: fwd={s_fwd:.2f}, bwd={s_bwd:.2f}, threshold={t})"`. The report loses the prose restatement but gains a defensible, inspectable verdict basis. (Alternative, deferred: keep one LLM call purely for the human-readable back-translation while NLI owns the verdict — costs a call; not worth it for v1.) `retried` is threaded unchanged.

---

## 7. Off / errors / determinism
- **Off (`nli_backend="none"` / `nli_scorer=None`) → today's LLM self-judge, byte-identical** entry-gate behavior (regression pin).
- **Scorer error → `no`** (fail-closed block), recorded; never crashes.
- **Determinism:** a frozen NLI model with a fixed threshold is deterministic given the inputs (no sampling temperature, unlike the LLM). The verdict basis (scores) is recorded for reproducibility. "Code-adjudicated" here is the *frozen/independent/reproducible* sense, not symbolic proof — a fail-closed detector with a fixed threshold, never ground truth.

---

## 8. Testing
- **Verdict mapping (fake scorer, no model):** both directions ≥ t → `yes`; one ≥ t → `narrowed`; neither → `no`; exactly-at-threshold boundary (`>=`) pinned.
- **Fail-closed:** a scorer that raises → `no`; `nli_scorer=None` → the existing LLM path runs (assert the LLM was called, NLI was not).
- **Entry-gate integration (`run_entry_gates`, fake scorer + FakeLLM):** NLI `no` → re-formalize once → NLI re-check → still `no` → run aborts with `verdict_class=="ill_posed"`, reason `unfaithful_drift`; NLI `narrowed` (persisting) → `unfaithful_narrowed`; NLI `yes` → run proceeds to decomposer. These mirror the existing entry-gate tests with the NLI verdict source.
- **Regression:** `nli_backend="none"` → the existing faithfulness tests pass unchanged.
- **Threshold knob:** the same score pair yields `yes` at `t=0.4` and `narrowed`/`no` at `t=0.7` — confirms the conservative-threshold lever.
- (No test loads a real model; `MNLIScorer`'s lazy import is covered by a thin "constructs and lazy-imports on first call" test guarded/skipped if the optional dep is absent.)

---

## 9. Cardinal-rule fit
Frozen, independent, reproducible adjudication at a **block-only** gate: NLI can only remove a run (ill-posed), never grant credit, so importing a model's judgment here does not relocate say-so into a verdict. Off-by-default (no forced heavy dep; LLM fallback). NOT reused as a grounder credit-gate (NF-D3). `artifact.py` untouched.

---

## 10. Files
- `valagents/nli.py` (new) — `NLIScorer` Protocol, `MNLIScorer` (`roberta-large-mnli`, lazy import; optional `AlignScoreScorer`), the verdict-from-scores helper.
- `valagents/agents/faithfulness.py` — `faithfulness_check(…, nli_scorer=None)`; NLI path + LLM fallback + fail-soft.
- `valagents/scheduler.py` — thread `nli_scorer` through `run_entry_gates` and `run`.
- `valagents/cli.py` — build the scorer from `cfg.faithfulness` and pass it.
- `valagents/config.py` — `FaithfulnessCfg{nli_backend, nli_threshold}`; `Config.faithfulness`.
- `requirements.txt` — `transformers`/`torch` (for `roberta-large-mnli`) as an **optional** extra (opt-in, not a hard install); `alignscore` optional.
- A small probe harness/script + a few labeled `(seed, formal)` pairs for §5b (not a unit test — a one-time validation gate before enablement).

---

## 11. Decision log
- **NF-D1 (fail-closed block-only filter)** NLI feeds the existing entry gate, which only blocks (ill-posed). It never grants. Safe to import a model's judgment here.
- **NF-D2 (bidirectional, count passing directions)** 2/1/0 directions ≥ threshold → yes/narrowed/no. Avoids committing to which entailment direction means "narrowed."
- **NF-D3 (NOT a grounder credit-gate)** Standing Tier-2 decision, paper-confirmed (NLI collapses on partial support). NLI grants nothing; a future contradicts-downgrade-only grounder use is a separate slice.
- **NF-D4 (injected, optional, LLM fallback; model = `roberta-large-mnli`)** `nli_scorer=None` → today's LLM self-judge. Concrete scorer wraps `roberta-large-mnli` (clean `P(entailment)`, lighter than AlignScore; AlignScore an optional alternative), lazy-imported, built by the CLI only when `nli_backend != "none"`; tests inject a fake. Net-new dep is opt-in.
- **NF-D8 (concordance probe is a PREREQUISITE, not a flag)** NLI must beat the LLM self-judge on a small hand-labeled set of real physics `(seed, formal)` pairs before it is trusted/enabled; that probe sets `nli_threshold`. Until it passes, NLI-beats-LLM-on-physics is unproven (the AUC~92 is out-of-distribution). §5b.
- **NF-D5 (scores in `back_translation`)** NLI-on skips the LLM call; `back_translation` carries `fwd/bwd/threshold` (leaner; loses prose). [flag]
- **NF-D6 (conservative threshold knob)** `nli_threshold` default 0.5, tunable; biases toward blocking when uncertain (fail-closed). [flag — load-bearing given the partial-support collapse]
- **NF-D7 (scorer error → `no`)** Fail-closed (block), stricter than grounding's fail-soft-to-skip, because the gate's safe direction is to block.

---

## 11b. Reviewer-scrutiny flags (author's uncertainties — attack these)
1. **NF-D6 threshold is load-bearing and unvalidated here.** The paper shows NLI collapses on partial support; 0.5 is a guess. False-`yes` (admit drift) vs false-block (reject faithful) trade off around `t`. Should the default be higher (more blocks, safer) given fail-closed intent? Is a *single* threshold for both directions right, or should `narrowed` need a margin between `s_fwd` and `s_bwd`?
2. **NF-D2 count-directions vs a real subset test.** "Exactly one direction ≥ t → narrowed" assumes asymmetric entailment cleanly separates subset from drift. On dense physics text NLI may pass both directions spuriously (→ false `yes`) or neither (→ false `no`). Is the count rule robust, or does it need a confidence margin?
3. **NF-D5 losing the prose back-translation.** The report's human-readable restatement is a real artifact for the qual. Is dropping it (scores only) acceptable, or should one LLM call be kept for the prose while NLI owns the verdict?
4. ~~AlignScore vs MNLI~~ — **RESOLVED (NF-D4):** default `roberta-large-mnli` (clean `P(entailment)`, lighter); AlignScore optional.
5. ~~Default off means the near-random judge stays the default~~ — **ACKNOWLEDGED (NF-D5b):** off-by-default keeps the dep optional; landing the fix requires enabling `nli_backend="mnli"` after the §5b probe. Stated explicitly, not silent.
6. ~~Domain mismatch~~ — **ELEVATED to a prerequisite (§5b / NF-D8):** the concordance probe on real physics pairs must pass before trusting/enabling NLI; it also sets the threshold. No longer a flag — a gate.
