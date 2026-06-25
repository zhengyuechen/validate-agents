# validate-agents — Popper E-Value Gate Design (calibrated evidence accumulation)

- **Date:** 2026-06-25
- **Status:** Draft, REVISED after adversarial review. **Key change: v1 SURFACES the e-value as a computed field; it does NOT gate the verdict on it.** The gate-migration (replacing `independent_sources ≥ 1` with `E ≥ 1/α`) is **deferred** behind a concordance check (E-tracks-correctness on a labeled gold set), κ>0.5, and a correlation-robust cross-lens combination. So v1 does **not** change `_evaluate`'s verdict logic; `artifact.py` gains only a new computed `evidence_strength` field. Rationale in §3/§5.
- **Builds on:** the gate in `valagents/artifact.py` (`CheckRecord`, `AtomicClaim.status`, `_has_independent_external_check`, `_evaluate`, `verdict_class`, all `@computed_field` with no setter), `valagents/config.py` (`GateCfg`), the per-lens CheckRecord producers (`agents/grounder.py`, `agents/prover.py`, `computation.py` `verdict_to_check`, `sandbox/runner.py`).
- **Source:** the Popper card + "Design sketch: the e-value gate" in `docs/2026-06-25_papers_for_validate_agents_report.md` (arXiv:2502.09858). The strategic upgrade, after the three cheap wins.
- **One-line goal:** Build the **calibrated evidence machinery** — each statistical check emits a p-value under an explicit null → e-value `e = κ·p^(κ−1)` → aggregate `E = ∏ eₛ` — and **surface E as an evidence-strength signal** (a computed field the arbiter/basis read, a calibrated successor to the source-count). It is the principled *form* of the deferred ≥2 question (corroboration **strength**, not **tally**), but v1 **does not gate the verdict on it**: the gate stays count-based until E is concordance-validated (§5/§6). The headline is deliberately NOT "any-time-valid Type-I guarantee" — that guarantee requires independence the system cannot witness (§5), and claiming it would be exactly the false-authority the cardinal rule forbids.

---

## 1. Why count→e-value (the problem with the bar)

Today a claim is externally validated when some passing check has `independent_sources >= 1` (`artifact.py:155`, `:218`). Two checks each with `independent_sources=1` count the same as one; a simulation passing 9/10 grid points counts the same as 6/10; the deferred ≥2 bar is an arbitrary tally. The count throws away **strength of evidence** and cannot principledly combine many noisy/correlated checks.

Popper's calibrator turns each test into an **e-value** (`eᵢ = κ·pᵢ^(κ−1)`, `κ∈(0,1)`, assumption-free), and accumulates **multiplicatively** (`Eᵢ = ∏ eₛ`). In Popper's setting (a *sequential* sequence of *independent* tests) the product is a super-martingale under the null, giving any-time-valid Type-I control. **That guarantee does NOT survive transport into this gate:** our checks are neither sequential nor witnessably independent (§5), so `E[∏ e] > 1` under correlation and the Type-I bound is lost. v1 therefore takes the e-value as a *better-calibrated evidence-strength heuristic than a raw count* — surfaced, not gated — and earns the gate-migration only after a concordance check validates it empirically (§6).

---

## 2. The e-value computation — additive, back-compatible (PP-D1)

This defines how the surfaced `evidence_strength` (E) is **computed**. (The `E ≥ 1/α` decision rule below is the *deferred gate-migration's* semantics, PP-D8 — v1 computes and surfaces E but the verdict does not use this rule.) **The e-value reduces *exactly* to a count when only counts exist, and grades evidence when p-values exist** — the property that would make the eventual gate-migration safe.

Per passing check, define its e-contribution `e(c)`:
```
e(c) = 1.0                          if c.verdict != "pass"        # no evidence
e(c) = κ · c.p_value ** (κ − 1)     if c.p_value is not None      # calibrated (statistical)
e(c) = E_EXACT                      elif c.independent_sources >= 1   # exact/count pass (back-compat)
e(c) = 1.0                          else                          # pass, but no evidence (e.g. mechanistic prover)
```
Per claim: `E = ∏ e(c)` over its checks → this is the surfaced `evidence_strength`. *(Deferred gate-migration rule: a claim would be externally validated when `E ≥ 1/α` — but v1 does NOT apply this; the count-based gate decides, §3.)*

- `E_EXACT` is a fixed strong contribution chosen so **one** exact/count pass validates: `E_EXACT ≥ 1/α` (with `α=0.1`, `E_EXACT=10` ⇒ one count-pass validates, exactly today's `≥1` semantics). So an existing `CheckRecord(verdict="pass", independent_sources=1, p_value=None)` → `e=E_EXACT ≥ 1/α` → validated; `independent_sources=0, p_value=None` → `e=1.0` → not validated. **Every existing gate test that depends on `independent_sources` is preserved unchanged.**
- A statistical check with `p_value` set grades continuously: a grounder with two independent sources at `p=0.5` each (§4) → `E = e(0.5)² > E_EXACT` → stronger than one; a simulation at `frac=9/10` gets a smaller p (bigger e) than `6/10`. This is the ≥2 story, made calibrated.

So the layer is **additive**: lenses that set `p_value` feed the calibrated path; everything else falls back to the count path = today. No lens is *required* to emit a p-value for the gate to keep working.

---

## 3. What changes in `artifact.py` — v1 SURFACES E, does NOT gate on it

The verdict logic (`_evaluate`, `AtomicClaim.status`, `_has_independent_external_check`) is **untouched in v1** — it stays count-based (`independent_sources >= 1`). The scary surgery on `AtomicClaim.status`'s `uncertain`/`fail`/`_math_uncertainty_is_nonblocking` branches is **not performed**; the back-compat-transition question is moot because no transition moves. What v1 adds:

- **`CheckRecord` gains `p_value: float | None = None`** — the check's p-value under its own null (None = non-statistical/exact).
- **`GateCfg` gains** `evalue_kappa: float = 0.7` (**> 0.5**, see §5 — at κ≤0.5 the e-value has infinite second moment, which makes correlated multiplication unbounded), `evalue_alpha: float = 0.1`, `evalue_exact: float = 10.0`; stamped onto the artifact by `_apply_gate_cfg` (`scheduler.py:12`) as data fields, exactly like `min_attack_categories`. Data, not setters.
- **A pure helper** `_claim_evalue(self, claim) -> float` computes `∏ e(c)` per §2 (the back-compat-from-count `e(c)` definition still gives a sensible E: a count-pass contributes `E_EXACT`, a p-valued check its calibrated e).
- **A new computed field `AtomicClaim.evidence_strength`** — `@computed_field @property` returning `_claim_evalue(self)`. **Read-only signal; nothing branches the verdict on it.** Crucially, to avoid re-introducing false authority through the *reader's* eye (PP-D8 removed it from the machine; a "E=42, validates-at-≥10" basis would smuggle it back in — a `1/α`-shaped number reads as a verdict): in v1 `evidence_strength` is surfaced **only to the artifact JSON / logs / the concordance harness (developer-facing)**, **NOT to the human-facing report**, and **never displayed against a `1/α` or "validates-at" threshold** — only as a bare, explicitly-labeled heuristic ("evidence-strength, higher = more corroboration; not a validation threshold"). It is **promoted into the human report only after the concordance check exists (§5b/§6)**, and then as a **gold-set percentile** ("80th pct of validated ideas") — honest *relative* strength, which structurally cannot even be computed without the gold set. So one gate ("does a labeled gold set say E means anything?") unlocks both human-facing display and gating. `status`/`verdict_class` are unchanged.
- **Gate purity preserved and strengthened:** the only new field is a pure `@computed_field` with no setter; the verdict-determining computed fields are byte-identical to today. Surfacing a number cannot launder into a verdict because nothing reads it for a verdict (v1).

**The deferred gate-migration** (a future slice, gated on §6's prerequisites) would then replace the two `independent_sources >= 1` sites (`artifact.py:155`, `:218`) with `evidence_strength >= 1/evalue_alpha` — but only after E is shown to track correctness and a correlation-robust combination replaces naive `∏`. v1 deliberately stops short of that.

---

## 4. The p-value per check type (the hard part — where the research risk lives)

The honest core: **defining a defensible null + p-value per check type is real work, and lives WITH each lens** (it sets `CheckRecord.p_value`), not in the gate (the gate only multiplies). v1 defines p-values where a defensible null exists and bypasses the rest to `E_EXACT`.

- **Grounder corroboration (PP-D3a):** null = "spurious topical co-occurrence" — a retrieved on-topic abstract carrying a passing on-property quote *by chance*. Model each passing source as a Bernoulli(q0) spurious hit ⇒ `p = q0 ** code_witnessed` (the dedup'd count, `grounder.py:100`). `q0` a config knob (default 0.5 → `p = 0.5^N`). N=1 → p=0.5; N=2 → p=0.25 → the calibrated ≥2 story. Set `p_value` in `ground_claim`. **NB (the independence assumption, made quantitative — §5):** `q0^N` treats the N sources as *independent*, which Tier-2 cannot witness (no authors in `Article`). Same residual the count carried, now numeric and thus more seductive — keep `q0` conservative (closer to 1), and remember this is a heuristic strength, not a calibrated guarantee.
- **Magnitude with uncertainty (PP-D3b):** `discriminating_margin` already computes `margin = |predicted − closest| / uncertainty` (`runner.py:~430`) — literally a z-score; `sensitivity_ratio` similarly. Null = "no effect beyond noise" ⇒ `p = normal_sf(margin)` (one-sided). The margin/ratio is currently only in the `computed` string; v1 returns it as a structured float and sets `p_value`. (`bound_check` is exact — no p; bypass.)
- **Simulation `robust_frac` (PP-D3c):** `passes`/`gsize` over the grid. Null = "passes by chance at rate p0" ⇒ one-sided binomial `p = P(X ≥ passes | Binomial(gsize, p0))`. **Negative-control** (`null_overrides`) has a *clean* null (the null arm should not discriminate; p0 is the chance rate of spurious discrimination); a **plain criterion** has an ill-defined null (default `p0=0.5` — "absent the mechanism a binary criterion passes ~half the time" — defensible but coarse; **flag**). The runner returns structured `passes`/`gsize`; the executor computes the binomial p and sets `p_value`.
- **Exact checks bypass (PP-D3d):** prover derivation pass, symbolic `simplify==0`, magnitude `bound_check` — deterministic, no sampling, no natural p-value → `p_value=None` → `e=E_EXACT` (one exact pass validates a math/definitional claim, preserving today). A mechanistic prover pass with `independent_sources=0` stays `e=1.0` (no external evidence) — exactly today (a mechanistic prover pass is not external support, `prover.py:50`).

**The gate never sees a null.** Each lens owns its null and emits a p-value (or None); the gate's job is purely `∏` + compare. Say-so stays out of the aggregation (the LLM designs the tests; the p-value is computed from execution counts/magnitudes/retrieval, all code).

---

## 5. The independence risk — why v1 surfaces E instead of gating on it (PP-D7, PP-D8)

`E = ∏ eₛ` carries a Type-I meaning **only if the checks are sequential and independent under the null.** Our checks are neither. Two failure layers, both fatal to the *guarantee* (not just anti-conservative):

**(a) Infinite second moment at κ≤0.5 (the knife-edge — verified).** For `e(p)=κ·p^(κ−1)`, `p~U(0,1)`: `E[e]=1` (valid e-value), but `E[e²]=κ²∫₀¹ p^(2κ−2)dp = κ²/(2κ−1)` for `κ>0.5`, and **diverges (∞) for κ≤0.5**. The original default `κ=0.5` sat *exactly* on the infinite-variance boundary. For correlated `e₁,e₂`, `E[e₁e₂]=1+Cov`, and `|Cov|≤√(Var·Var)` — so at κ≤0.5 a single correlated pair can blow E up *unboundedly*, not merely nudge it. **Fix: `evalue_kappa = 0.7` default (>0.5), so the second moment is finite (`E[e²]=0.49/0.4≈1.225`) and correlation damage is bounded.** Higher κ = weaker per-test e but robust to correlation — the right trade when independence is un-witnessed.

**(b) κ>0.5 bounds the damage but does NOT restore the guarantee.** Multiplying non-independent, non-sequential e-values still gives `E[∏]>1` under correlation. The independence assumption is baked in at *two* levels — within the grounder's `q0^N` (§4a) and across lenses (a simulation and its negative-control arm; a magnitude check and a grounding of the same number are correlated by construction). Tier-2 **cannot witness source independence** (no authors in `Article`). So a gate on `E ≥ 1/α` would wear the authority of a Type-I bound it cannot honor — exactly the false-authority the cardinal rule exists to prevent, and the same failure mode as `NLI-as-grounder-gate` and `refine-never-empty`, both already rejected. A heuristic dressed as a guarantee is *worse* than the honest count, because it invites trust it hasn't earned.

**Resolution (PP-D8 — the load-bearing decision): SURFACE E, do not GATE on it (v1).** Build the machinery and show E as an evidence-strength signal (a calibrated successor to the count), but keep the verdict count-based. Earn the gate-migration only after **all three** hold: (1) a **concordance check** shows E tracks correctness on a small labeled gold set of known-valid/known-refuted ideas (the Co-Scientist idea, §6); (2) `κ>0.5`; (3) a **correlation-robust** cross-lens combination (e.g. require *distinct lens types* to multiply, cap the factor count, or down-weight within-lineage evidence) replaces naive `∏`. The fully-sound version needs a non-say-so independence signal — same frontier as the ≥2 slice's "non-saturation subject signal." Until then, multiplying is a heuristic, labelled as one.

**Single-p-per-lens (kept regardless):** the grounder emits **one** `p_value` for its N sources (`q0^N`), never N separate per-source e's — so within-lens correlation isn't double-counted even in the surfaced E.

---

## 6. Scope (honest — this is a research direction, not a drop-in)

**v1 ships (machinery + SURFACE only — no gate change):** `CheckRecord.p_value`; `_claim_evalue` + the back-compat-from-count `e(c)` (§2); the `evidence_strength` computed field (§3) read by the arbiter/basis; the `GateCfg` knobs (**`evalue_kappa=0.7`**); the grounder p-value (`q0^N`, §4a — makes the ≥2 *strength* visible) and the magnitude p-value (`normal_sf(margin)`, §4b — nearly free, the z-score exists). Symbolic/prover/bound_check bypass to `E_EXACT`. **`_evaluate` and the verdict are unchanged.**

**Prerequisite — the SAME concordance check gates BOTH the gate-migration AND human-facing E:** assemble a small labeled gold set of known-valid / known-refuted ideas, run the pipeline, and confirm `evidence_strength` (E) *separates* them (high E on valid, low on refuted). Until it passes: E is **not** gated on (count-based gate) **and not shown to humans** (logs/JSON only). After it passes: E may gate the verdict (with the §5b correlation-robust combination) **and** may appear in the human report as a gold-set percentile. One gate unlocks both. This is the Co-Scientist "validate-the-validator" idea; per the cross-cutting principle it is the **prerequisite slice**, not an afterthought. It also calibrates `q0`/`κ`/`α`. **(This harness deserves its own spec — it is now load-bearing for Popper's migration, NLI's enablement, and interpretable E.)**

**Deferred (the gate-migration slice, after the prerequisite):** replace the two `independent_sources ≥ 1` sites with `evidence_strength ≥ 1/evalue_alpha`; introduce the correlation-robust cross-lens combination (§5b); the simulation binomial p (§4c — needs structured `passes`/`gsize` and a defensible `p0`; the plain-criterion null is the weakest, deferred because a coin-flip null fabricates a p-value); a non-say-so independence signal.

The staged point: **the machinery + the surfaced signal is mechanical and testable now and resolves ≥2 *visibly*; gating the verdict on it is the multi-week research, and is not done until E is empirically shown to track correctness.**

---

## 7. Cardinal-rule fit
The calibrator and the `∏ e` aggregation are **pure deterministic arithmetic** over numbers the lenses already produce (retrieval counts, z-scores, grid fractions); p-values come from lens-owned nulls computed in code, not LLM say-so; Popper's own LLM relevance-checker is **not** adopted (kept out). **In v1 the verdict does not read E**, so correlated multiplication cannot manufacture a false validation — the soundness risk (§5) is structurally closed for the verdict and reduced to a *presentation* concern (don't let a surfaced number imply a guarantee). The verdict-determining computed fields are unchanged; `α`/`κ` are config data; the gate stays a pure computed function. The un-witnessed-independence residual is documented and is the explicit precondition on the deferred gate-migration — not silently assumed away.

---

## 8. Files
- `valagents/artifact.py` — `CheckRecord.p_value`; `_claim_evalue` helper; **a new read-only `AtomicClaim.evidence_strength` `@computed_field`**; read `self.evalue_kappa`/`self.evalue_alpha`/`self.evalue_exact`. **The verdict-determining computed fields (`status`, `_has_independent_external_check`, `_evaluate`) are NOT modified in v1.** (Gate purity preserved; no setters; nothing branches the verdict on the new field.)
- `valagents/config.py` — `GateCfg.evalue_kappa=0.7`, `evalue_alpha=0.1`, `evalue_exact=10.0`, `evalue_q0=0.5` (grounder null); stamp in `scheduler._apply_gate_cfg` + the `IdeaArtifact` fields.
- `valagents/agents/grounder.py` — set `CheckRecord.p_value = q0 ** code_witnessed`.
- `valagents/computation.py` / `valagents/sandbox/runner.py` — return structured `margin`/`ratio`; `verdict_to_check` sets `p_value = normal_sf(margin)` for magnitude `discriminating_margin`/`sensitivity_ratio`.
- `valagents/agents/arbiter.py` / run-logs / artifact JSON — surface `evidence_strength` **developer-facing only** in v1 (logs/JSON/harness), as a bare labeled heuristic with NO `1/α` framing; **NOT in the human-facing `cli.py` report** until the concordance check passes (then as a gold-set percentile). PP-D8 / §3.
- Tests: `tests/test_artifact_*.py` — **add `evidence_strength` cases; the existing gate/verdict tests need NO changes (the gate is untouched)** — a regression check, not a rewrite.

---

## 9. Decision log
- **PP-D1 (additive, back-compat from count — the keystone)** `e(c)` = calibrated from `p_value` if present, else `E_EXACT` if `independent_sources≥1`, else `1.0`. Reduces *exactly* to today when only counts exist; every existing gate test is preserved. New statistical checks grade continuously.
- **PP-D2 (v1 SURFACES, does not gate — REVISED)** v1 adds a read-only `evidence_strength` `@computed_field` (= `_claim_evalue`) that the arbiter/basis surface; the verdict logic (`_evaluate`, `status`, `_has_independent_external_check`) is **unchanged** and stays count-based. Gate purity preserved (new field has no setter; nothing branches the verdict on it). The gate-migration to `E ≥ 1/α` is deferred (PP-D8).
- **PP-D3 (p-value lives with each lens)** Grounder `q0^N`; magnitude `normal_sf(margin)`; simulation binomial (deferred — weak plain-criterion null); exact checks (prover/symbolic/bound_check) bypass to `E_EXACT`. Each lens owns its null; the aggregator only multiplies.
- **PP-D4 (κ calibrator, κ>0.5 required)** `e=κ·p^(κ−1)`; **default `evalue_kappa=0.7`, NOT 0.5** — at κ≤0.5 the e-value's second moment is infinite (`E[e²]=κ²/(2κ−1)`), making correlated multiplication unbounded (§5a). The super-martingale/any-time-valid property holds only for sequential *independent* tests and does **not** transport to this gate (§5b) — `E` is a calibrated *heuristic strength*, not a Type-I guarantee.
- **PP-D5 (the principled-FORM of ≥2, surfaced)** E expresses corroboration *strength* not tally, and is shown as a calibrated successor to the count — but v1 does not gate on it; the count still decides. The ≥2 *resolution* arrives with the gate-migration (PP-D8), post-concordance.
- **PP-D8 (surface-don't-gate is the resolution of the independence risk — load-bearing)** Because naive `∏` over correlated, non-sequential, un-witnessably-independent checks gives `E[∏]>1`, gating on `E ≥ 1/α` would claim a Type-I authority the system can't deliver (false-authority — the cardinal-rule failure mode). So v1 surfaces E and keeps the count-based gate; the gate-migration requires (1) a concordance check that E tracks correctness on a labeled gold set, (2) κ>0.5, (3) a correlation-robust cross-lens combination. This decision also dissolves the prior scrutiny flags #1/#2/#5/#7 (no gate surgery, no back-compat-transition risk, no `E_EXACT`-in-gate, minimal test churn).
- **PP-D6 (honest staged scope)** v1 = layer + aggregation + gate + back-compat + grounder & magnitude p-values; simulation binomial + per-null tuning + independence-witnessing deferred. The nulls are the multi-week research; the gate arithmetic is mechanical now.
- **PP-D7 (independence caveat, bounded + documented)** `∏` assumes independence the system can't fully witness; mitigated by dedup + conservative knobs + one-p-per-lens; the fully-sound version needs a non-say-so independence signal, deferred. Surfaced in the basis, not silently assumed.

---

## 10. Testing
- **The gate is untouched — the critical regression pin:** run the *entire existing* `tests/test_artifact_*.py` suite **with no changes** and confirm green. v1 adds a field; it must move no verdict. (This is the payoff of PP-D8 — no back-compat-transition audit needed.)
- **`evidence_strength` (`_claim_evalue`) values:** a `CheckRecord(verdict="pass", independent_sources=1, p_value=None)` → `E=E_EXACT`; `independent_sources=0, p_value=None` → `E=1.0`; two grounder sources (`p=0.25`) → larger E than one (`p=0.5`); a magnitude `margin=3` (`p≈0.0013`) → larger E than `margin=1` (`p≈0.16`). These pin the SIGNAL; none gates the verdict.
- **κ second-moment guard (PP-D4):** assert `evalue_kappa > 0.5` is enforced (a config validator or a documented invariant test) so the infinite-variance regime can't be configured silently.
- **Gate purity:** `evidence_strength`/`status`/`verdict_class` are computed with no setter; reading `status` twice is idempotent; nothing writes any of them; the verdict is independent of `evidence_strength` (construct two artifacts differing only in `p_value` → identical `status`, differing `evidence_strength`).
- **One-p-per-lens (PP-D7):** the grounder emits a single `p_value` for its N sources (`q0^N`), not N multiplied e's.
- **Surfacing:** the arbiter record / report basis include `evidence_strength`.

---

## 11. Cardinal-rule risk register (read with §5)
- **Correlated-evidence multiplication** can over-count E — but in v1 it **cannot manufacture a false validation**, because the verdict does not read E (PP-D8). The residual is a *presentation* risk: a surfaced E with `1/α` framing could mislead a human into treating a claim as validated (flag #1). Mitigate at the label, not the math.
- **Un-witnessed independence** is a *known, documented* residual (same frontier as the ≥2 bar), and is the explicit reason the gate-migration is gated behind a concordance check — not a silent assumption.
- **Per-null defensibility** (q0, the normal-tail) is LLM-test-designed but code-computed; a wrong null mis-calibrates the *surfaced* E (not the verdict, in v1). v1 ships only the two best-justified nulls and surfaces, defers gating.
- **The κ infinite-variance trap** (§5a) is a config-level footgun closed by the `κ>0.5` default + invariant test.

---

## 12. Reviewer-scrutiny flags (post-revision)

**DISSOLVED by PP-D8 (surface-don't-gate) — recorded so the reviewer knows why they're gone:**
- ~~#1 back-compat keystone reproduces every transition~~ — moot; the gate is untouched, no transition moves.
- ~~#2 `AtomicClaim.status` restructuring~~ — not performed; the subtle math-bypass/uncertain/fail logic is unchanged.
- ~~#5 `E_EXACT` vs the math-bypass double-counting~~ — moot; no `E_EXACT` enters the gate (it only shapes the surfaced E).
- ~~#7 test churn across the gate-test files~~ — minimized; existing gate tests need no edits, only `evidence_strength` cases are added.
- #6 (surface-vs-gate) — **ADOPTED** as PP-D8; this was the right call.

**STILL LIVE — attack these:**
1. ~~Even a *surfaced* E can mislead~~ — **RESOLVED (§3/§6, PP-D8):** v1 surfaces E developer-facing only (logs/JSON), never in the human report, never against a `1/α`/"validates-at" threshold — only as a bare labeled heuristic; human-facing display is promoted only post-concordance as a gold-set percentile. The same concordance gate unlocks both human-facing E and gating.
2. **The concordance prerequisite is the real gate (§6).** Is the gold-set concordance check specified concretely enough to be the precondition it claims to be — what's "tracks correctness," how many labeled ideas, what separation threshold? Until that's pinned, the gate-migration has no defined trigger.
3. **Per-null defensibility (PP-D3):** `q0^N` (independence-assuming, §4a) and `normal_sf(margin)` (assumes the margin is a true z-score with a well-specified σ — is `uncertainty` a 1σ? if it's a bound or a guess, the p is wrong). Confirm the two shipped nulls are defensible enough to *surface*, even if not to gate.
4. **The plain-simulation null `p0=0.5` (PP-D3c):** correctly deferred? A coin-flip null fabricates a p-value where none is justified. Confirm deferral over shipping a weak null even just for surfacing.
5. **κ=0.7 default:** §5a shows κ>0.5 is required for finite variance; is 0.7 the right point on the per-test-strength vs correlation-robustness curve, or should it be higher given independence is wholly un-witnessed?
