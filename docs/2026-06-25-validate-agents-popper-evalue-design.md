# validate-agents — Popper E-Value Gate Design (calibrated evidence accumulation)

- **Date:** 2026-06-25
- **Status:** Draft for review (no user Q&A — design calls made by the author, flagged for scrutiny in §12). **This is the strategic, multi-week feature and the ONLY one that modifies `valagents/artifact.py`** — by design (it redefines the gate's aggregation), while preserving gate purity.
- **Builds on:** the gate in `valagents/artifact.py` (`CheckRecord`, `AtomicClaim.status`, `_has_independent_external_check`, `_evaluate`, `verdict_class`, all `@computed_field` with no setter), `valagents/config.py` (`GateCfg`), the per-lens CheckRecord producers (`agents/grounder.py`, `agents/prover.py`, `computation.py` `verdict_to_check`, `sandbox/runner.py`).
- **Source:** the Popper card + "Design sketch: the e-value gate" in `docs/2026-06-25_papers_for_validate_agents_report.md` (arXiv:2502.09858). The strategic upgrade, after the three cheap wins.
- **One-line goal:** Replace the gate's source-**count** bar (`independent_sources >= 1`) with **calibrated, multiplicative e-value evidence**: each statistical check emits a p-value under an explicit null → e-value `e = κ·p^(κ−1)` → aggregate `E = ∏ eₛ` → a claim is externally validated when `E ≥ 1/α`. This is the principled resolution of the deferred ≥2 question — corroboration **strength**, not **tally**, with a tunable any-time-valid Type-I guarantee that natively handles 1, 3, or 10 checks. The gate stays a pure computed function (just different arithmetic).

---

## 1. Why count→e-value (the problem with the bar)

Today a claim is externally validated when some passing check has `independent_sources >= 1` (`artifact.py:155`, `:218`). Two checks each with `independent_sources=1` count the same as one; a simulation passing 9/10 grid points counts the same as 6/10; the deferred ≥2 bar is an arbitrary tally. The count throws away **strength of evidence** and cannot principledly combine many noisy/correlated checks.

Popper's calibrator turns each test into an **e-value** (`eᵢ = κ·pᵢ^(κ−1)`, `κ∈(0,1)`, assumption-free), and accumulates **multiplicatively** (`Eᵢ = ∏ eₛ`). The product is a super-martingale under the null, so rejecting the null at `E ≥ 1/α` gives **any-time-valid Type-I control** — you can add or stop checks adaptively without α-inflation. Empirically holds α where Fisher's combination and an LLM-likelihood variant both blow it.

---

## 2. The headline design — additive, back-compatible (PP-D1)

**The e-value layer reduces *exactly* to today's behavior when only counts exist, and grades evidence when p-values exist.** This is the keystone that keeps the migration safe.

Per passing check, define its e-contribution `e(c)`:
```
e(c) = 1.0                          if c.verdict != "pass"        # no evidence
e(c) = κ · c.p_value ** (κ − 1)     if c.p_value is not None      # calibrated (statistical)
e(c) = E_EXACT                      elif c.independent_sources >= 1   # exact/count pass (back-compat)
e(c) = 1.0                          else                          # pass, but no evidence (e.g. mechanistic prover)
```
Per claim: `E = ∏ e(c)` over its checks. A claim is **externally validated** when `E ≥ 1/α`.

- `E_EXACT` is a fixed strong contribution chosen so **one** exact/count pass validates: `E_EXACT ≥ 1/α` (with `α=0.1`, `E_EXACT=10` ⇒ one count-pass validates, exactly today's `≥1` semantics). So an existing `CheckRecord(verdict="pass", independent_sources=1, p_value=None)` → `e=E_EXACT ≥ 1/α` → validated; `independent_sources=0, p_value=None` → `e=1.0` → not validated. **Every existing gate test that depends on `independent_sources` is preserved unchanged.**
- A statistical check with `p_value` set grades continuously: a grounder with two independent sources at `p=0.5` each (§4) → `E = e(0.5)² > E_EXACT` → stronger than one; a simulation at `frac=9/10` gets a smaller p (bigger e) than `6/10`. This is the ≥2 story, made calibrated.

So the layer is **additive**: lenses that set `p_value` feed the calibrated path; everything else falls back to the count path = today. No lens is *required* to emit a p-value for the gate to keep working.

---

## 3. What changes in `artifact.py` (and how purity is preserved)

- **`CheckRecord` gains `p_value: float | None = None`** — the check's p-value under its own null (None = non-statistical/exact).
- **`GateCfg` gains** `alpha: float = 0.1`, `evalue_kappa: float = 0.5`, `evalue_exact: float = 10.0` (≥ 1/α); stamped onto the artifact by `_apply_gate_cfg` (`scheduler.py:12`) as data fields `self.alpha` / `self.evalue_kappa` / `self.evalue_exact` (exactly like `min_attack_categories`). They are **data, not setters** — purity intact.
- **A pure helper** `_claim_evalue(self, claim) -> float` (method on the claim or a free function) computes `∏ e(c)` per §2.
- **Two gate sites change** (the only two `independent_sources >= 1` occurrences):
  - `AtomicClaim.status` (`artifact.py:155`): the `passes` filter + the final `if passes: return "pass"` become "the claim's external evidence clears the bar," i.e. `_claim_evalue(self) >= 1/self.alpha`. The surrounding structure (the `fail` short-circuit, the `uncertain` handling, `_math_uncertainty_is_nonblocking`, `has_proof_pass`) is **kept**; only the "is there ≥1 independent pass" test is replaced by "is E ≥ 1/α." `has_proof_pass` (used for the math-uncertainty bypass) stays a structural check over passing prover/executor checks.
  - `_has_independent_external_check` (`artifact.py:218`): `return self._claim_evalue(c) >= 1/self.alpha`.
- **Everything stays `@computed_field` with no setter.** `status`/`blocker`/`verdict_class`/`load_bearing`/`maturity` are unchanged in shape; only the arithmetic inside differs. The arbiter still *reads* `status` (`arbiter.py:10`), never writes it. **Gate purity invariant preserved** (verified: all gate outputs are computed properties; `α`/`κ` are stamped data).

---

## 4. The p-value per check type (the hard part — where the research risk lives)

The honest core: **defining a defensible null + p-value per check type is real work, and lives WITH each lens** (it sets `CheckRecord.p_value`), not in the gate (the gate only multiplies). v1 defines p-values where a defensible null exists and bypasses the rest to `E_EXACT`.

- **Grounder corroboration (PP-D3a):** null = "spurious topical co-occurrence" — a retrieved on-topic abstract carrying a passing on-property quote *by chance*. Model each independent passing source as an independent Bernoulli(q0) spurious hit ⇒ `p = q0 ** code_witnessed` (the dedup'd count, `grounder.py:100`). `q0` a config knob (default 0.5 → `p = 0.5^N`). N=1 → p=0.5 → e small; N=2 → p=0.25 → e larger → the calibrated ≥2 story. Set `p_value` in `ground_claim` alongside `independent_sources`.
- **Magnitude with uncertainty (PP-D3b):** `discriminating_margin` already computes `margin = |predicted − closest| / uncertainty` (`runner.py:~430`) — literally a z-score; `sensitivity_ratio` similarly. Null = "no effect beyond noise" ⇒ `p = normal_sf(margin)` (one-sided). The margin/ratio is currently only in the `computed` string; v1 returns it as a structured float and sets `p_value`. (`bound_check` is exact — no p; bypass.)
- **Simulation `robust_frac` (PP-D3c):** `passes`/`gsize` over the grid. Null = "passes by chance at rate p0" ⇒ one-sided binomial `p = P(X ≥ passes | Binomial(gsize, p0))`. **Negative-control** (`null_overrides`) has a *clean* null (the null arm should not discriminate; p0 is the chance rate of spurious discrimination); a **plain criterion** has an ill-defined null (default `p0=0.5` — "absent the mechanism a binary criterion passes ~half the time" — defensible but coarse; **flag**). The runner returns structured `passes`/`gsize`; the executor computes the binomial p and sets `p_value`.
- **Exact checks bypass (PP-D3d):** prover derivation pass, symbolic `simplify==0`, magnitude `bound_check` — deterministic, no sampling, no natural p-value → `p_value=None` → `e=E_EXACT` (one exact pass validates a math/definitional claim, preserving today). A mechanistic prover pass with `independent_sources=0` stays `e=1.0` (no external evidence) — exactly today (a mechanistic prover pass is not external support, `prover.py:50`).

**The gate never sees a null.** Each lens owns its null and emits a p-value (or None); the gate's job is purely `∏` + compare. Say-so stays out of the aggregation (the LLM designs the tests; the p-value is computed from execution counts/magnitudes/retrieval, all code).

---

## 5. The independence caveat (the key soundness risk — PP-D7)

`E = ∏ eₛ` is calibrated **only if the checks are independent under the null.** Multiplying e-values from **correlated** checks over-counts evidence and inflates E — a real way to manufacture a false validation. Sources of correlation:
- Two grounder "independent sources" that aren't truly independent (same lineage). Tier-2 already **cannot witness independence** (no authors in `Article`) — the count is "quote-verified retrieved sources," not provably independent. So `p = q0^N` *assumes* independence the system can't verify. This is the same residual the deferred ≥2 bar carries, now made quantitative (and thus more dangerous if trusted naively).
- A simulation and its negative-control arm (same dynamics), or a magnitude check and a grounding of the same number — correlated by construction.

**Mitigations (v1):** (a) dedup is already enforced (grounder `_dedup_articles`); (b) keep `κ` conservative and `α` not-too-small so a single correlated pair can't cross the bar cheaply; (c) **only multiply e-values across *different lens types* freely; within a lens, treat the lens's own p-value as the single contribution** (don't multiply per-source e-values *and* per-check e-values — the grounder emits ONE p for its N sources, §4, not N separate e's). (d) Document that true independence is un-witnessed; the calibration is "assuming the declared sources/checks are independent," surfaced in the basis. The honest, fully-sound version needs a non-say-so independence signal — **deferred** (same frontier as the ≥2 slice's "non-saturation subject signal").

---

## 6. Scope (honest — this is a research direction, not a drop-in)

**v1 ships:** the e-value layer + multiplicative aggregation + the two gate-site changes + back-compat-from-count (§2, the keystone) + the GateCfg knobs; **and** the grounder p-value (`q0^N`, §4a — the highest-value, it makes ≥2 calibrated) and the magnitude p-value (`normal_sf(margin)`, §4b — nearly free, the z-score already exists). Symbolic/prover/bound_check bypass to `E_EXACT`.

**v1.x / deferred:** the simulation binomial p (§4c — needs the runner to return structured `passes`/`gsize` and a defensible `p0` per criterion type; the plain-criterion null is the weakest); per-null tuning of `q0`/`p0`/`κ`/`α` against a labeled set (ties to the Co-Scientist concordance check); the independence-witnessing that makes multiplication fully sound (§5).

The point of the staged scope: **the gate-arithmetic change + back-compat is mechanical and testable now; the per-null definitions are the multi-week research.** Shipping the layer with grounder+magnitude p-values proves the mechanism and resolves ≥2 calibratedly without blocking on the hard nulls.

---

## 7. Cardinal-rule fit
The calibrator and `∏ eₛ ≥ 1/α` are **pure deterministic arithmetic** over numbers the lenses already produce (retrieval counts, z-scores, grid fractions). p-values come from lens-owned nulls computed in code, not LLM say-so. Popper's own LLM relevance-checker is **not** adopted (that is say-so — kept out). The gate stays a pure computed function; `α`/`κ` are config data. The one genuine soundness risk (correlated-evidence multiplication, §5) is bounded by dedup + conservative knobs + the single-p-per-lens rule, and the residual (un-witnessed independence) is documented and deferred — not silently assumed away.

---

## 8. Files
- `valagents/artifact.py` — `CheckRecord.p_value`; `_claim_evalue` helper; replace the two `independent_sources >= 1` gate sites with `E ≥ 1/α`; read `self.alpha`/`self.evalue_kappa`/`self.evalue_exact`. (Gate purity preserved; no setters.)
- `valagents/config.py` — `GateCfg.alpha/evalue_kappa/evalue_exact`; stamp in `scheduler._apply_gate_cfg` + the `IdeaArtifact` fields.
- `valagents/agents/grounder.py` — set `CheckRecord.p_value = q0 ** code_witnessed` (config `q0`).
- `valagents/computation.py` / `valagents/sandbox/runner.py` — return structured `margin`/`ratio` (and, deferred, `passes`/`gsize`); `verdict_to_check` sets `p_value = normal_sf(margin)` for magnitude.
- `valagents/config.py` — a small `EvidenceCfg` or extend `GroundCfg`/`GateCfg` for `q0`, `p0` defaults.
- Tests: `tests/test_artifact_gate.py`, `test_artifact_claim.py`, `test_verdict_class.py` — add e-value cases; **confirm every existing count-based case still passes via the back-compat default-e** (§2).

---

## 9. Decision log
- **PP-D1 (additive, back-compat from count — the keystone)** `e(c)` = calibrated from `p_value` if present, else `E_EXACT` if `independent_sources≥1`, else `1.0`. Reduces *exactly* to today when only counts exist; every existing gate test is preserved. New statistical checks grade continuously.
- **PP-D2 (gate change preserves purity)** Replace `independent_sources >= 1` with `_claim_evalue ≥ 1/α` at both sites; everything stays `@computed_field`, no setter; `α`/`κ`/`E_EXACT` are stamped data knobs.
- **PP-D3 (p-value lives with each lens)** Grounder `q0^N`; magnitude `normal_sf(margin)`; simulation binomial (deferred); exact checks (prover/symbolic/bound_check) bypass to `E_EXACT`. The gate only multiplies; nulls never enter the gate.
- **PP-D4 (κ calibrator + multiplicative E)** `e=κ·p^(κ−1)`, `κ∈(0,1)`; `E=∏e`; super-martingale ⇒ any-time-valid Type-I at `E≥1/α`.
- **PP-D5 (the principled ≥2)** ≥2 becomes "E from N corroborating sources ≥ 1/α" — strength, not tally; natively 1/3/10 checks.
- **PP-D6 (honest staged scope)** v1 = layer + aggregation + gate + back-compat + grounder & magnitude p-values; simulation binomial + per-null tuning + independence-witnessing deferred. The nulls are the multi-week research; the gate arithmetic is mechanical now.
- **PP-D7 (independence caveat, bounded + documented)** `∏` assumes independence the system can't fully witness; mitigated by dedup + conservative knobs + one-p-per-lens; the fully-sound version needs a non-say-so independence signal, deferred. Surfaced in the basis, not silently assumed.

---

## 10. Testing
- **`_claim_evalue` back-compat (the critical pins):** a single `CheckRecord(verdict="pass", independent_sources=1, p_value=None)` → `E=E_EXACT ≥ 1/α` → validated; `independent_sources=0, p_value=None` → `E=1 < 1/α` → not validated; mechanistic prover pass (`independent_sources=0`) → not external-validated — **all matching today's outcomes.** Re-run the entire existing gate suite and confirm green with no test changes (other than added e-value cases).
- **Calibrated grading:** two grounder sources (`p=0.25`) → larger E than one (`p=0.5`); a magnitude `margin=3` (`p≈0.001`) validates where `margin=1` (`p≈0.16`) does not (at fixed α).
- **Gate purity:** `status`/`verdict_class` remain computed with no setter; constructing an artifact and reading `status` twice is pure/idempotent; nothing writes `status`.
- **α/κ knobs:** the same checks validate at `α=0.1` and not at `α=0.001`; monotonicity of E in the knobs.
- **Independence pin (PP-D7):** a test asserting one-p-per-lens (the grounder emits a single `p_value` for its N sources, not N multiplied e's) so correlated within-lens evidence isn't double-counted.

---

## 11. Cardinal-rule risk register (read with §5)
- **Correlated-evidence multiplication** (PP-D7) is the one way this feature could manufacture a false validation. The reviewer must treat §5's mitigations as load-bearing, not decorative.
- **Un-witnessed independence** is a *known, documented* residual (same as the ≥2 frontier), not a silent assumption.
- **Per-null defensibility** (q0, p0, the normal-tail) is LLM-test-designed but code-computed; a wrong null mis-calibrates E. v1 keeps nulls conservative and ships only the two best-justified (grounder, magnitude).

---

## 12. Reviewer-scrutiny flags (author's uncertainties — attack these hardest)
1. **The back-compat keystone (PP-D1):** does `e(c) = E_EXACT if independent_sources≥1 else 1.0` (for `p_value=None`) reproduce **every** current gate transition? Enumerate the existing gate tests and confirm none flips. This is the make-or-break for a safe migration — if any count-based test changes outcome, the back-compat default is wrong.
2. **`AtomicClaim.status` restructuring (artifact.py:149–167):** the `passes`/`has_proof_pass`/`_math_uncertainty_is_nonblocking` logic is subtle. Replacing the `≥1` filter with an E-bar must NOT change the `uncertain`/`fail`/math-bypass branches. Does the proposed surgery keep those exact, or does it accidentally alter the math-claim non-blocking path?
3. **The independence assumption (§5, PP-D7) — the deepest risk.** Is multiplying e-values across lenses sound given the system can't witness independence? Is the "one p per lens" rule + conservative knobs enough, or does v1 over-claim calibration it can't back? Should v1 cap the number of multiplied factors, or require *distinct lens types* to multiply at all?
4. **The plain-simulation null `p0=0.5` (PP-D3c):** is a default coin-flip null defensible, or does it fabricate a p-value where none is justified? (This is why simulation p is deferred — confirm deferral is right vs shipping it with a weak null.)
5. **`E_EXACT` and the math-bypass interaction:** `E_EXACT≥1/α` makes one exact pass validate. But for a `mathematical` claim, the math-uncertainty bypass (`_math_uncertainty_is_nonblocking`) already lets a proof pass dominate uncertainties. Do `E_EXACT` and that bypass double-count or conflict?
6. **Is the gate the right place, or should E live on the artifact as a separate computed field** that the *arbiter* reads, leaving `status` count-based until ≥2 is formally adopted? (A less invasive staging: compute E everywhere, surface it, but don't gate on it until validated against a labeled set — ties to the deferred concordance check.)
7. **Test churn:** changing `CheckRecord`'s contribution semantics touches `tests/test_artifact_gate.py`, `test_artifact_claim.py`, `test_verdict_class.py`, `test_artifact_maturity.py`. The reviewer should confirm the plan budgets for auditing each, not just adding new cases.
