# validate-agents — Concordance Harness Design (validate-the-validator against a settled-science gold set)

- **Date:** 2026-06-25
- **Status:** Draft for review. **The gold-set section (§3) is a STRAWMAN written by an LLM and MUST be red-penned by a physicist** — it is the load-bearing, say-so-risky part; the harness around it (§4–§5) is mechanical.
- **Builds on:** `valagents/scheduler.py` (`run`), `valagents/cli.py` (`run_cli`), `valagents/artifact.py` (`IdeaArtifact.verdict_class`, `status`, the deferred `evidence_strength` from the Popper spec), `valagents/agents/faithfulness.py` (the NLI probe).
- **Source:** the Co-Scientist "validate-the-validator" / Elo-vs-GPQA concordance idea (report §"Nature Co-Scientist", item 7), promoted from afterthought to **prerequisite slice** by the adversarial reviews of the Popper and NLI specs. It is now load-bearing for **three** downstream unlocks (§1).
- **One-line goal:** A read-only measurement harness that runs the full validate-agents pipeline over a **labeled gold set of settled-science ideas** (known-valid / known-refuted by experiment+time, ground-truth not opinion) and scores whether the system's outputs — the terminal `verdict_class`, the surfaced `evidence_strength` (E), and the NLI faithfulness verdict — actually **separate** valid from refuted. Passing this harness is the precondition that unlocks (a) Popper's gate-migration, (b) NLI's enablement, and (c) human-facing display of E.

---

## 1. Why this is the prerequisite (it gates three things)

The cardinal rule's corollary, surfaced by the reviews: **don't trust a new detector or calibration without a concordance check against a small labeled gold set.** Three pending changes all rest on an unvalidated assumption that only this harness can discharge:
- **Popper gate-migration** (gate the verdict on `E ≥ 1/α`) — needs E shown to track correctness (`E` higher on valid than refuted), AND its human-facing display (Popper §3/§6).
- **NLI enablement** (trust the frozen NLI faithfulness verdict over the near-random LLM self-judge) — needs NLI shown to beat the LLM on real physics `(seed, formal)` pairs (NLI §5b/NF-D8). Without it, switching trades one near-random judge for another plus a heavy dependency.
- **The system's own credibility** — for the August qual, "the verdict tracks settled truth on cases we already know the answer to" is the single most persuasive evidence that the machine works.

So this harness is the **gate before** Popper's migration or NLI's enablement — not after.

---

## 2. Two evaluation tracks (one harness, two labeled sets)

The harness hosts **two** distinct labeled sets, because the three unlocks ask two different questions:

**Track A — Idea-validity** (unlocks Popper gate-migration + human-facing E + the credibility claim).
- Set: `{seed: str, label ∈ {valid, refuted}}` — a settled-science idea phrased as its historical *proposal* (no hindsight), with a ground-truth label from how experiment+time settled it (§3).
- Measures: (1) **verdict concordance** — does `verdict_class` align with the label (valid → `validated`/at least not `refuted|challenged`; refuted → `refuted|challenged`/not `validated`)? (2) **E separation** — is `evidence_strength` stochastically higher on valid than refuted (the Popper unlock)?

**Track B — Faithfulness** (unlocks NLI enablement; the §5b probe).
- Set: `{seed: str, formal: str, label ∈ {faithful, narrowed, drifted}}` — a seed paired with a formalization that is (by construction) faithful, a proper-subset narrowing, or a drifted restatement, hand-labeled. (Faithfulness is a *linguistic* judgment checkable by reading — **lower say-so risk than Track A's validity labels** — but still curated, not model-generated.)
- Measures: does the **NLI** faithfulness verdict agree with the human label **better than** the LLM self-judge, on this set? Sets the `nli_threshold`.

The two sets are separate files; an item in Track A is a whole idea, an item in Track B is a (seed, formalization) pair. (Track B's seeds *may* reuse Track A seeds, but its formalizations are deliberately authored to span faithful/narrowed/drifted.)

---

## 3. The gold set — criteria + STRAWMAN (the part a physicist must own)

> **This section is the load-bearing say-so risk.** "Who labels an idea valid?" is exactly the question the project exists to keep away from opinion. The cardinal-rule-clean answer: **labels come from settled science — how experiment and time adjudicated the idea — not from a present-day model or annotator's judgment.** The harness is only as trustworthy as this set; if the gold set is biased or wrong, every downstream unlock inherits the bias. So the criteria below are firm; the *items* are a strawman an LLM should not be trusted to curate.

### 3.1 Curation criteria (Track A — idea validity)

1. **Ground-truth label from settled history, not opinion.** The idea was proposed at time T; by now its fate is **experimentally settled** — confirmed (e.g. BCS, BEC) or refuted (e.g. luminiferous aether, cold fusion). The label is that historical experimental verdict, not "this seems right/wrong today."
2. **Time-lag / settled.** Exclude live controversies and anything whose status is still contested — the label must be uncontroversial among physicists. (This is the criterion a physicist enforces; an LLM cannot reliably judge "is this settled?")
3. **Pre-registered, no cherry-pick.** The set + the §5 unlock thresholds are **fixed before** running the system. You may not add/drop items after seeing the system's scores, or the concordance is meaningless (you'd be tuning the validator to the test). Record the set in git before the first run.
4. **Domain-matched.** Prefer condensed-matter / quantum ideas (the system's actual target — QSL noise, QM meta-symmetry, hole superconductivity), so the test is informative for the real use. Particle/GR ideas (Higgs, gravitational waves) are valid history but a different literature; use sparingly.
5. **Decomposable like a real target.** The seed must be the kind of idea the pipeline handles — a mechanism / magnitude / symbolic relation that decomposes into claims — not a bare fact.
6. **Seed phrased WITHOUT hindsight.** The seed is the *historical proposal* ("Superconductivity arises from phonon-mediated electron pairing"), never "X, which was later confirmed." Hindsight in the seed leaks the label into the system's input and invalidates the test.
7. **Balanced + small.** Roughly equal valid/refuted (so separation metrics aren't degenerate); small (each item is a full, expensive pipeline run — §4). v1 target: ~10–16 items, grown over time.
8. **Confound-aware (prefer matched pairs).** Where possible, pair a confirmed and a refuted idea from the *same subfield/era* so separation isn't confounded by literature volume or topic. (See §7 — settled ideas have saturated literature, which the Tier-2 co-saturation fail-closed may degrade.)

### 3.2 STRAWMAN gold items (LLM-proposed — **physicist: correct, cull, replace, and especially fix the labels/phrasing**)

Each: `{seed (historical proposal, no hindsight), label, provenance (the settled verdict), confound notes}`. **Do not trust these as authored; they are a starting point to red-pen.**

**VALID (later experimentally confirmed):**
- `seed:` "Superconductivity arises from a phonon-mediated attractive interaction that binds electrons into Cooper pairs condensing into a single coherent ground state." `label: valid` (BCS; confirmed, Nobel 1972). *Domain-matched (CM).*
- `seed:` "A dilute gas of bosons cooled below a critical temperature condenses macroscopically into the single-particle ground state." `label: valid` (BEC; predicted 1924–25, realized 1995). *Domain-matched.*
- `seed:` "A supercurrent flows between two superconductors separated by a thin insulating barrier, set by the superconducting phase difference across it." `label: valid` (Josephson effect; confirmed 1963). *Domain-matched; good matched-pair partner for a refuted-tunneling claim.*
- `seed:` "The quantized Hall conductance of a 2-D electron gas is exact and topologically protected, independent of sample detail." `label: valid` (IQHE; confirmed 1980). *Domain-matched.*
- `seed:` "Beta decay's missing energy and momentum are carried off by a light, neutral, weakly-interacting particle." `label: valid` (neutrino; proposed 1930, confirmed 1956). *Particle — use sparingly.*

**REFUTED (later experimentally falsified):**
- `seed:` "Light propagates through a stationary space-filling medium (the aether); the Earth's motion through it produces a measurable anisotropy in the speed of light." `label: refuted` (Michelson–Morley). *Clean refutation; classic.*
- `seed:` "Deuterium electrochemically loaded into palladium undergoes nuclear fusion at room temperature, releasing measurable excess heat beyond any chemical source." `label: refuted` (cold fusion, 1989). *Domain-adjacent; well-documented refutation literature.*
- `seed:` "Water confined in fine quartz capillaries forms a stable anomalous polymeric phase with markedly elevated viscosity and boiling point." `label: refuted` (polywater, 1960s). *Condensed-matter-adjacent; clean.*
- `seed:` "Many common materials emit a previously unknown radiation that increases the brightness of an electric spark and can be refracted by aluminum prisms." `label: refuted` (N-rays, Blondlot). *Clean — a self-deception case.*
- `seed:` "The universe is unchanging on large scales, with continuous creation of matter maintaining constant density as space expands." `label: refuted` (steady-state; refuted by the CMB + expansion). *Cosmology — use sparingly.*

**Matched-pair candidates to consider (physicist to confirm):** Josephson tunneling (valid) vs a specific refuted tunneling/transport claim in the same era; a confirmed vs a refuted high-Tc mechanism proposal.

**The physicist's job here:** (i) verify every label is genuinely settled and uncontroversial; (ii) rephrase each seed to the faithful historical proposal with zero hindsight; (iii) replace particle/GR/cosmology items with domain-matched CM/QM ones where possible; (iv) add 1–2 matched pairs; (v) decide the final N and freeze the set in git before any run.

### 3.3 Track-B faithfulness set (lower say-so risk)

`{seed, formal, label ∈ {faithful, narrowed, drifted}}`. For a handful of seeds (can reuse Track A's), author three formalizations each: a faithful sharpening, a proper-subset narrowing, and a drifted/contradicted restatement. The label is a *reading* judgment (is this restatement the same claim?), which a careful annotator can make defensibly — but still curated by a human, never model-generated. ~8–12 triples.

---

## 4. The harness (mechanical, read-only)

`scripts/concordance.py` (a batch research script, **NOT** a pytest test — it runs full network+LLM pipelines and is far too slow/non-deterministic for CI):
1. **Load** the frozen gold sets (`gold/validity_set.yaml`, `gold/faithfulness_set.yaml`).
2. **Track A:** for each item, run `run(seed, llm, cfg, backend)` (the full pipeline) `K` times (default `K=3`, low temperature) — LLM non-determinism is real, so collect a **distribution**, not a point. Record per run: `verdict_class`, `status`, the artifact-level `evidence_strength` rollup, the blocker reason. Report the **modal** verdict + the E distribution per item.
3. **Track B:** for each `(seed, formal)` triple, call **both** `faithfulness_check` with the LLM path AND with the injected NLI scorer; record both verdicts vs the label.
4. **Compute metrics** (§5), write a timestamped concordance report (`results/concordance/<date>.md`) with per-item rows, the confusion matrices, the separation scores **with confidence intervals**, and the explicit pass/fail against the pre-registered thresholds.
5. **Read-only:** the harness never writes back into any artifact or verdict; it measures. It imports `run`/`faithfulness_check` and reads their outputs.

Cost honesty: ~`N × K` full pipeline runs + the Track-B calls — a **periodic research run**, not CI; budget accordingly. The non-determinism (and small N) is why everything is reported with distributions/CIs, not single numbers.

---

## 5. Metrics + the unlock thresholds (pre-registered)

Per track, with **wide-CI honesty** (small N ⇒ noisy; thresholds are conservative and the set grows over time):

- **A1 — Verdict concordance (sanity):** balanced accuracy of `verdict_class`-vs-label (valid↔validated-ish, refuted↔refuted/challenged). A floor (e.g. ≥ 0.7 balanced accuracy with the CI lower bound above chance) is a sanity gate — if the system can't even separate settled cases, nothing downstream should ship.
- **A2 — E separation (the Popper unlock):** **ROC-AUC of `evidence_strength` as a classifier of valid-vs-refuted.** Unlock the Popper gate-migration + human-facing E only if `AUC ≥ 0.80` AND the CI lower bound clears 0.5 by a margin. Below → keep E developer-facing-only and the gate count-based (exactly the Popper-spec default).
- **B — NLI-beats-LLM (the NLI unlock):** on the faithfulness set, NLI's agreement with the human label must **exceed** the LLM self-judge's by a pre-set margin (and clear an absolute floor). Unlock NLI enablement only then; the threshold that maximizes NLI's agreement on the set becomes `nli_threshold`.

All thresholds are **frozen before the first run** (§3.1.3). The report states pass/fail per unlock; no silent dropping of items, no post-hoc threshold tuning.

---

## 6. Cardinal-rule fit
The harness is **measurement, not a new gate** — read-only on the system, pure-code metrics over ground-truth labels. The labels are **settled-science ground truth**, the one labeling source that isn't say-so; the gold set is curated by a human physicist and **frozen/pre-registered** so the validator can't be tuned to its own test. The harness adds the corollary the reviews demanded: a new detector/calibration (E, NLI) earns trust only by **separating known-valid from known-refuted on cases we already know** — and until it does, the system stays on its conservative defaults (count-based gate, LLM-fallback faithfulness, E hidden from humans).

---

## 7. Confounds & honest limitations (state them, don't hide them)
- **Settled ideas have saturated literature.** A confirmed idea retrieves abundant supporting literature; the Tier-2 **co-saturation fail-closed** may *under*-credit it (property co-saturates → uncertain). So Track A partly tests the *grounder's* behavior on saturated topics, not just the gate — a confirmed item landing `uncertain` may be the co-saturation residual, not a concordance failure. Matched pairs (§3.1.8) and reading the per-item basis mitigate; the report must flag co-saturated items.
- **Refuted ideas still have literature** (historical discussion + the refutation). The grounder should surface the contradiction (→ contradiction guard → challenged/uncertain); a refuted item landing `validated` is a true concordance failure (the harness's main signal).
- **Small N ⇒ noisy AUC.** 10–16 items gives wide CIs; the thresholds are conservative and the set must grow. A single passing run is necessary, not sufficient — re-run as the set grows.
- **LLM non-determinism** ⇒ per-item distributions, not points (§4).
- **The gold set could itself become the new un-witnessed assumption** if curated lazily (LLM-picked, opinion-labeled, hindsight-phrased, or cherry-picked). §3's criteria + the pre-registration discipline exist precisely to prevent that; the physicist's red-pen is the safeguard.
- **Seed-phrasing leakage:** any hindsight in a seed leaks the label; §3.1.6 + review of each seed.

---

## 8. Files
- `gold/validity_set.yaml` (Track A), `gold/faithfulness_set.yaml` (Track B) — the frozen, git-committed gold sets (physicist-curated).
- `scripts/concordance.py` — the batch harness (load → run pipeline ×K → collect signals → metrics → report). Read-only.
- `valagents/concordance.py` (optional) — pure-code metric helpers (`balanced_accuracy`, `roc_auc`, bootstrap CI) so they're unit-testable independent of the expensive runs.
- `results/concordance/<date>.md` — the timestamped report (per-item rows, confusion matrices, separation + CIs, pass/fail vs pre-registered thresholds).
- (Reads `valagents.scheduler.run`, `valagents.agents.faithfulness.faithfulness_check`, the Popper `evidence_strength`.)

---

## 9. Decision log
- **CN-D1 (prerequisite, not afterthought)** Passing this harness gates Popper's migration, NLI's enablement, and human-facing E — all three. It is the next spec, ahead of those unlocks.
- **CN-D2 (two tracks)** Idea-validity (verdict + E) and faithfulness (NLI-vs-LLM) are distinct labeled sets in one harness.
- **CN-D3 (labels = settled-science ground truth)** Confirmed/refuted-by-experiment-and-time, uncontroversial among physicists; never present-day model/annotator opinion. The one non-say-so labeling source.
- **CN-D4 (pre-registered, frozen, no cherry-pick)** The set + thresholds are fixed in git before the first run; no post-hoc add/drop/tune. Otherwise the validator is tuned to its own test.
- **CN-D5 (read-only measurement)** The harness never feeds a verdict; pure-code metrics over ground truth.
- **CN-D6 (conservative thresholds, wide-CI honesty, grow the set)** Small N ⇒ noisy; unlock thresholds are conservative with CI lower-bound requirements; the set grows; a single pass is necessary not sufficient.
- **CN-D7 (domain-matched + decomposable + matched-pairs)** Prefer CM/QM, idea-shaped, paired-by-subfield items so separation isn't confounded by literature volume/topic.
- **CN-D8 (co-saturation confound acknowledged)** Track A partly probes the grounder on saturated topics; co-saturated valid items may under-credit — flagged in the report, not silently scored as failures.

---

## 10. Build slices
1. **Pure metric helpers** (`valagents/concordance.py`): `balanced_accuracy`, `roc_auc`, bootstrap CI — unit-tested on synthetic labels/scores (no pipeline runs).
2. **The harness** (`scripts/concordance.py`): load frozen sets, run `run`×K (Track A) + dual `faithfulness_check` (Track B), collect signals, emit the report. Tested with a **FakeLLM-scripted** mini gold set (so the harness logic is covered without real models/network); the real run is a manual research invocation.
3. **The gold sets** (`gold/*.yaml`): physicist-curated, frozen, committed. (This is the human work, not code; the spec's §3 is its requirements doc.)

---

## 11. Testing
- **Metric helpers (pure unit):** `roc_auc` on separable vs inseparable synthetic scores; `balanced_accuracy` on a known confusion matrix; bootstrap CI width shrinks with N; degenerate cases (all-one-label) handled.
- **Harness logic (FakeLLM, no network):** a scripted 2-valid/2-refuted mini set where the FakeLLM produces a known verdict/E per seed → the harness computes the expected concordance/AUC and writes the report; Track B with a fake NLI scorer + scripted LLM → the comparison table is correct. (This tests the *plumbing*, not the science.)
- **Read-only pin:** running the harness does not mutate any artifact/store/verdict.
- **No test runs a real pipeline or model** (cost/non-determinism); the real concordance run is a documented manual step.

---

## 12. Reviewer-scrutiny flags (author's uncertainties — attack these)
1. **THE GOLD SET (§3) — the whole spec rests here.** Are the §3.1 criteria sufficient to keep labeling out of opinion? Are the strawman items genuinely settled, domain-appropriate, hindsight-free, and balanced — or has the LLM mislabeled/misphrased some (likely)? Is "settled by experiment+time" the right ground-truth, and is the pre-registration discipline enough to prevent tuning-to-test? **This needs a physicist, not a reviewer-agent, for the items; the agent should attack the *criteria + the discipline*.**
2. **Track A confound = grounder co-saturation (§7).** Because settled ideas have saturated literature, Track A may measure the Tier-2 co-saturation residual as much as the gate/E. Does that make E-separation on Track A an unreliable unlock signal — should the gold set deliberately favor ideas with *thinner* literature (closer to the system's novel-idea home turf), even at some loss of "settledness"?
3. **Small-N validity.** Is ~10–16 items enough for an AUC≥0.80 unlock to mean anything, given the CI? Should the unlock require a minimum N (e.g. ≥20) regardless of AUC?
4. **Does Track A even test the right thing for the qual?** The system targets *novel* ideas (unknown truth); the gold set is *settled* (known truth). Concordance on settled cases is necessary evidence, but does high settled-concordance transfer to novel ideas, or is there a distribution gap (settled = saturated literature; novel = thin)? State the limit of what passing this harness licenses.
5. **Track B label objectivity.** Faithfulness labels are a reading judgment — defensible, but is a single annotator enough, or does it need inter-annotator agreement to not be one person's say-so?
