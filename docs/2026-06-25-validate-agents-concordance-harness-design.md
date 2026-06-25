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

> **This section is the load-bearing say-so risk, and it has a regress.** "Who labels an idea valid?" is exactly the question the project exists to keep away from opinion. A first instinct — "a physicist red-pens the LLM's items" — does **not** close the gap if the physicist's ratification is *their own judgment*: that is a second say-so stacked on the first, and the harness that grounds every downstream detector would itself be grounded in opinion — say-so all the way down.

> **The fix — the cardinal rule applied recursively to the gold set (CN-D3):** every label must be **witnessable against a cited authoritative source**, not asserted. Replace the parenthetical gloss `(BCS; confirmed, Nobel 1972)` with a **provenance citation** — the Nobel/award citation, the definitive replication-failure paper (for refuted items), or a consensus review/encyclopedia statement — plus a recorded **human sign-off** (who, when, against which sources). Then a human *ratifies the label by checking the source* (a verification), not by trusting a model or their own recall; the LLM's role (and the authoring physicist's) is **draft-and-stress-test, never authority**. This is identical to the grounder's discipline (a claim's support is a quote ∈ the cited source, not the model's word) — here, a label's support is the verdict of a cited settled-science source. The criteria below are firm; the *items* are a strawman, and even a physicist's blessing of an item counts only insofar as it is a check against the cited source.

### 3.1 Curation criteria (Track A — idea validity)

1. **Ground-truth label, witnessable against a cited source (not opinion).** The idea was proposed at time T; by now its fate is **experimentally settled** — confirmed or refuted. Each item carries a **`provenance` citation** to the authoritative source establishing the verdict (award citation; the definitive replication-failure paper; a consensus review) and a **`signoff`** recording the human who ratified the label *by checking that source*, when, and against which source. The label's authority is the cited source, not the model or the annotator (§3.0). Schema in §3.1b.
2. **Time-lag / settled.** Exclude live controversies and anything whose status is still contested — the label must be uncontroversial *and have a citable consensus source* (criterion 1). (A physicist enforces this; an LLM cannot reliably judge "is this settled?" — which is why the citation, not the model, carries the label.)
3. **Pre-registered, WITNESSABLY (not honor-system).** The set + the §5 unlock thresholds are fixed before running the system; you may not add/drop items or tune thresholds after seeing scores (that tunes the validator to its own test). Because "frozen in git" is amendable, the proof must be witnessable: the concordance report records the gold-set file's **git blob hash** (`git hash-object gold/validity_set.yaml`) and the pre-registered thresholds; a reviewer re-hashes the committed file and confirms the match. Post-hoc tampering changes the hash → detectable. Don't ask the reader to trust "we didn't tune" — prove it (§3.1b).
4. **Domain-matched.** Prefer condensed-matter / quantum ideas (the system's actual target — QSL noise, QM meta-symmetry, hole superconductivity), so the test is informative for the real use. Particle/GR ideas (Higgs, gravitational waves) are valid history but a different literature; use sparingly.
5. **Decomposable like a real target.** The seed must be the kind of idea the pipeline handles — a mechanism / magnitude / symbolic relation that decomposes into claims — not a bare fact.
6. **Seed phrased WITHOUT hindsight.** The seed is the *historical proposal* ("Superconductivity arises from phonon-mediated electron pairing"), never "X, which was later confirmed." Hindsight in the seed leaks the label into the system's input and invalidates the test.
7. **Balanced + small.** Roughly equal valid/refuted (so separation metrics aren't degenerate); small (each item is a full, expensive pipeline run — §4). v1 target: ~10–16 items, grown over time.
8. **Confound-aware (prefer matched pairs).** Where possible, pair a confirmed and a refuted idea from the *same subfield/era* so separation isn't confounded by literature volume or topic.
9. **Two strata (load-bearing — drives what the unlock licenses).** Split the set:
   - **(i) famous-saturated CM/QM** — blockbuster settled ideas (BCS, BEC). They serve the **sanity + qual-credibility** story, but they are the *worst case* for both confounds (§7): their literature is saturated (co-saturation may under-credit) and their "settled" status is over-determined.
   - **(ii) proposal-era-thin, now-settled CM/QM** — ideas proposed and then settled *relatively quickly*, so the **proposal-era literature was thin** (matching the system's novel-idea home turf) while the verdict is now ground-truth. **Stratum (ii) is the one whose E-separation actually licenses the Popper unlock for the real target distribution** (novel = thin). Curating (ii) is the harder physicist work.
10. **The binding curation constraint — refuted CM/QM is scarce (named, for the physicist to mine).** The *valid* side is easy (BCS/BEC/Josephson are clean CM/QM). The *refuted* side skews non-CM — aether (optics/foundational), steady-state (cosmology), cold fusion (electrochem/nuclear). Clean, settled, CM/QM refutations are genuinely rare: most are either recent and **not** settled (LK-99-class — trades off "settled"), or old/obscure pre-BCS mechanism proposals. This scarcity is the binding constraint and is *why* matched-pairs (criterion 8) and stratum (ii) are hard. An LLM can name the category; it cannot author "X is a settled CM refutation" without that being the say-so we're removing — the physicist mines it against citable sources.

### 3.1b Provenance schema + witnessable pre-registration (the discipline mechanism — author-owned)

Each gold item (YAML):
```yaml
- id: bcs
  stratum: famous_saturated        # famous_saturated | thin_settled
  seed: "Superconductivity arises from a phonon-mediated attractive interaction that binds
         electrons into Cooper pairs condensing into a single coherent ground state."
  label: valid                     # valid | refuted
  provenance:
    settled_by: "experimental confirmation; field consensus"
    sources:                       # the AUTHORITATIVE record(s) a human checks to ratify the label
      - "Bardeen, Cooper, Schrieffer, Phys. Rev. 108, 1175 (1957)"
      - "Nobel Prize in Physics 1972 — official citation"
    doi_or_url: ["10.1103/PhysRev.108.1175", "https://www.nobelprize.org/prizes/physics/1972/"]
  signoff:
    by: "<physicist name>"
    date: "<YYYY-MM-DD>"
    ratified_against: "checked the Nobel citation + the 1957 paper abstract; label witnessed, not recalled"
```
- The **label's authority is `provenance.sources`**, ratified by `signoff` — never the seed author's assertion. A reviewer (human or agent) can audit any label by reading the cited source.
- **Pre-registration is recorded in the report, not promised:** the harness writes `gold_blob_sha = git hash-object gold/validity_set.yaml`, the `gold_commit`, and the frozen thresholds into the concordance report header **before** consuming any run result. Re-running `git hash-object` on the committed file must reproduce `gold_blob_sha`; a mismatch proves the set was edited after pre-registration. (This is pure-discipline mechanism, fully author-ownable; the physicist only fills `provenance`/`signoff`.)

### 3.2 STRAWMAN gold items (LLM-proposed — **physicist: correct, cull, replace, and especially fix the labels/phrasing**)

Each: `{seed (historical proposal, no hindsight), label, confound notes}` — provenance/sign-off (§3.1b) to be filled by the physicist. **Do not trust these as authored; they are a starting point to red-pen.** **A stress-test already caught a criterion-#6 (no-hindsight) violation in this very strawman** — concrete proof an LLM doesn't reliably enforce its own criteria, and that the human/source ratification is load-bearing (the original IQHE seed below smuggled the post-1982 *topological* understanding into a 1980 proposal; corrected).

**VALID (later experimentally confirmed):**
- `seed:` "Superconductivity arises from a phonon-mediated attractive interaction that binds electrons into Cooper pairs condensing into a single coherent ground state." `label: valid` (BCS; confirmed, Nobel 1972). *Domain-matched (CM).*
- `seed:` "A dilute gas of bosons cooled below a critical temperature condenses macroscopically into the single-particle ground state." `label: valid` (BEC; predicted 1924–25, realized 1995). *Domain-matched.*
- `seed:` "A supercurrent flows between two superconductors separated by a thin insulating barrier, set by the superconducting phase difference across it." `label: valid` (Josephson effect; confirmed 1963). *Domain-matched; good matched-pair partner for a refuted-tunneling claim.*
- `seed:` "The Hall resistance of a 2-D electron system in the quantum regime is quantized to extraordinary precision, at values independent of the material and the sample geometry." `label: valid` (IQHE; 1980). *Domain-matched.* **CORRECTED — the original strawman ("exact and topologically protected, independent of sample detail") leaked the post-1982 TKNN topological understanding into a 1980 proposal (criterion #6 violation); this rephrasing is the hindsight-free historical claim.*
- `seed:` "Beta decay's missing energy and momentum are carried off by a light, neutral, weakly-interacting particle." `label: valid` (neutrino; proposed 1930, confirmed 1956). *Particle — use sparingly.*

**REFUTED (later experimentally falsified):**
- `seed:` "Light propagates through a stationary space-filling medium (the aether); the Earth's motion through it produces a measurable anisotropy in the speed of light." `label: refuted` (Michelson–Morley). *Clean refutation; classic.*
- `seed:` "Deuterium electrochemically loaded into palladium undergoes nuclear fusion at room temperature, releasing measurable excess heat beyond any chemical source." `label: refuted` (cold fusion, 1989). *Domain-adjacent; well-documented refutation literature.* **CAVEAT (criterion #2 borderline): mainstream-refuted, but a residual LENR fringe makes it less cleanly "settled" than aether/N-rays — the physicist may cut it, or keep it with this caveat in `provenance`.*
- `seed:` "Water confined in fine quartz capillaries forms a stable anomalous polymeric phase with markedly elevated viscosity and boiling point." `label: refuted` (polywater, 1960s). *Condensed-matter-adjacent; clean.*
- `seed:` "Many common materials emit a previously unknown radiation that increases the brightness of an electric spark and can be refracted by aluminum prisms." `label: refuted` (N-rays, Blondlot). *Clean — a self-deception case.*
- `seed:` "The universe is unchanging on large scales, with continuous creation of matter maintaining constant density as space expands." `label: refuted` (steady-state; refuted by the CMB + expansion). *Cosmology — use sparingly.*

**Matched-pair candidates to consider (physicist to confirm):** Josephson tunneling (valid) vs a specific refuted tunneling/transport claim in the same era; a confirmed vs a refuted high-Tc mechanism proposal.

**The physicist's job here:** (i) verify every label is genuinely settled and uncontroversial; (ii) rephrase each seed to the faithful historical proposal with zero hindsight; (iii) replace particle/GR/cosmology items with domain-matched CM/QM ones where possible; (iv) add 1–2 matched pairs; (v) decide the final N and freeze the set in git before any run.

### 3.3 Track-B faithfulness set (lower say-so risk)

`{seed, formal, label ∈ {faithful, narrowed, drifted}}`. For a handful of seeds (can reuse Track A's), author three formalizations each: a faithful sharpening, a proper-subset narrowing, and a drifted/contradicted restatement. The label is a *reading* judgment (is this restatement the same claim?) — lower say-so risk than Track A's validity labels, but still curated, never model-generated. **`≥2` annotators per item with a recorded inter-annotator agreement statistic (CN-D9):** faithfulness is linguistic and cheap to double-label, and the whole project is about removing single-person say-so — so the label is the annotators' consensus, and the agreement statistic is reported (a low statistic means the item is ambiguous and should be cut, not forced). ~8–12 triples.

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

Per track, with **wide-CI honesty** (small N ⇒ noisy):

- **A1 — Verdict concordance (sanity), with an ABSTAIN class.** A valid settled idea often *correctly* lands `needs_experiment` (the system is conservative — it demands the experiment that history then supplied); that is **not** a concordance failure. So A1 is **balanced accuracy over DECISIVE outcomes only** — `validated` vs `refuted|challenged` — treating `needs_experiment`/`draft`/`promising` as **abstain**, with the **abstain rate reported separately**. The cardinal failure is `refuted-item → validated`; `valid-item → abstain` is not penalized. Forcing the cautious middle into valid-or-refuted would distort the metric and punish correct caution. Sanity floor on the decisive subset (e.g. balanced accuracy ≥ 0.7, CI lower bound above chance).
- **A2 — E separation (the Popper unlock), gated on the CI LOWER BOUND of the NON-co-saturated stratum.** **ROC-AUC of `evidence_strength` as a valid-vs-refuted classifier**, but: (a) AUC ≥ 0.80 *point* on N≈10–16 is nearly meaningless (CI half-width ~0.2), so the unlock is on the **bootstrap CI lower bound ≥ 0.65** (self-regulating: it forces enough items to tighten the CI — the statistically honest form of "min N"); (b) **stratify by whether Tier-2 co-saturation fired** for the item (a measured covariate, §7) and **unlock on the non-co-saturated stratum's AUC** — so a known orthogonal bug (co-saturation under-crediting a valid item) can neither sink nor inflate the unlock decision. Below the bar → keep E developer-facing-only and the gate count-based (the Popper default).
- **B — NLI-beats-LLM (the NLI unlock).** On the faithfulness set, NLI's agreement with the (≥2-annotator, §3.3) consensus label must **exceed** the LLM self-judge's by a pre-set margin and clear an absolute floor; the threshold maximizing NLI's agreement becomes `nli_threshold`. Report inter-annotator agreement alongside.

**Licensing (state next to the thresholds, so a pass isn't over-read — esp. for the qual):** passing concordance on a *settled* gold set is **necessary, not sufficient**. *Failing* it kills the migration/enablement; *passing* it licenses them only as **"validated on settled cases, with transfer to novel (thin-literature) ideas an explicit unmeasured residual"** (§7, flag #4) — partially mitigated by the stratum-(ii) thin-literature items but not eliminated.

All thresholds + the gold-set blob hash are **frozen and recorded in the report** before any run result is consumed (§3.1b). No silent item-dropping, no post-hoc tuning.

---

## 6. Cardinal-rule fit
The harness is **measurement, not a new gate** — read-only on the system, pure-code metrics over ground-truth labels. The labels are **settled-science ground truth**, the one labeling source that isn't say-so; the gold set is curated by a human physicist who **ratifies each label against a cited authoritative source** (§3.0/CN-D3 — not their own judgment) and is **frozen/pre-registered witnessably** (blob hash, CN-D4) so the validator can't be tuned to its own test. The harness adds the corollary the reviews demanded: a new detector/calibration (E, NLI) earns trust only by **separating known-valid from known-refuted on cases we already know** — and until it does, the system stays on its conservative defaults (count-based gate, LLM-fallback faithfulness, E hidden from humans). **This is the cardinal rule one level up:** the rule keeps a *verdict* off model say-so; the harness keeps the *validator-of-the-verdict* off say-so too (labels ⟸ cited sources), and where even that is impossible — validating on genuinely novel (unknown-truth) ideas — the residual is stated and bounded (§12 flag #4), never fabricated. The discipline is self-consistent to the top.

---

## 7. Confounds & honest limitations (state them, don't hide them)
- **Settled ideas have saturated literature → co-saturation is a MEASURED COVARIATE, not just a caveat.** A confirmed idea retrieves abundant supporting literature; the Tier-2 **co-saturation fail-closed** may *under*-credit it (property co-saturates → `prop_distinctive` empties → uncertain). So Track A partly tests the *grounder* on saturated topics, not the gate. **Design response (not just a flag):** instrument the grounder to record per-claim whether co-saturation fired (Guard-2-empty), have the harness collect it, and **stratify E-separation by it, unlocking on the non-co-saturated stratum** (§5/A2). A confirmed item landing `uncertain` *because* co-saturation fired is recorded as such and excluded from the unlock AUC — so a known orthogonal bug can't sink or spuriously inflate the unlock. (This needs a small grounder instrumentation: surface a `co_saturated` flag on the grounder `CheckRecord` or its basis.)
- **Refuted ideas still have literature** (historical discussion + the refutation). The grounder should surface the contradiction (→ contradiction guard → challenged/uncertain); a refuted item landing `validated` is a true concordance failure (the harness's main signal).
- **Small N ⇒ noisy AUC.** 10–16 items gives wide CIs; the thresholds are conservative and the set must grow. A single passing run is necessary, not sufficient — re-run as the set grows.
- **LLM non-determinism** ⇒ per-item distributions, not points (§4).
- **The gold set could itself become the new un-witnessed assumption** if curated lazily (LLM-picked, opinion-labeled, hindsight-phrased, or cherry-picked). §3's criteria + the pre-registration discipline exist precisely to prevent that; the physicist's red-pen is the safeguard.
- **Seed-phrasing leakage:** any hindsight in a seed leaks the label; §3.1.6 + review of each seed.

---

## 8. Files
- `gold/validity_set.yaml` (Track A), `gold/faithfulness_set.yaml` (Track B) — frozen, git-committed, physicist-curated, with the §3.1b `provenance`/`signoff`/`stratum` fields.
- `scripts/concordance.py` — the batch harness (load → record `git hash-object` blob SHA + frozen thresholds → run pipeline ×K → collect signals incl. the `co_saturated` covariate → stratified metrics → report). Read-only.
- `valagents/concordance.py` — pure-code metric helpers (`balanced_accuracy` over decisive outcomes, `roc_auc`, bootstrap CI, stratified-AUC) — unit-testable independent of the expensive runs.
- `valagents/agents/grounder.py` — **small instrumentation:** surface a `co_saturated` signal (whether the Tier-2 Guard-2 `prop_distinctive`-empty path fired) on the grounder `CheckRecord` (a new optional field or a basis marker) so the harness can stratify (CN-D8). This is the only production change; it's read-only/diagnostic and does not affect the verdict.
- `results/concordance/<date>.md` — the report: header with `gold_blob_sha` + `gold_commit` + frozen thresholds (witnessable pre-registration); per-item rows; confusion matrices; **stratified** separation + CIs; abstain rate; pass/fail per unlock.
- (Reads `valagents.scheduler.run`, `valagents.agents.faithfulness.faithfulness_check`, the Popper `evidence_strength`, the grounder `co_saturated`.)

---

## 9. Decision log
- **CN-D1 (prerequisite, not afterthought)** Passing this harness gates Popper's migration, NLI's enablement, and human-facing E — all three. It is the next spec, ahead of those unlocks.
- **CN-D2 (two tracks)** Idea-validity (verdict + E) and faithfulness (NLI-vs-LLM) are distinct labeled sets in one harness.
- **CN-D3 (labels witnessable against a cited source — the regress fix)** Each label carries a `provenance` citation to the authoritative settled-science source + a human `signoff` ratifying it *against that source*. The label's authority is the source, not any model's or annotator's judgment — closing the "second say-so stacked on the first" regress (§3.0/§3.1b). Identical discipline to the grounder (support = quote ∈ source).
- **CN-D4 (witnessable pre-registration, not honor-system)** The gold-set git **blob hash** + frozen thresholds are recorded in the concordance report before any result is consumed; a reviewer re-hashes the committed file to detect post-hoc tampering. Don't ask for trust — prove no tune-to-test (§3.1b).
- **CN-D5 (read-only measurement)** The harness never feeds a verdict; pure-code metrics over ground truth.
- **CN-D6 (CI-lower-bound unlock + abstain class)** Unlock on the **bootstrap CI lower bound** (A2 ≥ 0.65), not point AUC (self-regulating on N). A1 scores **decisive outcomes only** (`validated` vs `refuted|challenged`), treating `needs_experiment`/`draft` as **abstain** (reported separately) — a valid idea correctly landing `needs_experiment` is not a failure.
- **CN-D7 (domain-matched + decomposable + matched-pairs)** Prefer CM/QM, idea-shaped, paired-by-subfield items so separation isn't confounded by literature volume/topic.
- **CN-D8 (co-saturation = measured covariate, stratified unlock)** Instrument the grounder's `co_saturated` flag; stratify E-separation by it and **unlock on the non-co-saturated stratum** so a known orthogonal Tier-2 bug can't sink or inflate the decision (§7/§5/A2).
- **CN-D9 (Track-B ≥2 annotators + agreement)** Faithfulness labels are double-annotated with a reported agreement statistic; low agreement → cut the item, don't force it.
- **CN-D10 (two strata)** (i) famous-saturated (sanity + qual credibility) and (ii) proposal-era-thin now-settled CM/QM (matches the novel home turf); **stratum (ii)'s E-separation is what licenses the Popper unlock for the real distribution.** Refuted CM/QM scarcity (criterion 10) is the binding constraint on building (ii).
- **CN-D11 (necessary-not-sufficient licensing)** Passing concordance on settled cases *permits* the unlocks as "validated on settled cases, novel-transfer an unmeasured residual"; failing *kills* them. Stated next to the thresholds so a pass isn't over-read (§5).

---

## 10. Build slices
1. **Pure metric helpers** (`valagents/concordance.py`): `roc_auc`, **stratified `roc_auc`** (AUC on a subset mask), `balanced_accuracy` **over decisive outcomes only** (abstain class), bootstrap CI — unit-tested on synthetic labels/scores (no pipeline runs).
2. **Grounder `co_saturated` instrumentation** (`valagents/agents/grounder.py`): surface whether Guard-2 (`prop_distinctive` empty) fired, on the `CheckRecord` — read-only/diagnostic, must not change any verdict. Tested: a co-saturated claim sets the flag; a normal pass does not; the existing grounder tests stay green (the flag is additive).
3. **The harness** (`scripts/concordance.py`): record the gold-set blob SHA + frozen thresholds in the report header; load frozen sets, run `run`×K (Track A) collecting `verdict_class`/`evidence_strength`/`co_saturated`, dual `faithfulness_check` (Track B); compute stratified metrics; emit the report. Tested with a **FakeLLM-scripted** mini gold set (harness logic covered without real models/network); the real run is a manual research invocation.
4. **The gold sets** (`gold/*.yaml`): physicist-curated, with `provenance`/`signoff`/`stratum`, frozen + committed. (Human work, not code; §3 is its requirements doc. **Blocked on the physicist by design** — esp. stratum (ii).)

---

## 11. Testing
- **Metric helpers (pure unit):** `roc_auc` on separable vs inseparable synthetic scores; **stratified `roc_auc`** returns the AUC of a masked subset; **`balanced_accuracy` over decisive outcomes** treats `needs_experiment`/`draft` as abstain and reports the abstain rate (a valid→needs_experiment item does not lower the score); bootstrap **CI lower bound** behaves (shrinks with N; an inseparable set's lower bound stays < 0.65); degenerate (all-one-label) handled.
- **Grounder `co_saturated` flag:** a co-saturated claim (property fully subtracted → Guard-2 empty) sets the flag; a normal on-property pass does not; **the verdict is unchanged either way** (additive/diagnostic); existing grounder tests stay green.
- **Harness logic (FakeLLM, no network):** a scripted mini set where the FakeLLM produces a known `verdict_class`/`evidence_strength`/`co_saturated` per seed → the harness computes the expected stratified AUC + decisive balanced-accuracy + abstain rate, and the report header records the gold-set blob SHA; Track B with a fake NLI scorer + scripted LLM → the NLI-vs-LLM comparison table is correct. (Tests the *plumbing*, not the science.)
- **Witnessable pre-registration:** the report records `git hash-object` of the gold file; a mutated gold file yields a different SHA (the tamper-detection property).
- **Read-only pin:** running the harness does not mutate any artifact/store/verdict.
- **No test runs a real pipeline or model** (cost/non-determinism); the real concordance run is a documented manual step.

---

## 12. Reviewer-scrutiny flags (author's uncertainties — attack these)
1. **THE GOLD SET (§3) — still the binding dependency, now with stronger discipline.** The criteria + discipline are hardened (provenance-citation + sign-off close the regress, CN-D3; witnessable blob-hash pre-registration, CN-D4). What remains irreducibly the **physicist's** work, not a reviewer-agent's: (a) the items + labels (ratified against cited sources, not asserted); (b) **stratum (ii)** — proposal-era-thin, now-settled CM/QM — which is the unlock-licensing stratum and the hardest to mine given the **refuted-CM scarcity** (criterion 10). The agent should attack the *criteria + discipline*; the physicist fills `provenance`/`signoff` and curates (ii).
2. ~~Track A co-saturation confound~~ — **ADDRESSED (design change):** co-saturation is now a measured covariate; E-separation is stratified and the unlock is on the non-co-saturated stratum (CN-D8/§5/§7). Requires the small grounder `co_saturated` instrumentation (§8).
3. ~~Small-N AUC validity~~ — **ADDRESSED:** unlock on the bootstrap **CI lower bound ≥ 0.65** (self-regulating on N), not point AUC (CN-D6/§5/A2).
4. **(IRREDUCIBLE ceiling, not an under-solved gap) Settled-vs-novel.** You cannot validate a validator on *novel* ideas, because "novel" means **unknown-truth** and validation requires **known-truth** — the two are definitionally exclusive, and **no gold set, ever, can close that.** Stratum (ii) narrows the *literature-thinness* gap; it cannot close the *known-vs-unknown* gap. Trying to "fix" it further would mean fabricating ground truth for novel cases — precisely the say-so this whole edifice refuses. So the honest ceiling is exactly what's built: **validate on settled/thin, license narrowly** (passing permits the unlocks for settled cases with novel-transfer an explicit, bounded, unmeasured residual), with that sentence next to the thresholds (§5). This is the cardinal rule **one level up** — code witnesses what it can (concordance on known cases); the irreducible residual (novel transfer) stays loud and bounded, never fabricated. Not a flag to close — a ceiling to state.
5. ~~Track B single-annotator~~ — **ADDRESSED:** ≥2 annotators + reported agreement (CN-D9/§3.3).
