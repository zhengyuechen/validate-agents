# Grounding Tier-2 (grounder `[A1]` support adjudication) Implementation Plan

> **BUILD AMENDMENT (T2-D12 — supersedes the floor design in Tasks 2–3 below).** The final whole-branch review found a cardinal-rule leak in the as-planned floor: it had the model emit `asserted_property`/`subject_phrase`, and the model could under-declare the property (or absorb property words into the subject) to collapse the distinctive set and launder an off-property quote. **The shipped floor is fully CODE-derived:** `prop_distinctive = _content_tokens(claim_statement) − _retrieval_saturated_tokens(articles)` (saturation-only; no model property/subject). `_support_quote_valid` signature is `(quote, source_text, claim_statement, subject_tokens, min_tokens)`; the `GROUNDER_CLAIM` prompt emits a citations JSON only. Authoritative record: spec §3/§4/§5 + decision log **T2-D12**, and the shipped code (`valagents/grounding.py`, `valagents/agents/grounder.py`). The Task 2/3 code blocks below show the *pre-T2-D12* require-ALL-with-`asserted_property` design — read them for the admissibility gate, dedup, caps, contradiction guard, and basis suffix (all unchanged), but take the floor's inputs from the amendment.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the grounder lens's retrieval-existence credit with a code-witnessed one — a real, verbatim, sentence-bounded, on-property quote from a real retrieved abstract — plus a load-bearing contradiction guard and a deduped count, while claiming only what code witnesses (presence + on-property topicality + anti-fabrication, NOT entailment, NOT independence).

**Architecture:** Three tasks. Tasks 1–2 add pure-code helpers to `valagents/grounding.py` (no network, no LLM): an admissibility gate shared by both citation directions, and a supports-only on-property floor with a UNION subject subtractor. Task 3 rewires `valagents/agents/grounder.py` `ground_claim` to make the LLM emit per-citation `{label, direction, quote}` JSON + `asserted_property` + `subject_phrase`, run the pure gate per citation, dedup, cap the count, and force a verdict downgrade when a real contradiction is cited. The gate (`artifact.py`) and `map_support_to_verdict` are untouched.

**Tech Stack:** Python 3, Pydantic v2, pytest (asyncio), conda env `cosci-reproduce`. Reuses Tier-1 grounding helpers (`_norm`, `_content_tokens`) and the agent JSON pattern (`value_grounder._extract_json`, `parse.checked_body`).

## Global Constraints

- **NEVER modify `valagents/artifact.py`** (gate purity — `_evaluate`, `status`, `_has_independent_external_check`, `verdict_class`, the ≥1 bar) **and NEVER modify `map_support_to_verdict` in `valagents/agents/base.py`.** Tier-2 changes only (a) how `independent_sources` is computed and (b) the contradiction downgrade — both inside `ground_claim`.
- **Pure helpers take primitives, not `cfg`** (`min_tokens: int`, `frac: float`) — consistent with the existing Tier-1 sibling `_quote_valid(quote, fetched_text, value, unit, referent, min_tokens)`. The agent path (`ground_claim`) unpacks `cfg.grounding.quote_min_tokens` / `cfg.grounding.subject_saturation_frac` and passes the primitives.
- **`subject_saturation_frac` default is exactly `0.6`.** `quote_min_tokens` default stays `6` (reused, not re-added).
- **Fail-closed everywhere:** missing/unparseable citations JSON, quote not in the abstract, cross-sentence splice, non-substantial quote, off-property quote, Guard 1/2 failure, or `SUPPORT != "supported"` → `code_witnessed=0` or the existing `map_support_to_verdict` downgrade → `uncertain`.
- **Commits: plain messages, NO attribution trailers** — no `Co-Authored-By`, no `Claude-Session`, no "Generated with Claude", nothing. Audit each commit message before committing.
- **Test command (full suite):** `conda run -n cosci-reproduce python -m pytest tests/ -q`. Single file: `conda run -n cosci-reproduce python -m pytest tests/<file> -q`.
- **Honest boundary (must hold in code comments + basis text):** code witnesses presence + on-property topicality + anti-fabrication; it does NOT witness entailment or independence. The `supports`/`contradicts` direction is the model's loud label.
- Every new file starts with `from __future__ import annotations`. Match the terse comment style of `grounding.py` (cite the spec section, e.g. `§5`, `§6`).
- **Spec:** `docs/2026-06-25-validate-agents-spec3-grounding-tier2-design.md`.

---

### Task 1: Admissibility gate (`_sentence_bounded`, `_quote_admissible`) — pure code

**Files:**
- Modify: `valagents/grounding.py` (add helpers after `_quote_valid`, around line 95)
- Test: `tests/test_grounding_support.py` (create)

**Interfaces:**
- Consumes: `_norm` (existing, `grounding.py:41`).
- Produces:
  - `_split_sentences(text: str) -> list[str]`
  - `_sentence_bounded(quote: str, source_text: str) -> bool`
  - `_quote_admissible(quote: str, source_text: str, min_tokens: int) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grounding_support.py`:

```python
from valagents.grounding import _split_sentences, _sentence_bounded, _quote_admissible

ABS = ("We study YbZn2GaO5, a quantum-spin-liquid candidate. "
       "The measured noise PSD is temperature-independent below 1 K. "
       "Samples were grown by the floating-zone method, e.g. as in prior work.")


def test_split_does_not_break_on_abbreviation_or_decimal():
    # "e.g." (lowercase after the period) and "1 K." (digit/space, no capital after) must NOT split mid-sentence.
    sents = _split_sentences("The PSD saturates at 0.5 K. We grew samples, e.g. by floating zone.")
    assert sents == ["The PSD saturates at 0.5 K.", "We grew samples, e.g. by floating zone."]


def test_sentence_bounded_true_within_one_sentence():
    assert _sentence_bounded("noise PSD is temperature-independent below 1 K", ABS) is True


def test_sentence_bounded_false_across_boundary():
    # spans the '. ' boundary between sentence 2 and 3 — the splice-inversion attack
    assert _sentence_bounded("below 1 K. Samples were grown", ABS) is False


def test_quote_admissible_full():
    assert _quote_admissible("the measured noise PSD is temperature-independent below 1 K", ABS, 6) is True


def test_quote_admissible_fabricated_not_in_bytes():
    assert _quote_admissible("the PSD shows clear 1/f temperature dependence here", ABS, 6) is False


def test_quote_admissible_too_short():
    assert _quote_admissible("temperature-independent below", ABS, 6) is False


def test_quote_admissible_cross_sentence_rejected():
    assert _quote_admissible("temperature-independent below 1 K. Samples were grown by", ABS, 6) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_support.py -q`
Expected: FAIL with `ImportError: cannot import name '_split_sentences'`.

- [ ] **Step 3: Implement the helpers**

In `valagents/grounding.py`, insert after `_quote_valid` (after line 94, before the `# The conditions parser's OWN ladders` block):

```python
# --- Tier-2 (§4): admissibility gate shared by BOTH citation directions (anti-fab + sentence-bound + substantial).
# Split on sentence-ending punctuation followed by whitespace + a capital letter. '0.5 K' (no space after the
# '.') and 'e.g. the' (lowercase after) do NOT split — abbreviation/decimal tolerance.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]


def _sentence_bounded(quote: str, source_text: str) -> bool:
    """True iff the normalized quote lies within a single sentence of source_text — kills the cross-sentence
    splice inversion ('...no order is observed. The lattice...' → 'order is observed. The lattice')."""
    nq = _norm(quote)
    if not nq:
        return False
    return any(nq in _norm(sent) for sent in _split_sentences(source_text))


def _quote_admissible(quote: str, source_text: str, min_tokens: int) -> bool:
    """§4 shared gate (BOTH directions): the quote is a real, single-sentence, substantial passage of
    source_text. Anti-fabrication + sentence-bounded + substantial. Does NOT judge direction or property."""
    nq = _norm(quote)
    if not nq or nq not in _norm(source_text):       # anti-fabrication (both sides _norm'd)
        return False
    if not _sentence_bounded(quote, source_text):    # no cross-sentence splice
        return False
    if len(nq.split()) < min_tokens:                 # substantial
        return False
    return True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_support.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Run the full grounding suite (no regressions)**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_quote.py tests/test_grounding_units.py tests/test_grounding_conditions.py -q`
Expected: PASS (all existing grounding tests green — the new helpers are additive).

- [ ] **Step 6: Commit**

```bash
git add valagents/grounding.py tests/test_grounding_support.py
git commit -m "grounding tier-2: admissibility gate (sentence-bounded + anti-fab + substantial)"
```

---

### Task 2: On-property floor (`_retrieval_saturated_tokens`, `_support_quote_valid`) — pure code

**Files:**
- Modify: `valagents/grounding.py` (add helpers after `_quote_admissible`)
- Test: `tests/test_grounding_support.py` (extend)

**Interfaces:**
- Consumes: `_norm`, `_content_tokens` (existing), `_quote_admissible` (Task 1).
- Produces:
  - `_retrieval_saturated_tokens(articles, frac: float) -> set[str]` — `articles` is any sequence of objects with a `.summary` str attribute (`web_search.Article`).
  - `_support_quote_valid(quote: str, source_text: str, claim_statement: str, asserted_property: str, subject_tokens: set[str], min_tokens: int) -> bool`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_grounding_support.py`:

```python
from valagents.grounding import _retrieval_saturated_tokens, _support_quote_valid, _content_tokens


class _Art:
    def __init__(self, summary):
        self.summary = summary


def test_retrieval_saturated_picks_common_tokens():
    arts = [_Art("ybzn2gao5 spin liquid noise"),
            _Art("ybzn2gao5 magnetization study"),
            _Art("ybzn2gao5 heat capacity")]
    sat = _retrieval_saturated_tokens(arts, 0.6)   # 'ybzn2gao5' in 3/3 ; others in 1/3
    assert "ybzn2gao5" in sat and "noise" not in sat


def test_retrieval_saturated_empty_corpus():
    assert _retrieval_saturated_tokens([], 0.6) == set()


# claim: "the noise PSD of YbZn2GaO5 is temperature-independent"
CLAIM = "the noise PSD of YbZn2GaO5 is temperature-independent"
PROP = "temperature-independent"
SUBJ_UNION = {"ybzn2gao5", "psd", "noise"}   # union = saturated ∪ subject_phrase tokens (caller-formed)
SRC = "We report that the noise PSD is temperature-independent below 1 K in this material."


def test_support_quote_valid_on_property():
    assert _support_quote_valid(
        "the noise PSD is temperature-independent below 1 K in this material",
        SRC, CLAIM, PROP, SUBJ_UNION, 6) is True


def test_support_quote_off_property_rejected():
    # a synthesis sentence sharing only the SUBJECT formula, no distinctive property token → fails the floor
    off = "single crystals of YbZn2GaO5 were grown by the floating-zone method in this study"
    assert _support_quote_valid(off, off, CLAIM, PROP, SUBJ_UNION, 6) is False


def test_support_quote_guard1_property_not_in_claim():
    # asserted_property carries a token absent from the claim → fail-closed
    assert _support_quote_valid(
        "the noise PSD is field-independent below 1 K in this material",
        "the noise PSD is field-independent below 1 K in this material",
        CLAIM, "field-independent", SUBJ_UNION, 6) is False


def test_support_quote_guard2_property_all_subject():
    # property is entirely subject tokens (property-as-subject) → prop_distinctive empty → fail
    assert _support_quote_valid(
        "the noise PSD of YbZn2GaO5 is reported in this material below 1 K",
        "the noise PSD of YbZn2GaO5 is reported in this material below 1 K",
        "the noise PSD of YbZn2GaO5", "noise PSD", {"ybzn2gao5", "psd", "noise"}, 6) is False


def test_support_quote_compound_fragment_rejected():
    # T2-D11 CRITICAL regression: a compound property ('temperature-independent' → {temperature, independent})
    # must NOT be credited by a quote sharing only ONE fragment. require-ALL closes this; any-overlap leaked.
    off_observable = "The magnetization shows strong temperature variation across the measured range here"   # only 'temperature'
    unrelated_sense = "The reported results were independent of the specific growth batch used here today"     # only 'independent'
    genuine = "We report the noise PSD is temperature-independent below 1 K in this material here"            # both
    assert _support_quote_valid(off_observable, off_observable, CLAIM, PROP, SUBJ_UNION, 6) is False
    assert _support_quote_valid(unrelated_sense, unrelated_sense, CLAIM, PROP, SUBJ_UNION, 6) is False
    assert _support_quote_valid(genuine, genuine, CLAIM, PROP, SUBJ_UNION, 6) is True


def test_support_quote_subject_subtraction_aids_recall():
    # Under require-ALL the subject subtraction is a RECALL aid: a genuine quote stating the full distinctive
    # property but OMITTING the subject formula still passes once the subject is subtracted; without
    # subtraction the same quote false-rejects (it would be forced to restate 'ybzn2gao5').
    leaky_prop = "YbZn2GaO5 temperature-independent"                       # property names the subject too (⊆ CLAIM)
    genuine = "the noise PSD is temperature-independent below 1 K in this material"   # omits 'YbZn2GaO5'
    assert _support_quote_valid(genuine, genuine, CLAIM, leaky_prop, set(), 6) is False          # no subtraction → forced to restate subject → reject
    assert _support_quote_valid(genuine, genuine, CLAIM, leaky_prop, {"ybzn2gao5"}, 6) is True   # subject subtracted → {temperature, independent} both present → pass
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_support.py -q`
Expected: FAIL with `ImportError: cannot import name '_retrieval_saturated_tokens'`.

- [ ] **Step 3: Implement the helpers**

In `valagents/grounding.py`, add immediately after `_quote_admissible` (from Task 1). Add `from collections import Counter` to the imports near the top of the file (after `import re`, `import unicodedata` at lines 32–33):

```python
from collections import Counter


def _retrieval_saturated_tokens(articles, frac: float) -> set[str]:
    """§5: content tokens appearing in >= `frac` of the retrieved abstracts — the entity/topic tokens
    retrieval already maximized (code-derived, ungameable). SUBTRACTED from the asserted property."""
    n = len(articles)
    if n == 0:
        return set()
    counts: Counter = Counter()
    for a in articles:
        counts.update(_content_tokens(a.summary))    # document frequency: each token once per abstract
    threshold = frac * n
    return {tok for tok, c in counts.items() if c >= threshold}


def _support_quote_valid(quote: str, source_text: str, claim_statement: str, asserted_property: str,
                         subject_tokens: set[str], min_tokens: int) -> bool:
    """§4/§5 supports-only gate = _quote_admissible AND the on-property floor. The quote must contain
    EVERY distinctive property token (require-ALL, not any) — a compound property like
    'temperature-independent' tokenizes to {temperature, independent}; requiring only one fragment lets an
    off-property quote ('temperature variation' of a different observable; 'independent' in an unrelated
    sense) earn credit (the T2-D11 false-credit). `subject_tokens` is the caller-formed union
    (retrieval-saturated ∪ subject_phrase); subtracting it is now a RECALL aid (don't force the quote to
    restate the subject), not the soundness mechanism. Witnesses on-property topicality, NOT entailment —
    a polarity flip carrying the full property phrase passes; direction stays the model's label."""
    if not _quote_admissible(quote, source_text, min_tokens):
        return False
    prop = _content_tokens(asserted_property)
    if not prop <= _content_tokens(claim_statement):       # Guard 1: property must be claim-derived
        return False
    prop_distinctive = prop - subject_tokens               # subtract subject (recall aid, not soundness)
    if not prop_distinctive:                               # Guard 2: non-vacuous
        return False
    return prop_distinctive <= _content_tokens(quote)      # require ALL distinctive tokens (T2-D11)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_support.py -q`
Expected: PASS (15 passed — 7 from Task 1 + 8 new).

- [ ] **Step 5: Commit**

```bash
git add valagents/grounding.py tests/test_grounding_support.py
git commit -m "grounding tier-2: on-property floor (require-ALL distinctive tokens, union subject subtractor)"
```

---

### Task 3: Grounder prompt + `ground_claim` rewrite (citations JSON, gate, dedup, contradiction guard)

**Files:**
- Modify: `valagents/config.py` (add `subject_saturation_frac` to `GroundCfg`, line 8–13)
- Modify: `valagents/prompts.py` (`GROUNDER_CLAIM`, lines 81–100)
- Modify: `valagents/agents/grounder.py` (`ground_claim`, lines 26–90; imports)
- Modify: `tests/test_agent_lenses.py` (update fixtures relying on retrieval-existence credit)
- Test: `tests/test_grounding_support_agent.py` (create — the Tier-2 agent-path tests)

**Interfaces:**
- Consumes: `_quote_admissible`, `_support_quote_valid`, `_retrieval_saturated_tokens`, `_content_tokens`, `_norm` (grounding.py, Tasks 1–2); `_extract_json` (`value_grounder.py:26`); `checked_body` (`parse.py:110`); `as_int`, `map_support_to_verdict`, `build_messages` (base.py); `references.normalize_id`, `references.detect_kind` (references.py:38, 29); `_extract_label` (existing, grounder.py:20).
- Produces: rewritten `async ground_claim(claim, formal_claim, backend, llm, cfg, tick=0) -> CheckRecord` (signature unchanged) + `_dedup_articles(articles) -> list`.

- [ ] **Step 1: Add the config knob**

In `valagents/config.py`, inside `GroundCfg` (after line 12 `quote_min_tokens`):

```python
    subject_saturation_frac: float = 0.6  # token in >= this fraction of retrieved abstracts = subject/topic (Tier-2 §5)
```

- [ ] **Step 2: Update the grounder prompt**

In `valagents/prompts.py`, replace the `GROUNDER_CLAIM` tail block (the `End with exactly:` line and below, lines 99–100) with a tail that drops the now-redundant `SOURCES` field and adds a citations JSON block. Replace:

```python
End with exactly:
CLAIM: {cid} | SUPPORT: supported|unsupported|uncertain | INDEPENDENT_SOURCES: <n> | SOURCES: <[A1], [A2], ...|none> | BASIS: <...>"""
```

with:

```python
End with exactly this line:
CLAIM: {cid} | SUPPORT: supported|unsupported|uncertain | INDEPENDENT_SOURCES: <n> | BASIS: <...>

Then, on the next lines, a JSON block naming the claim's property and your per-source citations:
```json
{{"asserted_property": "<what the claim asserts about its subject, copied from the sub-claim's own words>",
  "subject_phrase": "<the entity/material/system the sub-claim is about>",
  "citations": [{{"label": "A1", "direction": "supports", "quote": "<one verbatim sentence copied from that abstract>"}}]}}
```
- Copy each quote verbatim as a single complete sentence from the cited abstract; do not paraphrase or splice.
- direction is "supports" if that sentence backs the sub-claim, "contradicts" if it states the opposite.
- Cite only labels present in RETRIEVED LITERATURE; omit citations if none apply (use an empty list)."""
```

(Note: braces in the JSON example are doubled `{{ }}` because the template is consumed by `str.format`.)

- [ ] **Step 3: Write the failing agent-path tests**

Create `tests/test_grounding_support_agent.py`:

```python
import json
from valagents.config import Config
from valagents.agents.grounder import ground_claim
from valagents.artifact import AtomicClaim, FormalClaim
from valagents.web_search import Article
from tests.fake_llm import FakeLLM


def _cfg():
    return Config(default_model="fake")


CLAIM = AtomicClaim(id="c1", statement="the noise PSD of YbZn2GaO5 is temperature-independent",
                    type="empirical")
FC = FormalClaim(statement="PSD(T) = const")

# Three real abstracts: A1 carries an on-property supporting sentence; A2 a synthesis-only sentence;
# A3 a contradicting sentence. URLs distinct arXiv ids.
A_SUPPORT = Article(
    title="Noise spectroscopy of YbZn2GaO5",
    summary=("We study YbZn2GaO5, a quantum-spin-liquid candidate. "
             "The measured noise PSD is temperature-independent below 1 K."),
    url="http://arxiv.org/abs/2501.00001v1", published="2025-01-01")
A_SYNTH = Article(
    title="Crystal growth of YbZn2GaO5",
    summary="Single crystals of YbZn2GaO5 were grown by the floating-zone method in this study.",
    url="http://arxiv.org/abs/2501.00002v1", published="2025-01-02")
A_CONTRA = Article(
    title="Strong T-dependence in YbZn2GaO5",
    summary="In YbZn2GaO5 the noise PSD shows a strong temperature dependence below 1 K.",
    url="http://arxiv.org/abs/2501.00003v1", published="2025-01-03")


class _Backend:
    def __init__(self, arts):
        self._arts = arts

    async def search(self, query, max_results=10):
        return list(self._arts)


def _body(tail, payload):
    return tail + "\n```json\n" + json.dumps(payload) + "\n```"


def _llm(tail, payload):
    return FakeLLM(lambda a, m: _body(tail, payload))


async def test_supports_with_on_property_quote_passes():
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: direct on-property support"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports",
                              "quote": "The measured noise PSD is temperature-independent below 1 K."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "pass" and rec.independent_sources == 1


async def test_pass_basis_carries_honest_boundary():
    # §8: a credited pass must disclose IN THE BASIS that the credit is presence+topicality, not entailment/independence
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: direct on-property support"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports",
                              "quote": "The measured noise PSD is temperature-independent below 1 K."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "pass"
    assert "not code-witnessed" in rec.basis and "grounder credit" in rec.basis


async def test_fabricated_quote_uncertain():
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: claimed"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports",
                              "quote": "The PSD shows a clear 1/f temperature dependence nowhere in this abstract."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain" and rec.independent_sources == 0


async def test_off_property_synthesis_quote_uncertain():
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: claimed"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A2", "direction": "supports",
                              "quote": "Single crystals of YbZn2GaO5 were grown by the floating-zone method in this study."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain" and rec.independent_sources == 0


async def test_contradiction_guard_forces_uncertain_not_pass():
    # a valid supports AND a valid contradicts → the contradiction guard downgrades pass → uncertain (not fail)
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: mixed"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [
                   {"label": "A1", "direction": "supports",
                    "quote": "The measured noise PSD is temperature-independent below 1 K."},
                   {"label": "A3", "direction": "contradicts",
                    "quote": "In YbZn2GaO5 the noise PSD shows a strong temperature dependence below 1 K."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain"
    assert rec.basis.startswith("CONTRADICTION:")


async def test_dedup_preprint_and_published_count_once():
    # same arXiv id, two version URLs → one distinct work → code_witnessed capped to 1.
    # The backend carries 4 articles so 'independent' (in the 2 duplicates only, 2 of 4 < 0.6*4=2.4)
    # stays distinctive — otherwise the property co-saturates and nothing passes.
    a_v2 = Article(title=A_SUPPORT.title, summary=A_SUPPORT.summary,
                   url="http://arxiv.org/abs/2501.00001v2", published="2025-02-01")
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | BASIS: two cites same work"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [
                   {"label": "A1", "direction": "supports",
                    "quote": "The measured noise PSD is temperature-independent below 1 K."},
                   {"label": "A2", "direction": "supports",
                    "quote": "The measured noise PSD is temperature-independent below 1 K."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, a_v2, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.independent_sources == 1


async def test_thin_corpus_formula_leak_uncertain():
    # §10 day-one probe: the model SMUGGLES the subject formula into the property
    # ('YbZn2GaO5 temperature-independent') and labels a formula-only synthesis sentence "supports".
    # On a thin corpus the formula is in only 1 of 3 abstracts (saturation misses it). The agent ALWAYS
    # forms the union (saturated ∪ subject_phrase), which subtracts the formula → off-property → uncertain.
    thin = [A_SYNTH,
            Article(title="x", summary="unrelated spin liquid candidate magnetization", url="http://arxiv.org/abs/2501.01001v1", published="2025"),
            Article(title="y", summary="another frustrated magnet heat capacity", url="http://arxiv.org/abs/2501.01002v1", published="2025")]
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: claimed"
    payload = {"asserted_property": "YbZn2GaO5 temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports",
                              "quote": "Single crystals of YbZn2GaO5 were grown by the floating-zone method in this study."}]}
    rec = await ground_claim(CLAIM, FC, _Backend(thin), _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain" and rec.independent_sources == 0


async def test_backend_off_uncertain():
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 3 | BASIS: claimed"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports", "quote": "anything"}]}
    rec = await ground_claim(CLAIM, FC, None, _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain" and rec.independent_sources == 0


async def test_gate_purity_pass_requires_independent_at_least_one():
    # a VALID on-property supports quote (code_witnessed=1) but the LLM self-reports 0 independent →
    # min(0,1)=0 → the ≥1 bar still bites → uncertain. (3-article backend so 'independent' stays distinctive;
    # a 1-article corpus would co-saturate the whole property and fail for a different reason.)
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 0 | BASIS: direct"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports",
                              "quote": "The measured noise PSD is temperature-independent below 1 K."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain" and rec.independent_sources == 0
```

- [ ] **Step 4: Run the new tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_support_agent.py -q`
Expected: FAIL (the rewrite isn't in place — current `ground_claim` ignores the JSON block, builds sources from the now-absent `SOURCES` tail field, and reaches `pass` via retrieval-existence, so several asserts fail).

- [ ] **Step 5: Rewrite `ground_claim` and add `_dedup_articles`**

In `valagents/agents/grounder.py`, replace the imports (lines 6–10) and the entire `ground_claim` function (lines 26–90) with:

```python
from valagents.artifact import CheckRecord, Source, Novelty, AtomicClaim, FormalClaim
from valagents.parse import checked, checked_body
from valagents.prompts import GROUNDER_CLAIM, GROUNDER_NOVELTY
from valagents.agents.base import build_messages, map_support_to_verdict, as_int, choice
from valagents.agents.value_grounder import _extract_json
from valagents.grounding import (
    _quote_admissible, _support_quote_valid, _retrieval_saturated_tokens, _content_tokens, _norm,
)
from valagents.web_search import search_articles
from valagents import references
```

```python
def _dedup_articles(articles: list) -> list:
    """§7: collapse to distinct works. Key on normalize_id(url) for recognized arXiv/DOI ids, else the
    normalized title. Order-preserving. Protects the deferred ≥2 bar from per-version double counting."""
    seen: set[str] = set()
    out: list = []
    for a in articles:
        key = references.normalize_id(a.url) if references.detect_kind(a.url) != "unknown" else _norm(a.title)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


async def ground_claim(
    claim: AtomicClaim, formal_claim, backend, llm, cfg, tick: int = 0
) -> CheckRecord:
    """Ground a single atomic claim against retrieved literature (Tier-2: code-witnessed support).

    The LLM emits a SUPPORT/INDEPENDENT_SOURCES/BASIS tail PLUS a citations JSON (per-source
    {label, direction, quote}) + asserted_property + subject_phrase. Pure code (grounding.py) adjudicates
    each quote: anti-fabrication + sentence-bound + substantial (both directions), plus an on-property
    floor (supports only). The credited count is the number of DISTINCT retrieved works carrying a passing
    SUPPORTS quote, capped by both retrieval and the model's self-report. A passing CONTRADICTS quote forces
    a pass→uncertain downgrade. Code witnesses presence + on-property topicality + anti-fabrication — NOT
    entailment, NOT independence (the direction is the model's loud label). artifact.py is untouched.
    """
    formatted, articles = await search_articles(backend, claim.statement)
    label_to_article = {f"A{i}": a for i, a in enumerate(articles, start=1)}

    user = GROUNDER_CLAIM.format(
        ctype=claim.type, statement=claim.statement, articles=formatted or "(none)", cid=claim.id,
    )
    tail, body = await checked_body(
        "grounder",
        build_messages("You ground claims in literature.", user),
        ["CLAIM", "SUPPORT", "INDEPENDENT_SOURCES", "BASIS"],
        llm=llm,
    )
    if tail is None:
        return CheckRecord(lens="grounder", verdict="uncertain", basis="(unparseable)", tick=tick)

    g = cfg.grounding
    data = _extract_json(body) or {}
    asserted_property = str(data.get("asserted_property", ""))
    subject_phrase = str(data.get("subject_phrase", ""))
    raw_citations = data.get("citations")
    citations = raw_citations if isinstance(raw_citations, list) else []

    # §5: subtract the UNION of code-saturated topic tokens and the LLM-named subject tokens (thin-corpus fix).
    subject_tokens = _retrieval_saturated_tokens(articles, g.subject_saturation_frac) | _content_tokens(subject_phrase)

    passing: list = []
    contradicted = False
    contradiction_quote = ""
    for c in citations:
        if not isinstance(c, dict):
            continue
        label = _extract_label(str(c.get("label", "")))
        art = label_to_article.get(label) if label else None
        if art is None:
            continue
        quote = str(c.get("quote", ""))
        direction = str(c.get("direction", "")).strip().lower()
        if direction == "contradicts":
            if _quote_admissible(quote, art.summary, g.quote_min_tokens):   # §6: admissible only, NO property floor
                contradicted = True
                if not contradiction_quote:
                    contradiction_quote = quote
        elif direction == "supports":
            if _support_quote_valid(quote, art.summary, claim.statement,
                                    asserted_property, subject_tokens, g.quote_min_tokens):
                passing.append(art)

    deduped = _dedup_articles(passing)
    code_witnessed = min(len(deduped), len(articles))               # §7 code cap (cannot exceed retrieval)
    independent_sources = min(as_int(tail["independent_sources"]), code_witnessed)   # model may downgrade, never inflate
    verdict = map_support_to_verdict(tail["support"], independent_sources)
    if contradicted and verdict == "pass":                          # §6 contradiction guard (force-downgrade)
        verdict = "uncertain"

    srcs = [Source(locator=a.url, title=a.title, url=a.url, year=str(a.published)[:4], relation="independent")
            for a in deduped]
    basis = tail["basis"]
    if independent_sources >= 1:
        # §8 honest boundary IN THE ARTIFACT A HUMAN READS: the credited count is presence + on-property
        # topicality, NOT entailment, NOT independence. The field name (`independent_sources`) and
        # relation="independent" are kept for gate-compat, so the basis is the only place this is said.
        basis = (f"{basis} [grounder credit: {independent_sources} retrieved source(s) carrying a "
                 f"code-witnessed verbatim on-property passage; entailment & independence are the model's "
                 f"label, not code-witnessed]")
    if contradicted:
        basis = f"CONTRADICTION: {contradiction_quote} — {basis}"   # prefix kept FIRST (math-claim handling unchanged)

    return CheckRecord(
        lens="grounder", verdict=verdict, basis=basis, sources=srcs,
        independent_sources=independent_sources, tick=tick,
    )
```

Note: `_parse_source_labels` (grounder.py:13) is now unused (the `SOURCES` tail field is gone). Delete it. Keep `_extract_label` (still used for citation labels).

- [ ] **Step 6: Run the new agent tests to verify they pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_support_agent.py -q`
Expected: PASS (9 passed).

- [ ] **Step 7: Update the existing `ground_claim` fixtures that relied on retrieval-existence credit**

In `tests/test_agent_lenses.py`, FOUR tests change (the say-so→code-witnessed correction — credit now requires a passing on-property quote, not mere retrieval). First, replace the `FakeBackend` class (lines 13–21) so its articles carry real abstracts and distinct URLs. It needs FIVE articles: two supporting + one contradicting + two subject-only fillers, so the property tokens (`not`/`saturated`, in 2 of 5 = 0.4 < 0.6) stay distinctive (a 2-article all-supporting corpus would co-saturate the property and nothing could pass):

```python
class FakeBackend:
    """No-network backend returning canned Articles with real abstracts (Tier-2: quotes must be code-checkable)."""
    async def search(self, query: str, max_results: int = 10) -> list[Article]:
        return [
            Article(title="Alpha Saturation in Proteins",
                    summary="We report that alpha is not saturated under physiological conditions in this work.",
                    url="https://arxiv.org/abs/1234.5678", published="2022-03-15"),
            Article(title="Saturation Mechanisms",
                    summary="A second independent group finds that alpha is not saturated below threshold here.",
                    url="https://arxiv.org/abs/2345.6789", published="2021-07-01"),
            Article(title="Saturation Observed",
                    summary="In contrast, alpha reaches clear saturation at high concentration in our samples.",
                    url="https://arxiv.org/abs/3456.7890", published="2023-01-01"),
            Article(title="Alpha Kinetics",
                    summary="The alpha pathway kinetics were characterized in detail in this study.",
                    url="https://arxiv.org/abs/4567.8901", published="2023-02-01"),
            Article(title="Alpha Review",
                    summary="A broad review of the alpha regulatory system is presented here today.",
                    url="https://arxiv.org/abs/5678.9012", published="2023-03-01"),
        ]
```

(The claim `CM` is `statement="alpha not saturated"`. `asserted_property="not saturated"` ⊆ that claim; `subject_phrase="alpha"`. A small helper keeps the bodies readable — add it near the top of the file, after the imports:)

```python
import json as _json


def _grounder_body(tail: str, payload: dict) -> str:
    return tail + "\n```json\n" + _json.dumps(payload) + "\n```"
```

Update `test_grounder_supported_with_independent` (lines 35–42) — two passing supports → distinct works → pass with `independent_sources == 2`:

```python
@pytest.mark.asyncio
async def test_grounder_supported_with_independent(cfg):
    """Two retrieved works each carry a code-checked on-property supporting quote → pass, 2 independent."""
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | BASIS: ok"
    payload = {"asserted_property": "not saturated", "subject_phrase": "alpha", "citations": [
        {"label": "A1", "direction": "supports",
         "quote": "We report that alpha is not saturated under physiological conditions in this work."},
        {"label": "A2", "direction": "supports",
         "quote": "A second independent group finds that alpha is not saturated below threshold here."}]}
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: _grounder_body(tail, payload)), cfg)
    assert rec.verdict == "pass" and rec.independent_sources == 2 and rec.lens == "grounder"
    assert rec.sources[0].title == "Alpha Saturation in Proteins"
    assert rec.sources[0].url == "https://arxiv.org/abs/1234.5678"
```

Rewrite `test_grounder_contradiction_is_recorded_not_refuting` (lines 45–55) — under Tier-2 a contradiction is surfaced via the code-driven guard and the `CONTRADICTION:` basis prefix, NOT as a credited source (so it no longer indexes `sources[0]`):

```python
@pytest.mark.asyncio
async def test_grounder_contradiction_is_recorded_not_refuting(cfg):
    """A code-admissible contradicting quote forces uncertain (not fail) and is surfaced loud in basis."""
    tail = "CLAIM: c1 | SUPPORT: unsupported | INDEPENDENT_SOURCES: 1 | BASIS: retrieved work disagrees"
    payload = {"asserted_property": "not saturated", "subject_phrase": "alpha", "citations": [
        {"label": "A3", "direction": "contradicts",
         "quote": "In contrast, alpha reaches clear saturation at high concentration in our samples."}]}
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: _grounder_body(tail, payload)), cfg)
    assert rec.verdict == "uncertain"          # not refuting — the grounder never auto-fails a novel claim
    assert rec.basis.startswith("CONTRADICTION:")
```

Update `test_grounder_sources_carry_metadata` (lines 100–111) — sources now come from passing deduped articles:

```python
@pytest.mark.asyncio
async def test_grounder_sources_carry_metadata(cfg):
    """Sources in the CheckRecord carry title/url/year from the retrieved Articles whose quotes passed."""
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: direct"
    payload = {"asserted_property": "not saturated", "subject_phrase": "alpha", "citations": [
        {"label": "A1", "direction": "supports",
         "quote": "We report that alpha is not saturated under physiological conditions in this work."}]}
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: _grounder_body(tail, payload)), cfg)
    assert rec.verdict == "pass"
    assert len(rec.sources) == 1
    src = rec.sources[0]
    assert src.title == "Alpha Saturation in Proteins"
    assert src.url == "https://arxiv.org/abs/1234.5678"
    assert src.year == "2022"
```

Rewrite `test_grounder_unmatched_label_bare_source` (lines 114–124) — the `SOURCES` tail field and bare-source behavior are gone; a citation to a non-retrieved label is simply dropped:

```python
@pytest.mark.asyncio
async def test_grounder_unmatched_label_dropped(cfg):
    """A citation whose label was not retrieved is dropped — it cannot manufacture credit."""
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: ok"
    payload = {"asserted_property": "not saturated", "subject_phrase": "alpha", "citations": [
        {"label": "A9", "direction": "supports",
         "quote": "Some sentence about an article that was never retrieved at all here."}]}
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: _grounder_body(tail, payload)), cfg)
    assert rec.verdict == "uncertain" and rec.independent_sources == 0 and rec.sources == []
```

The other two `ground_claim` tests stay green unchanged — do NOT edit them: `test_grounder_downgrades_without_independent_source` (backend=None, no citations JSON → `code_witnessed=0` → uncertain) and `test_grounder_no_backend_trusts_no_llm_count` (backend=None, no citations → `min(3,0)=0` → uncertain). Both already assert only the verdict (and `independent_sources`), never index `sources`, so the rewrite leaves them green.

- [ ] **Step 8: Run the updated file to verify it passes**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_agent_lenses.py -q`
Expected: PASS (all green — the four updated/rewritten grounder tests plus the unchanged prover/grounder ones).

- [ ] **Step 9: Run the full suite (no regressions anywhere)**

Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS (the prior 401 Tier-1 tests + the new Tier-2 tests; no failures). If any unrelated test references the old `SOURCES` tail or `_parse_source_labels`, fix that reference (it is the same say-so→code-witnessed correction) and note it in the report.

- [ ] **Step 10: Commit**

```bash
git add valagents/config.py valagents/prompts.py valagents/agents/grounder.py tests/test_agent_lenses.py tests/test_grounding_support_agent.py
git commit -m "grounding tier-2: code-witnessed grounder support (citations JSON, dedup, contradiction guard)"
```

---

## Self-Review

**1. Spec coverage** (spec §1–§12):
- §3 pipeline (citations JSON, subject_subtract union, per-citation gate, code cap, min(llm,code), contradiction guard) → Task 3 Step 5. ✓
- §4 shared `_quote_admissible` + supports-only `_support_quote_valid` → Tasks 1, 2. ✓
- §5 property floor (**require-ALL distinctive tokens**, T2-D11) + subject subtractor (recall aid) + `_retrieval_saturated_tokens` + Guards 1/2 → Task 2. ✓
- §6 contradiction guard (admissible-only, force pass→uncertain, CONTRADICTION: basis) → Task 3 Step 5 + `test_contradiction_guard_forces_uncertain_not_pass`. ✓
- §7 dedup (`normalize_id`/title) + code cap + min(llm,code) → Task 3 `_dedup_articles` + `test_dedup_preprint_and_published_count_once`. ✓
- §8 honest rename (documentation/basis, field name unchanged) → basis suffix in Task 3 Step 5 (gated `independent_sources >= 1`) + `test_pass_basis_carries_honest_boundary`; `relation="independent"` kept (no schema change). The honest boundary now lives in the artifact a human reads, not just docstrings. ✓
- §9 fail-closed + gate purity (`artifact.py` / `map_support_to_verdict` untouched) → Global Constraints + `test_gate_purity_pass_requires_independent_at_least_one` + `test_backend_off_uncertain`. ✓
- §10 tests: compound-fragment false-credit regression (T2-D11) → Task 2 `test_support_quote_compound_fragment_rejected`; subject-subtraction-as-recall → `test_support_quote_subject_subtraction_aids_recall`; thin-corpus formula-leak → Task 3 `test_thin_corpus_formula_leak_uncertain` (now closed by require-ALL); co-saturation fail-closed → Guard 2 (`test_support_quote_guard2_property_all_subject`). ✓
- §12 slices 1 (pure gate+floor) / 2 (agent wiring) → Tasks 1+2 / Task 3. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows full code; every run step shows the exact command + expected result. ✓

**Soundness now rests on require-ALL distinctive tokens (T2-D11), not on subtraction:** a supporting quote must contain EVERY distinctive property token, so a quote sharing only one fragment of a compound property (`temperature` of a different observable; `independent` in an unrelated sense) cannot earn credit. Subject subtraction (`saturated ∪ subject_phrase`) is now a RECALL aid — it removes subject tokens so a genuine quote isn't forced to restate the subject formula — not the soundness mechanism. Two consequences for the fixtures and the honest boundary: (1) **co-saturation** still fail-closes (if the whole property co-saturates and is subtracted, `prop_distinctive` empties → Guard 2 → uncertain) — disclosed in §2/§5/§10; tests that must reach `pass` use corpora where a property token stays below the saturation threshold. (2) **single-token-distinctive residual:** when the distinctive set is one ambiguous word (a co-saturated compound reduced to `{independent}`, or an inherently one-word property), require-ALL = any, so that one word can still be matched in an unrelated sense — irreducible without semantics, overlaps the entailment residual, disclosed. The earlier "refine-never-empty" idea was rejected (T2-D10): restoring saturation-removed tokens reopens a rich-corpus leak; the genuine co-saturation fix needs a non-saturation subject signal, deferred to the ≥2 slice (T2-D9). The agent-path fixtures co-saturate `temperature`, so `prop_distinctive` there is `{independent}` and require-ALL = any — the require-ALL fix is exercised directly by the Task 2 unit test `test_support_quote_compound_fragment_rejected`, not the agent tests.

**3. Type consistency:** `_quote_admissible(quote, source_text, min_tokens)`, `_support_quote_valid(quote, source_text, claim_statement, asserted_property, subject_tokens, min_tokens)`, `_retrieval_saturated_tokens(articles, frac)` — same signatures in their definitions (Tasks 1–2) and their call sites (Task 3). `checked_body` returns `(tail, body)`; `tail` keys are lowercased (`tail["support"]`, `tail["independent_sources"]`, `tail["basis"]`) matching `parse._row`. `_extract_label`/`_extract_json`/`references.normalize_id`/`references.detect_kind` used with their real signatures. `Source`/`CheckRecord`/`Article` fields match `artifact.py`/`web_search.py`. ✓

**Note on a deliberate spec refinement:** the spec §4/§3 wrote some pure helpers as taking `cfg`; this plan has them take primitives (`min_tokens`, `frac`) to match the established Tier-1 sibling `_quote_valid(..., min_tokens)` and keep unit tests cfg-free. The agent path unpacks `cfg.grounding.*`. This is recorded in Global Constraints so the reviewer treats it as intended, not a deviation.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-06-25-validate-agents-spec3-grounding-tier2.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
