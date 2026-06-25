# Grounding (Spec 3, Tier 1: magnitude sourced values) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the magnitude lens's asserted `*_source` numeric inputs (`sensitivity`/`bound`/`closest_prior`) against their named, resolvable sources — the model reads (value + verbatim quote + conditions), code judges (quote ∈ fetched bytes ∧ code-owned unit conversion ∧ numeric match ∧ quantity-overlap ∧ conditions-compatible) — so a grounded value earns `independent_sources` credit (the say-so credit is otherwise stripped), an unconfirmable one is shown but doesn't clear the gate, and a literature-contradicted one suppresses its claim/attack.

**Architecture:** A new pure-code adjudicator module `valagents/grounding.py` (scale-table, quote/quantity/conditions gates, the four-outcome `ground_value`), a network fetch + a `value_grounder` extraction agent, three new `ComputationPlan` fields + designer prompt, and a per-path wiring in `run_magnitude_checks` that strips the LLM-`bound_source` say-so credit and replaces it with grounding. The gate (`artifact.py`) is never touched.

**Tech Stack:** Python, Pydantic v2, the existing `arxiv`/`httpx` deps, the existing `references.py` resolvers and `web_search` layer. Tests run under conda env `cosci-reproduce`: `conda run -n cosci-reproduce python -m pytest tests/ -q`.

## Global Constraints

- **Commit messages carry NO attribution trailer** — no `Co-Authored-By: Claude`, `Claude-Session:`, "Generated with Claude". Plain messages only; audit every commit (`git log -1 --format='%b'`).
- **NEVER modify `valagents/artifact.py`** (gate purity — `Source`, `CheckRecord`, `_evaluate`, `verdict_class`, the `internally_validated` rule stay as-is).
- **The model reads, code judges (F1/F3 for reading).** The adjudicator is pure code with NO LLM; the LLM only extracts. Every LLM assertion (quote, number, unit, referent, conditions) is checked against the fetched bytes / code-owned tables.
- **Code owns the unit conversion** (G-D3) — the LLM extracts in the source's own units; `convert` does the arithmetic via `SCALE_TABLE`. **Whole-token lookup only** (G-D9): a token with a denominator (`emu/g`, `emu·mol⁻¹`, …) is out-of-table → `unconfirmed`.
- **Conditions predicate uses its OWN T/field ladders, never `SCALE_TABLE.convert`** (G-D5b/F3 — `SCALE_TABLE`'s `K` is energy-via-k_B). G-D5c symmetry: a non-zero source clause on a v1-axis the claim doesn't constrain → not confirmed.
- **Four outcomes** (G-D6): `supports` (conditions-confirmed ∧ ratio<`supports_factor`), `contradicts` (conditions-confirmed ∧ ratio≥`contradict_factor`), `inconclusive` (gates pass, else), `unconfirmed` (a gate failed). Err toward `unconfirmed`/`inconclusive`; the harmful direction is a false `supports`.
- **Say-so credit STRIPPED (G-D6/G-D10):** a `bound_check` PASS earns `independent_sources=1` ONLY via grounding-`supports`; off/non-supports → `0`. **Symbolic `verdict_to_check` path UNCHANGED.**
- **Per-path wiring (G-D6):** `bound_check` → claim path (the only `independent_sources` path); `sensitivity_ratio`/`discriminating_margin` → attack path (`contradicts`→suppress the attack; `supports`/`inconclusive`→loud source in basis; no `independent_sources`).
- **On/off is an injected `resolver` (default `None`), not the backend string** — `None` → grounding skipped. Existing tests pass no resolver → grounding off.
- **Grounding adjudicates, never replaces** — never re-runs the magnitude arithmetic or swaps the asserted value.
- Config knobs: `supports_factor=2.0`, `contradict_factor=10.0`, `quote_min_tokens=6`, `reference_rel_tol=1e-3`.

---

## File Structure

- `valagents/config.py` — `GroundCfg` gains four float/int knobs (Task 1).
- `valagents/grounding.py` — **new**, the pure-code honesty core: `SCALE_TABLE`, `convert`, `_norm`, `_parse_floats`, `_quote_valid`, `_quantity_overlap`, `_conditions_compatible`, `GroundingResult`, `ground_value` (Tasks 2–5).
- `valagents/grounding_fetch.py` — **new**, `fetch_source_text(locator, resolver=None)` (Task 6, network; isolated so it's swappable in tests).
- `valagents/agents/value_grounder.py` — **new**, the extraction agent + `ground_plan(...)` orchestrator (Task 6).
- `valagents/prompts.py` — `VALUE_GROUNDER` prompt (Task 6); `MAGNITUDE_DESIGNER` gains the three new fields (Task 7).
- `valagents/computation.py` — `ComputationPlan` gains `source_quantity`/`claim_conditions`/`source_unit`; `verdict_to_check` magnitude branch (Task 7 fields; Task 8 strip).
- `valagents/agents/magnitude_designer.py` — `design_magnitude` emits the three new fields (Task 7).
- `valagents/scheduler.py` — `run_magnitude_checks` gains a `resolver` param + per-path grounding wiring; `run()` threads the resolver (Task 8).
- `tests/test_grounding_*.py` — new test files per task.

The implementer should `cat` each existing file before editing and follow its style.

---

## Task 1: GroundCfg knobs

**Files:**
- Modify: `valagents/config.py:8-9` (the `GroundCfg` class)
- Test: `tests/test_grounding_adjudicator.py` (create)

**Interfaces:**
- Produces: `GroundCfg.supports_factor: float = 2.0`, `contradict_factor: float = 10.0`, `quote_min_tokens: int = 6`, `reference_rel_tol: float = 1e-3`. Read by `ground_value` (Task 5) and the factor-verification test (Task 2).

- [ ] **Step 1: Write the failing test**

Create `tests/test_grounding_adjudicator.py`:

```python
from valagents.config import GroundCfg

def test_groundcfg_knobs():
    g = GroundCfg()
    assert g.supports_factor == 2.0 and g.contradict_factor == 10.0
    assert g.quote_min_tokens == 6 and g.reference_rel_tol == 1e-3
```

- [ ] **Step 2: Run it (FAIL — AttributeError)**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_adjudicator.py::test_groundcfg_knobs -v`
Expected: FAIL — `'GroundCfg' object has no attribute 'supports_factor'`.

- [ ] **Step 3: Add the fields**

In `valagents/config.py`, replace the `GroundCfg` class body:

```python
class GroundCfg(BaseModel):
    backend: str = "arxiv"          # arxiv | none | tavily
    supports_factor: float = 2.0    # ratio < this AND conditions-confirmed -> supports (G-D7)
    contradict_factor: float = 10.0 # ratio >= this AND conditions-confirmed -> contradicts (G-D7)
    quote_min_tokens: int = 6       # min word-tokens in a substantial referent-binding quote (§6)
    reference_rel_tol: float = 1e-3 # G-D9 scale-table both-directions reference-test tolerance
```

- [ ] **Step 4: Run it (PASS) + full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_adjudicator.py::test_groundcfg_knobs -v`
Expected: PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass (the new fields default; nothing reads them yet).

- [ ] **Step 5: Commit**

```bash
git add valagents/config.py tests/test_grounding_adjudicator.py
git commit -m "feat(grounding): GroundCfg knobs (supports_factor, contradict_factor, quote_min_tokens, reference_rel_tol)"
```

---

## Task 2: SCALE_TABLE + `convert` + the G-D9 factor-verification test

**Files:**
- Create: `valagents/grounding.py`
- Test: `tests/test_grounding_units.py` (create)

**Interfaces:**
- Produces: `SCALE_TABLE: dict[str, tuple[str, float]]` (token → (dimension, factor_to_canonical)); `convert(value: float, from_token: str, to_token: str) -> float | None` (None if either token is out-of-table or dimensions differ — whole-token exact-key lookup, never substring).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grounding_units.py`:

```python
import math
from valagents.grounding import SCALE_TABLE, convert

def test_convert_same_dimension():
    assert math.isclose(convert(1.0, "meV", "K"), 11.6045, rel_tol=1e-3)   # energy via k_B
    assert math.isclose(convert(12.0, "meV", "K"), 139.25, rel_tol=1e-3)
    assert convert(1.0, "Oe", "T") == 1e-4

def test_convert_cross_dimension_is_none():
    assert convert(1.0, "µB", "T") is None        # moment vs field

def test_convert_out_of_table_is_none():
    assert convert(1.0, "emu/g", "µB") is None     # whole-token: emu/g is NOT a key
    assert convert(1.0, "emu", "µB") == 1.078e20    # bare emu IS a moment

def test_scale_table_factor_references():
    # G-D9: every entry passes a both-directions reference test within reference_rel_tol (1e-3).
    refs = [("meV", "K", 11.6045), ("cm^-1", "K", 1.4388), ("Oe", "T", 1e-4), ("emu", "µB", 1.078e20)]
    for a, b, expected in refs:
        got = convert(1.0, a, b)
        assert got is not None
        assert max(got / expected, expected / got) - 1 < 1e-3

def test_factor_reference_catches_a_wrong_factor():
    # a deliberately-wrong meV factor must fail the reference check (the guard)
    bad = 1.5e-22 / SCALE_TABLE["K"][1]            # ~10.9, off from 11.6045 by ~6%
    assert max(bad / 11.6045, 11.6045 / bad) - 1 >= 1e-3
```

- [ ] **Step 2: Run them (FAIL — no module)**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_units.py -v`
Expected: FAIL — `ModuleNotFoundError: valagents.grounding`.

- [ ] **Step 3: Create `valagents/grounding.py` with the table + convert**

```python
"""Spec 3 grounding — the pure-code honesty core (no LLM, no network). The LLM reads (extracts a value +
verbatim quote + conditions); THIS module judges. Code owns the unit conversion (G-D3) and the regime
relevance (G-D5); a wrong scale-table factor is caught by the G-D9 both-directions reference test."""
from __future__ import annotations

# token -> (dimension, factor to the dimension's canonical unit). WHOLE-TOKEN exact-key lookup only (G-D9):
# a token with a denominator (emu/g, emu·mol⁻¹, ...) is simply absent -> out-of-table -> unconfirmed.
# Energy factors are FULL-PRECISION (4 sig figs fail the G-D9 reference test at rel-tol 1e-3).
SCALE_TABLE: dict[str, tuple[str, float]] = {
    # energy (canonical: Joule)
    "J": ("energy", 1.0), "eV": ("energy", 1.602177e-19), "meV": ("energy", 1.602177e-22),
    "K": ("energy", 1.380649e-23), "cm^-1": ("energy", 1.986446e-23),
    # magnetic field (canonical: Tesla; Oe/Gauss at the vacuum B-equivalence — §12 loud residual)
    "T": ("field", 1.0), "mT": ("field", 1e-3), "Gauss": ("field", 1e-4), "G": ("field", 1e-4), "Oe": ("field", 1e-4),
    # magnetic moment (canonical: Bohr magneton). bare emu only — emu/g, emu/cm^3, emu·mol⁻¹ are NOT moments.
    "µB": ("moment", 1.0), "mu_B": ("moment", 1.0), "emu": ("moment", 1.078e20), "J/T": ("moment", 1.078e23),
    # magnetic flux (canonical: flux quantum; identity within the dimension only — NO cross to field)
    "Φ0": ("flux", 1.0), "Phi0": ("flux", 1.0), "mΦ0": ("flux", 1e-3), "µΦ0": ("flux", 1e-6),
}


def convert(value: float, from_token: str, to_token: str) -> float | None:
    """Convert within a single physical dimension via SCALE_TABLE (CODE owns this — the LLM never converts).
    Returns None if either token is out-of-table (whole-token exact key) or the dimensions differ."""
    f = SCALE_TABLE.get(from_token)
    t = SCALE_TABLE.get(to_token)
    if f is None or t is None or f[0] != t[0]:
        return None
    return value * f[1] / t[1]
```

- [ ] **Step 4: Run them (PASS) + full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_units.py -v`
Expected: all PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add valagents/grounding.py tests/test_grounding_units.py
git commit -m "feat(grounding): SCALE_TABLE + code-owned convert + G-D9 factor-verification test (whole-token, full-precision energy)"
```

---

## Task 3: `_quote_valid` + `_quantity_overlap` (the quote & quantity gates)

**Files:**
- Modify: `valagents/grounding.py`
- Test: `tests/test_grounding_quote.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `_norm(s: str) -> str` (NFKC, collapse whitespace, casefold).
  - `_parse_floats(s: str) -> list[float]` (all numeric tokens incl. sci-notation/`×10^`).
  - `_quote_valid(quote, fetched_text, extracted_value, unit_token, referent, min_tokens) -> bool` — §6: quote ∈ normalized fetched bytes; quote contains the extracted numeral (any surface form, within 1e-6 rel), the **full** `unit_token` (whole token), and the `referent`; and has ≥ `min_tokens` whitespace word-tokens.
  - `_quantity_overlap(referent, source_quantity) -> bool` — ≥1 shared content token after stop-word removal (the necessary quantity floor; symbol↔prose misses → False → caller `unconfirmed`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grounding_quote.py`:

```python
from valagents.grounding import _quote_valid, _quantity_overlap, _parse_floats

TEXT = "We find the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 0.4 K below the transition."

def test_parse_floats():
    got = _parse_floats("value 1.2e-3 µB and 0.4 K")
    assert any(abs(x - 1.2e-3) < 1e-9 for x in got) and any(abs(x - 0.4) < 1e-9 for x in got)

def test_quote_valid_full():
    q = "the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 0.4 K"
    assert _quote_valid(q, TEXT, 1.2e-3, "µB", "ordered moment per Yb ion", 6) is True

def test_quote_not_in_bytes():
    assert _quote_valid("a fabricated 1.2e-3 µB ordered moment never written here", TEXT, 1.2e-3, "µB", "moment", 6) is False

def test_quote_degenerate_bare_number():
    assert _quote_valid("1.2e-3", TEXT, 1.2e-3, "µB", "moment", 6) is False   # no unit, no referent, too short

def test_quote_missing_unit():
    q = "the ordered moment per Yb ion saturates at 1.2e-3 at T = 0.4 K"      # fabricated (no µB) -> not in bytes anyway
    assert _quote_valid(q, TEXT, 1.2e-3, "µB", "moment", 6) is False

def test_quantity_overlap():
    assert _quantity_overlap("ordered moment per Yb ion", "Yb³⁺ effective magnetic moment") is True
    assert _quantity_overlap("applied magnetic field", "Yb³⁺ magnetic moment") is False   # field vs moment
    assert _quantity_overlap("µ_eff", "effective moment") is False                        # symbol↔prose miss (safe)
```

- [ ] **Step 2: Run them (FAIL — import error)**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_quote.py -v`
Expected: FAIL — cannot import `_quote_valid`.

- [ ] **Step 3: Implement the helpers**

Append to `valagents/grounding.py`:

```python
import re
import unicodedata

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*[×x]\s*10\s*\^?\s*[-+]?\d+)?")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "per", "in", "at", "and", "or", "to", "for", "with", "is", "are",
         "we", "find", "this", "that", "by", "on", "from", "as", "its"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "")).strip().casefold()


def _parse_floats(s: str) -> list[float]:
    out: list[float] = []
    for m in _NUM_RE.finditer(s or ""):
        tok = m.group(0)
        try:
            if "10" in tok and ("×" in tok or "x" in tok):     # a×10^b surface form
                mant, _, exp = re.split(r"[×x]\s*10\s*\^?", tok)
                out.append(float(mant) * 10 ** int(re.sub(r"\s", "", exp)))
            else:
                out.append(float(tok))
        except (ValueError, IndexError):
            continue
    return out


def _content_tokens(s: str) -> set[str]:
    return {w for w in _WORD_RE.findall(_norm(s)) if w not in _STOP}


def _quantity_overlap(referent: str, source_quantity: str) -> bool:
    """§5 G-D5a quantity gate: ≥1 shared content token. Necessary floor; symbol↔prose misses -> False (safe)."""
    return bool(_content_tokens(referent) & _content_tokens(source_quantity))


def _numeral_present(quote: str, value: float, rtol: float = 1e-6) -> bool:
    scale = max(abs(value), 1e-300)
    return any(abs(x - value) / scale < rtol for x in _parse_floats(quote))


def _quote_valid(quote: str, fetched_text: str, extracted_value: float,
                 unit_token: str, referent: str, min_tokens: int) -> bool:
    """§6: the quote asserts *this quantity has this value* and is literally in the source.
    Requires: quote ∈ normalized fetched bytes; quote carries the extracted numeral, the FULL unit token
    (whole token), and the referent; and ≥ min_tokens whitespace word-tokens. A bare-number / number-only
    quote fails — that is the anti-fabrication strength, not cosmetic."""
    nq = _norm(quote)
    if not nq or nq not in _norm(fetched_text):                 # anti-fabrication
        return False
    if len(nq.split()) < min_tokens:                            # substantial
        return False
    if not _numeral_present(quote, extracted_value):            # carries the value
        return False
    if _norm(unit_token) not in nq:                             # carries the FULL unit token
        return False
    if not _content_tokens(referent) & _content_tokens(quote):  # carries the referent
        return False
    return True
```

- [ ] **Step 4: Run them (PASS) + full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_quote.py -v`
Expected: all PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add valagents/grounding.py tests/test_grounding_quote.py
git commit -m "feat(grounding): _quote_valid (substantial, full-compound-token, in-bytes) + _quantity_overlap"
```

---

## Task 4: `_conditions_compatible` (the conditions predicate — own ladders, G-D5c symmetry, `=` clause)

**Files:**
- Modify: `valagents/grounding.py`
- Test: `tests/test_grounding_conditions.py` (create)

**Interfaces:**
- Consumes: `_norm`.
- Produces: `_conditions_compatible(claim_conditions: str, source_conditions: str) -> bool`. Parses each side into `{axis: (op, value_canonical)}` over v1 axes (temperature, field) using **its own** ladders (`_TEMP_UNITS`, `_FIELD_UNITS`), NEVER `SCALE_TABLE`. Returns True iff every claim clause is satisfied by a source point on that axis AND no non-zero source clause sits on a v1-axis the claim does not constrain (G-D5c). `=` claim clause → exact match only (no tolerance). Empty/unparseable claim or any miss → False.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grounding_conditions.py`:

```python
from valagents.grounding import _conditions_compatible as cc

def test_temperature_within():
    assert cc("T < 1 K", "T = 0.4 K") is True
    assert cc("T < 1 K", "T = 300 K") is False

def test_sub_kelvin_ladder():
    assert cc("T < 1 K", "400 mK") is True          # mK ladder, NOT SCALE_TABLE's energy-K
    assert cc("T < 1 K", "T = 400 mK") is True

def test_field_axis():
    assert cc("B < 0.1 T", "B = 5 mT") is True
    assert cc("B < 0.1 T", "B = 5 T") is False

def test_g_d5c_claim_silent_field_nonzero_source():
    # claim silent on field, source pins B=5 T -> NOT confirmed (the F1 hole)
    assert cc("T < 1 K", "T = 0.4 K, B = 5 T") is False
    # source B=0 (baseline) is permissive
    assert cc("T < 1 K", "T = 0.4 K, B = 0 T") is True
    assert cc("T < 1 K", "T = 0.4 K, B = 0") is True

def test_equals_clause_exact_only():
    assert cc("T = 0.3 K", "T = 0.3 K") is True
    assert cc("T = 0.3 K", "T = 0.4 K") is False

def test_absent_or_unparseable_not_confirmed():
    assert cc("T < 1 K", "measured at low temperature") is False   # no parseable source clause
    assert cc("", "T = 0.4 K") is False                            # empty claim -> not confirmed
```

- [ ] **Step 2: Run them (FAIL — import error)**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_conditions.py -v`
Expected: FAIL — cannot import `_conditions_compatible`.

- [ ] **Step 3: Implement the predicate**

Append to `valagents/grounding.py`:

```python
# The conditions parser's OWN ladders (G-D5b/F3) — NEVER SCALE_TABLE (whose K is energy-via-k_B, T is Tesla).
_TEMP_UNITS = {"k": 1.0, "mk": 1e-3, "µk": 1e-6, "uk": 1e-6}
_FIELD_UNITS = {"t": 1.0, "mt": 1e-3, "g": 1e-4, "gauss": 1e-4, "oe": 1e-4}
_AXIS_BY_SYMBOL = {"t": "temperature", "temp": "temperature", "temperature": "temperature",
                   "b": "field", "h": "field", "field": "field"}
_AXIS_BY_UNIT = {**{u: "temperature" for u in _TEMP_UNITS}, **{u: "field" for u in _FIELD_UNITS}}
_CLAUSE_RE = re.compile(
    r"(?:(?P<sym>[a-zµ]+)\s*(?P<op><=|>=|=|<|>|~)\s*)?(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>[a-zµ^\-0-9]+)")


def _to_canonical(value: float, unit: str, axis: str) -> float | None:
    table = _TEMP_UNITS if axis == "temperature" else _FIELD_UNITS
    f = table.get(unit)
    return None if f is None else value * f


def _parse_clauses(s: str) -> dict[str, tuple[str, float]] | None:
    """Parse 'T < 1 K, B = 0' into {axis: (op, canonical_value)}. A bare point ('400 mK') -> op '='.
    Returns {} if nothing parsed. Unit disambiguates the energy-K vs temperature-K collision: a clause's
    unit is resolved in the conditions ladders, and the axis is taken from the symbol (if present) else the unit."""
    out: dict[str, tuple[str, float]] = {}
    for m in _CLAUSE_RE.finditer(_norm(s)):
        unit = m.group("unit")
        sym = m.group("sym")
        axis = _AXIS_BY_SYMBOL.get(sym) if sym else None
        if axis is None:
            axis = _AXIS_BY_UNIT.get(unit)
        if axis is None:
            continue
        canon = _to_canonical(float(m.group("val")), unit, axis)
        if canon is None:
            continue
        out[axis] = (m.group("op") or "=", canon)
    return out


def _satisfies(op: str, claim_val: float, source_val: float) -> bool:
    if op == "<":  return source_val < claim_val
    if op == "<=": return source_val <= claim_val
    if op == ">":  return source_val > claim_val
    if op == ">=": return source_val >= claim_val
    # '=' and '~' : exact point match only (NO regime tolerance — would reopen the wrong-conditions hole)
    return abs(source_val - claim_val) <= 1e-9 * max(abs(claim_val), 1e-300)


def _conditions_compatible(claim_conditions: str, source_conditions: str) -> bool:
    """§5 G-D5b/c: the source regime must lie within the claim regime on every v1 axis the claim constrains,
    AND must not pin a non-zero value on a v1 axis the claim leaves free (G-D5c). Err to False (not confirmed)."""
    claim = _parse_clauses(claim_conditions)
    source = _parse_clauses(source_conditions)
    if not claim or source is None:
        return False
    for axis, (op, cval) in claim.items():
        if axis not in source:
            return False
        if not _satisfies(op, cval, source[axis][1]):
            return False
    for axis, (_, sval) in source.items():                 # G-D5c symmetry
        if axis not in claim and abs(sval) > 0.0:
            return False
    return True
```

- [ ] **Step 4: Run them (PASS) + full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_conditions.py -v`
Expected: all PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add valagents/grounding.py tests/test_grounding_conditions.py
git commit -m "feat(grounding): _conditions_compatible (own T/field ladders, G-D5c symmetry, exact-= clause)"
```

---

## Task 5: `GroundingResult` + `ground_value` (the four-outcome decision tree)

**Files:**
- Modify: `valagents/grounding.py`
- Test: `tests/test_grounding_adjudicator.py` (append to the Task-1 file)

**Interfaces:**
- Consumes: `convert`, `_quote_valid`, `_quantity_overlap`, `_conditions_compatible`, `_parse_floats` (all in `grounding.py`); `GroundCfg` knobs.
- Produces:
  - `GroundingResult` dataclass: `status: str` (`"supports"|"contradicts"|"inconclusive"|"unconfirmed"`), `reason: str = ""`, `converted_value: float | None = None`, `quote: str = ""`, `referent: str = ""`, `source_conditions: str = ""`, `source_unit_token: str = ""`.
  - `ground_value(asserted_value: str, source_unit: str, source_quantity: str, claim_conditions: str, extraction: dict | None, fetched_text: str, cfg) -> GroundingResult`. `extraction` is the value_grounder output `{"extracted_value","source_unit_token","referent","source_conditions","verbatim_quote"}` or `None` (= `not_found`). The four-outcome tree of §5. No exceptions escape (any parse failure → `unconfirmed`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_grounding_adjudicator.py`:

```python
from valagents.config import Config
from valagents.grounding import ground_value, GroundingResult

def _cfg():
    return Config(default_model="fake")

TEXT = "We find the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 0.4 K below 1 K."

def _ext(value="1.2e-3", unit="µB", referent="ordered moment per Yb ion",
         cond="T = 0.4 K", quote="the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 0.4 K"):
    return {"extracted_value": value, "source_unit_token": unit, "referent": referent,
            "source_conditions": cond, "verbatim_quote": quote}

def _g(**kw):
    a = dict(asserted_value="1.2e-3", source_unit="µB", source_quantity="Yb³⁺ effective magnetic moment",
             claim_conditions="T < 1 K", extraction=_ext(), fetched_text=TEXT)
    a.update(kw)
    return ground_value(a["asserted_value"], a["source_unit"], a["source_quantity"],
                        a["claim_conditions"], a["extraction"], a["fetched_text"], _cfg())

def test_supports():
    r = _g()
    assert r.status == "supports" and r.quote and r.converted_value is not None

def test_not_found_unconfirmed():
    assert _g(extraction=None).status == "unconfirmed"

def test_fabricated_quote_unconfirmed():
    assert _g(extraction=_ext(quote="a 1.2e-3 µB moment not in the source text at all here")).status == "unconfirmed"

def test_out_of_table_unit_unconfirmed():
    assert _g(extraction=_ext(unit="emu/g",
              quote="the magnetization per gram is 1.2e-3 emu/g at T = 0.4 K low temperature"),
              source_unit="µB").status == "unconfirmed"

def test_quantity_mismatch_unconfirmed():
    r = _g(source_quantity="applied magnetic field strength")   # referent 'moment' vs 'field' -> no overlap
    assert r.status == "unconfirmed"

def test_wrong_conditions_inconclusive():
    r = _g(extraction=_ext(cond="T = 300 K", quote="the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 300 K"),
           fetched_text="the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 300 K room temperature")
    assert r.status == "inconclusive" and r.reason == "conditions_unconfirmed"

def test_numeric_inconclusive():
    r = _g(asserted_value="4e-4")     # ratio 1.2e-3/4e-4 = 3.0 in [2,10) -> inconclusive
    assert r.status == "inconclusive" and r.reason == "numeric_inconclusive"

def test_contradicts_same_regime():
    r = _g(asserted_value="1e-4")     # ratio 12 >= 10, conditions compatible -> contradicts
    assert r.status == "contradicts"

def test_no_false_contradict_wrong_regime():
    r = _g(asserted_value="1e-4",
           extraction=_ext(cond="T = 300 K", quote="the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 300 K"),
           fetched_text="the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 300 K room temperature")
    assert r.status == "inconclusive"   # ratio gross BUT conditions incompatible -> not a contradiction

def test_code_owns_conversion():
    r = _g(asserted_value="139", source_unit="K",
           extraction=_ext(value="12", unit="meV", referent="exchange energy J",
                           cond="T = 0.4 K", quote="the exchange energy J is 12 meV at T = 0.4 K low temperature"),
           source_quantity="exchange energy", claim_conditions="T < 1 K",
           fetched_text="the exchange energy J is 12 meV at T = 0.4 K low temperature")
    assert r.status == "supports"   # 12 meV -> 139.25 K, ratio ~ 1
```

- [ ] **Step 2: Run them (FAIL)**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_adjudicator.py -v`
Expected: FAIL — cannot import `ground_value`/`GroundingResult`.

- [ ] **Step 3: Implement the tree**

Append to `valagents/grounding.py`:

```python
from dataclasses import dataclass


@dataclass
class GroundingResult:
    status: str                       # supports | contradicts | inconclusive | unconfirmed
    reason: str = ""
    converted_value: float | None = None
    quote: str = ""
    referent: str = ""
    source_conditions: str = ""
    source_unit_token: str = ""


def _one_float(s) -> float | None:
    vals = _parse_floats(str(s))
    return vals[0] if vals else None


def ground_value(asserted_value: str, source_unit: str, source_quantity: str, claim_conditions: str,
                 extraction: dict | None, fetched_text: str, cfg) -> GroundingResult:
    """§5 four-outcome adjudicator. Pure code; no exception escapes (any failure -> unconfirmed). The harmful
    direction is a false 'supports' — guarded by quote+unit+quantity+conditions gates BEFORE the numeric step."""
    g = cfg.grounding
    if not extraction:
        return GroundingResult("unconfirmed", reason="not_found")
    ev = _one_float(extraction.get("extracted_value"))
    unit_token = extraction.get("source_unit_token", "")
    referent = extraction.get("referent", "")
    src_cond = extraction.get("source_conditions", "")
    quote = extraction.get("verbatim_quote", "")
    if ev is None:
        return GroundingResult("unconfirmed", reason="unparseable_value")
    if not _quote_valid(quote, fetched_text, ev, unit_token, referent, g.quote_min_tokens):
        return GroundingResult("unconfirmed", reason="quote_invalid")
    converted = convert(ev, unit_token, source_unit)            # CODE owns the conversion
    if converted is None:
        return GroundingResult("unconfirmed", reason="unit_out_of_table")
    if not _quantity_overlap(referent, source_quantity):
        return GroundingResult("unconfirmed", reason="quantity_mismatch")
    # SHOWABLE from here (quote+unit+quantity all valid) — carry the loud fields on every remaining outcome.
    loud = dict(converted_value=converted, quote=quote, referent=referent,
                source_conditions=src_cond, source_unit_token=unit_token)
    asserted = _one_float(asserted_value)
    if asserted is None or asserted == 0.0 or converted == 0.0:
        return GroundingResult("inconclusive", reason="numeric_inconclusive", **loud)
    ratio = max(abs(converted / asserted), abs(asserted / converted))
    cond_ok = _conditions_compatible(claim_conditions, src_cond)
    if cond_ok and ratio < g.supports_factor:
        return GroundingResult("supports", **loud)
    if cond_ok and ratio >= g.contradict_factor:
        return GroundingResult("contradicts", reason="gross_disagreement", **loud)
    return GroundingResult("inconclusive",
                           reason=("numeric_inconclusive" if cond_ok else "conditions_unconfirmed"), **loud)
```

- [ ] **Step 4: Run them (PASS) + full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_adjudicator.py -v`
Expected: all PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add valagents/grounding.py tests/test_grounding_adjudicator.py
git commit -m "feat(grounding): GroundingResult + ground_value four-outcome tree (supports/contradicts/inconclusive/unconfirmed)"
```

---

## Task 6: Fetch + the `value_grounder` extraction agent + `ground_plan`

**Files:**
- Create: `valagents/grounding_fetch.py`, `valagents/agents/value_grounder.py`
- Modify: `valagents/prompts.py` (add `VALUE_GROUNDER`)
- Test: `tests/test_grounding_agent.py` (create)

**Interfaces:**
- Produces:
  - `fetch_source_text(locator: str, resolver=None) -> tuple[str, dict] | None` — resolve the locator via `references.DefaultResolver` (or an injected `resolver`) and return `(text, {"locator","title","url","year"})` or `None`. **v1 fetches the abstract** (arXiv summary / crossref abstract / URL→text) — full-text PDF is a v1.x lever (raw-PDF text breaks the verbatim-quote check; clean abstract text is what the quote machinery needs). Network; injected in tests.
  - `async ground_plan(plan, resolver, llm, cfg) -> GroundingResult | None` — orchestrator: returns `None` if `resolver is None` (grounding off); else picks the kind's `(value, locator, source_quantity, claim_conditions, source_unit)`, fetches, runs the extraction agent, calls `ground_value`. Returns `unconfirmed` (not None) if the fetch/extraction fails but grounding is on.
  - `async extract_value(text, source_quantity, source_unit, llm) -> dict | None` (in value_grounder.py) — the LLM extraction; returns the 5-field dict or `None`.

- [ ] **Step 1: Write the failing tests** (deterministic — fake resolver + FakeLLM)

Create `tests/test_grounding_agent.py`:

```python
import json
from valagents.config import Config
from valagents.agents.value_grounder import ground_plan
from valagents.computation import ComputationPlan
from tests.fake_llm import FakeLLM

def _cfg():
    return Config(default_model="fake")

ABSTRACT = "We report the ordered moment per Yb ion of 1.2e-3 µB at T = 0.4 K in the candidate QSL."

def _resolver(text=ABSTRACT):
    class R:
        async def fetch(self, locator):    # the injected fetch contract used by ground_plan
            return (text, {"locator": locator, "title": "T", "url": "u", "year": "2024"})
    return R()

EXTRACTION = {"extracted_value": "1.2e-3", "source_unit_token": "µB",
              "referent": "ordered moment per Yb ion", "source_conditions": "T = 0.4 K",
              "verbatim_quote": "the ordered moment per Yb ion of 1.2e-3 µB at T = 0.4 K"}

def _llm(ext=EXTRACTION):
    body = "```json\n" + json.dumps(ext) + "\n```"
    return FakeLLM(lambda a, m: body if a == "value_grounder" else "")

def _bplan():
    return ComputationPlan(kind="magnitude", comparison_kind="bound_check",
        predicted_effect="1e-3", bound="1.2e-3", bound_source="arXiv:2104.01234",
        source_quantity="Yb³⁺ effective magnetic moment", claim_conditions="T < 1 K", source_unit="µB")

async def test_ground_plan_off_when_no_resolver():
    assert await ground_plan(_bplan(), None, _llm(), _cfg()) is None

async def test_ground_plan_supports():
    r = await ground_plan(_bplan(), _resolver(), _llm(), _cfg())
    assert r is not None and r.status == "supports"

async def test_ground_plan_fabricated_quote_unconfirmed():
    bad = {**EXTRACTION, "verbatim_quote": "a moment of 1.2e-3 µB nowhere in the abstract text"}
    r = await ground_plan(_bplan(), _resolver(), _llm(bad), _cfg())
    assert r is not None and r.status == "unconfirmed"

async def test_ground_plan_unresolvable_unconfirmed():
    class Dead:
        async def fetch(self, locator):
            return None
    r = await ground_plan(_bplan(), Dead(), _llm(), _cfg())
    assert r is not None and r.status == "unconfirmed"
```

- [ ] **Step 2: Run them (FAIL)**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_agent.py -v`
Expected: FAIL — cannot import `ground_plan`.

- [ ] **Step 3: Implement fetch, the prompt, the agent, and `ground_plan`**

Add to `valagents/prompts.py`:

```python
VALUE_GROUNDER = """You READ a source and report what it says about ONE physical quantity. You do NOT judge \
whether it supports any claim, you do NOT convert units, you do NOT infer — code does the judging. Report the \
source's PRIMARY value for the quantity, in the SOURCE's own units, with a verbatim quote.

SOURCE TEXT:
{text}

QUANTITY TO FIND: {source_quantity} (reported in units compatible with {source_unit})

Output a SINGLE JSON object in a ```json fenced block with these keys (or {{"not_found": true}} if the source \
does not report this quantity):
- extracted_value: the number the source reports for this quantity, in the source's OWN units (a string)
- source_unit_token: the unit exactly as written in the source, INCLUDING any denominator (e.g. "emu/g", not "emu")
- referent: the source's own name/symbol for the quantity, as it appears in your quote
- source_conditions: the source's stated conditions for this value (e.g. "T = 0.4 K, B = 0"), as written
- verbatim_quote: a contiguous span COPIED EXACTLY from the source text containing the number, the unit, and the referent
Do not convert, do not paraphrase the quote, do not pick a value to match any target."""
```

Create `valagents/grounding_fetch.py`:

```python
"""Spec 3 grounding — locator -> source text (network; agent layer only, never the sandbox). v1 fetches the
ABSTRACT (arXiv summary / crossref abstract / URL text); full-text PDF is a v1.x lever (raw PDF text breaks
the verbatim-quote check). Isolated so tests inject a fake. Fail-soft: any error -> None."""
from __future__ import annotations
from valagents.references import detect_kind, normalize_id


async def fetch_source_text(locator: str) -> tuple[str, dict] | None:
    """The real (live) network fetch. Wrapped by LiveFetcher; tests never call this — they inject a fake
    fetcher with the same `async fetch(locator)` contract."""
    kind = detect_kind(locator)
    try:
        if kind == "arxiv":
            import asyncio, arxiv, re
            m = re.search(r"(\d{4}\.\d{4,5})", locator)
            if not m:
                return None
            res = await asyncio.to_thread(list, arxiv.Client().results(arxiv.Search(id_list=[m.group(1)])))
            if not res:
                return None
            r = res[0]
            return (f"{r.title}\n{r.summary}",
                    {"locator": normalize_id(locator), "title": r.title, "url": r.entry_id, "year": str(r.published)[:4]})
        if kind == "doi":
            import httpx, re
            doi = re.search(r"10\.\d{1,9}/[^\s]+", locator).group(0).rstrip(".,;)")
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"https://api.crossref.org/works/{doi}", timeout=15)
                resp.raise_for_status()
                msg = resp.json()["message"]
            text = (msg.get("title") or [""])[0] + "\n" + (msg.get("abstract") or "")
            return (text, {"locator": doi.lower(), "title": (msg.get("title") or [""])[0],
                           "url": f"https://doi.org/{doi}", "year": ""})
        if "http" in locator:
            import httpx, re
            async with httpx.AsyncClient() as client:
                resp = await client.get(locator, timeout=15)
                resp.raise_for_status()
                text = re.sub(r"<[^>]+>", " ", resp.text)
            return (text, {"locator": locator, "title": "", "url": locator, "year": ""})
    except Exception:
        return None
    return None
```

Create `valagents/agents/value_grounder.py`:

```python
"""Value-Grounder: READS a source and extracts a value + verbatim quote + conditions (F1 — reading only,
never judging). ground_plan orchestrates fetch -> extract -> code adjudication (ground_value)."""
from __future__ import annotations
import json
import re
from valagents.prompts import VALUE_GROUNDER
from valagents.agents.base import build_messages
from valagents.grounding import ground_value, GroundingResult
from valagents.grounding_fetch import fetch_source_text

# kind -> (value_attr, source_attr) for the single sourced value of each comparison_kind
_KIND_FIELDS = {
    "bound_check": ("bound", "bound_source"),
    "sensitivity_ratio": ("sensitivity", "sensitivity_source"),
    "discriminating_margin": ("closest_prior_effect", "closest_prior_source"),
}


class LiveFetcher:
    """The injected resolver the CLI builds when grounding is on. Wraps the real network fetch so the
    on/off is the *presence of this object*, not the backend string (tests inject a fake fetcher instead)."""
    async def fetch(self, locator: str):
        return await fetch_source_text(locator)


def _extract_json(text: str):
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL) or re.findall(r"(\{.*\})", text, re.DOTALL)
    for block in reversed(blocks):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    return None


async def extract_value(text: str, source_quantity: str, source_unit: str, llm) -> dict | None:
    user = VALUE_GROUNDER.format(text=text[:8000], source_quantity=source_quantity, source_unit=source_unit)
    body = await llm.complete("value_grounder", build_messages("You read sources and report values verbatim.", user))
    data = _extract_json(body)
    if not isinstance(data, dict) or data.get("not_found"):
        return None
    keys = ("extracted_value", "source_unit_token", "referent", "source_conditions", "verbatim_quote")
    return {k: str(data.get(k, "")) for k in keys}


async def ground_plan(plan, resolver, llm, cfg) -> GroundingResult | None:
    """Ground the plan's single sourced value. `resolver` is a fetcher with `async fetch(locator)` (the CLI's
    LiveFetcher, or a fake in tests). **`resolver is None` → grounding OFF → return None**, regardless of the
    backend string (the on/off is the injected dependency). When ON, a fetch/extraction failure yields a
    GroundingResult('unconfirmed'), never None."""
    if resolver is None:
        return None
    fields = _KIND_FIELDS.get(plan.comparison_kind)
    if fields is None:
        return None
    value = getattr(plan, fields[0], "")
    locator = getattr(plan, fields[1], "")
    if not value or not locator:
        return GroundingResult("unconfirmed", reason="missing_value_or_locator")
    fetched = await resolver.fetch(locator)
    if not fetched:
        return GroundingResult("unconfirmed", reason="unresolvable")
    text, _meta = fetched
    extraction = await extract_value(text, plan.source_quantity, plan.source_unit, llm)
    return ground_value(value, plan.source_unit, plan.source_quantity, plan.claim_conditions,
                        extraction, text, cfg)
```

- [ ] **Step 4: Run them (PASS) + full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_agent.py -v`
Expected: all PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add valagents/grounding_fetch.py valagents/agents/value_grounder.py valagents/prompts.py tests/test_grounding_agent.py
git commit -m "feat(grounding): fetch_source_text + value_grounder extraction agent + ground_plan orchestrator (abstract-level fetch v1)"
```

---

## Task 7: Data model + designer (three new `ComputationPlan` fields)

**Files:**
- Modify: `valagents/computation.py:25-29` (add fields), `valagents/agents/magnitude_designer.py`, `valagents/prompts.py` (`MAGNITUDE_DESIGNER`)
- Test: `tests/test_grounding_designer.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `ComputationPlan` fields `source_quantity: str = ""`, `claim_conditions: str = ""`, `source_unit: str = ""`. `design_magnitude` parses `SOURCE_QUANTITY`/`CLAIM_CONDITIONS`/`SOURCE_UNIT` (common to all kinds) and sets them on the plan; backward-compatible (absent → `""`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_grounding_designer.py`:

```python
import json
from valagents.computation import ComputationPlan
from valagents.agents.magnitude_designer import design_magnitude
from valagents.prompts import MAGNITUDE_DESIGNER

def test_plan_has_grounding_fields():
    p = ComputationPlan(kind="magnitude", comparison_kind="bound_check",
                        source_quantity="Yb moment", claim_conditions="T < 1 K", source_unit="µB")
    assert p.source_quantity == "Yb moment" and p.claim_conditions == "T < 1 K" and p.source_unit == "µB"

def test_prompt_teaches_grounding_fields():
    for tok in ("SOURCE_QUANTITY", "CLAIM_CONDITIONS", "SOURCE_UNIT", "resolvable"):
        assert tok in MAGNITUDE_DESIGNER
```

*(A full design_magnitude end-to-end test with a FakeLLM emitting the new tail keys is added here too — mirror the existing `tests/test_magnitude_model.py` FakeLLM pattern, asserting `p.source_quantity`/`claim_conditions`/`source_unit` are populated for a bound_check plan.)*

- [ ] **Step 2: Run it (FAIL)**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_designer.py -v`
Expected: FAIL — unexpected kwargs / tokens absent.

- [ ] **Step 3: Add the fields, the parse keys, and the prompt**

In `valagents/computation.py`, after the `closest_prior_source` line (line 25), add to the magnitude block:

```python
    source_quantity: str = ""        # grounding: what the sourced value measures (the referent target)
    claim_conditions: str = ""       # grounding: the claim's regime for that value (e.g. "T < 1 K")
    source_unit: str = ""            # grounding: the asserted unit (the conversion target, e.g. "µB")
```

In `valagents/agents/magnitude_designer.py`, add the three keys to `_COMMON` and set them on every plan via `common`:

```python
_COMMON = ["COMPARISON_KIND", "PREDICTED_EFFECT", "CONFIRM_IF", "REFUTE_IF",
           "SOURCE_QUANTITY", "CLAIM_CONDITIONS", "SOURCE_UNIT"]
```

and in `design_magnitude`'s `common = dict(...)` add:

```python
    common = dict(kind="magnitude", confirm_if=t["confirm_if"], refute_if=t["refute_if"],
                  target_claim_id=art.load_bearing, discriminating=bool(prediction.discriminates_from),
                  criterion="magnitude",
                  source_quantity=t.get("source_quantity", ""), claim_conditions=t.get("claim_conditions", ""),
                  source_unit=t.get("source_unit", ""))
```

In `valagents/prompts.py`, update `MAGNITUDE_DESIGNER`: change the SOURCE instruction to require a resolvable locator + the new fields, and add the three keys to each tail template, e.g. extend the bound_check line:

```python
COMPARISON_KIND: bound_check | PREDICTED_EFFECT: <n> | BOUND: <n> | BOUND_SOURCE: <arXiv id / DOI / URL> | SOURCE_QUANTITY: <what the bound measures> | CLAIM_CONDITIONS: <the claim's regime, e.g. T < 1 K> | SOURCE_UNIT: <unit of the bound, e.g. µB> | CONFIRM_IF: <...> | REFUTE_IF: <...>
```

(and the analogous additions to the `sensitivity_ratio` and `discriminating_margin` tail lines), plus change line 326 to: "never invent a threshold/sensitivity/bound without naming a **resolvable** SOURCE (arXiv id / DOI / URL), the QUANTITY it reports, the CONDITIONS, and the UNIT."

- [ ] **Step 4: Run it (PASS) + full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_designer.py -v`
Expected: PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass (existing magnitude tests: the new tail keys are parsed via `t.get(...)` so an old FakeLLM body without them still builds; if any existing designer test asserts an exact tail, update it to include the new keys).

- [ ] **Step 5: Commit**

```bash
git add valagents/computation.py valagents/agents/magnitude_designer.py valagents/prompts.py tests/test_grounding_designer.py
git commit -m "feat(grounding): ComputationPlan source_quantity/claim_conditions/source_unit + MAGNITUDE_DESIGNER (resolvable locator)"
```

---

## Task 8: Gate integration (say-so strip + per-path wiring + injected resolver)

**Files:**
- Modify: `valagents/computation.py:73-89` (`verdict_to_check` magnitude branch — the strip), `valagents/scheduler.py:254-298` (`run_magnitude_checks` + `run()` threading)
- Test: `tests/test_grounding_gate.py` (create); update `tests/test_magnitude_integration.py`

**Interfaces:**
- Consumes: `ground_plan` (Task 6), `GroundingResult`.
- Produces:
  - `verdict_to_check(v, tick=0, grounding=None)` — magnitude-`bound_check` branch sets `independent_sources` from `grounding` (`supports` → 1 + real `Source(locator=grounding-locator)`; else → 0), stripping the old pass→1 auto-credit; the **symbolic** else-branch is unchanged (`indep = 1 if pass`).
  - `run_magnitude_checks(store, llm, cfg, tick=0, resolver=None)` — grounds each plan via `ground_plan`; `bound_check` → `verdict_to_check(verdict, grounding=result)` + `contradicts`→skip; attack kinds → `contradicts`→suppress + `supports`/`inconclusive`→loud source in basis.
  - `run()` passes `resolver=` into `run_magnitude_checks` (the CLI builds it from `cfg.grounding`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grounding_gate.py` (mirror `tests/test_magnitude_integration.py`'s FakeLLM + store harness; the magnitude designer emits a bound_check plan, and an injected fake resolver+LLM grounds it):

```python
# Pattern (fill in the store/prediction harness exactly as test_magnitude_integration.py does):
# - A bound_check plan the idea PASSES (predicted <= bound).
# - resolver=None  -> verdict_to_check sets independent_sources == 0 (SAY-SO STRIP: no auto-credit).
# - resolver+LLM that grounds to SUPPORTS -> the BND claim's check has independent_sources == 1 + a real Source.
# - grounding CONTRADICTS -> the bound_check claim is SKIPPED (no BND claim injected).
# - a sensitivity_ratio plan whose sensitivity CONTRADICTS -> NO magnitude attack added.
# - a symbolic verdict_to_check pass still sets independent_sources == 1 (strip didn't leak).

from valagents.computation import ComputationPlan, ComputationVerdict, ComputationResult, verdict_to_check
from valagents.grounding import GroundingResult

def _bound_verdict(matched="confirm"):
    plan = ComputationPlan(kind="magnitude", comparison_kind="bound_check", bound="1e-3",
                           bound_source="arXiv:2104.01234")
    return ComputationVerdict(verdict="pass", measured="ok", plan=plan,
                              result=ComputationResult(ok=True, matched=matched))

def test_say_so_strip_no_grounding():
    rec = verdict_to_check(_bound_verdict(), grounding=None)        # grounding OFF
    assert rec.independent_sources == 0 and rec.sources == []       # the auto-credit is stripped

def test_grounding_supports_earns_credit():
    g = GroundingResult("supports", quote="…1e-3 µB…", converted_value=1e-3)
    rec = verdict_to_check(_bound_verdict(), grounding=g)
    assert rec.independent_sources == 1 and len(rec.sources) == 1

def test_symbolic_credit_unchanged():
    sym = ComputationVerdict(verdict="pass", measured="0", plan=ComputationPlan(kind="symbolic", expected="0"),
                             result=ComputationResult(ok=True, matched="confirm"))
    rec = verdict_to_check(sym)                                     # no grounding arg
    assert rec.independent_sources == 1                            # symbolic path untouched
```

*(Add the end-to-end `run_magnitude_checks` tests in `tests/test_grounding_gate.py` using the FakeLLM router pattern from `test_magnitude_integration.py`: a `resolver=None` run leaves the BND claim's check at `independent_sources==0`; a supports-grounding run reaches `independent_sources==1`; a contradicts-grounding run injects no BND claim; a contradicts sensitivity_ratio adds no attack.)*

- [ ] **Step 2: Run them (FAIL)**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_gate.py -v`
Expected: FAIL — `verdict_to_check` has no `grounding` param / current behavior sets indep=1.

- [ ] **Step 3: The say-so strip in `verdict_to_check`**

In `valagents/computation.py`, replace `verdict_to_check`:

```python
def verdict_to_check(v: "ComputationVerdict", tick: int = 0, grounding=None):
    """Map an executed ComputationVerdict to a CheckRecord(lens='executor'). No LLM (F3).
    Magnitude-bound_check: independent_sources comes ONLY from grounding-supports (the LLM-bound_source
    say-so auto-credit is STRIPPED, G-D6/G-D10). Symbolic: unchanged (indep = 1 on pass)."""
    from valagents.artifact import CheckRecord, Source
    if v.plan.kind == "magnitude" and v.plan.comparison_kind == "bound_check":
        supported = grounding is not None and getattr(grounding, "status", None) == "supports"
        indep = 1 if supported else 0
        gnote = ""
        if grounding is not None:
            gnote = f"; grounding={grounding.status}" + (f" (quote: {grounding.quote})" if grounding.quote else "")
        basis = (f"computed {v.measured or '?'}; bound = {v.plan.bound} "
                 f"(source: {v.plan.bound_source or 'n/a'}); matched = {v.result.matched}{gnote}")
        sources = [Source(locator=v.plan.bound_source, relation="independent")] if supported else []
        return CheckRecord(lens="executor", verdict=v.verdict, basis=basis,
                           independent_sources=indep, sources=sources, tick=tick)
    indep = 1 if v.verdict == "pass" else 0
    basis = (f"computed limit = {v.measured or '?'}; expected = {v.plan.expected} "
             f"(source: {v.plan.expected_source or 'n/a'}); matched = {v.result.matched}")
    src = v.plan.expected_source
    sources = ([Source(locator=src, relation="independent")] if src else [])
    return CheckRecord(lens="executor", verdict=v.verdict, basis=basis,
                       independent_sources=indep, sources=sources, tick=tick)
```

- [ ] **Step 4: Wire grounding into `run_magnitude_checks`**

In `valagents/scheduler.py`, change the signature to `async def run_magnitude_checks(store, llm, cfg, tick: int = 0, resolver=None)`, import `ground_plan`, and ground each plan before the verdict mapping:

```python
        from valagents.agents.value_grounder import ground_plan
        grounding = await ground_plan(plan, resolver, llm, cfg)   # None iff grounding OFF
        if grounding is not None and grounding.status == "contradicts":
            store.record({"event": "magnitude_grounding", "kind": plan.comparison_kind, "status": "contradicts"})
            continue                                               # suppress: input is literature-contradicted
```

(place this immediately after `verdict = run_plan(...)` and the existing `if verdict.verdict == "uncertain": continue`). Then in the `bound_check` branch, pass grounding to `verdict_to_check`:

```python
            store.add_check(claim_id, verdict_to_check(verdict, tick=tick, grounding=grounding))
```

and in the attack branch, fold a grounded source into the basis (append to the `Attack.basis` when `grounding` has a quote). Finally, in `run()` (line 242), build the resolver from `cfg.grounding.backend` and pass it (do NOT stash on the pydantic `cfg` — v2 rejects undeclared attrs):

```python
    resolver = None
    if cfg.grounding.backend != "none":
        from valagents.agents.value_grounder import LiveFetcher
        resolver = LiveFetcher()
    await run_magnitude_checks(store, llm, cfg, tick=tick + 500, resolver=resolver)
```

Tests call `run_magnitude_checks` directly with `resolver=<fake>` or `resolver=None`. Keep `run_simulation_checks`/symbolic untouched.

- [ ] **Step 5: Run them (PASS) + full suite + update the flipped integration test**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_grounding_gate.py -v`
Expected: all PASS.
Update `tests/test_magnitude_integration.py`: the bound_check test that relied on `bound_source` alone clearing `_has_independent_external_check` now asserts `independent_sources == 0` with `resolver=None` (the say-so strip is the correction). 
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass (the one flipped assertion updated; symbolic/simulation/gate-purity untouched).

- [ ] **Step 6: Commit**

```bash
git add valagents/computation.py valagents/scheduler.py tests/test_grounding_gate.py tests/test_magnitude_integration.py
git commit -m "feat(grounding): gate integration — strip bound_source say-so credit, per-path grounding wiring, injected resolver"
```

---

## Self-Review (against the spec, rev4)

**Spec coverage:**
- §2 honesty mechanism (read/judge) → Tasks 5 (`ground_value`) + 6 (extraction). ✅
- §3 resolvable locator + fetch → Task 6 `fetch_source_text` (abstract-level v1, full-text deferred — noted). ✅
- §4 extraction agent, anti-anchoring (no value, no claim_conditions shown) → Task 6 `VALUE_GROUNDER`. ✅
- §5 adjudicator (gates + four-outcome tree) → Tasks 2 (units), 3 (quote/quantity), 4 (conditions), 5 (tree). ✅
- §6 quote substantiality (numeral+unit+referent+min_tokens, full-compound-token) → Task 3 `_quote_valid`. ✅
- §7 scale-table whole-token + full-precision + G-D9 reference test → Task 2. ✅
- §8 gate integration (per-path; say-so strip; symbolic untouched) → Task 8. ✅
- §9 three plan fields + designer → Task 7. ✅
- §10 config + injected resolver + grounding-off=strip → Tasks 1, 6, 8. ✅
- §11 probes (wrong-conditions, G-D5c, sub-Kelvin, emu-denominator, no-false-contradict, interior+boundary, factor-verification, say-so strip, symbolic-intact) → distributed across Tasks 2–8. ✅
- §12 loud residual → carried in `basis` (Task 8); semantic-conditions/symbol-prose/Oe stay loud. ✅
- §13 decision log G-D1..G-D10 → each maps to a task constraint. ✅

**Gaps / flags:** (a) **Abstract-level fetch (Task 6)** — v1 fetches abstracts, not full-text PDF, so `supports` is rare in practice (the spec §3 realism note already concedes this; full-text is a v1.x lever). This is the one scope choice in the plan beyond the spec text — fail-closed (fewer supports, more inconclusive), not a soundness change. (b) The `§11` interior/boundary contradicts pins and the no-false-contradict-from-wrong-regime probe live in Task 5's tests; the wrong-conditions/G-D5c/sub-Kelvin probes in Task 4's; the emu-denominator in Tasks 2/5. The Task-5 and Task-4 reviewers carry the no-ground-truth load.

**Placeholder scan:** the only deferred-detail markers are the explicit "mirror `test_magnitude_integration.py`'s harness" notes in Tasks 7/8's end-to-end tests — the unit-level code and assertions are complete; the harness wiring is named, not invented, because it must match the existing FakeLLM/store pattern verbatim. No `TBD`/`add validation`/etc.

**Type consistency:** `GroundingResult` fields and `ground_value`/`ground_plan`/`verdict_to_check(grounding=)` signatures are consistent across Tasks 5/6/8; `convert`/`_quote_valid`/`_quantity_overlap`/`_conditions_compatible` signatures match their call sites in `ground_value`.

## Review Routing (for the controller)

- **Tasks 4 and 5 are the no-ground-truth logic** (`_conditions_compatible`, `ground_value`) — same class as the bounded slice's convergence predicate, where the real bugs lived. Point those reviewers at the **adversarial-construction** of a false `supports` (a quote/conditions/unit input that passes every gate and is still wrong), not just at code style.
- **Task 8** carries the cardinal-rule teeth (the say-so strip) — the reviewer must confirm the symbolic `verdict_to_check` path is byte-identical and only the magnitude-`bound_check` branch changed, and that `resolver=None` yields `independent_sources=0` (strip) not the old `1`.
