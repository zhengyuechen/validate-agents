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


import re
import unicodedata

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*[×x]\s*10\s*\^?\s*[-+]?\d+)?")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "per", "in", "at", "and", "or", "to", "for", "with", "is", "are",
         "we", "find", "this", "that", "by", "on", "from", "as", "its", "magnetic"}


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


# The conditions parser's OWN ladders (G-D5b/F3) — NEVER SCALE_TABLE (whose K is energy-via-k_B, T is Tesla).
# N1: µk key uses GREEK MU (U+03BC) — the post-NFKC form that _norm produces from both MICRO SIGN and GREEK MU.
_TEMP_UNITS = {"k": 1.0, "mk": 1e-3, "μk": 1e-6, "uk": 1e-6, "kelvin": 1.0, "millikelvin": 1e-3}
# koe: 1 kOe = 1000 Oe × 1e-4 T/Oe = 0.1 T; kg: 1 kG = 1000 G × 1e-4 T/G = 0.1 T; tesla/millitesla spelled-out.
_FIELD_UNITS = {"t": 1.0, "mt": 1e-3, "g": 1e-4, "gauss": 1e-4, "oe": 1e-4,
                "tesla": 1.0, "millitesla": 1e-3, "koe": 1e-1, "kg": 1e-1}
_AXIS_BY_SYMBOL = {"t": "temperature", "temp": "temperature", "temperature": "temperature",
                   "b": "field", "h": "field", "field": "field"}
_AXIS_BY_UNIT = {**{u: "temperature" for u in _TEMP_UNITS}, **{u: "field" for u in _FIELD_UNITS}}
_CLAUSE_RE = re.compile(
    r"(?:(?P<sym>[a-zµμ]+)\s*(?P<op><=|>=|=|<|>|~)\s*)?(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>[a-zµμ^\-0-9]+)")


def _to_canonical(value: float, unit: str, axis: str) -> float | None:
    table = _TEMP_UNITS if axis == "temperature" else _FIELD_UNITS
    f = table.get(unit)
    return None if f is None else value * f


def _parse_clauses(s: str) -> dict[str, list[tuple[str, float | None]]]:
    """Parse 'T < 1 K, B = 0' into {axis: [(op, canonical_value), ...]}.
    A bare point ('400 mK') -> op '='. Returns {} if nothing parsed.
    Unit disambiguates the energy-K vs temperature-K collision: a clause's unit is resolved in the
    conditions ladders, and the axis is taken from the symbol (if present) else the unit.
    If the axis is identified but the unit is not in the ladder, records (op, None) — the clause is
    NOT dropped, so downstream logic can fail closed (C1 fix). Multiple clauses on one axis are all
    collected — last-write-wins is removed (M1 fix)."""
    out: dict[str, list[tuple[str, float | None]]] = {}
    for m in _CLAUSE_RE.finditer(_norm(s)):
        unit = m.group("unit")
        sym = m.group("sym")
        axis = _AXIS_BY_SYMBOL.get(sym) if sym else None
        if axis is None:
            axis = _AXIS_BY_UNIT.get(unit)
        if axis is None:
            continue  # genuinely not a v1 clause (no axis identified at all)
        canon = _to_canonical(float(m.group("val")), unit, axis)
        # canon may be None if axis identified but unit not in ladder — still record it
        out.setdefault(axis, []).append((m.group("op") or "=", canon))
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
    AND must not pin a non-zero value on a v1 axis the claim leaves free (G-D5c). Err to False (not confirmed).

    Both claim and source are now lists of (op, value|None) per axis (M1 fix: all points checked).
    A None value anywhere on a claim axis → False (claim unparseable).
    A None source value on a claim-constrained axis → False (can't confirm).
    A None source value on an unconstrained axis → False (C1 fix: fail-closed on uncanonicalizable field)."""
    claim = _parse_clauses(claim_conditions)
    source = _parse_clauses(source_conditions)
    if not claim:
        return False  # empty or unparseable claim → not confirmed
    # Check each claim axis is satisfied by ALL source points on that axis
    for axis, claim_clauses in claim.items():
        # If any claim value is None, the claim itself is unparseable on this axis
        for op, cval in claim_clauses:
            if cval is None:
                return False
        if axis not in source:
            return False
        for _, sval in source[axis]:
            if sval is None:
                return False  # can't confirm: source axis identified but not canonicalizable
            for op, cval in claim_clauses:
                if not _satisfies(op, cval, sval):  # type: ignore[arg-type]
                    return False
    # G-D5c symmetry: source axes not in claim must all be zero (or baseline)
    for axis, source_points in source.items():
        if axis not in claim:
            for _, sval in source_points:
                if sval is None or abs(sval) > 0.0:
                    return False
    return True
