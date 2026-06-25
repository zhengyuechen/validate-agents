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
