"""Prompt templates. Each ends with a mandatory machine-readable tail (parsed strictly)."""

FORMALIZER = """Restate the following idea as a precise, falsifiable claim. Identify the variables \
and what they range over; the scope and the regime of validity; the conditions under which it is \
asserted to hold. Do not add mechanism or evidence — only sharpen the statement.

IDEA: {raw_idea}

End your response with exactly:
CLAIM: <one sentence> | VARIABLES: <…> | REGIME: <…> | FALSIFIABLE: yes|no"""

FAITHFULNESS = """Here is a seed idea and a formal claim a colleague derived from it.
SEED: {raw_idea}
FORMAL CLAIM: {formal}
Back-translate the FORMAL CLAIM into plain language, then judge whether it is what the SEED asked —
not a narrowing, not an adjacent claim. End with exactly:
FAITHFUL: yes|narrowed|no | BACK_TRANSLATION: <plain-language restatement of the formal claim>"""

DECOMPOSER = """Decompose this claim into atomic, independently-checkable sub-claims with dependency \
edges. Tag each by type. Do not invent support — only expose structure.
CLAIM: {formal}
Output ONE line per sub-claim, exactly:
CLAIM: <id> | TYPE: definitional|mathematical|empirical|mechanistic | DEPENDS_ON: <ids|none> | STATEMENT: <…>"""

ENTAILMENT = """Does the conjunction of these sub-claims logically establish the formal claim, or is a \
load-bearing step missing?
FORMAL CLAIM: {formal}
SUB-CLAIMS:
{subclaims}
End with exactly:
COVERS: complete|gap | MISSING: <description|none>"""

GROUNDER_CLAIM = """Assess whether the literature supports this sub-claim, and identify INDEPENDENT \
sources (distinct authors/groups — not the same lab citing itself).
SUB-CLAIM ({ctype}): {statement}
RETRIEVED LITERATURE:
{articles}
End with exactly:
CLAIM: {cid} | SUPPORT: supported|unsupported|uncertain | INDEPENDENT_SOURCES: <n> | SOURCES: <[A1], [A2], …|none> | BASIS: <…>"""

GROUNDER_NOVELTY = """Position this claim against the closest prior work and name the delta — the \
specific thing it asserts that prior work does not.
CLAIM: {formal}
RETRIEVED LITERATURE:
{articles}
End with exactly:
CLOSEST_PRIOR: <…> | DELTA: <…> | POSITION: new|special_case|restatement"""

PROVER = """Build the chain from premises to this sub-claim. For a definitional claim, check it is \
coherent and non-circular; for mathematical/mechanistic, sketch and check the derivation/causal chain. \
Flag gaps rather than paper over them.
SUB-CLAIM ({ctype}): {statement}
End with exactly:
DERIVATION: complete|gapped | GAPS: <ids|none> | FATAL_GAP: yes|no"""
