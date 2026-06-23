"""Prompt templates. Each ends with a mandatory machine-readable tail (parsed strictly)."""

FORMALIZER = """Restate the following idea as a precise, falsifiable claim. Identify the variables \
and what they range over; the scope and the regime of validity; the conditions under which it is \
asserted to hold. Do not add mechanism or evidence — only sharpen the statement.

IDEA: {raw_idea}

End your response with exactly:
CLAIM: <one sentence> | VARIABLES: <…> | REGIME: <…> | FALSIFIABLE: yes|no"""
