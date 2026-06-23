"""IdeaArtifact schema + the computed gate. status/maturity/load_bearing/blocker are
computed properties with NO setter — the gate is code, never an LLM (I1)."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, computed_field

class Source(BaseModel):
    locator: str
    author: str | None = None
    group: str | None = None
    relation: Literal["independent", "same_author", "same_group", "self_citation", "unknown"] = "unknown"

class CheckRecord(BaseModel):
    lens: Literal["grounder", "prover", "redteam"]
    verdict: Literal["pass", "fail", "uncertain"]
    basis: str = ""
    sources: list[Source] = []
    independent_sources: int = 0
    tick: int = 0

class FormalClaim(BaseModel):
    statement: str
    variables: list[str] = []
    scope: str = ""
    regime: str = ""
    falsifiable: bool

class Faithfulness(BaseModel):
    verdict: Literal["yes", "narrowed", "no"]
    back_translation: str = ""
    retried: bool = False

class Coverage(BaseModel):
    verdict: Literal["complete", "gap"]
    missing: str | None = None

class AttackSurface(BaseModel):
    attempted: list[str] = []
    skipped: list[str] = []

class Novelty(BaseModel):
    closest_prior: list[str] = []
    delta: str = ""
    position: Literal["new", "special_case", "restatement"] = "new"

class Prediction(BaseModel):
    observable: str
    effect_size: str = ""
    discriminates_from: str = ""
    measurable: bool = False

class Attack(BaseModel):
    type: str
    severity: Literal["fatal", "major", "minor"]
    status: Literal["survived", "landed"]
    target_claim_id: str | None = None
    basis: str = ""

class Gap(BaseModel):
    description: str
    claim_id: str
    fatal: bool = False

class Derivation(BaseModel):
    steps: list[str] = []
    gaps: list[Gap] = []

class ValidationPlan(BaseModel):
    decisive_test: str
    controls: list[str] = []
    confirm_if: str = ""
    refute_if: str = ""
    cost: Literal["low", "medium", "high"] = "medium"

class AtomicClaim(BaseModel):
    id: str
    statement: str
    type: Literal["definitional", "mathematical", "empirical", "mechanistic"]
    depends_on: list[str] = []
    load_bearing: bool = True
    checks: list[CheckRecord] = []
    exhausted: bool = False

    @computed_field
    @property
    def status(self) -> str:
        if any(c.verdict == "fail" for c in self.checks):
            return "fail"
        if any(c.verdict == "uncertain" for c in self.checks):
            return "uncertain"
        if any(c.verdict == "pass" and c.independent_sources >= 1 for c in self.checks):
            return "pass"
        return "pending"
