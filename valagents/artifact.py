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
    title: str | None = None
    url: str | None = None
    year: str | None = None

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

class IdeaCompletion(BaseModel):
    status: Literal["incomplete", "completed_candidate", "polished_research_plan"] = "incomplete"
    completed_idea: str = ""
    mechanism: str = ""
    assumptions: list[str] = []
    weakest_link: str = ""

class TheoryBridge(BaseModel):
    theory_family: str = ""
    nearest_theories: list[str] = []
    extends: str = ""
    challenges: str = ""
    recovers_known_limits: str = ""
    departure_point: str = ""
    expert_translation: str = ""

class PriorArtPositioning(BaseModel):
    closest_prior: str = ""
    similarity: str = ""
    difference: str = ""
    what_is_new: str = ""
    must_cite: list[str] = []

class KnownLimit(BaseModel):
    limit: str
    recovered: Literal["yes", "no", "unclear"] = "unclear"
    failure_if_not: str = ""
    repair_needed: str = ""

class ConvincingCase(BaseModel):
    elevator_version: str = ""
    technical_version: str = ""
    why_existing_theory_leaves_room: str = ""
    why_plausible: str = ""
    skeptic_tests: list[str] = []

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
    role: Literal["background", "bridge", "novel_core", "assumption", "prediction", "test_condition"] = "novel_core"
    depends_on: list[str] = []
    load_bearing: bool = True
    checks: list[CheckRecord] = []
    exhausted: bool = False

    def _math_grounder_uncertainty_is_nonblocking(self, check: CheckRecord) -> bool:
        if self.type != "mathematical" or check.lens != "grounder":
            return False
        basis = (check.basis or "").strip().upper()
        return not basis.startswith(("CONTRADICTION:", "COUNTEREXAMPLE:", "REFUTES:"))

    @computed_field
    @property
    def status(self) -> str:
        if any(c.verdict == "fail" for c in self.checks):
            return "fail"
        uncertainties = [c for c in self.checks if c.verdict == "uncertain"]
        passes = [c for c in self.checks if c.verdict == "pass" and c.independent_sources >= 1]
        if uncertainties:
            has_proof_pass = any(c.lens == "prover" for c in passes)
            if (
                self.type == "mathematical"
                and has_proof_pass
                and all(self._math_grounder_uncertainty_is_nonblocking(c) for c in uncertainties)
            ):
                return "pass"
            return "uncertain"
        if passes:
            return "pass"
        return "pending"


DRAFT = "draft"
INTERNALLY_VALIDATED = "internally_validated"
NEEDS_EXPERIMENT = "needs_experiment"
REFUTED = "refuted"


class IdeaArtifact(BaseModel):
    raw_idea: str
    formal_claim: FormalClaim | None = None
    faithfulness: Faithfulness | None = None
    coverage: Coverage | None = None
    claim_graph: list[AtomicClaim] = []
    derivation: Derivation | None = None
    novelty: Novelty | None = None
    completion: IdeaCompletion | None = None
    theory_bridge: TheoryBridge | None = None
    prior_art_positioning: PriorArtPositioning | None = None
    known_limits: list[KnownLimit] = []
    convincing_case: ConvincingCase | None = None
    predictions: list[Prediction] = []
    attacks: list[Attack] = []
    attack_surface: AttackSurface | None = None
    validation_plan: ValidationPlan | None = None
    version_id: int = 0
    parent_version: int | None = None
    repairs_spent: int = 0
    repair_cap: int = 3
    min_attack_categories: int = 2
    fanout_N: int = 2
    finalized: bool = False

    # ---- helpers (pure; read recorded state only) ----
    def root_ancestors(self) -> list[AtomicClaim]:
        return [c for c in self.claim_graph if c.load_bearing]

    def _landed(self, severity: str) -> bool:
        return any(a.status == "landed" and a.severity == severity for a in self.attacks)

    def _any_landed(self) -> bool:
        return any(a.status == "landed" for a in self.attacks)

    def _thin_attack_surface(self) -> bool:
        s = self.attack_surface
        if s is None or "magnitude" not in s.attempted:
            return True
        return len(set(s.attempted)) < self.min_attack_categories

    def _has_independent_external_check(self, c: AtomicClaim) -> bool:
        return any(ck.verdict == "pass" and ck.independent_sources >= 1 for ck in c.checks)

    def _b(self, reason: str, claim_id: str | None = None) -> dict:
        return {"claim_id": claim_id, "reason": reason}

    def _evaluate(self) -> tuple[str, dict | None]:
        rs = self.root_ancestors()
        # ===== ENTRY GATES =====
        if self.formal_claim is None and self.finalized:
            return NEEDS_EXPERIMENT, self._b("unformalizable")
        if self.formal_claim and not self.formal_claim.falsifiable:
            return NEEDS_EXPERIMENT, self._b("not_falsifiable")
        if self.faithfulness and self.faithfulness.retried and self.faithfulness.verdict == "no":
            return NEEDS_EXPERIMENT, self._b("unfaithful_drift")
        if self.faithfulness and self.faithfulness.retried and self.faithfulness.verdict == "narrowed":
            return NEEDS_EXPERIMENT, self._b("unfaithful_narrowed")
        if (self.formal_claim and self.faithfulness and self.faithfulness.verdict == "yes"
                and not self.claim_graph and self.finalized):
            return NEEDS_EXPERIMENT, self._b("ill_formed")
        # ===== REFUTATION =====
        for c in rs:
            if c.status == "fail":
                return REFUTED, self._b("failed", c.id)
        if self._landed("fatal"):
            a = next(a for a in self.attacks if a.status == "landed" and a.severity == "fatal")
            return NEEDS_EXPERIMENT, self._b("severe_objection", a.target_claim_id)
        # ===== NEEDS EXPERIMENT =====
        for c in rs:
            if c.status == "uncertain":
                return NEEDS_EXPERIMENT, self._b("inconclusive", c.id)
        if self._landed("major") and self.finalized:
            a = next(a for a in self.attacks if a.status == "landed" and a.severity == "major")
            return NEEDS_EXPERIMENT, self._b("open_objection", a.target_claim_id)
        for c in rs:
            if c.status == "pending" and c.exhausted:
                return NEEDS_EXPERIMENT, self._b("uncovered", c.id)
        if self.coverage and self.coverage.verdict == "gap":
            return NEEDS_EXPERIMENT, self._b("decomposition_gap")
        if self._thin_attack_surface() and self.finalized:
            return NEEDS_EXPERIMENT, self._b("thin_attack_surface")
        # ===== VALIDATED: STRICT =====
        if (rs and all(c.status == "pass" for c in rs)
                and all(self._has_independent_external_check(c) for c in rs)
                and (self.faithfulness and self.faithfulness.verdict == "yes")
                and (self.coverage and self.coverage.verdict == "complete")
                and not self._thin_attack_surface()
                and not self._landed("fatal") and not self._landed("major")):
            return INTERNALLY_VALIDATED, None
        return DRAFT, None

    @computed_field
    @property
    def status(self) -> str:
        s, _ = self._evaluate()
        return s

    @computed_field
    @property
    def blocker(self) -> dict | None:
        _, b = self._evaluate()
        return b

    @computed_field
    @property
    def load_bearing(self) -> str | None:
        b = self._evaluate()[1]
        if b and b.get("claim_id"):
            return b["claim_id"]
        rs = self.root_ancestors()
        if not rs:
            return None
        deps = {c.id: 0 for c in self.claim_graph}
        adj = {c.id: c.depends_on for c in self.claim_graph}
        def reaches(start, target, seen=None):
            seen = seen or set()
            for d in adj.get(start, []):
                if d == target or (d not in seen and reaches(d, target, seen | {d})):
                    return True
            return False
        for c in self.claim_graph:
            for other in self.claim_graph:
                if other.id != c.id and reaches(other.id, c.id):
                    deps[c.id] += 1
        return max(rs, key=lambda c: (deps[c.id], c.id)).id

    @computed_field
    @property
    def maturity(self) -> float:
        # Display/ranking scalar (spec 2.3). Reads the verdict set directly --
        # claim_graph, attacks, attack_surface, predictions, validation_plan, coverage --
        # and NEVER the artifact's own gate verdict (self.status). One-directional by rule.
        rs = self.root_ancestors()
        if not rs:
            return 0.0
        verified = sum(1 for c in rs if c.status == "pass") / len(rs)
        b = 0
        if any(p.measurable and p.discriminates_from for p in self.predictions):
            b += 1
        if self.validation_plan is not None:
            b += 1
        if self.attack_surface is not None and not self._thin_attack_surface():
            b += 1
        if self.coverage is not None and self.coverage.verdict == "complete":
            b += 1
        n_minor = sum(1 for a in self.attacks
                      if a.status == "landed" and a.severity == "minor")
        penalty = 0.05 * min(n_minor, 4)   # per-attack, summed penalty saturates at 0.2
        return max(0.0, min(1.0, 0.7 * verified + 0.3 * (b / 4) - penalty))
