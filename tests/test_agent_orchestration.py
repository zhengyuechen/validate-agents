from valagents.agents.arbiter import arbitrate
from valagents.agents.repairer import repair
from valagents.artifact import (IdeaArtifact, FormalClaim, Faithfulness, Coverage,
                                AttackSurface, AtomicClaim, CheckRecord)
from tests.fake_llm import FakeLLM


def validated_art():
    PASS = CheckRecord(lens="grounder", verdict="pass", independent_sources=1)
    return IdeaArtifact(raw_idea="s", formal_claim=FormalClaim(statement="x", falsifiable=True),
                        faithfulness=Faithfulness(verdict="yes"), coverage=Coverage(verdict="complete"),
                        attack_surface=AttackSurface(attempted=["magnitude", "confound"]),
                        claim_graph=[AtomicClaim(id="c1", statement="s", type="empirical",
                                                 checks=[PASS], exhausted=True)], finalized=True)


async def test_arbiter_agrees_with_computed(cfg):
    body = "STATUS: internally_validated | LOAD_BEARING: c1 | DECISIVE_TEST: none needed"
    out = await arbitrate(validated_art(), FakeLLM(lambda a, m: body), cfg)
    assert out["agrees"] is True
    assert out["status"] == "internally_validated"


async def test_arbiter_disagreement_flagged_computed_wins(cfg):
    # Arbiter narrates validated, but a fatal attack means computed == refuted
    from valagents.artifact import Attack
    art = validated_art()
    art.attacks = [Attack(type="counterexample", severity="fatal", status="landed", target_claim_id="c1")]
    body = "STATUS: internally_validated | LOAD_BEARING: c1 | DECISIVE_TEST: x"
    out = await arbitrate(art, FakeLLM(lambda a, m: body), cfg)
    assert art.status == "refuted" and out["agrees"] is False   # computed wins; mismatch surfaced
    assert out["status"] == "refuted"


async def test_repair_returns_new_statements(cfg):
    body = ("REPAIR: tightened scope | TARGETS: c1 | RATIONALE: fix mechanism\n"
            "CLAIM: c1 | STATEMENT: revised text")
    out = await repair(validated_art(), ["c1"], FakeLLM(lambda a, m: body), cfg)
    assert out is not None
    assert out["new_statements"]["c1"] == "revised text"
