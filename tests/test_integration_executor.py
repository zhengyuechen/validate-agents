from valagents.scheduler import inject_limit_checks
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, KnownLimit
from valagents.config import Config
from tests.fake_llm import FakeLLM

def cfg():
    return Config(default_model="fake")

def store_with_limit(limit_text="recovers Newtonian gravity in the weak field"):
    art = IdeaArtifact(raw_idea="seed",
                       formal_claim=FormalClaim(statement="modified gravity law", falsifiable=True),
                       known_limits=[KnownLimit(limit=limit_text)])
    return ArtifactStore(art)

def router(designer_body, prover_body="DERIVATION: gapped | GAPS: none | FATAL_GAP: no"):
    def r(agent, messages):
        if agent == "computation_designer":
            return designer_body
        if agent == "prover":
            return prover_body
        return ""
    return r

async def test_recovered_limit_executor_pass_makes_claim_pass():
    s = store_with_limit()
    designer = ("EXPRESSION: G*M/r**2*(1+a/c**2) | VARIABLES: G,M,r,a,c | LIMIT_VARIABLE: c "
                "| LIMIT_POINT: oo | EXPECTED: G*M/r**2 | EXPECTED_SOURCE: Newtonian gravity "
                "| CONFIRM_IF: equals GM/r^2 | REFUTE_IF: differs")
    await inject_limit_checks(s, FakeLLM(router(designer)), cfg(), tick=0)
    L = next(c for c in s.current.claim_graph if c.origin == "limit_recovery")
    assert any(ck.lens == "executor" and ck.verdict == "pass" for ck in L.checks)
    assert L.status == "pass"

async def test_violated_limit_executor_fail_refutes():
    s = store_with_limit("must recover that 1/x vanishes at infinity")
    designer = ("EXPRESSION: 1/x | VARIABLES: x | LIMIT_VARIABLE: x | LIMIT_POINT: oo "
                "| EXPECTED: 1 | EXPECTED_SOURCE: (wrong target on purpose) "
                "| CONFIRM_IF: equals 1 | REFUTE_IF: differs")
    await inject_limit_checks(s, FakeLLM(router(designer)), cfg(), tick=0)
    L = next(c for c in s.current.claim_graph if c.origin == "limit_recovery")
    assert any(ck.lens == "executor" and ck.verdict == "fail" for ck in L.checks)
    assert L.status == "fail"
    assert s.current.status == "refuted"      # a load-bearing fail → refuted

async def test_no_plan_falls_back_to_prover():
    s = store_with_limit()
    # designer can't produce a tail → executor skipped; the prover (uncertain) verdict stands
    await inject_limit_checks(s, FakeLLM(router("no tail at all")), cfg(), tick=0)
    L = next(c for c in s.current.claim_graph if c.origin == "limit_recovery")
    assert not any(ck.lens == "executor" for ck in L.checks)   # no executor check added
    assert any(ck.lens == "prover" for ck in L.checks)          # prover fallback present
    assert L.status in ("uncertain", "pending")                # not falsely passed/failed
