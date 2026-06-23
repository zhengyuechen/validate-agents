from valagents.artifact import IdeaArtifact, AtomicClaim, CheckRecord
from valagents.store import ArtifactStore

def base():
    g = [AtomicClaim(id="c1", statement="s", type="empirical"),
         AtomicClaim(id="c2", statement="s", type="empirical")]
    return IdeaArtifact(raw_idea="seed", claim_graph=g)

def test_add_check_and_record_appends():
    s = ArtifactStore(base())
    s.add_check("c1", CheckRecord(lens="grounder", verdict="pass", independent_sources=1))
    s.record({"event": "check", "claim": "c1"})
    assert s.current.claim_graph[0].checks[0].verdict == "pass"
    assert s.events[-1]["claim"] == "c1"

def test_fork_freezes_prior_version():
    s = ArtifactStore(base())
    s.add_check("c1", CheckRecord(lens="grounder", verdict="pass", independent_sources=1))
    s.add_check("c2", CheckRecord(lens="grounder", verdict="pass", independent_sources=1))
    v1 = s.current
    s.fork_for_repair(["c2"])                # repair only c2's subgraph
    v2 = s.current
    assert v2.version_id == 1 and v2.parent_version == 0 and v2.repairs_spent == 1
    assert len(v1.claim_graph[1].checks) == 1     # v1 frozen, untouched
    assert v2.claim_graph[1].checks == []         # c2 cleared in v2
    assert len(v2.claim_graph[0].checks) == 1     # c1 carried forward
