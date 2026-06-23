import inspect
from tests.test_artifact_gate import art, claim
from valagents.artifact import IdeaArtifact, CheckRecord

def test_maturity_is_a_float_in_unit_interval():
    m = art().maturity
    assert isinstance(m, float) and 0.0 <= m <= 1.0

def test_status_does_not_depend_on_maturity():
    # maturity ⊥ status: status source must not reference `maturity`
    src = inspect.getsource(IdeaArtifact._evaluate)
    assert "maturity" not in src

def test_passing_claims_mature_higher_than_pending():
    high = art()  # all pass
    low = art(claim_graph=[claim("c1", checks=[])])  # pending
    assert high.maturity > low.maturity

def test_minor_attacks_lower_maturity_by_gradient():
    # per-attack penalty (summed, saturating at 0.2) — NOT a flat clamp:
    # more landed minor attacks => strictly lower maturity, up to the saturation point.
    from valagents.artifact import Attack
    minor = lambda: Attack(type="confound", severity="minor", status="landed")
    none = art()
    one = art(attacks=[minor()])
    three = art(attacks=[minor() for _ in range(3)])
    assert none.maturity > one.maturity > three.maturity


def test_maturity_body_does_not_reference_artifact_status():
    # maturity must read the verdict set, never the collapsed gate verdict
    from pathlib import Path
    src = Path("valagents/artifact.py").read_text()
    # Extract the maturity method body (from def maturity to next @computed_field or next def)
    start = src.find("    def maturity(self) -> float:")
    assert start != -1, "maturity method not found"
    # Find the next method or property decorator after maturity
    rest = src[start:]
    next_def = rest.find("\n    def ", 4)
    next_decorator = rest.find("\n    @", 4)
    candidates = [x for x in [next_def, next_decorator] if x > 0]
    end = min(candidates) if candidates else len(rest)
    maturity_src = rest[:end]
    # Skip docstring/comment lines (those starting with # or containing only strings)
    lines = maturity_src.split('\n')
    code_lines = [l for l in lines[1:] if l.strip() and not l.strip().startswith('#')]
    code_body = '\n'.join(code_lines)
    assert "self.status" not in code_body
