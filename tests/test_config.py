from valagents.config import Config, load_config

def test_config_defaults():
    c = Config(default_model="m")
    assert c.model_for("anything") == "m"
    assert c.gate.fanout_N == 2 and c.gate.min_attack_categories == 2 and c.gate.repair_cap == 3
    assert c.grounding.backend == "arxiv"

def test_grounding_query_planner_defaults():
    c = Config(default_model="m")
    assert c.grounding.query_planner is True
    assert c.grounding.widen_min_results == 3

def test_config_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("default_model: x\ngate:\n  fanout_N: 5\n")
    c = load_config(str(p))
    assert c.gate.fanout_N == 5 and c.model_for("grounder") == "x"
