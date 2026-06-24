from valagents import prompts


FORMAT_ARGS = {
    "raw_idea": "A seed idea about X causing Y in regime Z.",
    "formal": "For systems in regime Z, increasing X increases Y.",
    "subclaims": "CLAIM: c1 | TYPE: empirical | DEPENDS_ON: none | STATEMENT: X varies",
    "ctype": "mechanistic",
    "statement": "X causally changes Y through M.",
    "articles": "[A1] Example paper. [A2] Independent paper.",
    "cid": "c1",
    "delta": "Prior work measured X but did not claim X changes Y.",
    "artifact": "FORMAL: X changes Y\nCLAIM c1: X varies",
    "targets": "c1",
    "computed_status": "revise",
}


TEMPLATE_CONTRACTS = {
    "FORMALIZER": ["CLAIM:", "VARIABLES:", "REGIME:", "FALSIFIABLE:"],
    "FAITHFULNESS": ["FAITHFUL:", "BACK_TRANSLATION:"],
    "DECOMPOSER": ["CLAIM:", "TYPE:", "DEPENDS_ON:", "STATEMENT:"],
    "ENTAILMENT": ["COVERS:", "MISSING:"],
    "GROUNDER_CLAIM": [
        "CLAIM:",
        "SUPPORT:",
        "INDEPENDENT_SOURCES:",
        "SOURCES:",
        "BASIS:",
    ],
    "GROUNDER_NOVELTY": ["CLOSEST_PRIOR:", "DELTA:", "POSITION:"],
    "PROVER": ["DERIVATION:", "GAPS:", "FATAL_GAP:"],
    "PREDICTOR": ["OBSERVABLE:", "EFFECT_SIZE:", "DISCRIMINATES_FROM:", "MEASURABLE:"],
    "RED_TEAM": ["ATTACK:", "SEVERITY:", "STATUS:", "TARGET:", "BASIS:"],
    "VALIDATION_DESIGNER": ["TEST:", "CONFIRM_IF:", "REFUTE_IF:", "COST:"],
    "REPAIRER": ["REPAIR:", "TARGETS:", "RATIONALE:", "CLAIM:", "STATEMENT:"],
    "ARBITER": ["STATUS:", "LOAD_BEARING:", "DECISIVE_TEST:"],
}


def test_all_prompts_format_and_keep_strict_tail_labels():
    for name, labels in TEMPLATE_CONTRACTS.items():
        rendered = getattr(prompts, name).format(**FORMAT_ARGS)
        assert "{" not in rendered
        assert "}" not in rendered
        for label in labels:
            assert label in rendered, name


def test_red_team_rubric_defines_required_attack_settings():
    rendered = prompts.RED_TEAM.format(**FORMAT_ARGS)
    assert "fatal = a contradiction" in rendered
    assert "major = a material unresolved objection" in rendered
    assert "minor = a caveat" in rendered
    assert "magnitude must be attempted" in rendered.lower()
