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
    "DECOMPOSER": ["CLAIM:", "TYPE:", "ROLE:", "DEPENDS_ON:", "STATEMENT:"],
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
    "IDEA_COMPLETER": [
        "COMPLETION_STATUS:",
        "COMPLETED_IDEA:",
        "MECHANISM:",
        "ASSUMPTIONS:",
        "WEAKEST_LINK:",
    ],
    "THEORY_BRIDGE": [
        "THEORY_FAMILY:",
        "NEAREST_THEORIES:",
        "EXTENDS:",
        "CHALLENGES:",
        "RECOVERS_KNOWN_LIMITS:",
        "DEPARTURE_POINT:",
        "EXPERT_TRANSLATION:",
    ],
    "PRIOR_ART_POSITIONING": [
        "CLOSEST_PRIOR:",
        "SIMILARITY:",
        "DIFFERENCE:",
        "WHAT_IS_NEW:",
        "MUST_CITE:",
    ],
    "KNOWN_LIMITS": ["LIMIT:", "RECOVERED:", "FAILURE_IF_NOT:", "REPAIR_NEEDED:"],
    "CONVINCING_CASE": [
        "ELEVATOR_VERSION:",
        "TECHNICAL_VERSION:",
        "WHY_EXISTING_THEORY_LEAVES_ROOM:",
        "WHY_PLAUSIBLE:",
        "SKEPTIC_TESTS:",
    ],
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
