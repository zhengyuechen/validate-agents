"""Prompt templates. Each ends with a mandatory machine-readable tail (parsed strictly)."""

COMMON_RUBRIC = """Shared validation rules:
- Preserve the user's claim; do not launder it into an easier adjacent claim.
- Prefer explicit uncertainty, gaps, or no-support verdicts over confident invention.
- Distinguish evidence in the retrieved material from speculation or plausible mechanism.
- Keep the machine-readable tail as the final response text, with the exact labels requested.
- Do not assign the artifact's final status unless the prompt explicitly asks for a STATUS cross-check."""

FORMALIZER = COMMON_RUBRIC + """

Role: formalize the seed idea into one precise, falsifiable claim.
Checklist:
- Keep the same scope, variables, and intended phenomenon as the seed.
- Name the variables and what they range over.
- State the regime of validity: system, assumptions, limits, scale, and boundary conditions.
- A falsifiable claim must imply an observable or computationally checkable refutation condition.
- Do not add mechanism, citations, predictions, or evidence here; sharpen only the statement.
- If the seed is too vague to falsify, preserve that limitation and mark falsifiable as no.

IDEA: {raw_idea}

End your response with exactly:
CLAIM: <one sentence> | VARIABLES: <...> | REGIME: <...> | FALSIFIABLE: yes|no"""

FAITHFULNESS = COMMON_RUBRIC + """

Role: decide whether the formal claim faithfully preserves the seed idea.
Compare:
- Scope: same object, population, phenomenon, and intended scale.
- Variables: no missing load-bearing variable and no new unrequested variable that changes the claim.
- Mechanism: no invented causal pathway or substitution of a different mechanism.
- Regime: no hidden narrowing to a special case unless the seed already required it.
- Strength: no stronger guarantee, weaker hedge, or swapped conclusion.

Use "yes" only when the formal claim is the same claim in sharper language.
Use "narrowed" when it is a proper subset or special case of the seed.
Use "no" when it drifts, contradicts, adds a different mechanism, or changes the target.

SEED: {raw_idea}
FORMAL CLAIM: {formal}

Back-translate the FORMAL CLAIM into plain language, then judge whether it is what the SEED asked.
End with exactly:
FAITHFUL: yes|narrowed|no | BACK_TRANSLATION: <plain-language restatement of the formal claim>"""

DECOMPOSER = COMMON_RUBRIC + """

Role: expose the formal claim's logical structure as atomic sub-claims.
Checklist:
- Each sub-claim should be independently checkable and small enough to attack or support.
- Include definitional claims when terms, metrics, or mappings are load-bearing.
- Include mathematical claims when an equation, bound, scaling law, or derivation is required.
- Include empirical claims when the artifact depends on observations or measured facts.
- Include mechanistic claims when causal, physical, algorithmic, or process explanations are required.
- Assign each claim a role: background, bridge, novel_core, assumption, prediction, or test_condition.
- Dependencies should point only to earlier or otherwise necessary sub-claims.
- Do not invent literature support, validation status, or repairs.

CLAIM: {formal}

Output ONE line per sub-claim, exactly:
CLAIM: <id> | TYPE: definitional|mathematical|empirical|mechanistic | ROLE: background|bridge|novel_core|assumption|prediction|test_condition | DEPENDS_ON: <ids|none> | STATEMENT: <...>"""

ENTAILMENT = COMMON_RUBRIC + """

Role: check whether the sub-claims jointly imply the formal claim.
Checklist:
- Treat the sub-claims as the only premises; do not silently import missing facts.
- Verify the conclusion's scope, strength, variables, and regime match the formal claim.
- A missing definition, bridge law, empirical premise, or quantifier condition is a gap.
- Mark complete only when the conjunction would make the formal claim follow.

FORMAL CLAIM: {formal}
SUB-CLAIMS:
{subclaims}

End with exactly:
COVERS: complete|gap | MISSING: <description|none>"""

GROUNDER_CLAIM = COMMON_RUBRIC + """

Role: assess whether retrieved literature supports this specific sub-claim.
Checklist:
- Use only labels present in RETRIEVED LITERATURE, such as [A1] or [A2].
- Do not cite memory, outside knowledge, or unlabeled sources.
- "supported" requires direct support for the sub-claim, not merely topic similarity.
- "unsupported" applies only when retrieved work explicitly contradicts the sub-claim.
- "uncertain" applies when retrieved work is incomplete, indirect, ambiguous, absent, or merely lacks support.
- If retrieved work contradicts the sub-claim, write the contradiction explicitly in BASIS beginning with "CONTRADICTION:".
- Do not decide that a novel claim is false merely because existing literature has not yet supported it.
- Count independent sources as distinct author groups or experimental/theoretical lineages, not repeated papers from the same group.
- The BASIS should name the decisive support, contradiction, or missing evidence.

SUB-CLAIM ({ctype}): {statement}
RETRIEVED LITERATURE:
{articles}

End with exactly:
CLAIM: {cid} | SUPPORT: supported|unsupported|uncertain | INDEPENDENT_SOURCES: <n> | SOURCES: <[A1], [A2], ...|none> | BASIS: <...>"""

GROUNDER_NOVELTY = COMMON_RUBRIC + """

Role: position the formal claim against the closest retrieved prior work.
Checklist:
- Identify the closest prior even if it weakens the novelty case.
- The delta must be the exact assertion the prior work does not already make.
- Use "restatement" if the claim is already present in prior work under different words.
- Use "special_case" if it narrows, instantiates, or recombines known work without a new load-bearing assertion.
- Use "new" only when the retrieved prior does not contain the claim or its equivalent.
- If literature is absent or too weak, say so in CLOSEST_PRIOR/BASIS-style wording, but keep the exact final schema.

CLAIM: {formal}
RETRIEVED LITERATURE:
{articles}

End with exactly:
CLOSEST_PRIOR: <...> | DELTA: <...> | POSITION: new|special_case|restatement"""

PROVER = COMMON_RUBRIC + """

Role: test whether the sub-claim has a coherent derivation or internal argument.
Checklist by claim type:
- Definitional: check clarity, non-circularity, and whether the definition can bear later use.
- Mathematical: identify premises, transformations, missing lemmas, and boundary conditions.
- Mechanistic: trace the causal chain and note any unsupported transition or scale mismatch.
- Empirical: identify whether the empirical assertion has enough specified measurement context.

Use "complete" only when the derivation/argument reaches the sub-claim without hand-waving.
Use "gapped" when a premise, lemma, measurement condition, or causal bridge is missing.
Set FATAL_GAP to yes when the missing step is load-bearing enough that the sub-claim cannot currently stand, but a fatal gap is not by itself a falsification.
If the sub-claim is actually contradicted, begin GAPS with "CONTRADICTION:" or "COUNTEREXAMPLE:" and describe the contradiction.

SUB-CLAIM ({ctype}): {statement}

End with exactly:
DERIVATION: complete|gapped | GAPS: <ids|none> | FATAL_GAP: yes|no"""

PREDICTOR = COMMON_RUBRIC + """

Role: extract concrete falsifiable consequences of the claim.
Checklist:
- Each prediction should be observable, measurable, and discriminating.
- Name the null model, baseline, or closest existing model it distinguishes from.
- Give an effect size as a direction, threshold, scaling relation, or order-of-magnitude expectation.
- Mark MEASURABLE as no if the observable is vague, lacks an operational measurement, or cannot be distinguished from the comparator.
- Mark DETECTABLE as no if the effect is real in principle but currently unmeasurable (numerically inert guard).
- Prefer fewer high-value predictions over many generic consequences.

CLAIM: {formal}
DELTA vs prior work: {delta}

Output ONE line per prediction, exactly:
OBSERVABLE: <...> | EFFECT_SIZE: <...> | DISCRIMINATES_FROM: <...> | MEASURABLE: yes|no | DETECTABLE: yes|no|unclear"""

RED_TEAM = COMMON_RUBRIC + """

Role: be an adversarial reviewer trying to break the artifact.
Attempt all applicable attack categories:
- counterexample: construct a case satisfying the assumptions where the claim fails.
- failure_regime: find boundaries, limits, scales, or conditions where the claim should stop holding.
- confound: identify a simpler explanation or uncontrolled variable that could explain the same observation.
- magnitude: strip the framing and estimate whether the mechanism changes any measurable quantity at the relevant scale, including orders of magnitude.

Severity definitions:
- fatal = a contradiction, counterexample, or scale check appears to collapse a load-bearing claim and requires repair or decisive validation.
- major = a material unresolved objection requiring experiment, computation, or repair, but not a refutation.
- minor = a caveat or weakness that lowers confidence but does not block the claim.

Status rules:
- Use "landed" when the attack exposes a real unresolved problem.
- Use "survived" only when the artifact directly answers the attack in the supplied text.
- Do not soften a landed attack because it seems fixable; the repairer handles fixes.
- For an actual refutation, begin BASIS with "CONTRADICTION:", "COUNTEREXAMPLE:", or "REFUTES:" and state exactly what breaks.
- Magnitude must be attempted when any physical, statistical, computational, or measurable scale is relevant.

ARTIFACT:
{artifact}

First line, exactly: ATTEMPTED: <subset of counterexample, failure_regime, confound, magnitude>
Then ONE line per attack, exactly:
ATTACK: <type> | SEVERITY: fatal|major|minor | STATUS: survived|landed | TARGET: <claim_id|none> | BASIS: <...>"""

VALIDATION_DESIGNER = COMMON_RUBRIC + """

Role: propose the cheapest decisive validation.
Checklist:
- Prefer a computation, simulation, reanalysis, or sanity-check calculation over a new experiment when sufficient.
- The test should target the load-bearing uncertainty, not merely produce supportive context.
- CONFIRM_IF and REFUTE_IF must be concrete decision criteria.
- Cost should reflect equipment/time/access burden, not the importance of the claim.
- Avoid vague "collect more data" plans; specify what is measured or computed.
- DISCRIMINATES_FROM must name the closest prior/alternative the test distinguishes from (reference PRIOR-ART POSITION.CLOSEST if available).
- INFERENTIAL_STANDARD must state the decision threshold for empirical claims (e.g. power/sample size, pre-registration requirement, required controls).

ARTIFACT:
{artifact}

End with exactly:
TEST: <...> | CONFIRM_IF: <...> | REFUTE_IF: <...> | DISCRIMINATES_FROM: <...> | INFERENTIAL_STANDARD: <...> | COST: low|medium|high"""

IDEA_COMPLETER = COMMON_RUBRIC + """

Role: complete the seed into the strongest coherent candidate idea before judging validation.
Checklist:
- Preserve the original ambition; do not shrink the idea merely because it is new.
- Fill missing mechanism steps, assumptions, and boundary conditions explicitly.
- Make the idea understandable as a candidate theory/research program.
- If multiple completions are possible, choose the most coherent and testable version.
- Name the weakest link that most needs proof, simulation, or experiment.
- After the main tail, emit one ASSUMPTION line per identified assumption.
- Each assumption has a status: standard (well-established background), contested (disputed in the field), or novel_load_bearing (new and required for the main claim).

ARTIFACT:
{artifact}

End with exactly:
COMPLETION_STATUS: incomplete|completed_candidate|polished_research_plan | COMPLETED_IDEA: <...> | MECHANISM: <...> | WEAKEST_LINK: <claim_id or description>

Then one line per assumption, exactly:
ASSUMPTION: <text> | STATUS: standard|contested|novel_load_bearing"""

THEORY_BRIDGE = COMMON_RUBRIC + """

Role: connect the completed idea to existing theory so experts can understand and evaluate it.
Checklist:
- Name the theory family or formalism that makes the idea legible.
- Identify nearest existing theories, analogies, or limiting cases.
- State what the idea extends and what it challenges.
- State what known limits it must recover.
- Translate the idea into language an expert in the field already trusts.
- Do not claim support merely from analogy; mark bridges as conceptual scaffolding.

ARTIFACT:
{artifact}

End with exactly:
THEORY_FAMILY: <...> | NEAREST_THEORIES: <comma-separated theories|none> | EXTENDS: <...> | CHALLENGES: <...> | RECOVERS_KNOWN_LIMITS: <...> | DEPARTURE_POINT: <...> | EXPERT_TRANSLATION: <...>"""

PRIOR_ART_POSITIONING = COMMON_RUBRIC + """

Role: position the completed idea against prior art without reducing it to pass/fail support.
Checklist:
- Name the closest prior theory, paper, model, or research program from the artifact/retrieval context.
- Explain similarity and difference separately.
- State exactly what is new if the completion is right.
- List works/theories that must be cited or discussed.

ARTIFACT:
{artifact}

End with exactly:
CLOSEST_PRIOR: <...> | SIMILARITY: <...> | DIFFERENCE: <...> | WHAT_IS_NEW: <...> | MUST_CITE: <comma-separated items|none>"""

KNOWN_LIMITS = COMMON_RUBRIC + """

Role: check whether the completed idea recovers or respects known limiting cases and constraints.
Checklist:
- Include standard limits, conservation/symmetry/causality/no-signaling constraints, dimensional checks, or benchmark regimes when relevant.
- Mark RECOVERED as yes only when the artifact explicitly recovers the limit.
- Mark RECOVERED as no only when the artifact appears incompatible.
- Mark RECOVERED as unclear when the completion still needs derivation or evidence.
- State what fails if the limit is not recovered and what repair would be needed.

ARTIFACT:
{artifact}

Output ONE line per known limit, exactly:
LIMIT: <...> | RECOVERED: yes|no|unclear | FAILURE_IF_NOT: <...> | REPAIR_NEEDED: <...>"""

CONVINCING_CASE = COMMON_RUBRIC + """

Role: build the scientific case for the completed idea without hype.
Checklist:
- Give a short version and a technical version.
- Explain why existing theory leaves room for the proposal.
- Explain why the completed mechanism is plausible but still unvalidated.
- Name what would convince a skeptical expert.
- Do not call the idea true; this is an argument map plus validation agenda.

ARTIFACT:
{artifact}

End with exactly:
ELEVATOR_VERSION: <...> | TECHNICAL_VERSION: <...> | WHY_EXISTING_THEORY_LEAVES_ROOM: <...> | WHY_PLAUSIBLE: <...> | SKEPTIC_TESTS: <comma-separated tests|none>"""

STEELMAN_OBJECTION = COMMON_RUBRIC + """

Role: you are the idea's most capable critic. Build the STRONGEST honest case that it is wrong — steelman the objection, do not strawman it; if the idea is actually strong, say where it nonetheless remains vulnerable.
Checklist:
- Name the single most damaging objection and develop it fully; do not list many weak ones.
- Trace the concrete mechanism by which the idea most plausibly fails.
- Identify the established theory, result, or bound that most directly threatens it.
- State the decisive disconfirming observation or derivation that, if found, would kill it.
- Close with a one-line honest summary of what a careful skeptic would conclude.
- Do not strawman; engage with the strongest version of the idea.
- Do not call the idea false unless a demonstrated refutation exists; name vulnerabilities, not verdicts.

ARTIFACT:
{artifact}

End with exactly:
STRONGEST_OBJECTION: <...> | MECHANISM_OF_FAILURE: <...> | THREATENING_RESULT: <...> | WHAT_WOULD_KILL_IT: <...> | FAIR_SUMMARY: <...>"""

REPAIRER = COMMON_RUBRIC + """

Role: repair only the named target sub-claims after a landed attack or derivation gap.
Checklist:
- Change the target statements directly; do not rewrite unrelated claims.
- Preserve the original ambition when possible; do not merely weaken the claim to dodge the attack.
- Add the missing condition, mechanism, variable, or derivation bridge that addresses the stated failure.
- If a target cannot be repaired without changing the thesis, say that in the rationale and still emit the best revised target statement.
- The repaired sub-claim lines must use the target ids so downstream state can merge them.

ARTIFACT:
{artifact}
TARGETS: {targets}

End with the summary line, exactly:
REPAIR: <what changed> | TARGETS: <claim_ids> | RATIONALE: <...>
Then ONE line per repaired sub-claim, exactly:
CLAIM: <id> | STATEMENT: <revised statement text>"""

COMPUTATION_DESIGNER = """You DESIGN a symbolic check; you do NOT run or judge it — code does that, and \
you will never see the result. Given a known-limit-recovery claim, produce the structured plan to test \
whether the idea's expression reduces to the established result in the stated regime.

FORMAL CLAIM: {formal}
LIMIT-RECOVERY CLAIM: {statement}

Give the idea's expression and the variable taken to a limit; give the expected established result and \
where it comes from. Use plain SymPy-parseable math (e.g. G*M/r**2, c, oo). Do not output code.
End with exactly:
EXPRESSION: <expr> | VARIABLES: <comma-separated symbols> | LIMIT_VARIABLE: <symbol> | LIMIT_POINT: <oo|0|value> | EXPECTED: <expr> | EXPECTED_SOURCE: <where the known result comes from> | CONFIRM_IF: <…> | REFUTE_IF: <…>"""

ARBITER = COMMON_RUBRIC + """

Role: cross-check the computed outcome and identify what matters most.
Checklist:
- The computed status wins; do not re-argue evidence, attacks, novelty, or repairs.
- Choose the load-bearing claim whose failure would most directly collapse the artifact.
- The decisive test should be the single validation most likely to resolve that load-bearing uncertainty.
- If the computed status and artifact text appear inconsistent, reflect that in the STATUS cross-check without inventing a new computed result.

ARTIFACT:
{artifact}
COMPUTED STATUS: {computed_status}

End with exactly:
STATUS: <...> | LOAD_BEARING: <claim_id> | DECISIVE_TEST: <...>"""
