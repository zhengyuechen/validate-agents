# validate-agents

Placeholder for a validation-focused sibling of `../co-scientist-reproduce/`.

**Goal (as stated):** run the agent on a different goal whose purpose is to *validate* — to be
scoped (see open question below).

**Plan:** most of the existing multi-agent code is expected to be reusable, so this project will
migrate / adapt files from `../co-scientist-reproduce/` (e.g. the engine, supervisor, reflection
and ranking agents, the gate, the config and LLM layers) rather than start from scratch. Nothing has
been moved yet.

**Open question (blocks the migration):** what does "validate" mean here?
- (A) A test/eval harness that validates the Co-Scientist *system itself* (does the pipeline behave?).
- (B) Repurpose the agents so the research *task* is validation — scrutinise a given hypothesis/claim
  (reuse Reflection's full / deep-verification / simulation reviews + the novelty gate, drop or
  demote Generation).
- (C) Something else.
