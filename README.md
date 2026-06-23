# validate-agents

A depth-first idea-validation agent system — it grows one seed idea into a fully-specified, check-hardened artifact that terminates in exactly one of `internally_validated` / `needs_experiment` / `refuted`.

## Why

Two invariants are non-negotiable:

1. **Every check is a parsed verdict that gates, never just narrates.** The artifact status (`internally_validated`, `needs_experiment`, `refuted`) is a computed property with no setter. No agent ever writes it directly. Verdicts come from strict machine-readable tails; a missing or unparseable tail degrades to `uncertain`, never silently passes.

2. **"Validated" means the idea survived an external, independent check — never the model's own say-so.** `internally_validated` is structurally unreachable unless every load-bearing claim has a `CheckRecord` with `verdict == "pass"` and at least one independent source. `pending` is never `pass`.

The gate is **total**: every run ends in exactly one of the three honest verdicts. There is no fourth state a reader can round up to "validated." `draft` is non-terminal; the scheduler never stops there.

## The pipeline

Agents run over a claim DAG derived from the seed idea:

- **Formalizer** — pins the seed as a falsifiable, scoped formal claim (entry gate 1: not falsifiable → `refuted`)
- **Faithfulness** — independently back-translates the formal claim and checks it matches the seed (entry gate 2: drift/narrowing → one retry → `refuted`)
- **Decomposer** — decomposes the formal claim into typed atomic sub-claims with dependency edges
- **Entailment** — independently verifies the sub-claims jointly establish the formal claim (`gap` → caps below `internally_validated`)
- **Grounder** — grounds each claim in external literature; enforces the independence rule (≥1 independent source for `pass`)
- **Prover** — checks derivations and well-formedness; mandatory for mathematical/mechanistic/definitional claims
- **Predictor** — extracts falsifiable, discriminating predictions
- **Red-team** — adversarially attacks the artifact across four categories (counterexample, failure regime, confound, magnitude); magnitude is mandatory
- **Validation-designer** — proposes the single cheapest decisive test
- **Repairer** — on a landed attack or fatal gap, forks a new artifact version and re-runs only the affected subgraph (capped at 3 repairs)
- **Arbiter** — narrates the final verdict; the computed status always wins on disagreement

## Status

Design complete (`docs/`). Spec 1 (internal-validation spine) implementation in progress: 9 of 17 tasks complete; ~47 tests passing.

## Layout

```
valagents/
  artifact.py     # IdeaArtifact/AtomicClaim schema + total gate (status/blocker/load_bearing/maturity)
  parse.py        # strict machine-readable tail parsing
  store.py        # single-writer ArtifactStore, immutable version chain
  agents/         # one file per agent (formalizer, faithfulness, decomposer, entailment,
                  #   grounder, prover, predictor, redteam, validation_designer, repairer, arbiter)
  scheduler.py    # entry gates → per-claim lenses → fan-out → repair loop → finalize
  cli.py          # valagents "<seed>" → IdeaArtifact JSON + markdown report
  llm.py          # async LLM client (OpenRouter)
  run_log.py      # append-only JSONL event log
  web_search.py   # ArxivBackend, safe_search
  config.py       # typed config (roles→models, gate thresholds)
tests/            # deterministic FakeLLM; no network in any test
docs/             # design spec + implementation plan
```

## Develop

```bash
pip install -r requirements.txt
python -m pytest
```
