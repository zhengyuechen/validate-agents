# validate-agents Spec 2 — Symbolic Known-Limit Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn R3's known-limit-recovery claims from *reasoned* (Prover) into *executed* (SymPy): an LLM designs a structured `ComputationPlan`, code runs it in a sandbox and judges symbolic equality, and the result becomes a `CheckRecord(lens="executor")` the gate treats like any other check.

**Architecture:** A new `computation.py` (frozen plan / result / verdict models) + a `sandbox/` subprocess runner that builds and runs the SymPy computation from the *structured* plan (never LLM-emitted code), judged by `simplify(diff)==0` **in code**. A new `computation_designer` agent emits the structured plan and nothing else. Everything wires into the existing `inject_limit_checks` (the executor augments the Prover with graceful fallback). `IdeaArtifact._evaluate()` does not change — only `CheckRecord.lens` and the `claim.status` join learn the `"executor"` source.

**Tech Stack:** Python 3.11+, Pydantic v2, **SymPy** (new pinned dep), `subprocess` + `resource` rlimits, `pytest`. Tests run **real SymPy** (deterministic, offline); only the Computation-Designer (LLM) is faked.

**Source spec:** `docs/2026-06-23-validate-agents-spec2-design.md`. Section refs (e.g. "§5") point there.

## Global Constraints

Every task implicitly includes these.

- **F1 — structured plan, NO arbitrary code.** The Computation-Designer emits only the structured `ComputationPlan` fields. The Executor builds the SymPy call itself. No `exec`/`eval` of LLM output; no `sympify` (which can eval) — use `sympy.parsing.sympy_parser.parse_expr` with a restricted `local_dict`/`global_dict`.
- **F3 — code judges, the LLM never decides after execution.** The `pass`/`fail`/`uncertain` is `simplify(computed − expected) == 0` in code. The executor path (`run_plan`, `_verdict`, `verdict_to_check`) takes **no `llm` argument** and calls no model. A test asserts this.
- **F2 — augment, don't replace.** On a `limit_recovery` claim, keep the existing `prove_claim` check; add the executor check. Executor decisive verdict dominates via the join; executor `uncertain` → the Prover check stands.
- **F4 — isolation:** subprocess + `RLIMIT_CPU`/`RLIMIT_AS` (best-effort) + wall-clock `timeout` + minimal `env` + restricted parse. The runner imports only `json` + `sympy`.
- **Verdict mapping:** `ComputationResult.ok==False` (parse error / timeout / sympy failure) → `uncertain`; `matched=="confirm"` → `pass`; `matched=="refute"` → `fail`. An executor `pass` carries `independent_sources=1` (a computed equality is independent of the LLM).
- **Gate untouched:** `IdeaArtifact._evaluate()` must not reference `"executor"`. A test asserts `"executor" not in inspect.getsource(IdeaArtifact._evaluate)`.
- **Loud caveat (ship in the report basis):** the executor verifies *reduction to the preregistered `expected`*, not that `expected` is the literature-correct result. The `CheckRecord.basis` shows computed, expected, and `expected_source`.
- **Tests:** deterministic, no network. Run from the repo root with `python -m pytest`. SymPy must be installed (`pip install -r requirements.txt`).
- **Commits:** plain message, **no `Co-Authored-By`/`Claude-Session` trailer**. Stage only files you changed (do not `git add -A`).

---

## File Structure

```
valagents/
  computation.py        # NEW: ComputationPlan, ComputationResult, ComputationVerdict + verdict_to_check()
  sandbox/
    __init__.py         # NEW (empty)
    runner.py           # NEW: subprocess entry — reads plan JSON on stdin, runs SymPy, prints result JSON
    executor.py         # NEW: run_plan() — launches runner under limits, maps to ComputationVerdict, saves artifacts
  agents/
    computation_designer.py   # NEW: design_computation() — emits ComputationPlan ONLY (no verdict)
  prompts.py            # MODIFY: add COMPUTATION_DESIGNER
  artifact.py           # MODIFY: CheckRecord.lens += "executor"; claim.status join includes executor
  config.py             # MODIFY: add SandboxCfg + Config.sandbox
  scheduler.py          # MODIFY: inject_limit_checks runs the executor after the Prover (augment + fallback)
requirements.txt        # MODIFY: add sympy
tests/
  test_computation_model.py     test_claim_executor_join.py
  test_sandbox_executor.py      test_computation_designer.py
  test_integration_executor.py
```

---

## Task 1: Data model + gate seam + config + dependency

**Files:**
- Create: `valagents/computation.py`, `tests/test_computation_model.py`, `tests/test_claim_executor_join.py`
- Modify: `valagents/artifact.py` (`CheckRecord.lens`, `AtomicClaim.status` join), `valagents/config.py`, `requirements.txt`

**Interfaces:**
- Produces: `ComputationPlan`, `ComputationResult`, `ComputationVerdict` (Pydantic). `CheckRecord.lens` now includes `"executor"`. `Config.sandbox: SandboxCfg` with `.enabled/.wall_s/.cpu_s/.mem_mb`.

- [ ] **Step 1: Add the dependency.** Append `sympy` to `requirements.txt`. Run `pip install sympy` (or `pip install -r requirements.txt`). Verify: `python -c "import sympy; print(sympy.__version__)"`.

- [ ] **Step 2: Write failing tests** `tests/test_computation_model.py`:

```python
from valagents.computation import ComputationPlan, ComputationResult, ComputationVerdict

def test_plan_minimal():
    p = ComputationPlan(expression="G*M/r**2*(1+a/c**2)", variables=["G","M","r","a","c"],
                        limit_variable="c", limit_point="oo", expected="G*M/r**2")
    assert p.kind == "symbolic" and p.criterion == "symbolic_equality"

def test_verdict_wraps_plan_and_result():
    p = ComputationPlan(expression="x", variables=["x"], limit_variable="x", limit_point="0", expected="0")
    r = ComputationResult(ok=True, computed="0", matched="confirm")
    v = ComputationVerdict(verdict="pass", measured="0", plan=p, result=r)
    assert v.verdict == "pass" and v.plan.expected == "0"
```

  and `tests/test_claim_executor_join.py`:

```python
import inspect
from valagents.artifact import AtomicClaim, CheckRecord, IdeaArtifact

def _claim(checks, ctype="mathematical"):
    return AtomicClaim(id="c1", statement="s", type=ctype, checks=checks)

def test_executor_pass_makes_math_claim_pass_over_grounder_uncertainty():
    execpass = CheckRecord(lens="executor", verdict="pass", independent_sources=1)
    gunc = CheckRecord(lens="grounder", verdict="uncertain")
    assert _claim([execpass, gunc]).status == "pass"   # computed equality is a strongest-pass

def test_executor_fail_makes_claim_fail():
    assert _claim([CheckRecord(lens="executor", verdict="fail")]).status == "fail"

def test_executor_pass_needs_independent_source():
    weak = CheckRecord(lens="executor", verdict="pass", independent_sources=0)
    assert _claim([weak]).status == "pending"          # pass requires independent_sources>=1

def test_evaluate_does_not_reference_executor():
    assert "executor" not in inspect.getsource(IdeaArtifact._evaluate)
```

- [ ] **Step 3: Run → FAIL.** `python -m pytest tests/test_computation_model.py tests/test_claim_executor_join.py -v`

- [ ] **Step 4: Implement.** Create `valagents/computation.py`:

```python
"""Spec 2 execution models. A ComputationPlan is FROZEN before execution; the verdict
is produced in code (no LLM) — see the design doc F1/F3."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

class ComputationPlan(BaseModel):
    kind: Literal["symbolic"] = "symbolic"
    expression: str
    variables: list[str] = []
    limit_variable: str
    limit_point: str
    expected: str
    expected_source: str = ""
    criterion: Literal["symbolic_equality"] = "symbolic_equality"
    confirm_if: str = ""
    refute_if: str = ""

class ComputationResult(BaseModel):
    ok: bool
    computed: str = ""
    matched: Literal["confirm", "refute", "neither"] = "neither"
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    resource_use: dict = {}
    artifacts_path: str = ""

class ComputationVerdict(BaseModel):
    verdict: Literal["pass", "fail", "uncertain"]
    measured: str = ""
    plan: ComputationPlan
    result: ComputationResult
```

  In `valagents/config.py`, add (next to `GateCfg`):

```python
class SandboxCfg(BaseModel):
    enabled: bool = True
    wall_s: int = 10
    cpu_s: int = 10
    mem_mb: int = 512
```

  and add `sandbox: SandboxCfg = SandboxCfg()` to the `Config` model.

  In `valagents/artifact.py`: change `CheckRecord.lens` to `Literal["grounder", "prover", "redteam", "executor"]`. In `AtomicClaim.status`, find `has_proof_pass = any(c.lens == "prover" for c in passes)` and change to:

```python
            has_proof_pass = any(c.lens in ("prover", "executor") for c in passes)
```

- [ ] **Step 5: Run → PASS.** `python -m pytest tests/test_computation_model.py tests/test_claim_executor_join.py -v`, then full suite `python -m pytest -q`.

- [ ] **Step 6: Commit** (plain message, no trailer; stage only the touched files):

```bash
git add valagents/computation.py valagents/config.py valagents/artifact.py requirements.txt tests/test_computation_model.py tests/test_claim_executor_join.py
git commit -m "feat(computation): plan/result/verdict models + executor lens seam (gate unchanged)"
```

---

## Task 2: Sandbox runner + executor (real symbolic path)

**Files:**
- Create: `valagents/sandbox/__init__.py` (empty), `valagents/sandbox/runner.py`, `valagents/sandbox/executor.py`, `tests/test_sandbox_executor.py`

**Interfaces:**
- Consumes: `ComputationPlan`, `ComputationResult`, `ComputationVerdict` (Task 1); `Config.sandbox`.
- Produces: `valagents.sandbox.executor.run_plan(plan: ComputationPlan, cfg, artifacts_dir: str | None = None) -> ComputationVerdict` — takes **no `llm`** (F3). The runner is `valagents/sandbox/runner.py` (stdin JSON plan → stdout JSON result).

- [ ] **Step 1: Write failing tests** `tests/test_sandbox_executor.py` (real SymPy — these exercise actual computation):

```python
from valagents.computation import ComputationPlan
from valagents.config import Config
from valagents.sandbox.executor import run_plan
import inspect

def cfg():
    return Config(default_model="fake")

def plan(**kw):
    base = dict(expression="1/x", variables=["x"], limit_variable="x", limit_point="oo", expected="0")
    base.update(kw)
    return ComputationPlan(**base)

def test_recovers_limit_passes():
    v = run_plan(plan(), cfg())                      # limit(1/x, x, oo) == 0
    assert v.verdict == "pass" and v.result.matched == "confirm" and v.result.ok

def test_newtonian_recovery_passes():
    v = run_plan(plan(expression="G*M/r**2*(1+a/c**2)", variables=["G","M","r","a","c"],
                      limit_variable="c", limit_point="oo", expected="G*M/r**2"), cfg())
    assert v.verdict == "pass"

def test_wrong_limit_fails():
    v = run_plan(plan(expected="1"), cfg())          # limit(1/x,...) == 0, not 1
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_unparseable_or_malicious_expression_is_uncertain_not_executed():
    v = run_plan(plan(expression="__import__('os').system('echo hacked')"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok      # restricted parse rejects it; never runs code

def test_run_plan_takes_no_llm():                    # F3: code judges, no model in the loop
    sig = inspect.signature(run_plan)
    assert "llm" not in sig.parameters

def test_artifacts_saved(tmp_path):
    v = run_plan(plan(), cfg(), artifacts_dir=str(tmp_path / "c1"))
    assert (tmp_path / "c1" / "plan.json").exists() and (tmp_path / "c1" / "result.json").exists()

def test_disabled_sandbox_is_uncertain():
    c = cfg(); c.sandbox.enabled = False
    assert run_plan(plan(), c).verdict == "uncertain"
```

- [ ] **Step 2: Run → FAIL.** `python -m pytest tests/test_sandbox_executor.py -v`

- [ ] **Step 3: Implement `valagents/sandbox/__init__.py`** (empty file) and `valagents/sandbox/runner.py`:

```python
"""Sandbox runner (subprocess entry point). Reads a frozen ComputationPlan JSON on stdin,
runs the SymPy computation the plan describes, prints a result JSON on stdout.
Imports ONLY json + sympy. No network, no filesystem writes. NEVER execs LLM-provided code:
expressions are parsed with parse_expr over a restricted namespace, not sympify/eval."""
import json
import sys

_ALLOWED = ("sin", "cos", "tan", "exp", "log", "sqrt", "Abs", "sign",
            "pi", "E", "oo", "Rational", "Integer", "Float")

def _run(plan: dict) -> dict:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    syms = {name: sympy.Symbol(name) for name in plan.get("variables", [])}
    glob = {n: getattr(sympy, n) for n in _ALLOWED}
    local = dict(syms)
    expr = parse_expr(plan["expression"], local_dict=local, global_dict=glob, evaluate=True)
    expected = parse_expr(plan["expected"], local_dict=local, global_dict=glob, evaluate=True)
    lv = plan["limit_variable"]
    lvar = syms.get(lv, sympy.Symbol(lv))
    pt_raw = str(plan["limit_point"]).strip()
    point = {"oo": sympy.oo, "+oo": sympy.oo, "-oo": -sympy.oo}.get(
        pt_raw, parse_expr(pt_raw, local_dict=local, global_dict=glob, evaluate=True))
    computed = sympy.limit(expr, lvar, point)
    diff = sympy.simplify(computed - expected)
    holds = bool(diff == 0)
    return {"ok": True, "computed": str(computed),
            "matched": "confirm" if holds else "refute"}

def main() -> None:
    try:
        plan = json.load(sys.stdin)
        out = _run(plan)
    except Exception as e:                       # parse error, sympy failure, etc.
        out = {"ok": False, "matched": "neither", "error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout)

if __name__ == "__main__":
    main()
```

  and `valagents/sandbox/executor.py`:

```python
"""Run a frozen ComputationPlan in a subprocess under resource limits. Code judges; no LLM (F3)."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from valagents.computation import ComputationPlan, ComputationResult, ComputationVerdict

_RUNNER = str(Path(__file__).with_name("runner.py"))

def _preexec(cpu_s: int, mem_mb: int):
    def _set():
        import resource
        for res, lim in ((resource.RLIMIT_CPU, (cpu_s, cpu_s)),
                         (resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)):
            try:
                resource.setrlimit(res, lim)
            except (ValueError, OSError):        # RLIMIT_AS not enforced on some platforms (e.g. macOS)
                pass
    return _set

def _verdict(plan: ComputationPlan, result: ComputationResult) -> ComputationVerdict:
    if not result.ok:
        v = "uncertain"
    elif result.matched == "confirm":
        v = "pass"
    elif result.matched == "refute":
        v = "fail"
    else:
        v = "uncertain"
    return ComputationVerdict(verdict=v, measured=result.computed, plan=plan, result=result)

def _save(dirpath: str, plan: ComputationPlan, result: ComputationResult) -> str:
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(plan.model_dump_json(indent=2))
    (d / "result.json").write_text(result.model_dump_json(indent=2))
    (d / "stdout.txt").write_text(result.stdout or "")
    (d / "stderr.txt").write_text(result.stderr or "")
    return str(d)

def run_plan(plan: ComputationPlan, cfg, artifacts_dir: str | None = None) -> ComputationVerdict:
    if not cfg.sandbox.enabled:
        return _verdict(plan, ComputationResult(ok=False, error="sandbox disabled"))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, _RUNNER],
            input=plan.model_dump_json(), capture_output=True, text=True,
            timeout=cfg.sandbox.wall_s,
            preexec_fn=_preexec(cfg.sandbox.cpu_s, cfg.sandbox.mem_mb) if os.name == "posix" else None,
            env={"PATH": os.environ.get("PATH", "")},
        )
        wall = round(time.monotonic() - t0, 3)
        try:
            out = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            out = {"ok": False, "error": "unparseable runner output", "matched": "neither"}
        result = ComputationResult(
            ok=bool(out.get("ok")), computed=out.get("computed", ""),
            matched=out.get("matched", "neither"), stdout=proc.stdout, stderr=proc.stderr,
            error=out.get("error", ""), resource_use={"wall_s": wall})
    except subprocess.TimeoutExpired:
        result = ComputationResult(ok=False, error="timeout",
                                   resource_use={"wall_s": cfg.sandbox.wall_s})
    if artifacts_dir:
        result.artifacts_path = _save(artifacts_dir, plan, result)
    return _verdict(plan, result)
```

  Note for the implementer: a real wall-clock timeout maps through the same `ok=False → uncertain` path as `test_disabled_sandbox_is_uncertain` / the unparseable case; you do not need a flaky long-running test to cover it.

- [ ] **Step 4: Run → PASS.** `python -m pytest tests/test_sandbox_executor.py -v`, then full suite `python -m pytest -q`.

- [ ] **Step 5: Commit:**

```bash
git add valagents/sandbox/__init__.py valagents/sandbox/runner.py valagents/sandbox/executor.py tests/test_sandbox_executor.py
git commit -m "feat(sandbox): subprocess SymPy executor for structured plans (code judges, restricted parse, limits)"
```

---

## Task 3: Computation-Designer + verdict→CheckRecord + wire into inject_limit_checks

**Files:**
- Create: `valagents/agents/computation_designer.py`, `tests/test_computation_designer.py`
- Modify: `valagents/prompts.py` (add `COMPUTATION_DESIGNER`), `valagents/computation.py` (add `verdict_to_check`), `valagents/scheduler.py` (`inject_limit_checks`)

**Interfaces:**
- Consumes: `checked` (parse.py), `build_messages` (base.py), `ComputationPlan`, `run_plan` (Task 2), `CheckRecord`/`Source` (artifact.py), `prove_claim` (already in `inject_limit_checks`).
- Produces: `design_computation(claim, formal_claim, llm, cfg) -> ComputationPlan | None` (returns a plan ONLY — no verdict, F1); `verdict_to_check(v: ComputationVerdict, tick: int) -> CheckRecord` (no llm — F3).

- [ ] **Step 1: Write failing tests** `tests/test_computation_designer.py`:

```python
import inspect
from valagents.agents.computation_designer import design_computation
from valagents.computation import verdict_to_check, ComputationPlan, ComputationResult, ComputationVerdict
from valagents.artifact import FormalClaim, AtomicClaim
from valagents.config import Config
from tests.fake_llm import FakeLLM

def cfg():
    return Config(default_model="fake")

async def test_designer_emits_structured_plan_only():
    body = ("EXPRESSION: G*M/r**2*(1+a/c**2) | VARIABLES: G,M,r,a,c | LIMIT_VARIABLE: c "
            "| LIMIT_POINT: oo | EXPECTED: G*M/r**2 | EXPECTED_SOURCE: Newtonian gravity "
            "| CONFIRM_IF: limit equals GM/r^2 | REFUTE_IF: limit differs")
    claim = AtomicClaim(id="L1", statement="recovers Newtonian gravity", type="mathematical")
    plan = await design_computation(claim, FormalClaim(statement="x", falsifiable=True),
                                    FakeLLM(lambda a, m: body), cfg())
    assert isinstance(plan, ComputationPlan)
    assert plan.limit_variable == "c" and plan.expected == "G*M/r**2" and plan.variables == ["G","M","r","a","c"]

async def test_designer_returns_none_on_unparseable():
    claim = AtomicClaim(id="L1", statement="s", type="mathematical")
    plan = await design_computation(claim, FormalClaim(statement="x", falsifiable=True),
                                    FakeLLM(lambda a, m: "no tail"), cfg())
    assert plan is None

def test_designer_returns_no_verdict():                 # F1: it designs, it does not judge
    src = inspect.getsource(design_computation)
    assert "ComputationVerdict" not in src and "verdict" not in src.replace("# ", "")

def test_verdict_to_check_pass_is_independent():
    p = ComputationPlan(expression="x", variables=["x"], limit_variable="x", limit_point="0",
                        expected="0", expected_source="src")
    v = ComputationVerdict(verdict="pass", measured="0", plan=p,
                           result=ComputationResult(ok=True, computed="0", matched="confirm"))
    rec = verdict_to_check(v, tick=0)
    assert rec.lens == "executor" and rec.verdict == "pass" and rec.independent_sources == 1
    assert "expected = 0" in rec.basis and "src" in rec.basis      # caveat surfaced in basis

def test_verdict_to_check_takes_no_llm():               # F3
    assert "llm" not in inspect.signature(verdict_to_check).parameters
```

- [ ] **Step 2: Run → FAIL.** `python -m pytest tests/test_computation_designer.py -v`

- [ ] **Step 3: Implement.** Add to `valagents/prompts.py`:

```python
COMPUTATION_DESIGNER = """You DESIGN a symbolic check; you do NOT run or judge it — code does that, and \
you will never see the result. Given a known-limit-recovery claim, produce the structured plan to test \
whether the idea's expression reduces to the established result in the stated regime.

FORMAL CLAIM: {formal}
LIMIT-RECOVERY CLAIM: {statement}

Give the idea's expression and the variable taken to a limit; give the expected established result and \
where it comes from. Use plain SymPy-parseable math (e.g. G*M/r**2, c, oo). Do not output code.
End with exactly:
EXPRESSION: <expr> | VARIABLES: <comma-separated symbols> | LIMIT_VARIABLE: <symbol> | LIMIT_POINT: <oo|0|value> | EXPECTED: <expr> | EXPECTED_SOURCE: <where the known result comes from> | CONFIRM_IF: <…> | REFUTE_IF: <…>"""
```

  Create `valagents/agents/computation_designer.py`:

```python
"""Computation-Designer: emits a structured ComputationPlan for a known-limit-recovery claim.
It DESIGNS the check only — it returns no verdict and never sees the execution result (F1/F3)."""
from __future__ import annotations
from valagents.computation import ComputationPlan
from valagents.parse import checked
from valagents.prompts import COMPUTATION_DESIGNER
from valagents.agents.base import build_messages

_KEYS = ["EXPRESSION", "VARIABLES", "LIMIT_VARIABLE", "LIMIT_POINT",
         "EXPECTED", "EXPECTED_SOURCE", "CONFIRM_IF", "REFUTE_IF"]

async def design_computation(claim, formal_claim, llm, cfg) -> ComputationPlan | None:
    user = COMPUTATION_DESIGNER.format(
        formal=formal_claim.statement if formal_claim else "", statement=claim.statement)
    tail = await checked("computation_designer", build_messages("You design symbolic checks.", user),
                         _KEYS, llm=llm)
    if tail is None:
        return None
    variables = [v.strip() for v in tail["variables"].split(",") if v.strip()]
    try:
        return ComputationPlan(
            expression=tail["expression"], variables=variables,
            limit_variable=tail["limit_variable"], limit_point=tail["limit_point"],
            expected=tail["expected"], expected_source=tail["expected_source"],
            confirm_if=tail["confirm_if"], refute_if=tail["refute_if"])
    except Exception:
        return None
```

  Add `verdict_to_check` to `valagents/computation.py` (imports `CheckRecord`, `Source` from `valagents.artifact` — put the import inside the function to avoid a circular import, since `artifact.py` does not import `computation.py`):

```python
def verdict_to_check(v: "ComputationVerdict", tick: int = 0):
    """Map an executed ComputationVerdict to a CheckRecord(lens='executor'). No LLM (F3)."""
    from valagents.artifact import CheckRecord, Source
    indep = 1 if v.verdict == "pass" else 0
    basis = (f"computed limit = {v.measured or '?'}; expected = {v.plan.expected} "
             f"(source: {v.plan.expected_source or 'n/a'}); matched = {v.result.matched}")
    sources = ([Source(locator=v.plan.expected_source, relation="independent")]
               if v.plan.expected_source else [])
    return CheckRecord(lens="executor", verdict=v.verdict, basis=basis,
                       independent_sources=indep, sources=sources, tick=tick)
```

  In `valagents/scheduler.py`, in `inject_limit_checks`, after the existing `rec = await prove_claim(...)` + `store.add_check(claim_id, rec)`, add the executor augmentation:

```python
        # F2: augment the reasoned Prover with an EXECUTED symbolic check (Spec 2)
        plan = await design_computation(claim, art.formal_claim, llm, cfg)
        if plan is not None:
            from valagents.sandbox.executor import run_plan
            from valagents.computation import verdict_to_check
            adir = f"{cfg.results_dir}/computations/{claim_id}" if getattr(cfg, "results_dir", None) else None
            verdict = run_plan(plan, cfg, artifacts_dir=adir)
            store.add_check(claim_id, verdict_to_check(verdict, tick=tick))
            store.record({"event": "limit_executed", "claim": claim_id,
                          "verdict": verdict.verdict, "computed": verdict.measured})
            tick += 1
```

  (Add `from valagents.agents.computation_designer import design_computation` to the scheduler imports. Confirm `cfg.results_dir` exists; if not, pass `artifacts_dir=None`.)

- [ ] **Step 4: Run → PASS.** `python -m pytest tests/test_computation_designer.py -v`, then full suite.

- [ ] **Step 5: Commit:**

```bash
git add valagents/agents/computation_designer.py valagents/prompts.py valagents/computation.py valagents/scheduler.py tests/test_computation_designer.py
git commit -m "feat(executor): computation-designer + verdict->CheckRecord, wired into inject_limit_checks (augment+fallback)"
```

---

## Task 4: Negative integration tests (recovers → pass, violates → refuted, no-plan → Prover fallback)

**Files:**
- Create: `tests/test_integration_executor.py`

**Interfaces:**
- Consumes: `valagents.scheduler.run` (or `inject_limit_checks` directly), `ArtifactStore`, the agents, `FakeLLM`.

- [ ] **Step 1: Write the integration tests.** These drive `inject_limit_checks` end-to-end with a FakeLLM (Designer + Prover scripted) and the **real** executor, on an artifact that already has a `known_limits` entry so a `limit_recovery` claim is created.

```python
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
```

- [ ] **Step 2: Run → may FAIL** (depending on `inject_limit_checks` wiring from Task 3). Fix wiring bugs this surfaces in `scheduler.py` — do NOT weaken assertions. `python -m pytest tests/test_integration_executor.py -v`

- [ ] **Step 3: Run → PASS,** then the whole suite `python -m pytest -q` (all green, pristine).

- [ ] **Step 4: Commit:**

```bash
git add tests/test_integration_executor.py
git commit -m "test(executor): integration — recovers->pass, violates->refuted, no-plan->prover fallback"
```

---

## Self-Review (completed against the spec)

**Spec coverage:** §2 F1 (structured plan, no code) → Task 3 designer emits only fields + the `test_designer_returns_no_verdict` / restricted-parse tests; F2 augment+fallback → Task 3 wiring + Task 4 `test_no_plan_falls_back_to_prover`; F3 code judges → `run_plan`/`verdict_to_check` take no `llm` (tested) + `simplify(diff)==0` in the runner; F4 isolation → Task 2 subprocess+rlimits+timeout+restricted parse + `test_unparseable_or_malicious_expression_is_uncertain`. §3 preregistration → the frozen `ComputationPlan` is built before `run_plan`; the verdict is code. §4 data model → Task 1. §5 flow → Task 3 `inject_limit_checks`. §7 caveat → surfaced in `verdict_to_check` basis (tested). §8 gate untouched → `test_evaluate_does_not_reference_executor`. §9 testing (real SymPy) → Tasks 2/4. §10 build order → Tasks 1–4 are slices 1–5 (slice 6, lens 2, is a separate future plan).

**Placeholder scan:** none — every step has real code and exact commands.

**Type consistency:** `ComputationPlan`/`ComputationResult`/`ComputationVerdict` fields are identical across Tasks 1–4; `run_plan(plan, cfg, artifacts_dir=None)`, `design_computation(claim, formal_claim, llm, cfg)`, `verdict_to_check(v, tick)` signatures match every call site; `CheckRecord(lens="executor", …)` matches the Task-1 enum extension; the `claim.status` join change (`("prover","executor")`) is the exact string edit.

**Flagged for the implementer:** `inject_limit_checks` may already cap/iterate over `known_limits` (R3) — preserve that loop; the executor augmentation goes *inside* the per-limit loop after the Prover `add_check`. Confirm `cfg.results_dir` exists before using it for `artifacts_dir` (pass `None` if not).
