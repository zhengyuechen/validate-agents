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
        payload = plan.model_dump_json()
        if plan.kind == "simulation":
            d = json.loads(payload)
            d["_sim_ceilings"] = cfg.sim.model_dump()   # ceilings reach the subprocess; saved artifact stays the frozen plan
            payload = json.dumps(d)
        proc = subprocess.run(
            [sys.executable, _RUNNER],
            input=payload, capture_output=True, text=True,
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
    except Exception as e:
        result = ComputationResult(ok=False, error=f"subprocess failure: {type(e).__name__}: {e}")
    if artifacts_dir:
        result.artifacts_path = _save(artifacts_dir, plan, result)
    return _verdict(plan, result)
