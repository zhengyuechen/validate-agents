"""FastAPI app for launching and inspecting validate-agents runs."""
from __future__ import annotations

import asyncio
import glob
import json
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from valagents import run_log
from valagents.cli import _slug, run_cli
from valagents.config import Config, load_config
from valagents.web_search import build_backend

STATIC = Path(__file__).parent / "static"


def default_llm_factory(cfg: Config):
    from valagents.llm import OpenRouterClient

    return OpenRouterClient(cfg)


def _json_path(results_base: str, run_id: str) -> Path:
    return Path(results_base) / f"{run_id}.json"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _read_run(results_base: str, run_id: str) -> dict:
    base = Path(results_base)
    artifact_path = base / f"{run_id}.json"
    if not artifact_path.exists():
        raise HTTPException(404, f"run '{run_id}' not found")
    artifact = _load_json(artifact_path, {})
    report_path = base / f"{run_id}.md"
    bib_path = base / f"{run_id}.bib"
    return {
        "id": run_id,
        "seed": artifact.get("raw_idea", ""),
        "status": artifact.get("status"),
        "maturity": artifact.get("maturity"),
        "load_bearing": artifact.get("load_bearing"),
        "blocker": artifact.get("blocker"),
        "artifact": artifact,
        "report": report_path.read_text() if report_path.exists() else "",
        "bibtex": bib_path.read_text() if bib_path.exists() else "",
    }


def _run_summary(path: Path) -> dict | None:
    try:
        artifact = _load_json(path, {})
    except Exception:
        return None
    stat = path.stat()
    claims = artifact.get("claim_graph") or []
    return {
        "id": path.stem,
        "seed": artifact.get("raw_idea", ""),
        "status": artifact.get("status", "unknown"),
        "maturity": artifact.get("maturity"),
        "claims": len(claims),
        "load_bearing": artifact.get("load_bearing"),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


async def execute_run(run_id, seed, cfg, llm, backend, registry, results_base, references_path=None):
    """Run validation to completion and record status for the web poller."""
    rec = registry[run_id]
    rec["status"] = "running"
    run_log.bind(run_log.events_path(results_base, run_id))
    run_log.emit("run_started", seed=seed)
    try:
        out = await run_cli(
            seed,
            llm,
            cfg,
            backend=backend,
            out_dir=results_base,
            references_path=references_path,
            run_id=run_id,
        )
        rec.update(status="done", out=out["json_path"], error=None)
        run_log.emit(
            "run_done",
            status=out["artifact"].status,
            maturity=out["artifact"].maturity,
            claims=len(out["artifact"].claim_graph),
        )
    except Exception as exc:  # surface the failure to the UI instead of crashing the task
        rec.update(status="error", error=str(exc))
        run_log.emit("run_error", error=str(exc))


def create_app(
    llm_factory=default_llm_factory,
    config_path="config.yaml",
    results_base="results",
) -> FastAPI:
    app = FastAPI(title="Validate Agents")
    registry: dict[str, dict] = {}
    app.state.registry = registry

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/config")
    def get_config():
        with open(config_path) as f:
            return {"config": yaml.safe_load(f)}

    @app.put("/api/config")
    async def put_config(request: Request):
        data = await request.json()
        try:
            Config(**data)
        except Exception as exc:
            raise HTTPException(400, f"invalid config: {exc}")
        with open(config_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return {"ok": True}

    @app.get("/api/runs")
    def list_runs():
        out = []
        for name in sorted(glob.glob(f"{results_base}/*.json"), reverse=True):
            summary = _run_summary(Path(name))
            if summary is not None:
                out.append(summary)
        return out

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return _read_run(results_base, run_id)

    @app.post("/api/runs")
    async def launch_run(request: Request):
        body = await request.json()
        seed = (body.get("seed") or body.get("goal") or "").strip()
        if not seed:
            raise HTTPException(400, "a seed idea is required")
        references_path = body.get("references") or None
        cfg = load_config(config_path)
        try:
            llm = llm_factory(cfg)
        except Exception as exc:
            raise HTTPException(400, str(exc))
        backend = build_backend(cfg)

        base_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        slug = _slug(seed)
        run_id, n = f"{base_ts}_{slug}", 2
        while run_id in registry or _json_path(results_base, run_id).exists():
            run_id, n = f"{base_ts}-{n}_{slug}", n + 1
        registry[run_id] = {"status": "queued", "seed": seed, "error": None}
        asyncio.create_task(
            execute_run(run_id, seed, cfg, llm, backend, registry, results_base, references_path)
        )
        return {"run_id": run_id}

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, since: int = 0):
        evs = run_log.read_events(run_log.events_path(results_base, run_id), since=since)
        return {"events": evs, "next": since + len(evs)}

    @app.get("/api/runs/{run_id}/status")
    def run_status(run_id: str):
        rec = registry.get(run_id)
        if rec is None:
            if _json_path(results_base, run_id).exists():
                return {"status": "done"}
            raise HTTPException(404, f"run '{run_id}' not found")
        body = {"status": rec["status"], "error": rec.get("error")}
        if rec["status"] == "done" and _json_path(results_base, run_id).exists():
            body.update(_run_summary(_json_path(results_base, run_id)) or {})
        return body

    return app


app = create_app()
