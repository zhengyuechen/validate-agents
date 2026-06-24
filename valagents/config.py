"""Typed config for validate-agents. Roles → models/temps; gate thresholds."""
from __future__ import annotations
import os
import yaml
from pydantic import BaseModel
from dotenv import load_dotenv

class GroundCfg(BaseModel):
    backend: str = "arxiv"          # arxiv | none | tavily

class GateCfg(BaseModel):
    min_attack_categories: int = 2  # categories the Red-team must attempt for internally_validated
    fanout_N: int = 2               # diverse-type lenses on a load-bearing uncertain node before finalize
    repair_cap: int = 3             # max repair versions before finalize

class SandboxCfg(BaseModel):
    enabled: bool = True
    wall_s: int = 10
    cpu_s: int = 10
    mem_mb: int = 512

class Config(BaseModel):
    default_model: str
    models: dict[str, str] = {}
    temperature: dict[str, float] = {}
    grounding: GroundCfg = GroundCfg()
    gate: GateCfg = GateCfg()
    sandbox: SandboxCfg = SandboxCfg()
    results_dir: str = "results"

    def model_for(self, agent: str) -> str:
        return self.models.get(agent, self.default_model)

def load_config(path: str = "config.yaml") -> Config:
    load_dotenv()
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)

def require_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (add it to .env or the environment).")
    return key
