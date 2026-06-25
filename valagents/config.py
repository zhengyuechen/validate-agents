"""Typed config for validate-agents. Roles → models/temps; gate thresholds."""
from __future__ import annotations
import os
import yaml
from pydantic import BaseModel
from dotenv import load_dotenv

class GroundCfg(BaseModel):
    backend: str = "arxiv"          # arxiv | none | tavily
    supports_factor: float = 2.0    # ratio < this AND conditions-confirmed -> supports (G-D7)
    contradict_factor: float = 10.0 # ratio >= this AND conditions-confirmed -> contradicts (G-D7)
    quote_min_tokens: int = 6       # min word-tokens in a substantial referent-binding quote (§6)
    reference_rel_tol: float = 1e-3 # G-D9 scale-table both-directions reference-test tolerance

class GateCfg(BaseModel):
    min_attack_categories: int = 2  # categories the Red-team must attempt for internally_validated
    fanout_N: int = 2               # diverse-type lenses on a load-bearing uncertain node before finalize
    repair_cap: int = 3             # max repair versions before finalize

class SandboxCfg(BaseModel):
    enabled: bool = True
    wall_s: int = 10
    cpu_s: int = 10
    mem_mb: int = 512

class SimCfg(BaseModel):
    max_state_vars: int = 8
    max_expr_nodes: int = 200
    max_grid_points: int = 400
    max_steps: int = 200_000
    max_total_steps: int = 2_000_000
    min_grid_points: int = 4
    fixed_point_tol: float = 1e-6      # linear_stability equilibrium residual tolerance (absolute, LS-D8)
    min_points_per_axis: int = 5       # linear_stability per-swept-axis density floor (LS-D8)
    max_dt_halvings: int = 3           # bounded honesty check: dt-refinement depth (BP / B-D4)
    conv_rtol: float = 0.1             # bounded honesty check: convergence tolerance on t*/max_abs (B-D7)

class Config(BaseModel):
    default_model: str
    models: dict[str, str] = {}
    temperature: dict[str, float] = {}
    grounding: GroundCfg = GroundCfg()
    gate: GateCfg = GateCfg()
    sandbox: SandboxCfg = SandboxCfg()
    sim: SimCfg = SimCfg()
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
