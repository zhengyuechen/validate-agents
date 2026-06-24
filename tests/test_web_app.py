import json

from fastapi.testclient import TestClient

from valagents.artifact import FormalClaim, IdeaArtifact
from valagents.web.app import _coerce_references, create_app


def write_config(path, results_dir):
    path.write_text(
        "\n".join(
            [
                "default_model: fake/model",
                "models: {}",
                "temperature: {}",
                "grounding:",
                "  backend: none",
                "gate:",
                "  min_attack_categories: 2",
                "  fanout_N: 2",
                "  repair_cap: 3",
                f"results_dir: {results_dir}",
                "",
            ]
        )
    )


def test_web_app_serves_index_config_and_artifact(tmp_path):
    config_path = tmp_path / "config.yaml"
    results = tmp_path / "results"
    results.mkdir()
    write_config(config_path, results)

    art = IdeaArtifact(
        raw_idea="seed",
        formal_claim=FormalClaim(statement="X changes Y", regime="low noise", falsifiable=True),
        finalized=True,
    )
    (results / "run-1.json").write_text(art.model_dump_json(indent=2))
    (results / "run-1.md").write_text("# Validation Report\n")
    (results / "run-1.bib").write_text("@article{run1}\n")

    app = create_app(config_path=str(config_path), results_base=str(results))
    client = TestClient(app)

    assert "Validate Agents" in client.get("/").text
    assert client.get("/api/config").json()["config"]["grounding"]["backend"] == "none"

    runs = client.get("/api/runs").json()
    assert runs[0]["id"] == "run-1"
    assert runs[0]["seed"] == "seed"

    run = client.get("/api/runs/run-1").json()
    assert run["artifact"]["raw_idea"] == "seed"
    assert "Validation Report" in run["report"]
    assert "@article" in run["bibtex"]


def test_web_config_put_validates_and_writes(tmp_path):
    config_path = tmp_path / "config.yaml"
    results = tmp_path / "results"
    results.mkdir()
    write_config(config_path, results)

    app = create_app(config_path=str(config_path), results_base=str(results))
    client = TestClient(app)
    cfg = client.get("/api/config").json()["config"]
    cfg["grounding"]["backend"] = "arxiv"

    resp = client.put("/api/config", content=json.dumps(cfg))

    assert resp.status_code == 200
    assert "backend: arxiv" in config_path.read_text()


def test_web_references_accept_existing_path_or_inline_ids(tmp_path):
    refs = tmp_path / "refs.txt"
    refs.write_text("2401.12345\n")
    assert _coerce_references(str(refs), str(tmp_path), "run-1") == str(refs)

    inline = _coerce_references("2401.12345, 10.1234/example", str(tmp_path), "run-2")

    assert inline == str(tmp_path / ".references" / "run-2.txt")
    assert (tmp_path / ".references" / "run-2.txt").read_text().splitlines() == [
        "2401.12345",
        "10.1234/example",
    ]
