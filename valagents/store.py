"""Single-writer artifact store: immutable version chain + append-only event log.
Version-don't-mutate is what makes Spec-4 parallel + repair safe."""
from __future__ import annotations
from valagents.artifact import IdeaArtifact, CheckRecord
from valagents import run_log

class ArtifactStore:
    def __init__(self, initial: IdeaArtifact) -> None:
        self._versions: list[IdeaArtifact] = [initial]
        self.events: list[dict] = []

    @property
    def current(self) -> IdeaArtifact:
        return self._versions[-1]

    @property
    def versions(self) -> list[IdeaArtifact]:
        return list(self._versions)

    def record(self, event: dict) -> None:
        self.events.append(event)
        run_log.emit(event.get("event", "event"), **{k: v for k, v in event.items() if k != "event"})

    def _claim(self, claim_id: str):
        return next(c for c in self.current.claim_graph if c.id == claim_id)

    def add_check(self, claim_id: str, rec: CheckRecord) -> None:
        self._claim(claim_id).checks.append(rec)

    def set(self, field: str, value) -> None:
        setattr(self.current, field, value)

    def fork_for_repair(self, target_ids: list[str]) -> IdeaArtifact:
        cur = self.current
        nxt = cur.model_copy(deep=True)          # frozen snapshot of cur stays in _versions
        nxt.version_id = cur.version_id + 1
        nxt.parent_version = cur.version_id
        nxt.repairs_spent = cur.repairs_spent + 1
        nxt.finalized = False
        for c in nxt.claim_graph:
            if c.id in target_ids:
                c.checks = []
                c.exhausted = False
        self._versions.append(nxt)
        return nxt
