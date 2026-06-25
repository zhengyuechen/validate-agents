"""References and citations: resolve identifiers, merge sources, emit BibTeX."""
from __future__ import annotations

import json
import re
from typing import Literal, Protocol

from pydantic import BaseModel


_ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
_DOI_RE = re.compile(r"10\.\d{1,9}/[^\s]+", re.IGNORECASE)


class Reference(BaseModel):
    locator: str
    key: str = ""
    number: int = 0
    title: str = ""
    authors: list[str] = []
    year: str = ""
    url: str = ""
    origin: Literal["provided", "retrieved", "asserted"] = "retrieved"
    relation: str = "unknown"
    unresolved: bool = False
    cited_by: list[str] = []


def detect_kind(identifier: str) -> str:
    s = identifier.strip().lower()
    if "arxiv.org" in s or s.startswith("arxiv:") or _ARXIV_RE.fullmatch(s):
        return "arxiv"
    if "doi.org" in s or _DOI_RE.search(identifier):
        return "doi"
    return "unknown"


def normalize_id(identifier: str) -> str:
    s = identifier.strip()
    kind = detect_kind(s)
    if kind == "arxiv":
        match = _ARXIV_RE.search(s)
        if match:
            return "arxiv:" + match.group(1)
    if kind == "doi":
        match = _DOI_RE.search(s)
        if match:
            return match.group(0).rstrip(".,;)").lower()
    return s.lower()


class Resolver(Protocol):
    async def resolve(self, identifier: str) -> Reference | None:
        ...


class ArxivResolver:
    async def resolve(self, identifier: str) -> Reference | None:
        import asyncio
        import arxiv

        match = _ARXIV_RE.search(identifier)
        if not match:
            return None
        locator = normalize_id(identifier)
        try:
            results = await asyncio.to_thread(
                list,
                arxiv.Client().results(arxiv.Search(id_list=[match.group(1)])),
            )
        except Exception:
            results = []
        if not results:
            return Reference(locator=locator, url=identifier, origin="provided", unresolved=True)
        result = results[0]
        return Reference(
            locator=locator,
            title=result.title,
            authors=[author.name for author in result.authors],
            year=str(result.published)[:4],
            url=result.entry_id,
            origin="provided",
        )


class DoiResolver:
    async def resolve(self, identifier: str) -> Reference | None:
        import httpx

        match = _DOI_RE.search(identifier)
        if not match:
            return None
        doi = match.group(0).rstrip(".,;)")
        locator = doi.lower()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"https://api.crossref.org/works/{doi}", timeout=15)
                resp.raise_for_status()
                msg = resp.json()["message"]
        except Exception:
            return Reference(
                locator=locator,
                url=f"https://doi.org/{doi}",
                origin="provided",
                unresolved=True,
            )

        authors = [
            f"{author.get('given', '')} {author.get('family', '')}".strip()
            for author in msg.get("author", [])
        ]
        published = msg.get("published-print") or msg.get("published-online") or {}
        parts = published.get("date-parts", [[""]])
        return Reference(
            locator=locator,
            title=(msg.get("title") or [""])[0],
            authors=[author for author in authors if author],
            year=str(parts[0][0]),
            url=f"https://doi.org/{doi}",
            origin="provided",
        )


class DefaultResolver:
    def __init__(self) -> None:
        self._arxiv = ArxivResolver()
        self._doi = DoiResolver()

    async def resolve(self, identifier: str) -> Reference | None:
        kind = detect_kind(identifier)
        if kind == "arxiv":
            return await self._arxiv.resolve(identifier)
        if kind == "doi":
            return await self._doi.resolve(identifier)
        return None


def _read_ids(path: str) -> list[str]:
    with open(path) as f:
        text = f.read().strip()
    if not text:
        return []
    if text.startswith("["):
        return [str(value).strip() for value in json.loads(text) if str(value).strip()]
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


async def load_provided(path: str, resolver: Resolver) -> list[Reference]:
    refs = []
    for identifier in _read_ids(path):
        ref = await resolver.resolve(identifier)
        if ref is None:
            ref = Reference(
                locator=normalize_id(identifier),
                url=identifier,
                origin="provided",
                unresolved=True,
            )
        ref.locator = normalize_id(ref.locator)
        ref.origin = "provided"
        refs.append(ref)
    return refs


def collect_retrieved(artifact) -> list[Reference]:
    seen: dict[str, Reference] = {}
    refs: list[Reference] = []
    for claim in artifact.claim_graph:
        for check in claim.checks:
            for source in getattr(check, "sources", []):
                locator = normalize_id(source.locator)
                if locator not in seen:
                    ref = Reference(
                        locator=locator,
                        title=source.title or "",
                        url=source.url or "",
                        year=source.year or "",
                        relation=source.relation,
                        origin="retrieved",
                        cited_by=[claim.id],
                    )
                    seen[locator] = ref
                    refs.append(ref)
                elif claim.id not in seen[locator].cited_by:
                    seen[locator].cited_by.append(claim.id)
    return refs


def _bibtex_key(ref: Reference, number: int) -> str:
    first = ref.authors[0].split()[-1] if ref.authors else "ref"
    first = re.sub(r"[^a-z0-9]", "", first.lower()) or "ref"
    return f"{first}{ref.year or number}"


async def build_references(artifact, provided_path=None, resolver: Resolver | None = None) -> list[Reference]:
    by_locator = {ref.locator: ref for ref in collect_retrieved(artifact)}
    if provided_path:
        resolver = resolver or DefaultResolver()
        for provided in await load_provided(provided_path, resolver):
            if provided.locator in by_locator:
                provided.cited_by = by_locator[provided.locator].cited_by
            by_locator[provided.locator] = provided

    refs = sorted(by_locator.values(), key=lambda ref: (0 if ref.cited_by else 1, ref.locator))
    for number, ref in enumerate(refs, start=1):
        ref.number = number
        ref.key = _bibtex_key(ref, number)
    return refs


def markers_for_claim(refs: list[Reference], claim_id: str) -> list[int]:
    return sorted(ref.number for ref in refs if claim_id in ref.cited_by)


def to_bibtex(refs: list[Reference]) -> str:
    blocks = []
    for ref in refs:
        entry_type = "misc" if ref.unresolved or not ref.year else "article"
        fields = []
        if ref.title:
            fields.append(f"  title = {{{ref.title}}}")
        if ref.authors:
            fields.append(f"  author = {{{' and '.join(ref.authors)}}}")
        if ref.year:
            fields.append(f"  year = {{{ref.year}}}")
        if ref.url:
            fields.append(f"  url = {{{ref.url}}}")
        fields.append(f"  note = {{origin={ref.origin}; relation={ref.relation}}}")
        blocks.append(f"@{entry_type}{{{ref.key},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")
