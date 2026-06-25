"""Spec 3 grounding — locator -> source text (network; agent layer only, never the sandbox). v1 fetches the
ABSTRACT (arXiv summary / crossref abstract / URL text); full-text PDF is a v1.x lever (raw PDF text breaks
the verbatim-quote check). Isolated so tests inject a fake. Fail-soft: any error -> None."""
from __future__ import annotations
from valagents.references import detect_kind, normalize_id


async def fetch_source_text(locator: str) -> tuple[str, dict] | None:
    """The real (live) network fetch. Wrapped by LiveFetcher; tests never call this — they inject a fake
    fetcher with the same `async fetch(locator)` contract."""
    kind = detect_kind(locator)
    try:
        if kind == "arxiv":
            import asyncio, arxiv, re
            m = re.search(r"(\d{4}\.\d{4,5})", locator)
            if not m:
                return None
            res = await asyncio.to_thread(list, arxiv.Client().results(arxiv.Search(id_list=[m.group(1)])))
            if not res:
                return None
            r = res[0]
            return (f"{r.title}\n{r.summary}",
                    {"locator": normalize_id(locator), "title": r.title, "url": r.entry_id, "year": str(r.published)[:4]})
        if kind == "doi":
            import httpx, re
            doi = re.search(r"10\.\d{1,9}/[^\s]+", locator).group(0).rstrip(".,;)")
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"https://api.crossref.org/works/{doi}", timeout=15)
                resp.raise_for_status()
                msg = resp.json()["message"]
            text = (msg.get("title") or [""])[0] + "\n" + (msg.get("abstract") or "")
            return (text, {"locator": doi.lower(), "title": (msg.get("title") or [""])[0],
                           "url": f"https://doi.org/{doi}", "year": ""})
        if "http" in locator:
            import httpx, re
            async with httpx.AsyncClient() as client:
                resp = await client.get(locator, timeout=15)
                resp.raise_for_status()
                text = re.sub(r"<[^>]+>", " ", resp.text)
            return (text, {"locator": locator, "title": "", "url": locator, "year": ""})
    except Exception:
        return None
    return None
