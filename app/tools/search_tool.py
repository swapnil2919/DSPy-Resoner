import asyncio
import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from ..logger_config import logger


async def web_search(query: str, max_results: int = 3) -> dict:
    """Search the web using DuckDuckGo and return real results."""
    logger.info(f"web_search called: {query!r}")

    def _search() -> list:
        # Pin to DuckDuckGo only — ddgs's default backend rotation hits
        # unstable services (Grokipedia) that error out frequently.
        #
        # Two-tier resilience:
        #   1. "No results found" is treated as a legitimate empty result,
        #      not an error. DDG often surfaces rate-limit/empty pages with
        #      this same message; the model handles "no results" much
        #      better than a noisy tool error.
        #   2. Other DDGSExceptions get one retry with a 0.7s backoff —
        #      enough to clear most transient rate-limit blips.
        for attempt in range(2):
            try:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results, backend="duckduckgo"))
            except DDGSException as e:
                if "no results" in str(e).lower():
                    return []
                if attempt == 0:
                    logger.warning(f"ddgs failed (attempt 1/2): {e}; retrying")
                    time.sleep(0.7)
                    continue
                raise
        return []

    try:
        results = await asyncio.to_thread(_search)
        logger.info(f"web_search returned {len(results)} result(s)")
        return {"query": query, "results": results}
    except Exception as exc:
        logger.error("web_search failed", exc_info=True)
        return {"error": str(exc), "results": []}
