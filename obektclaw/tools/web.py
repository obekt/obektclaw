"""Web tools: fetch a URL, return text. No external dependencies — uses httpx."""
from __future__ import annotations

import re

import httpx

from .registry import Tool, ToolContext, ToolRegistry, ToolResult


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
MAX_FETCH_BYTES = 400_000


def web_fetch(args: dict, ctx: ToolContext) -> ToolResult:
    url = args.get("url")
    if not url:
        return ToolResult("missing 'url'", is_error=True)
    strip_html = bool(args.get("strip_html", True))
    try:
        with httpx.Client(follow_redirects=True, timeout=20.0, headers={"User-Agent": "obektclaw/0.1"}) as client:
            resp = client.get(url)
    except httpx.HTTPError as e:
        return ToolResult(f"fetch failed: {e}", is_error=True)
    body = resp.content[:MAX_FETCH_BYTES]
    try:
        text = body.decode(resp.encoding or "utf-8", errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
    if strip_html and "html" in resp.headers.get("content-type", "").lower():
        text = _TAG_RE.sub(" ", text)
        text = _WS_RE.sub(" ", text).strip()
    return ToolResult(f"HTTP {resp.status_code} {resp.headers.get('content-type','')}\n\n{text}",
                      is_error=resp.status_code >= 400)


def register(reg: ToolRegistry) -> None:
    reg.register(Tool(
        name="web_fetch",
        description="HTTP GET a URL and return the (optionally HTML-stripped) body. Capped at 400KB.",
        schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "strip_html": {"type": "boolean"},
            },
            "required": ["url"],
        },
        fn=web_fetch,
        category="web",
    ))
