"""Thin wrapper around the OpenAI Python SDK pointed at any compatible endpoint.

We assume the chat-completions API and OpenAI-style tool calling. Works with:
  - OpenAI itself
  - OpenRouter (https://openrouter.ai/api/v1)
  - vLLM with --enable-auto-tool-choice
  - Anthropic via OpenAI-compatible adapters
  - Ollama (http://localhost:11434/v1) for models that support tool use

Two client types:
  - LLMClient: Main agent for conversations and tool calling
  - ExtractionLLMClient: Entity/relationship extraction for Learning Loop (isolated context)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

from .logging_config import get_logger
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError

log = get_logger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string the model emitted


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall]
    raw: Any
    usage: TokenUsage | None = None


class LLMClient:
    def __init__(
        self, base_url: str, api_key: str, model: str, fast_model: str | None = None
    ):
        if not api_key:
            raise RuntimeError(
                "OBEKTCLAW_LLM_API_KEY is not set — copy .env.example to .env and fill it in"
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.fast_model = fast_model or model

    def chat(
        self,
        messages: list[dict],
        *,
        tools: Iterable[dict] | None = None,
        fast: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.fast_model if fast else self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        tools_list = list(tools) if tools else None
        if tools_list:
            kwargs["tools"] = tools_list
            kwargs["tool_choice"] = "auto"

        last_err: Exception | None = None
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                log.info(
                    "llm_call model=%s tokens=%d attempts=%d",
                    self.model,
                    resp.usage.total_tokens if resp.usage else 0,
                    attempt + 1,
                )
                break
            except (RateLimitError, APIConnectionError, APIError) as e:
                last_err = e
                log.warning(
                    "llm_call_error model=%s attempt=%d error=%s",
                    self.model,
                    attempt + 1,
                    e,
                )
                time.sleep(min(2**attempt, 8))
        else:
            log.error("llm_call_failed model=%s retries_exhausted", self.model)
            raise RuntimeError(f"LLM call failed after retries: {last_err}")

        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments or "{}",
                )
            )
        usage = None
        if resp.usage:
            usage = TokenUsage(
                prompt_tokens=resp.usage.prompt_tokens or 0,
                completion_tokens=resp.usage.completion_tokens or 0,
                total_tokens=resp.usage.total_tokens or 0,
            )
        return LLMResponse(content=content, tool_calls=calls, raw=resp, usage=usage)

    def chat_simple(
        self, system: str, user: str, *, fast: bool = True, temperature: float = 0.3
    ) -> str:
        """Convenience: single-shot system+user, return text."""
        resp = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            fast=fast,
            temperature=temperature,
        )
        return resp.content.strip()

    def chat_json(self, system: str, user: str, *, fast: bool = True) -> dict | None:
        """Single-shot expecting a JSON object back. Returns None on parse failure."""
        text = self.chat_simple(
            system + "\n\nReply with a single valid JSON object and nothing else.",
            user,
            fast=fast,
            temperature=0.2,
        )
        # try to peel a code fence if the model added one
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # last-ditch: extract the outermost {...}
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None


class ExtractionLLMClient:
    """LLM client for entity/relationship extraction in Learning Loop.

    Uses separate config (falls back to main LLM if not specified).
    Has isolated context/history separate from the main agent conversation.
    Used for structured JSON extraction, not tool calling.
    """

    def __init__(self, config: "Config"):
        """Initialize extraction LLM client with config fallbacks.

        Args:
            config: Config with extraction LLM settings (falls back to main LLM)
        """
        # Resolve extraction config with fallbacks to main LLM
        base_url = config.extraction_llm_base_url or config.llm_base_url
        api_key = config.extraction_llm_api_key or config.llm_api_key
        # Model fallback: extraction -> fast_model -> main_model
        model = config.extraction_llm_model or config.llm_fast_model or config.llm_model

        if not api_key:
            raise RuntimeError(
                "OBEKTCLAW_LLM_API_KEY (or OBEKTCLAW_EXTRACTION_LLM_API_KEY) is not set"
            )

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url

        # Isolated context history (separate from main agent)
        self._context_history: list[dict] = []

        log.info(
            "extraction_llm_initialized model=%s base_url=%s",
            model,
            base_url,
        )

    def extract(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> dict | None:
        """Run extraction and return parsed JSON.

        Uses isolated context - no shared history with main agent.
        """
        # Build messages with isolated context
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_err: Exception | None = None
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                log.info(
                    "extraction_llm_call model=%s tokens=%d attempt=%d",
                    self.model,
                    resp.usage.total_tokens if resp.usage else 0,
                    attempt + 1,
                )
                break
            except (RateLimitError, APIConnectionError, APIError) as e:
                last_err = e
                log.warning(
                    "extraction_llm_error model=%s attempt=%d error=%s",
                    self.model,
                    attempt + 1,
                    e,
                )
                time.sleep(min(2**attempt, 8))
        else:
            log.error("extraction_llm_failed model=%s retries_exhausted", self.model)
            raise RuntimeError(f"Extraction LLM call failed after retries: {last_err}")

        content = resp.choices[0].message.content or ""

        # Parse JSON from response
        return self._parse_json(content)

    def _parse_json(self, text: str) -> dict | None:
        """Parse JSON from LLM response, handling code fences and extraction."""
        text = text.strip()
        # Strip code fence if present
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last-ditch: extract outermost {...}
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    log.warning("extraction_llm_json_parse_failed text=%s", text[:200])
                    return None
            log.warning("extraction_llm_no_json_found text=%s", text[:200])
            return None

    def close(self) -> None:
        """Close the client (no persistent resources to clean up)."""
        self._context_history.clear()
