"""CPU-only local LLM using SmolLM2-135M-Instruct via llama-cpp-python.

SmolLM2-135M-Instruct is a small instruct model (~270MB) trained for:
- ChatML format conversations
- Tool calling capability
- JSON output for structured responses

Model is downloaded from HuggingFace on first use:
https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF

ChatML prompt format:
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Model configuration
MODEL_NAME = "SmolLM2-135M-Instruct"
MODEL_FILENAME = "SmolLM2-135M-Instruct-f16.gguf"
HF_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
HF_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{MODEL_FILENAME}"

# ChatML template tokens
CHATML_START = "<|im_start|>"
CHATML_END = "<|im_end|>\n"

# Singleton LLM instance and lock for thread safety
_llm_instance = None
_model_path = None
_llm_lock = threading.Lock()


@dataclass
class ToolCall:
    """Matches obektclaw.llm.ToolCall interface."""

    id: str
    name: str
    arguments: str  # raw JSON string


@dataclass
class TokenUsage:
    """Matches obektclaw.llm.TokenUsage interface."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Matches obektclaw.llm.LLMResponse interface."""

    content: str
    tool_calls: list[ToolCall]
    raw: Any
    usage: TokenUsage | None = None


def _get_model_path() -> Path:
    """Get the path where the model should be stored."""
    home = Path(os.environ.get("OBEKTCLAW_HOME") or Path.home() / ".obektclaw").expanduser()
    model_dir = home / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / MODEL_FILENAME


def _download_model_if_needed() -> Path:
    """Download the model from HuggingFace if not already present.

    Returns:
        Path to the model file
    """
    global _model_path
    if _model_path is not None and _model_path.exists():
        return _model_path

    model_path = _get_model_path()

    if model_path.exists():
        _model_path = model_path
        return model_path

    # Download the model
    print(f"Downloading {MODEL_NAME} from HuggingFace (~270MB F16)...")
    print(f"URL: {HF_URL}")

    import httpx

    try:
        with httpx.stream("GET", HF_URL, follow_redirects=True, timeout=300.0) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to download model: HTTP {resp.status_code}")

            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(model_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\rDownloading: {pct:.1f}%", end="", flush=True)

        print(f"\nModel downloaded to {model_path}")
        _model_path = model_path
        return model_path

    except Exception as e:
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Failed to download model: {e}")


def _get_llm():
    """Get or create the llama-cpp LLM instance.

    Returns:
        Llama instance

    Note: The LLM instance is shared across threads. Use _llm_lock for thread safety.
    """
    global _llm_instance

    with _llm_lock:
        if _llm_instance is not None:
            return _llm_instance

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for local LLM. "
                "Install it with: pip install llama-cpp-python"
            )

        from obektclaw.config import CONFIG

        model_path = _download_model_if_needed()

        # Initialize with CPU-only settings
        _llm_instance = Llama(
            model_path=str(model_path),
            n_ctx=4096,  # Context window for tool calling
            n_batch=512,  # Batch size for prompt processing
            n_threads=4,
            n_threads_batch=4,
            verbose=False,  # Suppress llama.cpp output
        )

        return _llm_instance

    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is required for local LLM. "
            "Install it with: pip install llama-cpp-python"
        )

    from obektclaw.config import CONFIG

    model_path = _download_model_if_needed()

    # Initialize with CPU-only settings
    _llm_instance = Llama(
        model_path=str(model_path),
        n_ctx=4096,  # Context window for tool calling
        n_batch=512,  # Batch size for prompt processing
        n_threads=4,
        n_threads_batch=4,
        verbose=False,  # Suppress llama.cpp output
    )

    return _llm_instance


def _format_chatml_messages(
    messages: list[dict],
    tools: list[dict] | None = None,
) -> str:
    """Format messages using ChatML template with tool support.

    Args:
        messages: List of message dicts with role and content
        tools: Optional list of tool definitions

    Returns:
        Formatted prompt string
    """
    prompt_parts = []

    # System message with tools if provided
    system_content = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
            break

    if tools:
        # Add tool definitions to system message
        tool_desc = _format_tools_for_prompt(tools)
        system_content = f"{system_content}\n\nYou have access to these tools:\n{tool_desc}\n\nWhen you need to use a tool, output a JSON object with 'name' and 'arguments' fields."

    if system_content:
        prompt_parts.append(f"{CHATML_START}system\n{system_content}{CHATML_END}")

    # User and assistant messages
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            prompt_parts.append(f"{CHATML_START}user\n{content}{CHATML_END}")
        elif role == "assistant":
            # Handle tool calls in assistant message
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                # Format tool calls as JSON
                for tc in tool_calls:
                    tc_json = {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": json.loads(
                            tc.get("function", {}).get("arguments", "{}") or "{}"
                        ),
                    }
                    prompt_parts.append(
                        f"{CHATML_START}assistant\n{json.dumps(tc_json)}{CHATML_END}"
                    )
            elif content:
                prompt_parts.append(f"{CHATML_START}assistant\n{content}{CHATML_END}")
        elif role == "tool":
            # Tool result message
            tool_call_id = msg.get("tool_call_id", "")
            prompt_parts.append(f"{CHATML_START}tool\n{content}{CHATML_END}")

    # Final assistant prompt
    prompt_parts.append(f"{CHATML_START}assistant\n")

    return "".join(prompt_parts)


def _format_tools_for_prompt(tools: list[dict]) -> str:
    """Format tool definitions for the prompt.

    Args:
        tools: List of OpenAI-style tool definitions

    Returns:
        Formatted tool descriptions
    """
    descriptions = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        # Format parameters
        param_desc = ""
        if params and "properties" in params:
            props = params["properties"]
            required = params.get("required", [])
            param_parts = []
            for pname, pdef in props.items():
                ptype = pdef.get("type", "any")
                pdesc = pdef.get("description", "")
                req_mark = " (required)" if pname in required else ""
                param_parts.append(f"    {pname}: {ptype}{req_mark} - {pdesc}")
            param_desc = "\n" + "\n".join(param_parts)

        descriptions.append(f"- {name}: {desc}{param_desc}")

    return "\n".join(descriptions)


def _parse_tool_calls(text: str) -> list[ToolCall]:
    """Parse tool calls from model output.

    SmolLM2-Instruct outputs tool calls as JSON objects.
    Looks for JSON with 'name' and 'arguments' fields.

    Args:
        text: Raw model output

    Returns:
        List of ToolCall objects
    """
    tool_calls = []

    # Try to find JSON tool call objects
    text = text.strip()

    # Look for JSON objects in the text
    start_idx = 0
    while start_idx < len(text):
        # Find start of JSON object
        json_start = text.find("{", start_idx)
        if json_start == -1:
            break

        # Find matching closing brace
        depth = 0
        json_end = json_start
        for i in range(json_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

        if json_end <= json_start:
            start_idx = json_start + 1
            continue

        # Parse JSON
        json_str = text[json_start:json_end]
        try:
            obj = json.loads(json_str)
            # Check if it's a tool call (has name field)
            if isinstance(obj, dict) and "name" in obj:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=obj.get("name", ""),
                        arguments=json.dumps(obj.get("arguments", {})),
                    )
                )
        except json.JSONDecodeError:
            pass

        start_idx = json_end

    return tool_calls


def _extract_json_from_response(text: str) -> dict:
    """Extract and parse JSON from model response.

    Handles:
    - Raw JSON
    - JSON wrapped in markdown code blocks
    - JSON with leading/trailing text

    Args:
        text: Raw model output

    Returns:
        Parsed JSON dict or empty dict on failure
    """
    text = text.strip()

    # Try to find JSON in markdown code blocks
    if "```" in text:
        start = text.find("```")
        if start != -1:
            after_start = text.find("\n", start)
            if after_start != -1:
                after_start += 1
            else:
                after_start = start + 3

            end = text.find("```", after_start)
            if end != -1:
                text = text[after_start:end].strip()
            else:
                text = text[after_start:].strip()

    # Try to find JSON object boundaries
    start_idx = text.find("{")
    if start_idx != -1:
        depth = 0
        for i, ch in enumerate(text[start_idx:], start_idx):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = text[start_idx : i + 1]
                    break

    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            return {}
        return result
    except json.JSONDecodeError:
        return {}


class LocalLLMClient:
    """CPU-only LLM client using SmolLM2-135M-Instruct.

    Provides the same interface as obektclaw.llm.LLMClient:
    - chat() with tool calling support
    - chat_simple() for single-shot text
    - chat_json() for structured JSON output
    """

    def __init__(self, model: str = MODEL_NAME, fast_model: str | None = None):
        """Initialize the local LLM client.

        Args:
            model: Model name (for logging, actual model is fixed)
            fast_model: Fast model name (same as model for local)
        """
        self.model = model
        self.fast_model = fast_model or model
        # Don't need API key for local LLM
        self._llm = None

    def _get_llm(self):
        """Lazy load the LLM instance."""
        if self._llm is None:
            self._llm = _get_llm()
        return self._llm

    def chat(
        self,
        messages: list[dict],
        *,
        tools: Iterable[dict] | None = None,
        fast: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Chat with the local LLM, supporting tool calling.

        Args:
            messages: List of message dicts (role, content)
            tools: Optional list of tool definitions
            fast: Use fast model (same as main for local)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with content and tool_calls
        """
        llm = self._get_llm()
        tools_list = list(tools) if tools else None

        # Format prompt with ChatML
        prompt = _format_chatml_messages(messages, tools_list)

        # Generate with thread safety (llama-cpp is not thread-safe)
        with _llm_lock:
            output = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=["<|im_end|>", "<|im_start|>"],
            )

        # Extract text
        if isinstance(output, dict) and "choices" in output:
            text = output["choices"][0].get("text", "")
        else:
            text = str(output)

        text = text.strip()

        # Parse tool calls if tools were provided
        tool_calls = []
        content = text

        if tools_list:
            tool_calls = _parse_tool_calls(text)
            # If we found tool calls, the content is empty or the JSON
            if tool_calls:
                # Remove the JSON from content
                content = ""
                # Or keep a brief response
                if not text.startswith("{"):
                    # There was text before the JSON
                    content = text.split("{")[0].strip()

        # Estimate token usage
        usage = TokenUsage(
            prompt_tokens=len(prompt) // 4,  # Rough estimate
            completion_tokens=len(text) // 4,
            total_tokens=(len(prompt) + len(text)) // 4,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw=output,
            usage=usage,
        )

    def chat_simple(
        self,
        system: str,
        user: str,
        *,
        fast: bool = True,
        temperature: float = 0.3,
    ) -> str:
        """Single-shot system+user, return text."""
        resp = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            fast=fast,
            temperature=temperature,
        )
        return resp.content.strip()

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        fast: bool = True,
    ) -> dict | None:
        """Single-shot expecting JSON output."""
        text = self.chat_simple(
            system + "\n\nReply with a single valid JSON object and nothing else.",
            user,
            fast=fast,
            temperature=0.2,
        )
        return _extract_json_from_response(text)


# ============================================================================
# Learning Loop extraction functions (kept for backward compatibility)
# ============================================================================


def extract_entities_local(
    turn_context: str,
    user_message: str,
    assistant_response: str,
) -> dict | None:
    """Extract entities, relations, and facts from a turn using local LLM.

    This is used by the Learning Loop for entity/fact extraction.

    Args:
        turn_context: Context about the conversation so far
        user_message: The user's message in this turn
        assistant_response: The assistant's response

    Returns:
        Dict with full retro schema, or empty dict if extraction failed
    """
    try:
        client = LocalLLMClient()
    except Exception as e:
        print(f"Failed to load local LLM: {e}")
        return {"entities": [], "relations": [], "facts": []}

    # Extraction prompt optimized for small models
    system_prompt = """You are a memory extraction system. Extract entities, relationships, and facts from the conversation.

Rules:
- Extract entities: tools, concepts, environments, projects, people mentioned
- Extract relations: user preferences (prefers/dislikes), usage (uses), dependencies
- Extract facts: persistent knowledge that should still be true a week later
- Do NOT extract ephemeral info: file paths from one-off questions, counts, temporary state
- Output valid JSON only, no explanation

Entity types: tool, concept, environment, project, person, workflow
Relation types: prefers, uses, dislikes, depends_on, related_to, owns, works_on, deployed_on

Output schema:
{"entities": [{"name": str, "type": str, "confidence": float, "properties": dict}], "relations": [{"subject": str, "predicate": str, "object": str, "confidence": float}], "facts": [{"content": str, "category": str, "confidence": float}]}"""

    user_prompt = f"""Context: {turn_context}

User: {user_message}

Assistant: {assistant_response}

Extract entities, relations, and facts. Return JSON only."""

    result = client.chat_json(system_prompt, user_prompt, fast=True)

    if not result:
        return {"entities": [], "relations": [], "facts": []}

    # Ensure expected keys exist (full retro schema)
    result.setdefault("entities", [])
    result.setdefault("relations", [])
    result.setdefault("facts", [])
    result.setdefault("user_model_updates", [])
    result.setdefault("new_skill", None)
    result.setdefault("skill_improvement", None)
    result.setdefault("notes", "extracted by local CPU LLM")

    return result


def is_local_extraction_enabled() -> bool:
    """Check if local extraction is enabled via config."""
    from obektclaw.config import CONFIG

    return CONFIG.use_local_extraction


def estimate_local_llm_ram() -> int:
    """Estimate RAM usage for local LLM.

    Returns:
        Estimated RAM in MB (~300MB for SmolLM2-135M-F16)
    """
    return 300


def close_local_llm() -> None:
    """Close and release the local LLM instance."""
    global _llm_instance
    if _llm_instance is not None:
        _llm_instance = None


def preload_model() -> bool:
    """Preload the model for faster first extraction.

    Returns:
        True if model loaded successfully, False otherwise
    """
    try:
        _get_llm()
        return True
    except Exception as e:
        print(f"Failed to preload model: {e}")
        return False
