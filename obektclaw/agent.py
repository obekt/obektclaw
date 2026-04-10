"""The obektclaw agent loop.

Each turn:
  1. Build a system prompt that includes the user model + top relevant
     persistent facts + relevant skill briefs (FTS5-retrieved by user input).
  2. Pull a tail of the recent conversation from session memory.
  3. Call the LLM with the tool catalogue.
  4. If the response has tool_calls, execute them and feed results back.
  5. Loop until the LLM stops calling tools (or we hit the step cap).
  6. Run the Learning Loop on the completed exchange.

The agent is intentionally synchronous and single-threaded; the gateways
serialize requests for a given session.
"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass

from .config import Config
from .llm import LLMClient, LLMResponse, TokenUsage
from .memory import PersistentMemory, SessionMemory, UserModel
from .memory.store import Store
from .skills import SkillManager
from .tools import ToolContext, ToolRegistry, build_default_registry


# ── Context window sizes by model name patterns ────────────────────────────
_CONTEXT_WINDOWS: list[tuple[str, int]] = [
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 8_192),
    ("gpt-3.5", 16_385),
    ("claude-3", 200_000),
    ("claude-2", 100_000),
    ("claude", 200_000),
    ("llama3", 8_192),
    ("llama-3.1", 128_000),
    ("llama-3.2", 128_000),
    ("mistral", 32_000),
    ("mixtral", 32_000),
    ("gemma", 8_192),
    ("qwen", 32_000),
    ("deepseek", 128_000),
    ("command-r", 128_000),
]


def _guess_context_window(model: str) -> int:
    m = model.lower()
    for pattern, size in _CONTEXT_WINDOWS:
        if pattern in m:
            return size
    return 128_000


SYSTEM_PROMPT = """You are obektclaw, a self-improving personal AI agent.

Operating principles:
- You have persistent memory and a library of skills. SEARCH them before
  attempting non-trivial tasks. Use `skill_search` and `memory_search`.
- When you discover a method that worked well and is likely to be reused,
  call `skill_create` to save it. When you find a problem with an existing
  skill, call `skill_improve` to fix it in place.
- When the user reveals a durable preference ("I always use httpx", "my
  prod server is on Hetzner"), call `memory_set_fact` to remember it.
- Prefer the smallest tool for the job. Don't run a 30s grep when a single
  read_file would do.
- Be concise in user-facing replies. Tool results are for you, not the user.
- Never invent file paths or URLs. Verify with a tool first.

You are running in a real environment; tools really execute. Be careful with
destructive operations and confirm with the user before doing anything you
can't easily undo.
"""


@dataclass
class Turn:
    user_text: str
    assistant_text: str
    tool_steps: int


class Agent:
    def __init__(
        self,
        *,
        config: Config,
        store: Store,
        skills: SkillManager,
        registry: ToolRegistry | None = None,
        llm: LLMClient | None = None,
        gateway: str = "cli",
        user_key: str = "default",
        run_learning_loop: bool = True,
        load_mcp: bool = True,
    ):
        self.config = config
        self.store = store
        self.skills = skills
        self.registry = registry or build_default_registry()
        self.llm = llm or LLMClient(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            fast_model=config.llm_fast_model,
        )
        self.gateway = gateway
        self.user_key = user_key
        self.run_learning_loop_flag = run_learning_loop
        self._mcp_servers: list | None = None

        self.session_id = store.open_session(gateway, user_key)
        self.session = SessionMemory(store, self.session_id)
        self.persistent = PersistentMemory(store)
        self.user_model = UserModel(store)

        # Auto-load MCP servers if config exists
        if load_mcp:
            mcp_config_path = config.home / "mcp.json"
            if mcp_config_path.exists():
                try:
                    from .mcp import attach_mcp_servers, load_mcp_config
                    specs = load_mcp_config(mcp_config_path)
                    if specs:
                        self._mcp_servers = attach_mcp_servers(self.registry, specs)
                except Exception as e:  # noqa: BLE001
                    msg = f"MCP failed to load: {e}"
                    print(f"[warning] {msg}", file=sys.stderr)
                    self.session.add("system", msg)

        # Context window management — explicit config wins, else guess from model name
        self.context_window = config.context_window or _guess_context_window(config.llm_model)
        self.last_usage: TokenUsage | None = None
        self.turn_tokens: int = 0  # total tokens used in current turn

        # Log MCP success if servers were loaded
        if self._mcp_servers:
            self.session.add("system", f"MCP: {len(self._mcp_servers)} server(s) loaded")

    # ----- public API -----
    def run_once(self, user_text: str, *, max_steps: int = 12) -> str:
        """Run a single user turn to completion and return the assistant's reply."""
        self.session.add("user", user_text)

        messages = self._build_messages(user_text)
        tool_steps = 0
        assistant_text = ""
        self.turn_tokens = 0

        for _ in range(max_steps):
            # Truncate mid-turn if context is getting full
            if self._context_pressure() > 0.75:
                messages = self._truncate_messages(messages)

            try:
                resp: LLMResponse = self.llm.chat(messages, tools=self.registry.to_openai_tools())
            except RuntimeError as e:
                err = str(e).lower()
                if "context" in err or "token" in err or "length" in err or "maximum" in err:
                    # Context overflow — truncate aggressively and retry once
                    messages = self._truncate_messages(messages)
                    try:
                        resp = self.llm.chat(messages, tools=self.registry.to_openai_tools())
                    except RuntimeError:
                        assistant_text = (
                            "(context window full — please start a new conversation "
                            "or ask a shorter question. Your memories and skills are preserved.)"
                        )
                        break
                else:
                    raise

            if resp.usage:
                self.last_usage = resp.usage
                self.turn_tokens += resp.usage.total_tokens

            assistant_msg: dict = {"role": "assistant", "content": resp.content or ""}
            if resp.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in resp.tool_calls
                ]
            messages.append(assistant_msg)

            if resp.content:
                assistant_text = resp.content

            if not resp.tool_calls:
                break

            ctx = self._tool_context()
            for tc in resp.tool_calls:
                tool_steps += 1
                result = self.registry.call(tc.name, tc.arguments, ctx)
                self.session.add(
                    "tool",
                    result.content,
                    tool_name=tc.name,
                    meta={"args": tc.arguments, "is_error": result.is_error},
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.content[:8000],
                })
        else:
            assistant_text = (assistant_text or "") + "\n\n(stopped: hit max tool steps)"

        if assistant_text:
            self.session.add("assistant", assistant_text)

        if self.run_learning_loop_flag:
            turn = Turn(user_text, assistant_text, tool_steps)
            def _run_learning_loop():
                try:
                    from .learning import LearningLoop
                    LearningLoop(self).run(turn)
                except Exception as e:  # noqa: BLE001
                    self.session.add("system", f"learning loop failed: {e}")
            threading.Thread(target=_run_learning_loop, daemon=True).start()

        return assistant_text or "(no answer)"

    def close(self) -> None:
        self.store.close_session(self.session_id)
        # Shut down MCP servers if any were loaded
        if self._mcp_servers:
            for srv in self._mcp_servers:
                try:
                    srv.stop()
                except Exception:
                    pass

    # ----- context management -----
    def _context_pressure(self) -> float:
        """Return 0.0–1.0 indicating how full the context window is."""
        if not self.last_usage:
            return 0.0
        used = self.last_usage.prompt_tokens + self.last_usage.completion_tokens
        return min(used / self.context_window, 1.0) if self.context_window else 0.0

    def _truncate_messages(self, messages: list[dict]) -> list[dict]:
        """Drop older conversation turns (keeping system + latest user) when
        context pressure is high. Uses a simple strategy: keep the system
        message, drop the oldest user/assistant pairs first, always keep the
        last user message.

        Adds a summary marker so the agent knows context was trimmed.
        """
        if len(messages) <= 3:
            return messages  # system + at most 1 exchange — nothing to drop

        # Reserve: system (first), current user (last)
        system_msg = messages[0]
        current_user = messages[-1]
        middle = messages[1:-1]

        # Drop the oldest half of middle messages
        keep = max(len(middle) // 2, 2)
        trimmed_middle = middle[-keep:]

        summary = {
            "role": "system",
            "content": (
                "[context trimmed: older messages were dropped to stay within the "
                "context window. Key facts are preserved in persistent memory.]"
            ),
        }
        return [system_msg, summary] + trimmed_middle + [current_user]

    # ----- prompt assembly -----
    def _build_messages(self, user_text: str) -> list[dict]:
        sys = self._compose_system_prompt(user_text)
        msgs: list[dict] = [{"role": "system", "content": sys}]

        # Scale history window based on context pressure
        pressure = self._context_pressure()
        if pressure > 0.8:
            history_limit = 8
        elif pressure > 0.6:
            history_limit = 16
        else:
            history_limit = 30

        # Recent in-session history (excluding the user message we just added,
        # which we'll re-add at the end so it lands at the bottom).
        recent = self.session.recent(limit=history_limit)
        for r in recent[:-1]:  # everything except the just-added user msg
            if r.role == "user":
                msgs.append({"role": "user", "content": r.content})
            elif r.role == "assistant":
                msgs.append({"role": "assistant", "content": r.content})
            elif r.role == "tool":
                # Skip raw tool turns from history — too noisy. The system
                # prompt's recall section is enough context.
                continue

        msgs.append({"role": "user", "content": user_text})
        return msgs

    def _compose_system_prompt(self, user_text: str) -> str:
        parts = [SYSTEM_PROMPT.strip()]

        # User model
        traits = self.user_model.render_for_prompt()
        parts.append(f"## What I know about the user\n{traits}")

        # Top persistent facts (a small fixed budget)
        facts = self.persistent.all_top(per_category=4)
        if facts:
            facts_block = "\n".join(f.render() for f in facts[:24])
            parts.append(f"## Persistent facts\n{facts_block}")

        # Always include a list of all available skills (cheap self-discovery)
        all_skills = self.skills.list_all()
        if all_skills:
            # Cap at 30 skills to keep token usage reasonable
            skills_list = [s.render_brief() for s in all_skills[:30]]
            if len(all_skills) > 30:
                skills_list.append(f"... and {len(all_skills) - 30} more (search with `skill_search`)")
            parts.append("## All available skills (load with `skill_load` if useful)\n" + "\n".join(skills_list))

        # FTS5 recall against the user's input (most relevant)
        relevant_skills = self.skills.search(user_text, limit=4)
        if relevant_skills:
            sk_block = "\n".join(s.render_brief() for s in relevant_skills)
            parts.append(
                "## Most relevant skills for this query\n" + sk_block
            )

        relevant_msgs = self.session.search_history(user_text, limit=4)
        if relevant_msgs:
            rec_block = "\n".join("- " + m.render() for m in relevant_msgs)
            parts.append(f"## Possibly relevant prior exchanges\n{rec_block}")

        return "\n\n".join(parts)

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            config=self.config,
            session=self.session,
            persistent=self.persistent,
            user_model=self.user_model,
            skills=self.skills,
            llm=self.llm,
        )
