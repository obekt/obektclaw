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
from .logging_config import get_logger
from .memory import PersistentMemory, SessionMemory, UserModel
from .memory.store import Store
from .model_context import get_context_window, save_user_model_override
from .skills import SkillManager
from .tools import ToolContext, ToolRegistry, build_default_registry

log = get_logger(__name__)


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
        status_fn: callable | None = None,
        session_id: int | None = None,
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
        self._status_fn = status_fn
        self._resumed = session_id is not None

        if session_id is not None:
            # Resume an existing session
            self.session_id = session_id
        else:
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

        # Context window management — explicit config wins, else detect from model name
        # Detection checks: user overrides → built-in exact → built-in patterns → default
        detected_window = get_context_window(config.llm_model, config.home)
        self.context_window = config.context_window or detected_window
        self.last_usage: TokenUsage | None = None
        self.turn_tokens: int = 0  # total tokens used in current turn

        # Log context window detection at startup
        source = "config" if config.context_window else "auto-detected"
        log.info("model=%s context_window=%d tokens=%s", config.llm_model, self.context_window, source)

        # Log MCP success if servers were loaded
        if self._mcp_servers:
            log.info("mcp_servers_loaded=%d", len(self._mcp_servers))
            self.session.add("system", f"MCP: {len(self._mcp_servers)} server(s) loaded")

    # ----- public API -----
    def _status(self, message: str) -> None:
        """Fire a status update to the gateway callback, if registered."""
        if self._status_fn:
            self._status_fn(message)

    def run_once(
        self,
        user_text: str,
        *,
        max_steps: int = 12,
        status_fn: callable | None = None,
    ) -> str:
        """Run a single user turn to completion and return the assistant's reply.

        Args:
            user_text: The user's message.
            max_steps: Maximum tool-use steps before stopping.
            status_fn: Optional callback fired with status strings during execution.
                       Called with "thinking...", "using <tool>", then "" when done.
        """
        self.session.add("user", user_text)

        messages = self._build_messages(user_text)
        tool_steps = 0
        assistant_text = ""
        self.turn_tokens = 0

        # Use the per-call callback, falling back to the instance default
        _fn = status_fn or self._status_fn
        if _fn:
            _fn("thinking...")

        for _ in range(max_steps):
            # Auto-compact context at 85% pressure (before truncation kicks in)
            if self._context_pressure() > self.COMPACTION_PRESSURE:
                self._status("compacting context...")
                compact_result = self.compact_context()
                if compact_result["compacted"]:
                    log.info("context_compacted turns=%d summary_words=%d tokens_saved=%d",
                             len(to_compact), compact_result["summary_length"], compact_result["tokens_saved"])
                    # Rebuild messages after compaction, but preserve the current
                    # turn's in-flight assistant/tool messages (they're not in
                    # session memory yet).
                    turn_tail: list[dict] = []
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            turn_tail = messages[i + 1 :]
                            break
                    messages = self._build_messages(user_text)
                    messages.extend(turn_tail)
                elif self._context_pressure() > 0.75:
                    # Compaction failed or didn't happen, fall back to truncation
                    messages = self._truncate_messages(messages)
                    log.warning("context_truncated pressure=%.2f", pressure)

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
                if _fn:
                    _fn(f"using {tc.name}")
                log.debug("tool_call tool=%s args=%s", tc.name, tc.arguments[:200])
                result = self.registry.call(tc.name, tc.arguments, ctx)
                log.info("tool_result tool=%s error=%s", tc.name, result.is_error)
                if result.is_error:
                    log.warning("tool_error tool=%s msg=%s", tc.name, result.content[:500])
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
            log.info("turn_complete user_len=%d tool_steps=%d tokens=%d",
                     len(user_text), tool_steps, self.turn_tokens)
            def _run_learning_loop():
                try:
                    from .learning import LearningLoop
                    LearningLoop(self).run(turn)
                except Exception as e:  # noqa: BLE001
                    self.session.add("system", f"learning loop failed: {e}")
            threading.Thread(target=_run_learning_loop, daemon=True).start()

        if _fn:
            _fn("")  # Clear status — we're done
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

    def switch_model(
        self,
        model: str,
        fast_model: str | None = None,
        context_window: int | None = None,
        persist: bool = True,
    ) -> dict:
        """Switch to a different model at runtime.
        
        Args:
            model: New model name for main reasoning/tool use.
            fast_model: New model name for Learning Loop (defaults to model).
            context_window: Override context window size (0 = auto-detect).
            persist: If True, save to ~/.obektclaw/models.json for future sessions.
        
        Returns:
            Dict with keys: model, fast_model, context_window, was_overridden
        """
        # Update LLM client
        fast = fast_model or model
        self.llm = LLMClient(
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            model=model,
            fast_model=fast,
        )
        
        # Determine context window
        if context_window and context_window > 0:
            # Explicit override
            new_window = context_window
            was_overridden = True
            if persist:
                save_user_model_override(self.config.home, model, context_window)
        else:
            # Auto-detect
            new_window = get_context_window(model, self.config.home)
            was_overridden = False
        
        self.context_window = new_window
        
        # Log the switch
        log.info("model_switched from=%s to=%s context_window=%d overridden=%s",
                 self.config.llm_model, model, new_window, was_overridden)
        self.session.add(
            "system",
            f"Model switched from {self.config.llm_model} → {model} "
            f"(context: {new_window:,} tokens)",
        )
        
        return {
            "model": model,
            "fast_model": fast,
            "context_window": new_window,
            "was_overridden": was_overridden,
        }

    # ----- context management -----

    # Compaction thresholds
    COMPACTION_PRESSURE = 0.85  # Auto-compact at 85% context usage
    COMPACTION_KEEP_TURNS = 6   # Keep last 6 turns (12 messages) raw
    COMPACTION_MAX_SUMMARY = 1000  # Max tokens for summary

    def _context_pressure(self) -> float:
        """Return 0.0–1.0 indicating how full the context window is."""
        if not self.last_usage:
            return 0.0
        # prompt_tokens reflects the current context size sent to the model
        used = self.last_usage.prompt_tokens
        return min(used / self.context_window, 1.0) if self.context_window else 0.0

    def compact_context(self, *, force: bool = False) -> dict:
        """Compact conversation history using LLM summarization.

        Reads the older conversation turns, generates a concise summary
        preserving goals/decisions/context, and replaces them with a single
        system message. Keeps recent turns raw for continuity.

        Args:
            force: If True, compact regardless of context pressure.

        Returns:
            Dict with keys: compacted (bool), reason (str), summary_length (int),
                           tokens_saved (int), error (str|None)
        """
        # Check if compaction is needed
        pressure = self._context_pressure()
        if not force and pressure < self.COMPACTION_PRESSURE:
            return {
                "compacted": False,
                "reason": f"Context pressure too low ({pressure:.0%} < {self.COMPACTION_PRESSURE:.0%})",
                "summary_length": 0,
                "tokens_saved": 0,
                "error": None,
            }

        # Get recent session history
        recent = self.session.recent(limit=100)
        if len(recent) <= self.COMPACTION_KEEP_TURNS * 2 + 2:
            return {
                "compacted": False,
                "reason": "Conversation too short to compact",
                "summary_length": 0,
                "tokens_saved": 0,
                "error": None,
            }

        # Split: keep recent turns raw, compact the rest
        keep_count = self.COMPACTION_KEEP_TURNS * 2  # user + assistant pairs
        to_compact = recent[:-keep_count] if keep_count < len(recent) else []

        if not to_compact:
            return {
                "compacted": False,
                "reason": "No old messages to compact",
                "summary_length": 0,
                "tokens_saved": 0,
                "error": None,
            }

        # Estimate tokens to compact
        old_tokens = sum(len(msg.content.split()) * 1.3 for msg in to_compact)

        # Build compacting prompt
        compact_prompt = (
            "Summarize this conversation history concisely. Preserve:\n"
            "- User's goals, requests, and preferences\n"
            "- Key decisions made and why\n"
            "- Important context about files, code, or environment\n"
            "- Any unresolved issues or ongoing work\n\n"
            "Omit: pleasantries, repetitive turns, resolved debugging.\n"
            "Be terse. Max 500 words.\n\n"
            "Conversation to summarize:\n"
        )

        messages_to_summarize = [
            {"role": msg.role, "content": msg.content}
            for msg in to_compact
            if msg.role in ("user", "assistant", "system")
        ]

        if not messages_to_summarize:
            return {
                "compacted": False,
                "reason": "No user/assistant messages to summarize",
                "summary_length": 0,
                "tokens_saved": 0,
                "error": None,
            }

        try:
            # Use fast model for compaction (saves cost)
            summary_resp = self.llm.chat(
                [
                    {"role": "system", "content": compact_prompt},
                    *messages_to_summarize,
                ],
                fast=True,
                max_tokens=self.COMPACTION_MAX_SUMMARY,
            )

            summary = summary_resp.content.strip()
            if not summary:
                return {
                    "compacted": False,
                    "reason": "LLM returned empty summary",
                    "summary_length": 0,
                    "tokens_saved": 0,
                    "error": None,
                }

            # Delete old messages from session memory
            for msg in to_compact:
                self.store.execute(
                    "DELETE FROM messages WHERE id = ?",
                    (msg.id,),
                )

            # Insert summary as a system message
            summary_msg = self.store.fetchone(
                """
                INSERT INTO messages (session_id, role, content, ts)
                VALUES (?, 'system', ?, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                (self.session_id, f"[compacted conversation summary]\n{summary}"),
            )

            # Log the compaction
            self.session.add(
                "system",
                f"Context compacted: {len(to_compact)} turns → summary "
                f"(~{int(old_tokens)} tokens → {len(summary.split())} tokens)",
            )

            return {
                "compacted": True,
                "reason": "Success",
                "summary_length": len(summary.split()),
                "tokens_saved": int(old_tokens - len(summary.split())),
                "error": None,
            }

        except Exception as e:
            return {
                "compacted": False,
                "reason": "Compaction failed",
                "summary_length": 0,
                "tokens_saved": 0,
                "error": str(e),
            }

    def _truncate_messages(self, messages: list[dict]) -> list[dict]:
        """Drop older conversation turns (keeping system + current turn) when
        context pressure is high. Uses a simple strategy: keep the system
        message, drop the oldest user/assistant pairs first, always keep the
        entire current turn (last user message + any assistant/tool messages).

        Adds a summary marker so the agent knows context was trimmed.
        """
        if len(messages) <= 3:
            return messages  # system + at most 1 exchange — nothing to drop

        # Find the last user message — that's the start of the current turn
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None or last_user_idx <= 1:
            return messages  # No user message or too short to trim

        system_msg = messages[0]
        current_turn = messages[last_user_idx:]  # user + assistant + tool results
        middle = messages[1:last_user_idx]

        # Drop the oldest half of middle messages
        keep = max(len(middle) // 2, 2)
        trimmed_middle = middle[-keep:] if len(middle) > keep else middle

        summary = {
            "role": "system",
            "content": (
                "[context trimmed: older messages were dropped to stay within the "
                "context window. Key facts are preserved in persistent memory.]"
            ),
        }
        return [system_msg, summary] + trimmed_middle + current_turn

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
