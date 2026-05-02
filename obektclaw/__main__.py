"""Top-level CLI dispatcher: `python -m obektclaw <subcommand>`."""

from __future__ import annotations

import json
import sys

from .config import CONFIG
from .logging_config import get_logger
from .memory.store import Store
from .skills import SkillManager

log = get_logger(__name__)


USAGE = """usage: python -m obektclaw [command] [args]

commands:
  start [mode]               start obektclaw (auto-detects gateways)
                             mode: "auto" (default) | "cli" | "tg"
  setup                      interactive setup wizard

  sessions list              list recent sessions
  sessions show <id>         show session details
  sessions export <id>       export session (--format md|json, --output file)
  sessions resume <id>       resume a past session in CLI

  skill list                 list known skills
  skill show <name>          print a skill body

  memory recent              dump recent messages
  memory search <query>      search persistent facts
  memory cleanup             remove stale facts
  memory status              check memory system health

  traits                     show your user model
  help                       show detailed help

First time? Run: python -m obektclaw start
"""


def _open() -> tuple[Store, SkillManager]:
    store = Store(CONFIG.db_path)
    skills = SkillManager(store, CONFIG.skills_dir, CONFIG.bundled_skills_dir)
    return store, skills


def _start_auto(mode: str | None = None) -> int:
    """Start obektclaw with auto-detected gateways.

    mode: "auto" (default), "cli", or "tg".
    """
    import threading
    from .config import CONFIG as _cfg
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    mode = (mode or "auto").lower()
    want_cli = mode in ("auto", "cli")
    want_tg = mode in ("auto", "tg") and _cfg.tg_token

    log.info("gateway_start mode=%s cli=%s telegram=%s", mode, want_cli, want_tg)

    if mode == "auto":
        # Announce what's starting
        gateways = []
        if want_cli:
            gateways.append("CLI")
        if want_tg:
            gateways.append("Telegram")
        if not gateways:
            gateways.append("CLI")  # fallback — always start something

        console.print(
            Panel(
                f"Starting obektclaw — {', '.join(gateways)} gateway(s)",
                style="bold cyan",
                padding=(0, 1),
            )
        )
        console.print()

        if not _cfg.tg_token:
            console.print("○ [dim]Telegram[/dim] — not configured")
            console.print("  Set OBEKTCLAW_TG_TOKEN in .env to enable it")

    # Start Telegram in background if requested
    if want_tg:

        def _run_telegram():
            from .gateways.telegram import run as tg_run

            tg_run()

        tg_thread = threading.Thread(
            target=_run_telegram, daemon=True, name="telegram-gateway"
        )
        tg_thread.start()
        console.print("✓ [green]Telegram gateway[/green] — running")

    # Start CLI in main thread (blocks until user exits)
    if want_cli:
        console.print("✓ [green]CLI gateway[/green] — running")
        console.print()
        try:
            from .gateways.cli import run as cli_run

            return cli_run()
        except KeyboardInterrupt:
            return 0

    # Telegram-only mode — block until interrupted or thread dies
    if want_tg:
        try:
            while tg_thread.is_alive():
                tg_thread.join(timeout=1)
        except KeyboardInterrupt:
            pass
        return 0

    return 0


def _resume_session(session_id: int, info: "SessionSummary") -> int:
    """Resume an old session in CLI mode."""
    from .gateways.cli import run_with_session

    return run_with_session(session_id, info)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    cmd, *rest = argv

    if cmd == "start":
        return _start_auto(rest[0] if rest else None)

    # Legacy aliases — route through the single entry point
    if cmd == "chat":
        return _start_auto("cli")

    if cmd == "tg":
        return _start_auto("tg")

    if cmd == "run":
        return _start_auto("auto")

    if cmd == "setup":
        # Interactive setup wizard
        print(
            """
╔═══════════════════════════════════════════════════════════╗
║                    obektclaw Setup                        ║
╚═══════════════════════════════════════════════════════════╝

Configuration:
  OBEKTCLAW_HOME: {}
  Database: {}
  Skills: {}

""".format(CONFIG.home, CONFIG.db_path, CONFIG.skills_dir)
        )

        # Check MCP
        mcp_config = CONFIG.home / "mcp.json"
        if mcp_config.exists():
            print("✓ MCP servers: configured")
        else:
            print("○ MCP servers: not configured")
            print("  Create {} to add external tools".format(mcp_config))

        # Check Telegram
        if CONFIG.tg_token:
            print("✓ Telegram: configured (run: python -m obektclaw tg)")
        else:
            print("○ Telegram: not configured")
            print("  Steps:")
            print("  1. Open Telegram and message @BotFather")
            print("  2. Send: /newbot")
            print("  3. Follow prompts to create bot")
            print("  4. Copy the API token")
            print("  5. Edit .env and add: OBEKTCLAW_TG_TOKEN=your_token")
            print("  6. Run: python -m obektclaw tg")
        print()
        return 0

    if cmd == "sessions":
        from .sessions import (
            list_sessions,
            get_session_info,
            get_session_messages,
            export_session_markdown,
            export_session_json,
        )

        store, skills = _open()
        try:
            if not rest:
                print(
                    "usage: sessions list | sessions show <id> | sessions export <id> | sessions resume <id>"
                )
                return 1
            sub = rest[0]

            if sub == "list":
                sessions = list_sessions(store, limit=20)
                if not sessions:
                    print("(no sessions)")
                    return 0
                print(
                    f"{'ID':>5}  {'Started':<17}  {'Dur':>5}  {'GW':<9}  {'Msgs':>4}  Preview"
                )
                print("-" * 80)
                for s in sessions:
                    print(
                        f"{s.id:>5}  {s.started_str:<17}  {s.duration_str:>5}  "
                        f"{s.gateway:<9}  {s.message_count:>4}  {s.preview}"
                    )
                return 0

            if sub == "show":
                if len(rest) < 2:
                    print("usage: sessions show <id>")
                    return 1
                try:
                    sid = int(rest[1])
                except ValueError:
                    print(f"invalid session id: {rest[1]}")
                    return 1
                info = get_session_info(store, sid)
                if info is None:
                    print(f"no such session: {sid}")
                    return 1
                print(f"Session #{info.id}")
                print(f"  Started:  {info.started_str}")
                if info.ended_str:
                    print(f"  Ended:    {info.ended_str}")
                print(f"  Duration: {info.duration_str}")
                print(f"  Gateway:  {info.gateway}")
                print(f"  User:     {info.user_key}")
                print(f"  Messages: {info.message_count}")
                print()
                messages = get_session_messages(store, sid)
                for msg in messages:
                    role = msg.tool_name or msg.role
                    content = msg.content[:200]
                    if len(msg.content) > 200:
                        content += "..."
                    print(f"  [{msg.ts_str}] {role}: {content}")
                return 0

            if sub == "export":
                if len(rest) < 2:
                    print(
                        "usage: sessions export <id> [--format md|json] [--output file]"
                    )
                    return 1
                try:
                    sid = int(rest[1])
                except ValueError:
                    print(f"invalid session id: {rest[1]}")
                    return 1
                # Parse optional flags
                fmt = "md"
                output_path = None
                i = 2
                while i < len(rest):
                    if rest[i] == "--format" and i + 1 < len(rest):
                        fmt = rest[i + 1]
                        i += 2
                    elif rest[i] == "--output" and i + 1 < len(rest):
                        output_path = rest[i + 1]
                        i += 2
                    else:
                        i += 1
                if fmt not in ("md", "json"):
                    print(f"unknown format: {fmt}  (use md or json)")
                    return 1
                if fmt == "md":
                    result = export_session_markdown(store, sid)
                else:
                    data = export_session_json(store, sid)
                    result = (
                        json.dumps(data, indent=2, ensure_ascii=False) if data else None
                    )
                if result is None:
                    print(f"no such session: {sid}")
                    return 1
                if output_path:
                    from pathlib import Path

                    Path(output_path).write_text(result)
                    print(f"exported session {sid} to {output_path}")
                else:
                    print(result)
                return 0

            if sub == "resume":
                if len(rest) < 2:
                    print("usage: sessions resume <id>")
                    return 1
                try:
                    sid = int(rest[1])
                except ValueError:
                    print(f"invalid session id: {rest[1]}")
                    return 1
                info = get_session_info(store, sid)
                if info is None:
                    print(f"no such session: {sid}")
                    return 1
                # Close this store — the CLI gateway will open its own
                store.close()
                return _resume_session(sid, info)

            print(f"unknown sessions subcommand: {sub}")
            return 1
        finally:
            store.close()

    if cmd == "skill":
        store, skills = _open()
        try:
            if not rest:
                print("usage: skill list | skill show <name>")
                return 1
            sub = rest[0]
            if sub == "list":
                for sk in skills.list_all():
                    print(sk.render_brief())
                return 0
            if sub == "show":
                if len(rest) < 2:
                    print("usage: skill show <name>")
                    return 1
                sk = skills.get(rest[1])
                if sk is None:
                    print(f"no such skill: {rest[1]}")
                    return 1
                print(sk.render())
                return 0
            print(f"unknown skill subcommand: {sub}")
            return 1
        finally:
            store.close()

    if cmd == "memory":
        from .memory import VectorMemory, GraphMemory, MemorySync
        from .memory.graph_memory import ENTITY_TYPES
        from .llm import LLMClient

        store, _skills = _open()
        try:
            if not rest:
                print(
                    "usage: memory recent | memory search <q> | memory cleanup | memory status"
                )
                return 1
            sub = rest[0]
            if sub == "status":
                # Memory system diagnostic (NEW: graph + vector)
                print("""
╔═══════════════════════════════════════════════════════════╗
║              Memory System Status                         ║
║         (Graph + Vector + Session Storage)                ║
╚═══════════════════════════════════════════════════════════╝
""")
                # SQLite counts (session storage)
                sessions = store.fetchone("SELECT COUNT(*) as c FROM sessions")["c"]
                traits = store.fetchone("SELECT COUNT(*) as c FROM user_traits")["c"]
                messages = store.fetchone("SELECT COUNT(*) as c FROM messages")["c"]
                skills_sqlite = store.fetchone("SELECT COUNT(*) as c FROM skills")["c"]

                print("Session Storage (SQLite):")
                print(f"  Sessions:    {sessions}")
                print(f"  Messages:    {messages}")
                print(f"  Traits:      {traits}")
                print(f"  Skills:      {skills_sqlite}")
                print()

                # Vector memory stats
                try:
                    vector = VectorMemory()
                    stats = vector.stats()
                    print("Vector Memory (ChromaDB):")
                    print(f"  Facts:       {stats['facts_count']}")
                    print(f"  Memories:    {stats['memories_count']}")
                    print(f"  Skills:      {stats['skills_count']}")
                    print(f"  Entities:    {stats['entities_count']}")
                    print(
                        f"  Embedding:   {stats['embedding_model']} ({stats['embedding_dimension']}d)"
                    )
                    print()
                except Exception as e:
                    print(f"Vector Memory: error - {e}")
                    print()

                # Graph memory stats
                try:
                    graph = GraphMemory(CONFIG.cog_home / CONFIG.graph_name)
                    entity_count = 0
                    for entity_type in ENTITY_TYPES:
                        entities = graph.get_entities_by_type(entity_type)
                        entity_count += len(entities)
                    relation_count = len(graph.get_all_relations(limit=1000))
                    print("Graph Memory (CogDB):")
                    print(f"  Entities:    {entity_count}")
                    print(f"  Relations:   {relation_count}")
                    print()
                except Exception as e:
                    print(f"Graph Memory: error - {e}")
                    print()

                # Consistency check
                try:
                    sync = MemorySync(graph, vector)
                    report = sync.check_consistency()
                    print("Cross-Store Consistency:")
                    print(f"  Graph entities:  {report['graph_entities']}")
                    print(f"  Vector entities: {report['vector_entities']}")
                    print(f"  Consistent:      {report['consistent']}")
                    if report["missing_in_vector"]:
                        print(
                            f"  Missing in vector: {len(report['missing_in_vector'])}"
                        )
                    if report["missing_in_graph"]:
                        print(f"  Missing in graph: {len(report['missing_in_graph'])}")
                    print()
                except Exception:
                    pass

                # FTS5 index check
                try:
                    store.fts_messages("test")
                    print("✓ FTS5 messages index: OK")
                except Exception as e:
                    print(f"✗ FTS5 messages index: {e}")

                try:
                    store.fts_skills("test")
                    print("✓ FTS5 skills index: OK")
                except Exception as e:
                    print(f"✗ FTS5 skills index: {e}")

                # Recent activity
                last_session = store.fetchone(
                    "SELECT datetime(started_at, 'unixepoch') as ts FROM sessions ORDER BY started_at DESC LIMIT 1"
                )
                if last_session:
                    print(f"\nLast session:  {last_session['ts']}")

                # WAL mode check
                wal = store.fetchone("PRAGMA journal_mode")
                print(f"\nJournal mode: {wal['journal_mode']} (WAL = good)")

                print("\nMemory system is healthy!")
                return 0
            if sub == "recent":
                row = store.fetchone(
                    "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1"
                )
                if row is None:
                    print("(no sessions)")
                    return 0
                msgs = store.recent_messages(int(row["id"]), limit=80)
                for m in msgs:
                    print(f"[{m['role']}] {m['content'][:200]}")
                return 0
            if sub == "search":
                q = " ".join(rest[1:]).strip()
                if not q:
                    print("usage: memory search <q>")
                    return 1
                # Use VectorMemory for semantic search
                try:
                    vector = VectorMemory()
                    print("# Facts (vector search)")
                    facts = vector.search_similar_facts(q, n_results=10)
                    for f in facts:
                        content = f.get("content", "")
                        category = f.get("metadata", {}).get("category", "unknown")
                        confidence = f.get("metadata", {}).get("confidence", 0.0)
                        print(f"  [{category}] (conf={confidence:.2f}) {content[:100]}")
                    print()
                    print("# Entities (vector search)")
                    entities = vector.search_similar_entities(q, n_results=5)
                    for e in entities:
                        desc = e.get("description", "")
                        etype = e.get("metadata", {}).get("entity_type", "unknown")
                        print(f"  [{etype}] {desc[:80]}")
                    print()
                    print("# Skills (vector search)")
                    skills = vector.search_similar_skills(q, n_results=5)
                    for s in skills:
                        name = s.get("metadata", {}).get("name", s.get("id", "unknown"))
                        desc = s.get("description", "")
                        print(f"  [{name}] {desc[:80]}")
                except Exception as e:
                    print(f"Vector search error: {e}")
                print("\n# Message archive (FTS5)")
                for r in store.fts_messages(q, limit=10):
                    print(f"- [{r['role']}] {r['content'][:160]}")
                return 0
            if sub == "cleanup":
                # Use VectorMemory for cleanup
                try:
                    vector = VectorMemory()
                    # Get recent facts for cleanup review
                    facts = vector.get_recent_facts(limit=50)
                    if not facts:
                        print("(no facts to clean up)")
                        return 0

                    # Ask the fast model to identify stale/contradictory facts
                    facts_list = "\n".join(
                        f"- {f.get('content', '')[:100]} (id={f.get('id', 'unknown')})"
                        for f in facts[:30]
                    )
                    system = """You are a memory hygiene assistant. Review the list of facts and identify any that are:
1. Stale — no longer true or outdated
2. Contradictory — conflict with other facts
3. Ephemeral — temporary state that shouldn't be stored (file paths, counts, etc.)

Reply with a JSON array of fact IDs to delete, e.g. ["fact_abc123", "fact_xyz789"].
If none should be deleted, return an empty array []."""
                    user = f"Review these facts and return JSON array of IDs to delete:\n\n{facts_list}"
                    llm = LLMClient(
                        base_url=CONFIG.llm_base_url,
                        api_key=CONFIG.llm_api_key,
                        model=CONFIG.llm_model,
                        fast_model=CONFIG.llm_fast_model,
                    )
                    result = llm.chat_json(system, user, fast=True)
                    if result is None or not isinstance(result, list):
                        print("LLM did not return a list of fact IDs to delete.")
                        print("Raw response:", result)
                        return 1
                    to_delete = set(str(k) for k in result if k)
                    if not to_delete:
                        print("No facts identified for deletion.")
                        return 0
                    print(f"Will delete {len(to_delete)} fact(s):")
                    for fact_id in to_delete:
                        print(f"  - {fact_id}")
                        try:
                            vector.delete_fact(fact_id)
                            print(f"Deleted: {fact_id}")
                        except Exception as e:
                            print(f"Failed to delete {fact_id}: {e}")
                    print("Cleanup complete.")
                except Exception as e:
                    print(f"Cleanup error: {e}")
                    return 1
                return 0
            print(f"unknown memory subcommand: {sub}")
            return 1
        finally:
            store.close()

    if cmd == "traits":
        from .memory import UserModel

        store, _skills = _open()
        try:
            print(UserModel(store).render_for_prompt())
        finally:
            store.close()
        return 0

    print(f"unknown command: {cmd}\n")
    print(USAGE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
