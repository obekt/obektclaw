"""Top-level CLI dispatcher: `python -m obektclaw <subcommand>`."""
from __future__ import annotations

import sys

from .config import CONFIG
from .memory.store import Store
from .skills import SkillManager


USAGE = """usage: python -m obektclaw <command> [args]

commands:
  chat                       interactive REPL (type /help for examples)
  tg                         start the Telegram bot
  setup                      interactive setup wizard
  
  skill list                 list known skills
  skill show <name>          print a skill body
  
  memory recent              dump recent messages
  memory search <query>      search persistent facts
  memory cleanup             remove stale facts
  memory status              check memory system health
  
  traits                     show your user model
  help                       show detailed help

First time? Run: python -m obektclaw chat
"""


def _open() -> tuple[Store, SkillManager]:
    store = Store(CONFIG.db_path)
    skills = SkillManager(store, CONFIG.skills_dir, CONFIG.bundled_skills_dir)
    return store, skills


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    cmd, *rest = argv

    if cmd == "chat":
        from .gateways.cli import run
        return run()

    if cmd == "tg":
        from .gateways.telegram import run
        return run()

    if cmd == "setup":
        # Interactive setup wizard
        print("""
╔═══════════════════════════════════════════════════════════╗
║                    obektclaw Setup                        ║
╚═══════════════════════════════════════════════════════════╝

Configuration:
  OBEKTCLAW_HOME: {}
  Database: {}
  Skills: {}

""".format(CONFIG.home, CONFIG.db_path, CONFIG.skills_dir))
        
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
        from .memory import PersistentMemory
        from .llm import LLMClient
        store, _skills = _open()
        try:
            if not rest:
                print("usage: memory recent | memory search <q> | memory cleanup | memory status")
                return 1
            sub = rest[0]
            if sub == "status":
                # Memory system diagnostic
                print("""
╔═══════════════════════════════════════════════════════════╗
║              Memory System Status                         ║
╚═══════════════════════════════════════════════════════════╝
""")
                # Counts
                sessions = store.fetchone("SELECT COUNT(*) as c FROM sessions")["c"]
                facts = store.fetchone("SELECT COUNT(*) as c FROM facts")["c"]
                traits = store.fetchone("SELECT COUNT(*) as c FROM user_traits")["c"]
                messages = store.fetchone("SELECT COUNT(*) as c FROM messages")["c"]
                skills = store.fetchone("SELECT COUNT(*) as c FROM skills")["c"]
                
                print(f"Sessions:    {sessions}")
                print(f"Facts:       {facts}")
                print(f"Traits:      {traits}")
                print(f"Messages:    {messages}")
                print(f"Skills:      {skills}")
                print()
                
                # FTS5 index check
                try:
                    store.fts_messages("test")
                    print("✓ FTS5 messages index: OK")
                except Exception as e:
                    print(f"✗ FTS5 messages index: {e}")
                
                try:
                    store.fts_facts("test")
                    print("✓ FTS5 facts index: OK")
                except Exception as e:
                    print(f"✗ FTS5 facts index: {e}")
                
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
                
                print("\nMemory system is healthy!" if facts >= 0 else "\nWarning: Check errors above")
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
                pm = PersistentMemory(store)
                print("# facts")
                for f in pm.search(q):
                    print(f.render())
                print("\n# message archive")
                for r in store.fts_messages(q, limit=10):
                    print(f"- [{r['role']}] {r['content'][:160]}")
                return 0
            if sub == "cleanup":
                pm = PersistentMemory(store)
                all_facts = []
                for cat in ("user", "project", "env", "preference", "general"):
                    all_facts.extend(pm.list_category(cat, limit=200))
                if not all_facts:
                    print("(no facts to clean up)")
                    return 0
                # Ask the fast model to identify stale/contradictory facts
                facts_list = "\n".join(f.render() for f in all_facts)
                system = """You are a memory hygiene assistant. Review the list of facts and identify any that are:
1. Stale — no longer true or outdated
2. Contradictory — conflict with other facts
3. Ephemeral — temporary state that shouldn't be stored (file paths, counts, etc.)

Reply with a JSON array of fact keys to delete, e.g. ["csv_file_path", "old_server"].
If none should be deleted, return an empty array []."""
                user = f"Review these facts and return JSON array of keys to delete:\n\n{facts_list}"
                llm = LLMClient(
                    base_url=CONFIG.llm_base_url,
                    api_key=CONFIG.llm_api_key,
                    model=CONFIG.llm_model,
                    fast_model=CONFIG.llm_fast_model,
                )
                result = llm.chat_json(system, user, fast=True)
                if result is None or not isinstance(result, list):
                    print("LLM did not return a list of facts to delete.")
                    print("Raw response:", result)
                    return 1
                to_delete = set(str(k) for k in result if k)
                if not to_delete:
                    print("No facts identified for deletion.")
                    return 0
                print(f"Will delete {len(to_delete)} fact(s):")
                for key in to_delete:
                    print(f"  - {key}")
                for cat in ("user", "project", "env", "preference", "general"):
                    for fact in pm.list_category(cat, limit=200):
                        if fact.key in to_delete:
                            pm.delete(cat, fact.key)
                            print(f"Deleted: ({cat}) {fact.key}")
                print("Cleanup complete.")
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
