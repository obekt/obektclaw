"""Telegram bot gateway using long polling. No external dependency beyond httpx.

The bot maintains one Agent instance per chat_id. Messages are processed
serially per chat (the agent is not thread-safe across simultaneous calls
in the same session).
"""
from __future__ import annotations

import os
import threading
import time
from queue import Queue, Empty

import httpx

from ..agent import Agent
from ..config import CONFIG, load_config
from ..memory.store import Store
from ..skills import SkillManager


API = "https://api.telegram.org/bot{token}/{method}"


class _ChatWorker(threading.Thread):
    def __init__(self, chat_id: int, store: Store, skills: SkillManager, send, send_chat_action):
        super().__init__(daemon=True)
        self.chat_id = chat_id
        self.queue: Queue[str] = Queue()
        self.send = send
        self.send_chat_action = send_chat_action
        self.agent = Agent(
            config=CONFIG, store=store, skills=skills,
            gateway="telegram", user_key=f"tg:{chat_id}",
        )

    _STOP_SENTINEL = "__STOP__"

    def run(self) -> None:
        while True:
            try:
                text = self.queue.get(timeout=2)  # Short timeout for responsive shutdown
            except Empty:
                continue
            if text == self._STOP_SENTINEL:
                break

            # Status callback — sends typing action to Telegram
            def _on_status(msg: str):
                if msg:
                    try:
                        self.send_chat_action(self.chat_id, "typing")
                    except Exception:
                        pass

            try:
                reply = self.agent.run_once(text, status_fn=_on_status)
            except Exception as e:  # noqa: BLE001
                reply = f"error: {e}"
            for chunk in _chunk(reply, 3500):
                self.send(self.chat_id, chunk)


def _chunk(s: str, n: int):
    for i in range(0, len(s), n):
        yield s[i : i + n]


def run() -> int:
    # Handle stale empty env vars — reload config from .env files if needed
    if not CONFIG.tg_token:
        # Clear stale empty vars and reload
        for key in ['OBEKTCLAW_TG_TOKEN', 'OBEKTCLAW_LLM_API_KEY', 'OBEKTCLAW_LLM_MODEL', 'OBEKTCLAW_LLM_FAST_MODEL']:
            if key in os.environ and not os.environ[key]:
                del os.environ[key]
        # Reload config by re-reading .env files
        from ..config import _read_env_file
        from pathlib import Path
        home = Path(os.environ.get("OBEKTCLAW_HOME") or Path.home() / ".obektclaw").expanduser()
        project_env = Path(__file__).resolve().parent.parent / ".env"
        _read_env_file(project_env)
        _read_env_file(home / ".env")
    
    if not CONFIG.tg_token:
        print("OBEKTCLAW_TG_TOKEN not set. Add it to ~/.obektclaw/.env and try again.")
        return 1
    key = CONFIG.llm_api_key.strip()
    if not key or key in {"your-api-key-here", "sk-xxx", "sk-your-key", ""}:
        print("OBEKTCLAW_LLM_API_KEY is not set. Edit ~/.obektclaw/.env with a real API key.")
        return 1
    store = Store(CONFIG.db_path)
    skills = SkillManager(store, CONFIG.skills_dir, CONFIG.bundled_skills_dir)
    workers: dict[int, _ChatWorker] = {}
    client = httpx.Client(timeout=60.0)

    def send(chat_id: int, text: str) -> None:
        try:
            client.post(
                API.format(token=CONFIG.tg_token, method="sendMessage"),
                json={"chat_id": chat_id, "text": text},
            )
        except httpx.HTTPError as e:
            print(f"[tg] send failed: {e}")

    def send_chat_action(chat_id: int, action: str) -> None:
        """Send a chat action (e.g. 'typing') to show the typing indicator."""
        try:
            client.post(
                API.format(token=CONFIG.tg_token, method="sendChatAction"),
                json={"chat_id": chat_id, "action": action},
            )
        except httpx.HTTPError:
            pass  # Silently ignore — typing indicator is non-critical

    offset: int | None = None
    print("obektclaw telegram bot running (long-poll)")
    try:
        while True:
            try:
                params = {"timeout": 30}
                if offset is not None:
                    params["offset"] = offset
                resp = client.get(
                    API.format(token=CONFIG.tg_token, method="getUpdates"),
                    params=params,
                )
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                print(f"[tg] poll error: {e}")
                time.sleep(2)
                continue
            for update in data.get("result", []):
                offset = int(update["update_id"]) + 1
                msg = update.get("message") or update.get("edited_message")
                if not msg:
                    continue
                chat_id = int(msg["chat"]["id"])
                if CONFIG.tg_allowed_chat_ids and chat_id not in CONFIG.tg_allowed_chat_ids:
                    continue
                text = msg.get("text") or ""
                if not text.strip():
                    continue
                worker = workers.get(chat_id)
                if worker is None:
                    worker = _ChatWorker(chat_id, store, skills, send, send_chat_action)
                    worker.start()
                    workers[chat_id] = worker
                worker.queue.put(text)
    except KeyboardInterrupt:
        pass
    finally:
        # Graceful shutdown: send sentinel to each worker
        for w in workers.values():
            w.queue.put(_ChatWorker._STOP_SENTINEL)
        # Wait for workers to finish (with timeout to avoid hanging)
        for w in workers.values():
            w.join(timeout=5)
            w.agent.close()
        store.close()
    return 0
