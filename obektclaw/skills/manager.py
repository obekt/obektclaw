"""Layer 3 — Skill memory.

Each Skill is a markdown file with YAML-ish frontmatter, e.g.:

    ---
    name: csv-to-database
    description: Clean a CSV and import it into SQLite/Postgres
    ---
    # Steps
    1. ...

Skills live on disk under ~/.obektclaw/skills/. The manager mirrors metadata
into the SQLite store so we can FTS5-search the skill corpus and rank by
usefulness, and so the Learning Loop can rewrite skills in-place when it
learns something new.
"""
from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ..logging_config import get_logger
from ..memory.store import Store

log = get_logger(__name__)


SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def slugify(name: str) -> str:
    s = name.strip().lower().replace(" ", "-")
    s = SLUG_RE.sub("", s)
    return s[:64] or "skill"


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path
    use_count: int = 0
    success_count: int = 0

    def render(self) -> str:
        return f"## Skill: {self.name}\n{self.description}\n\n{self.body}"

    def render_brief(self) -> str:
        return f"- {self.name}: {self.description}"


def parse_skill_file(path: Path) -> Skill | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    m = FRONTMATTER_RE.match(text)
    if not m:
        # Treat as a free-form skill: filename = name, first line = description.
        lines = text.strip().splitlines()
        desc = lines[0].lstrip("# ").strip() if lines else path.stem
        return Skill(name=path.stem, description=desc, body=text.strip(), path=path)

    front = m.group(1)
    body = text[m.end():].strip()
    name = path.stem
    desc = ""
    for raw in front.splitlines():
        if ":" not in raw:
            continue
        k, _, v = raw.partition(":")
        k = k.strip().lower()
        v = v.strip().strip('"').strip("'")
        if k == "name":
            name = v
        elif k == "description":
            desc = v
    return Skill(name=name, description=desc, body=body, path=path)


def write_skill_file(path: Path, name: str, description: str, body: str) -> None:
    front = f"---\nname: {name}\ndescription: {description}\n---\n\n"
    path.write_text(front + body.strip() + "\n")


class SkillManager:
    def __init__(self, store: Store, skills_dir: Path, bundled_dir: Path):
        self.store = store
        self.skills_dir = skills_dir
        self.bundled_dir = bundled_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._sync_bundled()
        self.reindex()

    def _sync_bundled(self) -> None:
        """Copy bundled skills into ~/.obektclaw/skills/ on first run."""
        if not self.bundled_dir.exists():
            return
        for src in self.bundled_dir.glob("*.md"):
            dest = self.skills_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)

    def reindex(self) -> None:
        """Walk the skills directory and refresh the SQLite mirror."""
        seen: set[str] = set()
        now = time.time()
        for path in sorted(self.skills_dir.glob("*.md")):
            sk = parse_skill_file(path)
            if sk is None:
                continue
            seen.add(sk.name)
            existing = self.store.fetchone(
                "SELECT name, use_count, success_count FROM skills WHERE name = ?",
                (sk.name,),
            )
            if existing is None:
                self.store.execute(
                    """
                    INSERT INTO skills (name, description, body, use_count, success_count, created_at, updated_at)
                    VALUES (?,?,?,0,0,?,?)
                    """,
                    (sk.name, sk.description, sk.body, now, now),
                )
            else:
                self.store.execute(
                    """
                    UPDATE skills SET description = ?, body = ?, updated_at = ?
                    WHERE name = ?
                    """,
                    (sk.description, sk.body, now, sk.name),
                )
        # Drop rows whose file disappeared.
        rows = self.store.fetchall("SELECT name FROM skills")
        for r in rows:
            if r["name"] not in seen:
                self.store.execute("DELETE FROM skills WHERE name = ?", (r["name"],))

    # ----- read API -----
    def list_all(self) -> list[Skill]:
        out: list[Skill] = []
        for path in sorted(self.skills_dir.glob("*.md")):
            sk = parse_skill_file(path)
            if sk is None:
                continue
            row = self.store.fetchone(
                "SELECT use_count, success_count FROM skills WHERE name = ?", (sk.name,)
            )
            if row is not None:
                sk.use_count = int(row["use_count"])
                sk.success_count = int(row["success_count"])
            out.append(sk)
        return out

    def get(self, name: str) -> Skill | None:
        path = self.skills_dir / f"{slugify(name)}.md"
        if not path.exists():
            # try literal name
            for p in self.skills_dir.glob("*.md"):
                sk = parse_skill_file(p)
                if sk and sk.name == name:
                    return sk
            return None
        return parse_skill_file(path)

    def search(self, query: str, limit: int = 5) -> list[Skill]:
        rows = self.store.fts_skills(query, limit=limit)
        out: list[Skill] = []
        for r in rows:
            sk = self.get(r["name"])
            if sk is not None:
                sk.use_count = int(r["use_count"])
                sk.success_count = int(r["success_count"])
                out.append(sk)
        return out

    # ----- write API (used by tools and the Learning Loop) -----
    def create(self, name: str, description: str, body: str) -> Skill:
        slug = slugify(name)
        path = self.skills_dir / f"{slug}.md"
        write_skill_file(path, slug, description, body)
        self.reindex()
        sk = self.get(slug)
        assert sk is not None
        log.info("skill_created name=%s path=%s", slug, path)
        return sk

    def improve(self, name: str, *, new_description: str | None = None, new_body: str | None = None,
                append: str | None = None) -> Skill | None:
        sk = self.get(name)
        if sk is None:
            return None
        desc = new_description if new_description is not None else sk.description
        if new_body is not None:
            body = new_body
        elif append is not None:
            body = sk.body.rstrip() + "\n\n" + append.strip()
        else:
            body = sk.body
        write_skill_file(sk.path, slugify(sk.name), desc, body)
        self.reindex()
        log.info("skill_improved name=%s mode=%s", sk.name, "body" if new_body else ("append" if append else "description"))
        return self.get(sk.name)

    def record_use(self, name: str, success: bool) -> None:
        if success:
            self.store.execute(
                "UPDATE skills SET use_count = use_count + 1, success_count = success_count + 1 WHERE name = ?",
                (name,),
            )
        else:
            self.store.execute(
                "UPDATE skills SET use_count = use_count + 1 WHERE name = ?",
                (name,),
            )
        log.debug("skill_used name=%s success=%s", name, success)
