"""Tests for obektclaw/skills/manager.py — skill parsing, creation, and improvement."""
import tempfile
from pathlib import Path

import pytest

from obektclaw.memory.store import Store
from obektclaw.skills.manager import (
    Skill,
    SkillManager,
    slugify,
    parse_skill_file,
    write_skill_file,
    FRONTMATTER_RE,
)


@pytest.fixture
def skills_env():
    """Create a temporary skills directory and store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        skills_dir = tmp / "skills"
        bundled_dir = tmp / "bundled"
        db_path = tmp / "test.db"
        
        skills_dir.mkdir()
        bundled_dir.mkdir()
        
        store = Store(db_path)
        manager = SkillManager(store, skills_dir, bundled_dir)
        yield store, manager, skills_dir, bundled_dir
        store.close()


class TestSlugify:
    """Test skill name slugification."""

    def test_simple_name(self):
        assert slugify("CSV Import") == "csv-import"

    def test_special_chars_removed(self):
        assert slugify("CSV-to-Database!") == "csv-to-database"

    def test_spaces_to_dashes(self):
        assert slugify("my cool skill") == "my-cool-skill"

    def test_lowercase(self):
        assert slugify("UPPERCASE") == "uppercase"

    def test_truncates_long_names(self):
        long = "a" * 100
        result = slugify(long)
        assert len(result) <= 64

    def test_empty_becomes_skill(self):
        assert slugify("!!!") == "skill"


class TestFrontmatterParsing:
    """Test YAML frontmatter parsing."""

    def test_valid_frontmatter(self):
        text = """---
name: csv-import
description: Import CSV files
---
# Body
Content here
"""
        m = FRONTMATTER_RE.match(text)
        assert m is not None
        assert "csv-import" in m.group(1)
        assert "Import CSV files" in m.group(1)

    def test_no_frontmatter(self):
        text = "# Just a heading\nContent"
        m = FRONTMATTER_RE.match(text)
        assert m is None

    def test_parse_skill_file_with_frontmatter(self, tmp_path: Path):
        path = tmp_path / "test.md"
        path.write_text("""---
name: test-skill
description: A test
---
## Steps
1. Do thing
""")
        skill = parse_skill_file(path)
        assert skill is not None
        assert skill.name == "test-skill"
        assert skill.description == "A test"
        assert "Do thing" in skill.body

    def test_parse_skill_file_without_frontmatter(self, tmp_path: Path):
        path = tmp_path / "test.md"
        path.write_text("# My Skill\nThis is the body")
        skill = parse_skill_file(path)
        assert skill is not None
        assert skill.name == "test"  # filename stem
        assert skill.description == "My Skill"  # first line
        assert "This is the body" in skill.body


class TestWriteSkillFile:
    """Test skill file creation."""

    def test_write_creates_frontmatter(self, tmp_path: Path):
        path = tmp_path / "new-skill.md"
        write_skill_file(path, "new-skill", "Does stuff", "# Body\nContent")
        text = path.read_text()
        assert text.startswith("---")
        assert "name: new-skill" in text
        assert "description: Does stuff" in text
        assert "# Body\nContent" in text

    def test_write_body_stripped(self, tmp_path: Path):
        path = tmp_path / "skill.md"
        write_skill_file(path, "s", "desc", "  padded body  \n")
        skill = parse_skill_file(path)
        assert skill.body == "padded body"


class TestSkillManager:
    """Test SkillManager operations."""

    def test_reindex_empty_dir(self, skills_env):
        store, manager, skills_dir, _ = skills_env
        manager.reindex()
        rows = store.fetchall("SELECT name FROM skills")
        assert len(rows) == 0

    def test_reindex_discovers_skills(self, skills_env):
        store, manager, skills_dir, _ = skills_env
        # Create a skill file
        write_skill_file(
            skills_dir / "import.md",
            "import",
            "Import data",
            "# Steps\n1. Read file"
        )
        manager.reindex()
        rows = store.fetchall("SELECT name FROM skills")
        assert len(rows) == 1
        assert rows[0]["name"] == "import"

    def test_list_all(self, skills_env):
        store, manager, skills_dir, _ = skills_env
        write_skill_file(
            skills_dir / "deploy.md",
            "deploy",
            "Deploy to prod",
            "Body here"
        )
        manager.reindex()
        skills = manager.list_all()
        assert len(skills) == 1
        assert skills[0].name == "deploy"
        assert skills[0].description == "Deploy to prod"

    def test_get_by_name(self, skills_env):
        store, manager, skills_dir, _ = skills_env
        write_skill_file(
            skills_dir / "csv-import.md",
            "csv-import",
            "Import CSV",
            "Steps..."
        )
        manager.reindex()
        skill = manager.get("csv-import")
        assert skill is not None
        assert skill.name == "csv-import"

    def test_get_returns_none_for_missing(self, skills_env):
        _, manager, _, _ = skills_env
        assert manager.get("nonexistent") is None

    def test_search_fts5(self, skills_env):
        store, manager, skills_dir, _ = skills_env
        write_skill_file(
            skills_dir / "csv-db.md",
            "csv-db",
            "Import CSV into SQLite database",
            "Read CSV, create table, insert rows"
        )
        manager.reindex()
        results = manager.search("csv database import")
        assert len(results) >= 1
        assert results[0].name == "csv-db"

    def test_create_skill(self, skills_env):
        _, manager, skills_dir, _ = skills_env
        skill = manager.create(
            name="test-skill",
            description="A test skill",
            body="# Steps\n1. Do test"
        )
        assert skill.name == "test-skill"
        assert (skills_dir / "test-skill.md").exists()

    def test_create_overwrites_existing_slug(self, skills_env):
        _, manager, skills_dir, _ = skills_env
        # Create with same slug
        skill1 = manager.create("My Skill", "Desc 1", "Body 1")
        skill2 = manager.create("my skill", "Desc 2", "Body 2")
        # Should be same file (same slug)
        assert skill1.path == skill2.path
        # Reindex should show updated version
        manager.reindex()
        skill = manager.get("my-skill")
        assert skill.description == "Desc 2"

    def test_improve_replace_body(self, skills_env):
        _, manager, skills_dir, _ = skills_env
        manager.create("old", "Old desc", "Old body")
        improved = manager.improve("old", new_body="New body")
        assert improved is not None
        assert improved.body == "New body"

    def test_improve_append(self, skills_env):
        _, manager, skills_dir, _ = skills_env
        manager.create("skill", "Desc", "Original body")
        improved = manager.improve("skill", append="## Addition\nNew content")
        assert improved is not None
        assert "Original body" in improved.body
        assert "New content" in improved.body

    def test_improve_replace_description(self, skills_env):
        _, manager, skills_dir, _ = skills_env
        manager.create("skill", "Old desc", "Body")
        improved = manager.improve("skill", new_description="New desc")
        assert improved is not None
        assert improved.description == "New desc"

    def test_improve_missing_skill_returns_none(self, skills_env):
        _, manager, _, _ = skills_env
        result = manager.improve("nonexistent", append="stuff")
        assert result is None

    def test_record_use(self, skills_env):
        store, manager, skills_dir, _ = skills_env
        manager.create("skill", "Desc", "Body")
        manager.record_use("skill", success=True)
        manager.record_use("skill", success=True)
        manager.record_use("skill", success=False)
        row = store.fetchone(
            "SELECT use_count, success_count FROM skills WHERE name = ?",
            ("skill",)
        )
        assert row["use_count"] == 3
        assert row["success_count"] == 2


class TestBundledSkills:
    """Test bundled skills sync on first run."""

    def test_sync_copies_missing(self, skills_env):
        store, manager, skills_dir, bundled_dir = skills_env
        # Create bundled skill
        write_skill_file(
            bundled_dir / "starter.md",
            "starter",
            "Starter skill",
            "Body"
        )
        # Create new manager (simulates first run)
        new_manager = SkillManager(store, skills_dir, bundled_dir)
        assert (skills_dir / "starter.md").exists()
        store.close()

    def test_sync_skips_existing(self, skills_env):
        store, manager, skills_dir, bundled_dir = skills_env
        # Create both bundled and user-modified
        bundled_dir.mkdir(exist_ok=True)
        write_skill_file(
            bundled_dir / "skill.md",
            "skill",
            "Bundled desc",
            "Bundled body"
        )
        write_skill_file(
            skills_dir / "skill.md",
            "skill",
            "User desc",
            "User body"
        )
        # Re-create manager
        new_manager = SkillManager(store, skills_dir, bundled_dir)
        # Should keep user version
        skill = new_manager.get("skill")
        assert skill.description == "User desc"


class TestSkillRender:
    """Test Skill rendering methods."""

    def test_render(self):
        skill = Skill(
            name="test",
            description="A test",
            body="## Steps\n1. Do it",
            path=Path("/tmp/test.md")
        )
        rendered = skill.render()
        assert "## Skill: test" in rendered
        assert "A test" in rendered
        assert "## Steps" in rendered

    def test_render_brief(self):
        skill = Skill(
            name="deploy",
            description="Deploy to production",
            body="Long body...",
            path=Path("/tmp/deploy.md")
        )
        brief = skill.render_brief()
        assert brief == "- deploy: Deploy to production"
