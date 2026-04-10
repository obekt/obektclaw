import os
import pytest
from pathlib import Path
from obektclaw.skills.manager import SkillManager, parse_skill_file

def test_parse_skill_file_oserror(tmp_path, monkeypatch):
    p = tmp_path / "no_perms.md"
    p.write_text("content")
    
    def mock_read_text(*args, **kwargs):
        raise OSError("Permission denied")
    
    monkeypatch.setattr(Path, "read_text", mock_read_text)
    assert parse_skill_file(p) is None

def test_parse_skill_file_description(tmp_path):
    p = tmp_path / "desc.md"
    p.write_text("---\ninvalid_line_no_colon\ndescription: my desc\n---\nbody")
    sk = parse_skill_file(p)
    assert sk.description == "my desc"

def test_manager_reindex_and_list_with_invalid_file(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    
    p = skills / "bad.md"
    p.write_text("something")
    
    import obektclaw.skills.manager
    # Store reference to original function
    orig_parse = obektclaw.skills.manager.parse_skill_file
    
    def mock_parse(path):
        if path.name == "bad.md":
            return None
        return orig_parse(path)
    
    monkeypatch.setattr("obektclaw.skills.manager.parse_skill_file", mock_parse)
    
    from obektclaw.memory.store import Store
    store = Store(tmp_path / "db.sqlite3")
    mgr = SkillManager(store, skills, bundled)
    
    mgr.reindex()  # should skip bad.md
    assert len(mgr.list_all()) == 0

def test_manager_reindex_update(tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    
    from obektclaw.memory.store import Store
    store = Store(tmp_path / "db.sqlite3")
    mgr = SkillManager(store, skills, bundled)
    
    mgr.create("myskill", "desc1", "body1")
    
    # modify file manually to trigger update branch in reindex
    p = skills / "myskill.md"
    p.write_text("---\nname: myskill\ndescription: desc2\n---\nbody2")
    
    mgr.reindex()
    sk = mgr.get("myskill")
    assert sk.description == "desc2"
    assert sk.body == "body2"

def test_manager_get_literal_name(tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    
    from obektclaw.memory.store import Store
    store = Store(tmp_path / "db.sqlite3")
    mgr = SkillManager(store, skills, bundled)
    
    # Create file with a name different from its stem
    p = skills / "file_stem.md"
    p.write_text("---\nname: LiteralName\n---\nbody")
    
    sk = mgr.get("LiteralName")
    assert sk is not None
    assert sk.name == "LiteralName"

def test_manager_search_with_sk_not_none(tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    
    from obektclaw.memory.store import Store
    store = Store(tmp_path / "db.sqlite3")
    mgr = SkillManager(store, skills, bundled)
    
    mgr.create("searchable", "desc", "searchword")
    
    res = mgr.search("searchword")
    assert len(res) == 1
    assert res[0].name == "searchable"
    
    # also simulate parse_skill_file returning None inside search by deleting the file
    (skills / "searchable.md").unlink()
    res2 = mgr.search("searchword")
    assert len(res2) == 0

def test_manager_improve_missing(tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    from obektclaw.memory.store import Store
    store = Store(tmp_path / "db.sqlite3")
    mgr = SkillManager(store, skills, bundled)
    assert mgr.improve("nonexistent_skill", append="foo") is None

def test_manager_create_existing(tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    from obektclaw.memory.store import Store
    store = Store(tmp_path / "db.sqlite3")
    mgr = SkillManager(store, skills, bundled)
    mgr.create("myskill", "desc1", "body1")
    mgr.create("myskill", "desc2", "body2")
    
    sk = mgr.get("myskill")
    assert sk.description == "desc2"
    assert sk.body == "body2"

def test_manager_delete_reindex(tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    from obektclaw.memory.store import Store
    store = Store(tmp_path / "db.sqlite3")
    mgr = SkillManager(store, skills, bundled)
    
    mgr.create("deletable", "desc", "body")
    assert mgr.get("deletable") is not None
    
    (skills / "deletable.md").unlink()
    mgr.reindex()
    assert mgr.get("deletable") is None

def test_sync_bundled_exists(tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    
    (bundled / "bundled1.md").write_text("body")
    (skills / "bundled1.md").write_text("already here")
    
    from obektclaw.memory.store import Store
    store = Store(tmp_path / "db.sqlite3")
    mgr = SkillManager(store, skills, bundled)
    
    # should not be overwritten
    assert (skills / "bundled1.md").read_text() == "already here"
