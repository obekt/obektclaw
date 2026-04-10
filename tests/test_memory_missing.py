from obektclaw.memory.store import Store
from obektclaw.memory.persistent import PersistentMemory, Fact
from obektclaw.memory.user_model import UserModel, Trait

def test_persistent_memory_upsert_and_delete(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    pm = PersistentMemory(store)
    
    # Insert with unknown category
    pm.upsert("key1", "val1", category="unknown_cat")
    facts = pm.list_category("general")
    assert len(facts) == 1
    assert facts[0].key == "key1"
    
    # Update existing
    pm.upsert("key1", "val2", category="general")
    facts = pm.list_category("general")
    assert len(facts) == 1
    assert facts[0].value == "val2"
    
    # Delete
    pm.delete("general", "key1")
    assert len(pm.list_category("general")) == 0

def test_persistent_memory_search(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    pm = PersistentMemory(store)
    pm.upsert("search_key", "search_value", category="general")
    
    res = pm.search("search_value")
    assert len(res) == 1
    assert res[0].key == "search_key"

def test_persistent_memory_all_top(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    pm = PersistentMemory(store)
    pm.upsert("k1", "v1", category="user")
    pm.upsert("k2", "v2", category="env")
    
    all_top = pm.all_top(per_category=2)
    assert len(all_top) == 2

def test_user_model_get_exists(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    um = UserModel(store)
    um.set("tooling_pref", "myval", evidence="myevidence")
    t = um.get("tooling_pref")
    assert t is not None
    assert t.layer == "tooling_pref"
    assert t.value == "myval"
    assert t.evidence == "myevidence"

def test_user_model_get_none(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    um = UserModel(store)
    assert um.get("nonexistent_layer") is None

def test_store_executemany(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    store.executemany("INSERT INTO user_traits (layer, value, evidence, updated_at) VALUES (?, ?, ?, ?)", [
        ("l1", "v1", None, 0.0),
        ("l2", "v2", None, 0.0),
    ])
    um = UserModel(store)
    assert len(um.all()) == 2

def test_fact_render():
    f = Fact(id=1, category="general", key="foo", value="bar", confidence=0.9)
    assert f.render() == "- (general) foo: bar"

def test_user_model_render_empty(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    um = UserModel(store)
    assert um.render_for_prompt() == "(no user model yet — this is an early conversation)"
