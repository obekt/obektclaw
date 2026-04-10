import os
from pathlib import Path
from obektclaw.config import _load_dotenv, _int_list

def test_load_dotenv_missing_file(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    _load_dotenv()  # Should return early, covering line 12

def test_load_dotenv_parsing(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "read_text", lambda self: "\n# comment\nNO_EQUALS_HERE\nMY_VAR=123\n")
    _load_dotenv()
    assert os.environ.get("MY_VAR") == "123"

def test_int_list():
    res = _int_list("1, , a, 3")
    assert res == (1, 3)
