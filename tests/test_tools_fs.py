import os
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obektclaw.tools.registry import ToolContext, ToolRegistry
from obektclaw.tools.fs import (
    _resolve,
    read_file,
    write_file,
    list_files,
    grep,
    register,
    MAX_READ_BYTES
)

@pytest.fixture
def mock_ctx(tmp_path: Path):
    config = MagicMock()
    config.workdir = tmp_path
    ctx = MagicMock(spec=ToolContext)
    ctx.config = config
    return ctx

def test_resolve(mock_ctx, tmp_path):
    assert _resolve(mock_ctx, "foo.txt") == tmp_path / "foo.txt"
    assert _resolve(mock_ctx, "/tmp/foo.txt") == Path("/tmp/foo.txt")
    assert _resolve(mock_ctx, "~/foo.txt").is_absolute()

def test_read_file_missing_path(mock_ctx):
    res = read_file({}, mock_ctx)
    assert res.is_error
    assert "missing 'path'" in res.content

def test_read_file_not_exists(mock_ctx):
    res = read_file({"path": "does_not_exist.txt"}, mock_ctx)
    assert res.is_error
    assert "no such file" in res.content

def test_read_file_is_directory(mock_ctx, tmp_path):
    res = read_file({"path": str(tmp_path)}, mock_ctx)
    assert res.is_error
    assert "is a directory" in res.content

def test_read_file_os_error(mock_ctx, tmp_path, monkeypatch):
    p = tmp_path / "test.txt"
    p.write_text("hello")
    def mock_read_bytes(*args, **kwargs):
        raise OSError("mock error")
    monkeypatch.setattr(Path, "read_bytes", mock_read_bytes)
    res = read_file({"path": "test.txt"}, mock_ctx)
    assert res.is_error
    assert "read error" in res.content

def test_read_file_too_large(mock_ctx, tmp_path):
    p = tmp_path / "large.txt"
    p.touch()
    def mock_read_bytes(*args, **kwargs):
        return b"a" * (MAX_READ_BYTES + 1)
    p.write_bytes(b"a" * (MAX_READ_BYTES + 1))
    res = read_file({"path": "large.txt"}, mock_ctx)
    assert res.is_error
    assert "file too large" in res.content

def test_read_file_success(mock_ctx, tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("hello world")
    res = read_file({"path": "test.txt"}, mock_ctx)
    assert not res.is_error
    assert res.content == "hello world"

def test_read_file_latin1_fallback(mock_ctx, tmp_path):
    p = tmp_path / "test.bin"
    p.write_bytes(b"\xff\xfe\x00")
    res = read_file({"path": "test.bin"}, mock_ctx)
    assert not res.is_error
    assert isinstance(res.content, str)

def test_write_file_missing_args(mock_ctx):
    res = write_file({"path": "test.txt"}, mock_ctx)
    assert res.is_error
    res2 = write_file({"content": "hi"}, mock_ctx)
    assert res2.is_error

def test_write_file_success(mock_ctx, tmp_path):
    res = write_file({"path": "subdir/test.txt", "content": "hi"}, mock_ctx)
    assert not res.is_error
    assert "wrote" in res.content
    assert (tmp_path / "subdir" / "test.txt").read_text() == "hi"

def test_list_files_not_exists(mock_ctx):
    res = list_files({"path": "does_not_exist"}, mock_ctx)
    assert res.is_error
    assert "not a directory" in res.content

def test_list_files_success(mock_ctx, tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("bb")
    (tmp_path / "dir").mkdir()
    
    res = list_files({"path": ".", "pattern": "*.txt"}, mock_ctx)
    assert not res.is_error
    assert "a.txt" in res.content
    assert "b.txt" in res.content
    assert "dir" not in res.content
    
    res2 = list_files({"path": "."}, mock_ctx)
    assert not res2.is_error
    assert "dir" in res2.content
    
    original_stat = Path.stat
    stat_calls = 0
    def mock_stat(self, *args, **kwargs):
        nonlocal stat_calls
        if self.name == "a.txt":
            stat_calls += 1
            if stat_calls == 3:
                raise OSError("mock error")
        return original_stat(self, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", mock_stat)
    res3 = list_files({"path": "."}, mock_ctx)
    assert not res3.is_error
    assert "dir" in res3.content
    assert "?" in res3.content

def test_grep_missing_pattern(mock_ctx):
    res = grep({}, mock_ctx)
    assert res.is_error
    assert "missing 'pattern'" in res.content

def test_grep_bad_regex(mock_ctx):
    res = grep({"pattern": "["}, mock_ctx)
    assert res.is_error
    assert "bad regex" in res.content

def test_grep_no_such_path(mock_ctx):
    res = grep({"pattern": "a", "path": "does_not_exist"}, mock_ctx)
    assert res.is_error
    assert "no such path" in res.content

def test_grep_file(mock_ctx, tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("hello\nworld\nhello again")
    res = grep({"pattern": "hello", "path": "test.txt"}, mock_ctx)
    assert not res.is_error
    assert "test.txt:1: hello" in res.content
    assert "test.txt:3: hello again" in res.content

def test_grep_dir(mock_ctx, tmp_path):
    (tmp_path / "test1.txt").write_text("hello\nworld")
    (tmp_path / "test2.log").write_text("world\nhello there")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "secret.txt").write_text("hello")
    
    res = grep({"pattern": "hello", "path": "."}, mock_ctx)
    assert not res.is_error
    assert "test1.txt" in res.content
    assert "test2.log" in res.content
    assert ".git" not in res.content

def test_grep_os_error_and_truncated(mock_ctx, tmp_path, monkeypatch):
    p = tmp_path / "test.txt"
    p.write_text("a\n" * 250)
    
    res = grep({"pattern": "a", "path": "test.txt"}, mock_ctx)
    assert not res.is_error
    assert "(truncated)" in res.content
    
    res2 = grep({"pattern": "b", "path": "test.txt"}, mock_ctx)
    assert not res2.is_error
    assert "(no matches)" in res2.content
    
    def mock_open(*args, **kwargs):
        raise OSError("mock")
    monkeypatch.setattr(Path, "open", mock_open)
    res3 = grep({"pattern": "a", "path": "test.txt"}, mock_ctx)
    assert "(no matches)" in res3.content

def test_register():
    reg = ToolRegistry()
    register(reg)
    tools = reg.all()
    names = {t.name for t in tools}
    assert {"read_file", "write_file", "list_files", "grep"}.issubset(names)
