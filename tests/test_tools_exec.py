import subprocess
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from obektclaw.tools.registry import ToolContext, ToolRegistry
from obektclaw.tools.exec import _truncate, bash, exec_python, register, MAX_OUTPUT

@pytest.fixture
def mock_ctx(tmp_path: Path):
    config = MagicMock()
    config.workdir = tmp_path
    config.bash_timeout = 2
    ctx = MagicMock(spec=ToolContext)
    ctx.config = config
    return ctx

def test_truncate():
    s1 = "a" * MAX_OUTPUT
    assert _truncate(s1) == s1
    s2 = "a" * (MAX_OUTPUT + 100)
    res = _truncate(s2)
    assert len(res) < len(s2)
    assert "(truncated" in res

def test_bash_missing_command(mock_ctx):
    res = bash({}, mock_ctx)
    assert res.is_error
    assert "missing 'command'" in res.content

def test_bash_timeout(mock_ctx, monkeypatch):
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="foo", timeout=1)
    monkeypatch.setattr(subprocess, "run", mock_run)
    res = bash({"command": "sleep 10"}, mock_ctx)
    assert res.is_error
    assert "timed out" in res.content

def test_bash_success_and_failure(mock_ctx, monkeypatch):
    mock_proc = MagicMock()
    mock_proc.stdout = "hello stdout"
    mock_proc.stderr = "hello stderr"
    mock_proc.returncode = 0
    
    def mock_run(*args, **kwargs):
        return mock_proc
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    res = bash({"command": "echo hello"}, mock_ctx)
    assert not res.is_error
    assert "hello stdout" in res.content
    assert "hello stderr" in res.content
    assert "EXIT: 0" in res.content

    mock_proc.returncode = 1
    mock_proc.stdout = ""
    mock_proc.stderr = "error"
    res2 = bash({"command": "false"}, mock_ctx)
    assert res2.is_error
    assert "STDERR:\nerror" in res2.content
    assert "EXIT: 1" in res2.content

def test_exec_python_missing_code(mock_ctx):
    res = exec_python({}, mock_ctx)
    assert res.is_error
    assert "missing 'code'" in res.content

def test_exec_python_timeout(mock_ctx, monkeypatch):
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python", timeout=1)
    monkeypatch.setattr(subprocess, "run", mock_run)
    res = exec_python({"code": "while True: pass"}, mock_ctx)
    assert res.is_error
    assert "timed out" in res.content

def test_exec_python_success_and_failure(mock_ctx, monkeypatch):
    mock_proc = MagicMock()
    mock_proc.stdout = "py out"
    mock_proc.stderr = "py err"
    mock_proc.returncode = 0
    
    def mock_run(*args, **kwargs):
        return mock_proc
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    res = exec_python({"code": "print('py out')"}, mock_ctx)
    assert not res.is_error
    assert "STDOUT:\npy out" in res.content
    assert "STDERR:\npy err" in res.content
    assert "EXIT: 0" in res.content

    mock_proc.returncode = 1
    mock_proc.stdout = ""
    mock_proc.stderr = "py err"
    res2 = exec_python({"code": "1/0"}, mock_ctx)
    assert res2.is_error
    assert "STDERR:\npy err" in res2.content
    assert "EXIT: 1" in res2.content

def test_register():
    reg = ToolRegistry()
    register(reg)
    tools = reg.all()
    names = {t.name for t in tools}
    assert {"bash", "exec_python"}.issubset(names)
