import pytest
from unittest.mock import MagicMock, patch
import httpx
from obektclaw.tools.registry import ToolContext, ToolRegistry
from obektclaw.tools.web import web_fetch, register

@pytest.fixture
def mock_ctx():
    return MagicMock(spec=ToolContext)

def test_web_fetch_missing_url(mock_ctx):
    res = web_fetch({}, mock_ctx)
    assert res.is_error
    assert "missing 'url'" in res.content

@patch("httpx.Client")
def test_web_fetch_httperror(mock_client_class, mock_ctx):
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__.return_value = mock_client
    mock_client.get.side_effect = httpx.HTTPError("conn error")
    
    res = web_fetch({"url": "http://example.com"}, mock_ctx)
    assert res.is_error
    assert "fetch failed: conn error" in res.content

@patch("httpx.Client")
def test_web_fetch_success_html_stripped(mock_client_class, mock_ctx):
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.content = b"<html><body>Hello <br> World</body></html>"
    mock_resp.encoding = "utf-8"
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.status_code = 200
    mock_client.get.return_value = mock_resp
    
    res = web_fetch({"url": "http://example.com", "strip_html": True}, mock_ctx)
    assert not res.is_error
    assert "Hello World" in res.content

@patch("httpx.Client")
def test_web_fetch_success_no_strip(mock_client_class, mock_ctx):
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.content = b"<html><body>Hello <br> World</body></html>"
    mock_resp.encoding = "utf-8"
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.status_code = 200
    mock_client.get.return_value = mock_resp
    
    res = web_fetch({"url": "http://example.com", "strip_html": False}, mock_ctx)
    assert not res.is_error
    assert "<html><body>Hello <br> World</body></html>" in res.content

@patch("httpx.Client")
def test_web_fetch_success_bad_encoding(mock_client_class, mock_ctx):
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.content = b"bad byte \xff"
    mock_resp.encoding = "invalid-encoding"
    mock_resp.headers = {"content-type": "text/plain"}
    mock_resp.status_code = 200
    mock_client.get.return_value = mock_resp
    
    res = web_fetch({"url": "http://example.com"}, mock_ctx)
    assert not res.is_error
    assert "bad byte" in res.content

@patch("httpx.Client")
def test_web_fetch_error_status(mock_client_class, mock_ctx):
    mock_client = MagicMock()
    mock_client_class.return_value.__enter__.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.content = b"Not Found"
    mock_resp.encoding = "utf-8"
    mock_resp.headers = {"content-type": "text/plain"}
    mock_resp.status_code = 404
    mock_client.get.return_value = mock_resp
    
    res = web_fetch({"url": "http://example.com"}, mock_ctx)
    assert res.is_error
    assert "Not Found" in res.content

def test_register():
    reg = MagicMock(spec=ToolRegistry)
    register(reg)
    assert reg.register.call_count == 1