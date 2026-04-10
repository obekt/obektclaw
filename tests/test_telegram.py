import pytest
import time
from unittest.mock import MagicMock, patch
from queue import Queue, Empty
from obektclaw.gateways.telegram import run, _chunk, _ChatWorker
import httpx

def test_chunk():
    s = "abcdef"
    assert list(_chunk(s, 2)) == ["ab", "cd", "ef"]
    assert list(_chunk(s, 4)) == ["abcd", "ef"]
    assert list(_chunk(s, 6)) == ["abcdef"]

def test_chat_worker():
    mock_store = MagicMock()
    mock_skills = MagicMock()
    mock_send = MagicMock()
    mock_send_chat_action = MagicMock()

    with patch("obektclaw.gateways.telegram.Agent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent_cls.return_value = mock_agent
        mock_agent.run_once.side_effect = ["reply 1", Exception("agent error")]

        worker = _ChatWorker(123, mock_store, mock_skills, mock_send, mock_send_chat_action)
        
        # We don't start the thread, just call run manually to test logic
        # But run is an infinite loop, so we need to mock queue.get
        worker.queue = MagicMock()
        worker.queue.get.side_effect = [
            "msg 1",
            Empty(), # Should continue
            "msg 2",
            None # Should break
        ]
        
        worker.run()
        
        assert mock_send.call_count == 2
        mock_send.assert_any_call(123, "reply 1")
        mock_send.assert_any_call(123, "error: agent error")

@patch("obektclaw.gateways.telegram.Store")
@patch("obektclaw.gateways.telegram.SkillManager")
@patch("obektclaw.gateways.telegram.httpx.Client")
@patch("obektclaw.gateways.telegram.CONFIG")
def test_run_no_token(mock_config, mock_client, mock_skills, mock_store, capsys):
    mock_config.tg_token = ""
    assert run() == 1
    out, err = capsys.readouterr()
    assert "OBEKTCLAW_TG_TOKEN not set" in out

@patch("obektclaw.gateways.telegram.Store")
@patch("obektclaw.gateways.telegram.SkillManager")
@patch("obektclaw.gateways.telegram.httpx.Client")
@patch("obektclaw.gateways.telegram.CONFIG")
@patch("obektclaw.gateways.telegram.time.sleep")
@patch("obektclaw.gateways.telegram._ChatWorker")
def test_run_success(mock_worker_cls, mock_sleep, mock_config, mock_client_cls, mock_skills, mock_store, capsys):
    mock_config.tg_token = "token"
    mock_config.tg_allowed_chat_ids = [123]
    
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    mock_worker = MagicMock()
    mock_worker_cls.return_value = mock_worker
    
    # We want to simulate a few polling iterations then a KeyboardInterrupt
    
    # Response 1: error
    # Response 2: valid update, but chat not allowed
    # Response 3: valid update, chat allowed, new worker
    # Response 4: valid update, chat allowed, existing worker
    # Response 5: valid update, edited_message
    # Response 6: KeyboardInterrupt
    
    def get_side_effect(*args, **kwargs):
        call_count = mock_client.get.call_count
        if call_count == 1:
            raise ValueError("test error")
        elif call_count == 2:
            resp = MagicMock()
            resp.json.return_value = {
                "result": [{"update_id": 1, "message": {"chat": {"id": 999}, "text": "hello"}}]
            }
            return resp
        elif call_count == 3:
            resp = MagicMock()
            resp.json.return_value = {
                "result": [{"update_id": 2, "message": {"chat": {"id": 123}, "text": "hello"}}]
            }
            return resp
        elif call_count == 4:
            resp = MagicMock()
            resp.json.return_value = {
                "result": [{"update_id": 3, "message": {"chat": {"id": 123}, "text": "  "}}] # Empty text
            }
            return resp
        elif call_count == 5:
            resp = MagicMock()
            resp.json.return_value = {
                "result": [{"update_id": 4, "edited_message": {"chat": {"id": 123}, "text": "hello 2"}}]
            }
            return resp
        elif call_count == 6:
            raise KeyboardInterrupt()
            
    mock_client.get.side_effect = get_side_effect
    
    # We need to test the inner send function. Since run() has a local function send(), we can't easily patch it.
    # But wait, run() initializes the worker with send. If we don't mock _ChatWorker, it will start a thread and call send.
    # Let's test send separately or let the worker thread run briefly.
    # Actually, we mocked _ChatWorker here, so we capture the send function!
    assert run() == 0

    assert mock_worker_cls.call_count == 1
    # send is at position 3, send_chat_action is at position 4
    send_fn = mock_worker_cls.call_args[0][3]
    send_chat_action_fn = mock_worker_cls.call_args[0][4]

    # Test send function
    send_fn(123, "test_msg")
    mock_client.post.assert_called_once()
    
    # Test send failure
    mock_client.post.side_effect = httpx.HTTPError("send err")
    send_fn(123, "test_msg")
    out, err = capsys.readouterr()
    assert "[tg] send failed: send err" in out

@patch("obektclaw.gateways.telegram.Store")
@patch("obektclaw.gateways.telegram.SkillManager")
@patch("obektclaw.gateways.telegram.httpx.Client")
@patch("obektclaw.gateways.telegram.CONFIG")
def test_run_empty_updates(mock_config, mock_client_cls, mock_skills, mock_store):
    mock_config.tg_token = "token"
    mock_config.tg_allowed_chat_ids = [] # Allow all
    
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    def get_side_effect(*args, **kwargs):
        call_count = mock_client.get.call_count
        if call_count == 1:
            resp = MagicMock()
            # empty result list
            resp.json.return_value = {"result": []}
            return resp
        elif call_count == 2:
            resp = MagicMock()
            # result without message or edited_message
            resp.json.return_value = {"result": [{"update_id": 1, "inline_query": {}}]}
            return resp
        elif call_count == 3:
            raise KeyboardInterrupt()
            
    mock_client.get.side_effect = get_side_effect
    
    assert run() == 0

