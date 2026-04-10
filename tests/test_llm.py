import json
import pytest
from unittest.mock import MagicMock, patch
from openai import APIError, APIConnectionError, RateLimitError
from obektclaw.llm import LLMClient, ToolCall, LLMResponse

def test_llm_client_init_no_key():
    with pytest.raises(RuntimeError, match="OBEKTCLAW_LLM_API_KEY is not set"):
        LLMClient(base_url="http://test", api_key="", model="test")

def test_llm_client_init():
    client = LLMClient(base_url="http://test", api_key="sk-test", model="test-model", fast_model="test-fast")
    assert client.model == "test-model"
    assert client.fast_model == "test-fast"

    client2 = LLMClient(base_url="http://test", api_key="sk-test", model="test-model")
    assert client2.fast_model == "test-model"

@pytest.fixture
def mock_openai_client():
    with patch("obektclaw.llm.OpenAI") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance
        yield mock_instance

def create_mock_completion(content, tool_calls=None):
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    
    if tool_calls is not None:
        mock_tool_calls = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc["id"]
            mock_tc.function.name = tc["name"]
            mock_tc.function.arguments = tc.get("arguments")
            mock_tool_calls.append(mock_tc)
        mock_choice.message.tool_calls = mock_tool_calls
    else:
        mock_choice.message.tool_calls = None
        
    mock_resp.choices = [mock_choice]
    return mock_resp

def test_llm_chat_success(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    
    mock_openai_client.chat.completions.create.return_value = create_mock_completion(
        content="Hello!",
        tool_calls=[{"id": "call_1", "name": "get_weather", "arguments": '{"loc": "SF"}'}]
    )
    
    resp = client.chat(
        messages=[{"role": "user", "content": "Hi"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
        fast=False
    )
    
    assert resp.content == "Hello!"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "call_1"
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == '{"loc": "SF"}'
    
    mock_openai_client.chat.completions.create.assert_called_once()
    kwargs = mock_openai_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "test-model"
    assert kwargs["tool_choice"] == "auto"
    assert len(kwargs["tools"]) == 1

def test_llm_chat_fast(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model", fast_model="test-fast")
    mock_openai_client.chat.completions.create.return_value = create_mock_completion(content="Fast")
    
    client.chat([{"role": "user", "content": "Hi"}], fast=True)
    kwargs = mock_openai_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "test-fast"

def test_llm_chat_missing_arguments(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    mock_openai_client.chat.completions.create.return_value = create_mock_completion(
        content="Hi",
        tool_calls=[{"id": "call_1", "name": "bad_tool", "arguments": None}]
    )
    resp = client.chat([{"role": "user", "content": "Hi"}])
    assert resp.tool_calls[0].arguments == "{}"

def test_llm_chat_retries(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    
    # Fail 3 times, succeed on 4th
    mock_req = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = [
        RateLimitError("Rate limit", response=MagicMock(), body=None),
        APIConnectionError(request=mock_req),
        APIError("API error", request=mock_req, body=None),
        create_mock_completion(content="Success!")
    ]
    
    with patch("obektclaw.llm.time.sleep") as mock_sleep:
        resp = client.chat([{"role": "user", "content": "Hi"}])
        
    assert resp.content == "Success!"
    assert mock_openai_client.chat.completions.create.call_count == 4
    assert mock_sleep.call_count == 3

def test_llm_chat_exhaust_retries(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    
    mock_openai_client.chat.completions.create.side_effect = RateLimitError("Rate limit", response=MagicMock(), body=None)
    
    with patch("obektclaw.llm.time.sleep"):
        with pytest.raises(RuntimeError, match="LLM call failed after retries"):
            client.chat([{"role": "user", "content": "Hi"}])
            
    assert mock_openai_client.chat.completions.create.call_count == 4

def test_llm_chat_simple(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    mock_openai_client.chat.completions.create.return_value = create_mock_completion(content="  Simple!  ")
    
    resp = client.chat_simple("sys", "usr")
    assert resp == "Simple!"

def test_llm_chat_json_success(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    mock_openai_client.chat.completions.create.return_value = create_mock_completion(content='{"key": "value"}')
    
    resp = client.chat_json("sys", "usr")
    assert resp == {"key": "value"}

def test_llm_chat_json_with_markdown(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    mock_openai_client.chat.completions.create.return_value = create_mock_completion(content='```json\n{"key": "value"}\n```')
    
    resp = client.chat_json("sys", "usr")
    assert resp == {"key": "value"}

def test_llm_chat_json_with_markdown_upper(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    mock_openai_client.chat.completions.create.return_value = create_mock_completion(content='```JSON\n{"key": "value"}\n```')
    
    resp = client.chat_json("sys", "usr")
    assert resp == {"key": "value"}
    
def test_llm_chat_json_fallback_extraction(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    mock_openai_client.chat.completions.create.return_value = create_mock_completion(content='Here is your json: \n{"key": "value"}\n Hope it helps!')
    
    resp = client.chat_json("sys", "usr")
    assert resp == {"key": "value"}

def test_llm_chat_json_fallback_extraction_failure(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    mock_openai_client.chat.completions.create.return_value = create_mock_completion(content='Here is your json: \n{"key": "value"\n Hope it helps!')
    
    resp = client.chat_json("sys", "usr")
    assert resp is None

def test_llm_chat_json_fallback_extraction_inner_failure(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    mock_openai_client.chat.completions.create.return_value = create_mock_completion(content='Here is your json: \n{invalid: value}\n Hope it helps!')
    
    resp = client.chat_json("sys", "usr")
    assert resp is None

def test_llm_chat_json_total_failure(mock_openai_client):
    client = LLMClient(base_url="", api_key="sk-test", model="test-model")
    mock_openai_client.chat.completions.create.return_value = create_mock_completion(content='No json here')
    
    resp = client.chat_json("sys", "usr")
    assert resp is None
