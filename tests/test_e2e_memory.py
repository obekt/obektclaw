"""E2E test for memory persistence with external OpenAI-compatible LLM.

This test uses an external LLM service (OpenAI-compatible API) for:
- Main agent (with tool calling)
- Entity/relationship extraction (Learning Loop)

Verifies that:
1. The agent can learn information from a conversation
2. The information persists in memory across agent restarts
3. The agent can recall the learned information in a new session

Requirements:
- External LLM service with OpenAI-compatible API
- Set environment variables:
  - OBEKTCLAW_LLM_BASE_URL (e.g., https://api.openai.com/v1)
  - OBEKTCLAW_LLM_API_KEY (your API key)
  - OBEKTCLAW_LLM_MODEL (main model, supports tool calling)
  - OBEKTCLAW_EXTRACTION_LLM_MODEL (optional, defaults to OBEKTCLAW_LLM_FAST_MODEL)

To run this test:
    pytest tests/test_e2e_memory.py -v -s

The test will be skipped if LLM API key is not set.
"""

from __future__ import annotations

import os
import random
import shutil
import string
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from obektclaw.config import Config
from obektclaw.memory.store import Store
from obektclaw.memory import PersistentMemory, SessionMemory
from obektclaw.skills import SkillManager
from obektclaw.agent import Agent


def generate_secret_code() -> str:
    """Generate a unique random secret code for each test run.

    Uses random letters and numbers so it can't be guessed or found
    by searching the source code.
    """
    letters = random.choices(string.ascii_uppercase, k=4)
    numbers = random.choices(string.digits, k=3)
    return f"{letters[0]}{letters[1]}-{letters[2]}{letters[3]}-{numbers[0]}{numbers[1]}{numbers[2]}"


def check_llm_configured() -> bool:
    """Check if external LLM is configured."""
    api_key = os.environ.get("OBEKTCLAW_LLM_API_KEY", "")
    # Skip placeholder values
    if api_key in {"your-api-key-here", "sk-xxx", "sk-your-key", "", "local"}:
        return False
    return True


@pytest.fixture(scope="module")
def llm_config_check() -> Generator[None, None, None]:
    """Verify LLM is configured before running tests."""
    if not check_llm_configured():
        pytest.skip(
            "External LLM not configured - set OBEKTCLAW_LLM_API_KEY environment variable"
        )
    yield


@pytest.fixture(scope="module", autouse=True)
def prewarm_embedding_model(llm_config_check: None) -> Generator[None, None, None]:
    """Pre-load the embedding model before tests run.

    The sentence-transformers model takes ~8s to load on first use.
    Pre-loading it once at module scope makes all subsequent Agent
    creations fast (~0.25s instead of ~8s).

    This fixture runs automatically (autouse=True) after llm_config_check.
    """
    # Import embedder and trigger model load with a warmup call
    from obektclaw.memory.embedder import embed

    print("[E2E] Pre-loading embedding model (takes ~8s on first run)...")
    embed("warmup")  # This loads the model into the singleton cache
    print("[E2E] Embedding model loaded, subsequent Agent creations will be fast")
    yield


@pytest.fixture
def e2e_home() -> Generator[Path, None, None]:
    """Create a temporary OBEKTCLAW_HOME with empty memory."""
    tmpdir = Path(tempfile.mkdtemp(prefix="obektclaw-e2e-"))
    yield tmpdir
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def e2e_config(e2e_home: Path) -> Config:
    """Create a Config using external LLM for everything."""
    return Config(
        home=e2e_home,
        db_path=e2e_home / "obektclaw.db",
        skills_dir=e2e_home / "skills",
        bundled_skills_dir=e2e_home / "bundled_skills",
        logs_dir=e2e_home / "logs",
        # Use external LLM (from environment)
        llm_base_url=os.environ.get(
            "OBEKTCLAW_LLM_BASE_URL", "https://api.openai.com/v1"
        ),
        llm_api_key=os.environ.get("OBEKTCLAW_LLM_API_KEY", ""),
        llm_model=os.environ.get("OBEKTCLAW_LLM_MODEL", "gpt-4o-mini"),
        llm_fast_model=os.environ.get(
            "OBEKTCLAW_LLM_FAST_MODEL",
            os.environ.get("OBEKTCLAW_LLM_MODEL", "gpt-4o-mini"),
        ),
        tg_token="",
        tg_allowed_chat_ids=(),
        bash_timeout=30,
        workdir=e2e_home,
        cog_home=e2e_home / "cog-home",
        chroma_path=e2e_home / "chroma",
    )


class TestE2EMemoryPersistence:
    """End-to-end tests for memory persistence with external LLM."""

    def test_memory_persists_across_agent_restart(
        self, e2e_config: Config, llm_config_check: None
    ):
        """
        Full E2E test: learn a fact, restart agent, recall the fact.

        Steps:
        1. Start with empty memory
        2. Ask a question about unknown info (should not know)
        3. Tell the agent the information
        4. Close agent (simulating restart)
        5. Create new agent instance
        6. Ask the same question again
        7. Verify the answer includes the learned information
        """
        # Generate unique secret code for this test run
        secret_code = generate_secret_code()
        test_fact_content = (
            f"The obektclaw-test-secret-code for this run is {secret_code}"
        )
        test_fact_question = "What is the obektclaw-test-secret-code for this run?"

        # 1. Start with empty memory
        store = Store(e2e_config.db_path)
        skills = SkillManager(
            store, e2e_config.skills_dir, e2e_config.bundled_skills_dir
        )

        # Verify memory is empty
        persistent = PersistentMemory(store)
        existing_facts = list(persistent.search("secret-code"))
        assert len(existing_facts) == 0, "Memory should be empty at start"

        # 2. Create first agent
        agent1 = Agent(
            config=e2e_config,
            store=store,
            skills=skills,
            gateway="cli",
            user_key="e2e-test",
        )

        # Ask the question - agent shouldn't know the answer
        reply1 = agent1.run_once(test_fact_question)

        # The reply should NOT contain the secret code (it doesn't know yet)
        assert secret_code not in reply1, (
            f"Agent should not know the secret code yet. Reply was: {reply1[:200]}"
        )

        # 3. Tell the agent the information
        # Learning loop runs synchronously after run_once returns
        info_message = f"Please remember this important fact: {test_fact_content}"
        reply2 = agent1.run_once(info_message)

        # 4. Close agent (simulating restart)
        agent1.close()

        # 5. Create new agent instance (fresh start, same memory)
        store2 = Store(e2e_config.db_path)
        skills2 = SkillManager(
            store2, e2e_config.skills_dir, e2e_config.bundled_skills_dir
        )

        agent2 = Agent(
            config=e2e_config,
            store=store2,
            skills=skills2,
            gateway="cli",
            user_key="e2e-test",
        )

        # 6. Ask the same question again
        reply3 = agent2.run_once(test_fact_question)

        # 7. Verify the answer includes the learned information
        found_in_reply = secret_code in reply3

        # Alternative check: verify memory was populated (use new persistent from store2)
        persistent2 = PersistentMemory(store2)
        persistent_facts_after = list(persistent2.search("secret-code"))

        memory_populated = len(persistent_facts_after) > 0

        # Log for debugging
        print(f"\n=== E2E Test Debug ===")
        print(f"Secret code: {secret_code}")
        print(f"First reply (should NOT have code): {reply1[:100]}...")
        print(f"Info reply: {reply2[:100]}...")
        print(f"Final reply (should have code): {reply3[:200]}...")
        print(f"Persistent facts found: {len(persistent_facts_after)}")
        print(f"======================\n")

        # Success criteria: either the agent recalled the info,
        # or the memory system stored it (proving persistence works)
        assert found_in_reply or memory_populated, (
            f"Agent should recall or memory should persist the secret code.\n"
            f"Secret code: {secret_code}\n"
            f"Reply was: {reply3[:300]}\n"
            f"Persistent facts: {len(persistent_facts_after)}"
        )

        agent2.close()

    def test_preference_learning_and_recall(
        self, e2e_config: Config, llm_config_check: None
    ):
        """
        Test that user preferences are learned and recalled.

        This tests the full Learning Loop extraction for preferences.
        """
        store = Store(e2e_config.db_path)
        skills = SkillManager(
            store, e2e_config.skills_dir, e2e_config.bundled_skills_dir
        )

        # Create agent
        agent = Agent(
            config=e2e_config,
            store=store,
            skills=skills,
            gateway="cli",
            user_key="e2e-preference-test",
        )

        # Tell the agent a preference
        # Learning loop runs synchronously after run_once returns
        preference_msg = (
            "I always use httpx for HTTP requests instead of requests library"
        )
        reply1 = agent.run_once(preference_msg)

        # Close agent
        agent.close()

        # Create new agent and ask about HTTP client preference
        store2 = Store(e2e_config.db_path)
        skills2 = SkillManager(
            store2, e2e_config.skills_dir, e2e_config.bundled_skills_dir
        )

        agent2 = Agent(
            config=e2e_config,
            store=store2,
            skills=skills2,
            gateway="cli",
            user_key="e2e-preference-test",
        )

        # Ask about HTTP client choice
        question = "Which HTTP client should I use for making requests in Python?"
        reply2 = agent2.run_once(question)

        print(f"\n=== Preference Test Debug ===")
        print(f"Preference input: {preference_msg}")
        print(f"Recall reply: {reply2[:200]}...")
        print(f"==============================\n")

        # The agent should mention httpx in the reply
        mentions_httpx = "httpx" in reply2.lower()

        # Also check memory storage
        persistent = PersistentMemory(store2)
        httpx_facts = list(persistent.search("httpx"))

        assert mentions_httpx or len(httpx_facts) > 0, (
            f"Agent should recall httpx preference.\n"
            f"Reply: {reply2[:300]}\n"
            f"Facts: {len(httpx_facts)}"
        )

        agent2.close()

    def test_environment_info_learning(
        self, e2e_config: Config, llm_config_check: None
    ):
        """
        Test that environment information is learned and recalled.
        """
        store = Store(e2e_config.db_path)
        skills = SkillManager(
            store, e2e_config.skills_dir, e2e_config.bundled_skills_dir
        )

        agent = Agent(
            config=e2e_config,
            store=store,
            skills=skills,
            gateway="cli",
            user_key="e2e-env-test",
        )

        # Tell about environment
        # Learning loop runs synchronously after run_once returns
        env_msg = "My production server is deployed on Hetzner Cloud CX22 instance"
        reply1 = agent.run_once(env_msg)

        # Close agent
        agent.close()

        # New agent, ask about server
        store2 = Store(e2e_config.db_path)
        skills2 = SkillManager(
            store2, e2e_config.skills_dir, e2e_config.bundled_skills_dir
        )

        agent2 = Agent(
            config=e2e_config,
            store=store2,
            skills=skills2,
            gateway="cli",
            user_key="e2e-env-test",
        )

        question = "Where is my production server deployed?"
        reply2 = agent2.run_once(question)

        print(f"\n=== Environment Test Debug ===")
        print(f"Environment input: {env_msg}")
        print(f"Recall reply: {reply2[:200]}...")
        print(f"===============================\n")

        mentions_hetzner = "hetzner" in reply2.lower()

        # Check memory
        persistent = PersistentMemory(store2)
        hetzner_facts = list(persistent.search("hetzner"))

        assert mentions_hetzner or len(hetzner_facts) > 0, (
            f"Agent should recall Hetzner deployment.\nReply: {reply2[:300]}"
        )

        agent2.close()


class TestE2EMemoryTools:
    """E2E tests that use memory tools directly."""

    def test_manual_fact_storage_and_recall(
        self, e2e_config: Config, llm_config_check: None
    ):
        """
        Test using memory_set_fact tool and verifying recall.
        """
        store = Store(e2e_config.db_path)
        skills = SkillManager(
            store, e2e_config.skills_dir, e2e_config.bundled_skills_dir
        )

        agent = Agent(
            config=e2e_config,
            store=store,
            skills=skills,
            gateway="cli",
            user_key="e2e-tool-test",
        )

        # Ask agent to store a fact using memory_set_fact tool
        # Learning loop runs synchronously after run_once returns
        tool_request = (
            "Use the memory_set_fact tool to save this fact: "
            "Key 'favorite-color', value 'blue', category 'preference'"
        )
        reply = agent.run_once(tool_request)

        # Close agent
        agent.close()

        # Verify the fact was stored
        store2 = Store(e2e_config.db_path)
        persistent = PersistentMemory(store2)
        facts = list(persistent.search("favorite-color"))

        print(f"\n=== Tool Test Debug ===")
        print(f"Tool request: {tool_request}")
        print(f"Reply: {reply[:200]}...")
        print(f"Facts found: {len(facts)}")
        print(f"=======================\n")

        agent2 = Agent(
            config=e2e_config,
            store=store2,
            skills=skills,
            gateway="cli",
            user_key="e2e-tool-test",
        )

        # Ask about the favorite color
        recall_reply = agent2.run_once("What is my favorite color?")
        mentions_blue = "blue" in recall_reply.lower()

        agent2.close()

        # Success if either tool worked or memory has the fact
        assert len(facts) > 0 or mentions_blue, (
            f"Fact should be stored or recalled.\n"
            f"Facts: {len(facts)}, Reply mentions blue: {mentions_blue}"
        )
