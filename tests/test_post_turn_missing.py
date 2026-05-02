import json
from obektclaw.post_turn import TurnExtractor
from unittest.mock import MagicMock


def test_turn_extractor_log_exception(tmp_path, monkeypatch):
    class MockAgent:
        config = MagicMock()
        config.logs_dir = tmp_path / "logs"

    # make log directory a file to cause OSError
    MockAgent.config.logs_dir.touch()

    agent = MockAgent()
    extractor = TurnExtractor(agent)

    # Shouldn't raise
    extractor._persist_extraction({"some": "extraction"})


def test_turn_extractor_apply_exceptions():
    agent = MagicMock()
    extractor = TurnExtractor(agent)

    # Setup mocks to raise exceptions
    agent.persistent.upsert.side_effect = ValueError
    agent.persistent.delete.side_effect = KeyError
    agent.user_model.set.side_effect = TypeError
    agent.skills.create.side_effect = TypeError
    agent.skills.improve.side_effect = KeyError

    result = {
        "facts": [{"key": "1", "value": "2"}],
        "deleted_facts": [{"key": "1"}],
        "user_model_updates": [{"layer": "1", "value": "2"}],
        "new_skill": {"name": "1", "description": "2"},
        "skill_improvement": {"name": "1", "append": "2"},
    }

    # Shouldn't raise
    extractor._apply(result)


def test_turn_extractor_apply_malformed_types():
    agent = MagicMock()
    extractor = TurnExtractor(agent)

    result = {
        "facts": None,  # or string
        "deleted_facts": None,
        "user_model_updates": None,
        "new_skill": "not a dict",
        "skill_improvement": "not a dict",
    }

    # Shouldn't raise
    extractor._apply(result)
