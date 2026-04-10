import json
from obektclaw.learning import LearningLoop
from unittest.mock import MagicMock

def test_learning_loop_log_exception(tmp_path, monkeypatch):
    class MockAgent:
        config = MagicMock()
        config.logs_dir = tmp_path / "logs"
    
    # make log directory a file to cause OSError
    MockAgent.config.logs_dir.touch()
    
    agent = MockAgent()
    loop = LearningLoop(agent)
    
    # Shouldn't raise
    loop._persist_retro({"some": "retro"})

def test_learning_loop_apply_exceptions():
    agent = MagicMock()
    loop = LearningLoop(agent)
    
    # Setup mocks to raise exceptions
    agent.persistent.upsert.side_effect = ValueError
    agent.persistent.delete.side_effect = KeyError
    agent.user_model.set.side_effect = TypeError
    agent.skills.create.side_effect = TypeError
    agent.skills.improve.side_effect = KeyError
    
    retro = {
        "facts": [{"key": "1", "value": "2"}],
        "deleted_facts": [{"key": "1"}],
        "user_model_updates": [{"layer": "1", "value": "2"}],
        "new_skill": {"name": "1", "description": "2"},
        "skill_improvement": {"name": "1", "append": "2"},
    }
    
    # Shouldn't raise
    loop._apply(retro)

def test_learning_loop_apply_malformed_types():
    agent = MagicMock()
    loop = LearningLoop(agent)
    
    retro = {
        "facts": None, # or string
        "deleted_facts": None,
        "user_model_updates": None,
        "new_skill": "not a dict",
        "skill_improvement": "not a dict",
    }
    
    # Shouldn't raise
    loop._apply(retro)
