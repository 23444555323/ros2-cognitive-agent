import torch
from cognitive_agent import CognitiveAgent

def test_integration_demo():
    agent = CognitiveAgent()
    agent.initialize()

    visual_input = torch.randn(1, 3, 64, 64)
    language_input = "Go to goal"

    broadcast = agent.step(visual_input, language_input)
    assert "vision" in broadcast
    assert "language" in broadcast

    # Trigger dream cycle
    future = agent.trigger_dream_cycle(agent.world_state, agent.goal, max_generations=2)
    result = future.result(timeout=2.0)
    assert result is not None
