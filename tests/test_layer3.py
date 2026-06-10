import numpy as np
from cognitive_agent.layer3_world_model import WorldModel, DroneState

def test_pybullet_grounding():
    model = WorldModel({"use_gui": False})
    state = DroneState(position=(0, 0, 1.0))

    # Test transition
    next_state = model.step(state, "TAKE_OFF")

    if not model.simulator.mock_mode:
        # In PyBullet, TAKE_OFF applies upward force, so Z should increase or have velocity
        assert next_state.velocity[2] > 0 or next_state.position[2] > 1.0
    else:
        assert next_state.position is not None
