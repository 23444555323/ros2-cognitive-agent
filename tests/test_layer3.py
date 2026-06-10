import numpy as np
from cognitive_agent.layer3_world_model import (
    DroneState, WorldModel, PhysicsSimulator
)

def test_layer3_instantiation():
    state = DroneState()
    model = WorldModel({"max_velocity": 5.0})
    sim = PhysicsSimulator()
    assert np.all(state.position == 0)

def test_layer3_methods():
    model = WorldModel({"max_velocity": 5.0})
    current = model.get_current_state()
    next_state = model.step(current, "MOVE_FORWARD")
    assert not np.array_equal(current.position, next_state.position)
