from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from .utils import log_info, log_error

try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False

class DroneState:
    """Realistic drone state variables."""
    def __init__(self, position: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.position = np.array(position)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(4) # Quaternion [x,y,z,w]

    def __repr__(self):
        return f"DroneState(pos={self.position}, vel={self.velocity})"

class PyBulletSimulator:
    """Robust Physics-Engine simulation using PyBullet."""
    def __init__(self, use_gui: bool = False):
        if not HAS_PYBULLET:
            self.mock_mode = True
            return

        self.mock_mode = False
        mode = p.GUI if use_gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load drone and environment
        self.plane_id = p.loadURDF("plane.urdf")
        # Load a basic box as a drone placeholder
        self.drone_id = p.loadURDF("sphere_1cm.urdf", [0, 0, 0.5])

        self.dt = 1./240.

    def step(self, current_state: DroneState, action: str) -> DroneState:
        if self.mock_mode:
            # Fallback to basic kinematics
            next_pos = current_state.position + current_state.velocity
            return DroneState(tuple(next_pos))

        # Reset drone to current state
        p.resetBasePositionAndOrientation(self.drone_id, current_state.position, [0,0,0,1])

        # Apply forces based on action
        force = [0, 0, 0]
        if action == "MOVE_FORWARD": force = [10, 0, 0]
        elif action == "MOVE_BACKWARD": force = [-10, 0, 0]
        elif action == "TAKE_OFF": force = [0, 0, 20]

        p.applyExternalForce(self.drone_id, -1, force, [0,0,0], p.WORLD_FRAME)
        p.stepSimulation()

        # Extract next state
        pos, ori = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)

        new_state = DroneState(pos)
        new_state.velocity = np.array(vel)
        new_state.orientation = np.array(ori)
        return new_state

class WorldModel:
    """f(state, action) -> next_state simulator with PyBullet integration."""
    def __init__(self, config: Dict[str, Any] = {}):
        self.simulator = PyBulletSimulator(use_gui=config.get("use_gui", False))
        self.current_state = DroneState()

    def get_current_state(self) -> DroneState:
        return self.current_state

    def step(self, state: DroneState, action: str) -> DroneState:
        return self.simulator.step(state, action)
