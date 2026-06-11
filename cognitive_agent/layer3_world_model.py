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
    """Robust Physics-Engine simulation using PyBullet with persistent sessions."""
    def __init__(self, use_gui: bool = False):
        if not HAS_PYBULLET:
            self.mock_mode = True
            log_error("PyBullet not installed. Using Mock Simulator.")
            return

        self.mock_mode = False
        mode = p.GUI if use_gui else p.DIRECT
        try:
            self.physics_client = p.connect(mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)

            # Load static environment
            self.plane_id = p.loadURDF("plane.urdf")
            # Load drone placeholder with real mass and inertia
            # In production: load specific drone URDF with joint constraints
            self.drone_id = p.loadURDF("sphere_1cm.urdf", [0, 0, 0.5], useFixedBase=False)
            p.changeDynamics(self.drone_id, -1, mass=1.0)

            self.dt = 1./240.
        except Exception as e:
            log_error(f"PyBullet initialization failed: {e}")
            self.mock_mode = True

    def step(self, current_state: DroneState, action: str) -> DroneState:
        """Runs a localized step in the physics engine instance."""
        if self.mock_mode:
            # Fallback to kinematics
            next_pos = current_state.position + current_state.velocity * 0.1
            return DroneState(tuple(next_pos))

        # Synchronize engine with agent's perceived state
        p.resetBasePositionAndOrientation(
            self.drone_id,
            current_state.position,
            current_state.orientation if np.any(current_state.orientation) else [0,0,0,1]
        )
        p.resetBaseVelocity(self.drone_id, current_state.velocity)

        # Apply physics-grounded forces
        # Linear forces (N) and Torques
        force = [0, 0, 0]
        if action == "MOVE_FORWARD": force = [10, 0, 0]
        elif action == "MOVE_BACKWARD": force = [-10, 0, 0]
        elif action == "TAKE_OFF": force = [0, 0, 15] # Oppose gravity (9.81)

        # Apply external force to the center of mass
        p.applyExternalForce(self.drone_id, -1, force, [0,0,0], p.WORLD_FRAME)

        # Simulation step
        p.stepSimulation()

        # Extract precise next-state matrices
        pos, ori = p.getBasePositionAndOrientation(self.drone_id)
        vel, ang_vel = p.getBaseVelocity(self.drone_id)

        new_state = DroneState(pos)
        new_state.velocity = np.array(vel)
        new_state.orientation = np.array(ori)
        return new_state

    def __del__(self):
        if not self.mock_mode:
            try:
                p.disconnect(self.physics_client)
            except Exception:
                pass

class WorldModel:
    """f(state, action) -> next_state simulator with PyBullet integration."""
    def __init__(self, config: Dict[str, Any] = {}):
        self.simulator = PyBulletSimulator(use_gui=config.get("use_gui", False))
        self.current_state = DroneState()

    def get_current_state(self) -> DroneState:
        return self.current_state

    def step(self, state: DroneState, action: str) -> DroneState:
        """Test 'dreams' against complex real-world gravity and constraints."""
        return self.simulator.step(state, action)
