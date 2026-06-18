import random
import threading
import concurrent.futures
import numpy as np
import json
import os
import time
from typing import List, Dict, Any, Callable, Optional
from .layer3_world_model import DroneState, WorldModel
from .utils import log_info, log_error

# Mock ROS 2 if not available
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    # Use standard action messages if possible or define placeholders
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    class Node:
        def __init__(self, name): self.name = name
    class ActionServer:
        def __init__(self, *args, **kwargs): pass

class Chromosome:
    """Action sequence genome with NumPy optimization."""
    def __init__(self, action_indices: np.ndarray):
        self.action_indices = action_indices # NumPy array of indices
        self.fitness = 0.0

    @property
    def actions(self) -> List[str]:
        possible_actions = ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "TAKE_OFF", "LAND"]
        return [possible_actions[i] for i in self.action_indices]

class HabitBuffer:
    """Habit Compilation Mechanism: Serializes and caches successful trajectories using JSON."""
    def __init__(self, cache_dir: str = "./habit_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.habits: Dict[str, List[str]] = {}
        self._load_habits()

    def _load_habits(self):
        path = os.path.join(self.cache_dir, "habits.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.habits = json.load(f)
            except Exception:
                self.habits = {}

    def _hash_state(self, state: DroneState) -> str:
        # Improved discretization for state matching including velocity and orientation
        rounded_pos = np.round(state.position, 1)
        rounded_vel = np.round(state.velocity, 1)
        # Use only a few digits of orientation to maintain some generalization
        rounded_ori = np.round(state.orientation, 2)

        state_tuple = (
            tuple(rounded_pos),
            tuple(rounded_vel),
            tuple(rounded_ori)
        )
        return str(state_tuple)

    def get_habit(self, state: DroneState) -> Optional[List[str]]:
        state_hash = self._hash_state(state)
        return self.habits.get(state_hash)

    def compile_habit(self, state: DroneState, actions: List[str]):
        state_hash = self._hash_state(state)
        self.habits[state_hash] = actions
        # Serialize to disk securely via JSON
        with open(os.path.join(self.cache_dir, "habits.json"), "w") as f:
            json.dump(self.habits, f)

class GeneticAlgorithm:
    """Vectorized Evolution loop for high performance."""
    def __init__(self, population_size: int, mutation_rate: float, world_model: WorldModel):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.world_model = world_model
        self.num_actions = 6
        self.seq_len = 10

    def _predict_vectorized(self, current_pos: np.ndarray, current_vel: np.ndarray, actions_batch: np.ndarray) -> tuple:
        """Analytical kinematics model for vectorized GA evaluation."""
        forces = np.zeros((len(actions_batch), 3))
        # Mapping indices to forces as defined in PyBulletSimulator
        forces[actions_batch == 0] = [10, 0, 0]  # MOVE_FORWARD
        forces[actions_batch == 1] = [-10, 0, 0] # MOVE_BACKWARD
        forces[actions_batch == 4] = [0, 0, 15]  # TAKE_OFF

        gravity = np.array([0, 0, -9.81])
        dt = 1.0 / 240.0 # Standard PyBullet timestep

        acc = (forces + gravity) # mass is 1.0
        next_vel = current_vel + acc * dt
        next_pos = current_pos + next_vel * dt

        # Simple ground constraint
        next_pos[:, 2] = np.maximum(next_pos[:, 2], 0.0)

        return next_pos, next_vel

    def evolve(self, current_state: DroneState, goal_state: DroneState, generations: int) -> Chromosome:
        pop = np.random.randint(0, self.num_actions, size=(self.population_size, self.seq_len))
        best_fitness = -1.0
        best_chromo = None

        goal_pos = np.array(goal_state.position)

        for _ in range(generations):
            # Vectorized fitness evaluation
            pos = np.tile(current_state.position, (self.population_size, 1))
            vel = np.tile(current_state.velocity, (self.population_size, 1))

            for t in range(self.seq_len):
                pos, vel = self._predict_vectorized(pos, vel, pop[:, t])

            dists = np.linalg.norm(pos - goal_pos, axis=1)
            fitnesses = 1.0 / (1.0 + dists)

            idx = np.argsort(fitnesses)[::-1]
            pop = pop[idx]
            fitnesses = fitnesses[idx]

            if fitnesses[0] > best_fitness:
                best_fitness = fitnesses[0]
                best_chromo = Chromosome(pop[0].copy())

            mutation_mask = np.random.rand(self.population_size, self.seq_len) < self.mutation_rate
            pop[mutation_mask] = np.random.randint(0, self.num_actions, size=np.sum(mutation_mask))

        return best_chromo

class DreamCycle(Node):
    """Orchestrates async dream evolution via ROS 2 Action Server."""
    def __init__(self, config: Dict[str, Any], world_model: WorldModel):
        super().__init__('dream_cycle')
        self.ga = GeneticAlgorithm(
            population_size=config.get("population_size", 100),
            mutation_rate=config.get("mutation_rate", 0.1),
            world_model=world_model
        )
        self.habit_buffer = HabitBuffer()
        self.executor_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        if HAS_ROS2:
            # Note: In a real ROS 2 environment, 'Dream' would be a generated interface.
            # self._action_server = ActionServer(
            #     self,
            #     'Dream',
            #     'dream_evolution',
            #     self.execute_callback)
            pass

    def execute_callback(self, goal_handle):
        """ROS 2 Action Server Callback."""
        log_info('Executing dream action...')
        # Real-time evolution results would be sent back to client
        # goal_handle.succeed()
        return None

    def trigger(self, current_state: DroneState, goal_state: DroneState, generations: int):
        habit_actions = self.habit_buffer.get_habit(current_state)
        if habit_actions:
            log_info("Habit found! Bypassing GA.")
            action_map = {"MOVE_FORWARD":0, "MOVE_BACKWARD":1, "TURN_LEFT":2, "TURN_RIGHT":3, "TAKE_OFF":4, "LAND":5}
            indices = np.array([action_map[a] for a in habit_actions])
            f = concurrent.futures.Future()
            f.set_result(Chromosome(indices))
            return f

        return self.executor_pool.submit(self.ga.evolve, current_state, goal_state, generations)
