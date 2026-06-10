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
        # Simple discretization for state matching
        rounded_pos = np.round(state.position, 1)
        return str(tuple(rounded_pos))

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

    def evolve(self, current_state: DroneState, goal_state: DroneState, generations: int) -> Chromosome:
        pop = np.random.randint(0, self.num_actions, size=(self.population_size, self.seq_len))
        best_fitness = -1.0
        best_chromo = None

        for _ in range(generations):
            fitnesses = np.zeros(self.population_size)
            # Fitness evaluation loop (Physically grounded)
            for i in range(self.population_size):
                temp_state = current_state
                for idx in pop[i]:
                    action = ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "TAKE_OFF", "LAND"][idx]
                    temp_state = self.world_model.step(temp_state, action)

                dist = np.linalg.norm(temp_state.position - goal_state.position)
                fitnesses[i] = 1.0 / (1.0 + dist)

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
