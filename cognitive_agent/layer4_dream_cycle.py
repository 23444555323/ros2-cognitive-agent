import random
import threading
import concurrent.futures
from typing import List, Dict, Any, Callable, Optional
from .layer3_world_model import DroneState, WorldModel

class Chromosome:
    """Action sequence genome."""
    def __init__(self, actions: List[str]):
        self.actions = actions
        self.fitness = 0.0

    def __repr__(self):
        return f"Chromosome(fitness={self.fitness:.3f}, actions={self.actions})"

class FitnessEvaluator:
    """Evaluate in world model."""
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def evaluate(self, chromosome: Chromosome, current_state: DroneState, goal_state: DroneState) -> float:
        # Simulate the action sequence and calculate fitness based on distance to goal
        sim_state = current_state
        for action in chromosome.actions:
            sim_state = self.world_model.step(sim_state, action)

        # Fitness is inverse distance to goal
        distance = float(((sim_state.position - goal_state.position)**2).sum()**0.5)
        chromosome.fitness = 1.0 / (1.0 + distance)
        return chromosome.fitness

class GeneticAlgorithm:
    """Evolution loop."""
    def __init__(self, population_size: int, mutation_rate: float, world_model: WorldModel):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.evaluator = FitnessEvaluator(world_model)
        self.possible_actions = ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "TAKE_OFF", "LAND"]

    def evolve(self, current_state: DroneState, goal_state: DroneState, generations: int) -> Chromosome:
        # Initial population
        population = [
            Chromosome([random.choice(self.possible_actions) for _ in range(5)])
            for _ in range(self.population_size)
        ]

        for _ in range(generations):
            # Evaluate
            for chromo in population:
                self.evaluator.evaluate(chromo, current_state, goal_state)

            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Simple evolution: Keep top 20%, replace others with mutated versions of top
            top_cutoff = self.population_size // 5
            new_population = population[:top_cutoff]

            while len(new_population) < self.population_size:
                parent = random.choice(population[:top_cutoff])
                child_actions = parent.actions.copy()
                if random.random() < self.mutation_rate:
                    # Mutate one action
                    idx = random.randint(0, len(child_actions)-1)
                    child_actions[idx] = random.choice(self.possible_actions)
                new_population.append(Chromosome(child_actions))

            population = new_population

        return population[0]

class AsyncDreamWorker:
    """Background thread for evolution."""
    def __init__(self, ga: GeneticAlgorithm):
        self.ga = ga

    def run_evolution(self, current_state: DroneState, goal_state: DroneState, generations: int) -> Chromosome:
        return self.ga.evolve(current_state, goal_state, generations)

class DreamCycle:
    """Orchestrates async dream evolution."""
    def __init__(self, config: Dict[str, Any], world_model: WorldModel):
        self.ga = GeneticAlgorithm(
            population_size=config.get("population_size", 30),
            mutation_rate=config.get("mutation_rate", 0.1),
            world_model=world_model
        )
        self.worker = AsyncDreamWorker(self.ga)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def trigger(self, current_state: DroneState, goal_state: DroneState, generations: int):
        return self.executor.submit(self.worker.run_evolution, current_state, goal_state, generations)
