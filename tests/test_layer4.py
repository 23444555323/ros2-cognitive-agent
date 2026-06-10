import numpy as np
from cognitive_agent.layer4_dream_cycle import GeneticAlgorithm, DroneState, Chromosome, HabitBuffer
from cognitive_agent.layer3_world_model import WorldModel

def test_vectorized_ga():
    wm = WorldModel()
    ga = GeneticAlgorithm(population_size=10, mutation_rate=0.1, world_model=wm)

    current = DroneState(position=(0,0,0))
    goal = DroneState(position=(5,5,5))

    best = ga.evolve(current, goal, generations=2)
    assert isinstance(best, Chromosome)
    assert len(best.action_indices) == 10

def test_habit_compilation():
    buffer = HabitBuffer(cache_dir="./test_habits")
    state = DroneState(position=(1.23, 4.56, 7.89))
    actions = ["MOVE_FORWARD", "TAKE_OFF"]

    buffer.compile_habit(state, actions)
    retrieved = buffer.get_habit(state)

    assert retrieved == actions

    # Test tolerance
    close_state = DroneState(position=(1.2, 4.6, 7.9))
    assert buffer.get_habit(close_state) == actions
