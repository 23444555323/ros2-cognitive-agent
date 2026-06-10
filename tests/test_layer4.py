from cognitive_agent.layer4_dream_cycle import (
    Chromosome, GeneticAlgorithm, DreamCycle
)
from cognitive_agent.layer3_world_model import WorldModel, DroneState

def test_layer4_instantiation():
    wm = WorldModel({})
    ga = GeneticAlgorithm(30, 0.1, wm)
    dream = DreamCycle({}, wm)
    assert ga.population_size == 30

def test_layer4_methods():
    wm = WorldModel({})
    ga = GeneticAlgorithm(10, 0.1, wm)
    best = ga.evolve(DroneState(), DroneState(position=(5,5,5)), generations=5)
    assert isinstance(best, Chromosome)
    assert len(best.actions) == 5
