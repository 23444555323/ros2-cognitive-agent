import torch
from cognitive_agent.layer2_rag_system import (
    MockVectorDatabase, SemanticMemoryInterface, ConstraintPropagator
)

def test_layer2_instantiation():
    db = MockVectorDatabase()
    rag = SemanticMemoryInterface()
    propagator = ConstraintPropagator()
    assert len(db.data) > 0

def test_layer2_methods():
    rag = SemanticMemoryInterface()
    context = rag.query("obstacle_identification", torch.randn(1, 256))
    assert "rules" in context

    propagator = ConstraintPropagator()
    constraints = propagator.propagate(context, priority=0.9)
    assert len(constraints) > 0
    assert constraints[0]["priority"] == 0.9
