"""
ROS 2 Cognitive Agent: 4-Layer Neuromorphic Architecture

A production-ready implementation of brain-inspired cognitive architecture
for autonomous robotic agents based on Global Workspace Theory (GWT).

Layers:
  1. Neuromorphic Core: Visual/Language/Motor lobes + Global Workspace
  2. RAG System: Semantic memory with ambiguity-triggered retrieval
  3. World Model: Physics simulator for safe off-policy prediction
  4. Dream Cycle: Genetic algorithm evolution in asynchronous background thread
"""

from .agent import CognitiveAgent
from .layer1_neuromorphic_core import (
    VisualLobe,
    LanguageLobe,
    MotorLobe,
    ThreadSafeGlobalWorkspace,
    MetacognitiveMonitor,
)
from .layer2_rag_system import (
    MockVectorDatabase,
    SemanticMemoryInterface,
)
from .layer3_world_model import (
    DroneState,
    WorldModel,
    PhysicsSimulator,
)
from .layer4_dream_cycle import (
    Chromosome,
    GeneticAlgorithm,
    DreamCycle,
)

__all__ = [
    "CognitiveAgent",
    "VisualLobe",
    "LanguageLobe",
    "MotorLobe",
    "ThreadSafeGlobalWorkspace",
    "MetacognitiveMonitor",
    "MockVectorDatabase",
    "SemanticMemoryInterface",
    "DroneState",
    "WorldModel",
    "PhysicsSimulator",
    "Chromosome",
    "GeneticAlgorithm",
    "DreamCycle",
]

__version__ = "0.1.0"
