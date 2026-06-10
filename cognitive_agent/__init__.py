"""
ROS 2 Cognitive Agent: 4-Layer Neuro-Symbolic Cognitive Architecture
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
    SemanticMemoryInterface,
)
from .layer3_world_model import (
    DroneState,
    WorldModel,
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
    "SemanticMemoryInterface",
    "DroneState",
    "WorldModel",
    "Chromosome",
    "GeneticAlgorithm",
    "DreamCycle",
]

__version__ = "0.2.0"
