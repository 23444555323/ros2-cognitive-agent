import torch
from .layer1_neuromorphic_core import (
    VisualLobe,
    LanguageLobe,
    MotorLobe,
    ThreadSafeGlobalWorkspace,
    MetacognitiveMonitor,
)
from .layer2_rag_system import SemanticMemoryInterface, ConstraintPropagator
from .layer3_world_model import WorldModel, DroneState
from .layer4_dream_cycle import DreamCycle

class CognitiveAgent:
    """
    Orchestrates the 4-layer cognitive architecture.
    """
    def __init__(self, latent_dim=256, world_model_config=None, ga_config=None):
        self.latent_dim = latent_dim
        self.visual_lobe = VisualLobe(latent_dim)
        self.language_lobe = LanguageLobe(latent_dim)
        self.motor_lobe = MotorLobe()
        self.workspace = ThreadSafeGlobalWorkspace()
        self.monitor = MetacognitiveMonitor()
        self.rag = SemanticMemoryInterface()
        self.propagator = ConstraintPropagator()

        wm_config = world_model_config or {"max_velocity": 5.0, "friction_coefficient": 0.1}
        self.world_model = WorldModel(wm_config)

        ga_conf = ga_config or {"population_size": 30, "mutation_rate": 0.1}
        self.dream_cycle = DreamCycle(ga_conf, self.world_model)

        self.state = "INITIALIZING"
        self.world_state = DroneState()
        self.goal = DroneState(position=(10.0, 10.0, 0.0))

    def initialize(self):
        self.state = "READY"

    def step(self, visual_input, language_input):
        # Layer 1: Forward Pass
        visual_encoding = self.visual_lobe.process(visual_input)
        language_encoding = self.language_lobe.process(language_input)

        # Write to workspace
        self.workspace.write("vision", visual_encoding)
        self.workspace.write("language", language_encoding)

        # Metacognitive Monitor update
        self.monitor.update(prediction_error=self.visual_lobe.ambiguity, task_demand=0.5)

        # Global Workspace Broadcast
        broadcast = self.workspace.get_broadcast()

        # Motor Lobe execution
        action = self.motor_lobe.execute(visual_encoding)

        # Update our internal world state
        self.world_state = self.world_model.step(self.world_state, action)

        return broadcast

    def trigger_dream_cycle(self, current_state, goal_state, max_generations):
        return self.dream_cycle.trigger(current_state, goal_state, max_generations)

    def set_motor_state(self, state):
        self.motor_lobe.state = state

    def inject_semantic_context(self, context, priority):
        self.propagator.propagate(context, priority)
