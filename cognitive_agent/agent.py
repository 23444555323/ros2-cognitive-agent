import torch
import time
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
from .utils import log_info

# Mock ROS 2 if not available
try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

class CognitiveAgent:
    """
    Orchestrates the 4-layer Neuro-Symbolic Cognitive Architecture.
    """
    def __init__(self, latent_dim=256, world_model_config=None, ga_config=None):
        self.latent_dim = latent_dim
        self.workspace = ThreadSafeGlobalWorkspace()

        # ROS 2 Nodes
        self.visual_lobe = VisualLobe(self.workspace, latent_dim)
        self.motor_lobe = MotorLobe()
        self.rag = SemanticMemoryInterface()

        # Local Components
        self.language_lobe = LanguageLobe(latent_dim)
        self.monitor = MetacognitiveMonitor()
        self.propagator = ConstraintPropagator()

        wm_config = world_model_config or {"max_velocity": 5.0, "friction_coefficient": 0.1}
        self.world_model = WorldModel(wm_config)

        ga_conf = ga_config or {"population_size": 100, "mutation_rate": 0.1}
        self.dream_cycle = DreamCycle(ga_conf, self.world_model)

        self.state = "INITIALIZING"
        self.world_state = DroneState()
        self.goal = DroneState(position=(5.0, 5.0, 2.0))
        self.is_dreaming = False

    def initialize(self):
        self.state = "READY"
        log_info("Neuro-Symbolic Cognitive Agent Ready.")

    def spin(self):
        """Native ROS 2 execution loop."""
        if HAS_ROS2:
            executor = MultiThreadedExecutor()
            executor.add_node(self.visual_lobe)
            executor.add_node(self.motor_lobe)
            executor.add_node(self.rag)
            executor.add_node(self.dream_cycle)
            executor.spin()
        else:
            log_info("ROS 2 not found. Manual step required.")

    def step(self, visual_input=None, language_input=""):
        # Layer 1: Neuro Perception (Manual trigger fallback)
        if visual_input is not None:
            visual_encoding = self.visual_lobe.process(visual_input)
            self.workspace.write("vision", visual_encoding, salience=0.8)
        else:
            visual_encoding = self.workspace.read("vision")
            if visual_encoding is None:
                visual_encoding = torch.zeros(1, self.latent_dim)

        # Layer 2: Symbolic Override / RAG
        semantic_context = None
        if self.visual_lobe.ambiguity > 0.7:
            semantic_context = self.rag.query("Identify environmental features")
            self.propagator.propagate(semantic_context, priority=0.9)

        language_encoding = self.language_lobe.process(language_input, semantic_context)
        self.workspace.write("language", language_encoding, salience=0.6)

        # Metacognitive Monitor Update
        prediction_error = float(self.visual_lobe.ambiguity)
        self.monitor.update(prediction_error=prediction_error, task_demand=0.5)

        # Autonomous Dreaming Trigger
        if self.monitor.ne_surprise > 0.8 and not self.is_dreaming:
            log_info(f"Surprise {self.monitor.ne_surprise:.2f} above threshold. Triggering Dream Cycle.")
            self.is_dreaming = True
            self.set_motor_state("HOLD")

            dream_future = self.trigger_dream_cycle(self.world_state, self.goal, generations=50)

            def dream_done_callback(future):
                try:
                    best_chromo = future.result()
                    log_info(f"Dream complete. Evolved actions: {best_chromo.actions}")
                    self.dream_cycle.habit_buffer.compile_habit(self.world_state, best_chromo.actions)
                finally:
                    self.set_motor_state("NORMAL")
                    self.is_dreaming = False

            dream_future.add_done_callback(dream_done_callback)

        # Motor Lobe execution
        action = self.motor_lobe.execute(visual_encoding)

        # Update our internal world state
        self.world_state = self.world_model.step(self.world_state, action)

        return self.workspace.get_broadcast()

    def trigger_dream_cycle(self, current_state, goal_state, generations):
        return self.dream_cycle.trigger(current_state, goal_state, generations)

    def set_motor_state(self, state):
        self.motor_lobe.state = state

    def inject_semantic_context(self, context, priority):
        self.propagator.propagate(context, priority)
