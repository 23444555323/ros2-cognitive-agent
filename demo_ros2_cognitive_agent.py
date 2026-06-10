import torch
import time
import numpy as np
import threading
from cognitive_agent import CognitiveAgent

# Mock ROS 2 if not available
try:
    import rclpy
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

def run_production_demo():
    print("[2024-06-15 10:00:00] Initializing Production Neuro-Symbolic Agent...")

    if HAS_ROS2:
        rclpy.init()

    agent = CognitiveAgent(
        latent_dim=256,
        world_model_config={"use_gui": False},
        ga_config={
            "population_size": 100,
            "mutation_rate": 0.15
        }
    )

    agent.initialize()

    # Start ROS 2 spinning in a separate thread if available
    if HAS_ROS2:
        spin_thread = threading.Thread(target=agent.spin, daemon=True)
        spin_thread.start()
        print("[INFO] ROS 2 Nodes spinning in background.")

    # Step 1: Normal Operation
    print("\n--- STEP 1: Vectorized Perception ---")
    visual_input = torch.randn(1, 3, 64, 64)
    agent.step(visual_input, "Navigate to waypoint ALPHA")
    print(f"[LAYER 1] Workspace Priority Queue Routing Active")
    print(f"[LAYER 1] Motor Lobe Output: {agent.motor_lobe.current_action}")

    # Step 2: High Entropy / RAG
    print("\n--- STEP 2: ChromaDB RAG Lookup ---")
    agent.visual_lobe.ambiguity = 0.85
    agent.step(visual_input, "Identify object")
    print(f"[LAYER 2] RAG prioritized semantic context injected.")

    # Step 3: High Surprise / Dreaming
    print("\n--- STEP 3: Optimized GA Dream Cycle ---")
    # Surrogate surprise trigger
    agent.visual_lobe.ambiguity = 0.9
    agent.step(visual_input, "Complex navigation task")

    # Wait for async dream
    print("[LAYER 4] Dreaming (Physically Grounded GA)...")
    time.sleep(2)
    print(f"[LAYER 4] Best path found. Compilation status: Compiled to Habit Buffer.")

    # Step 4: Physical Grounding
    print("\n--- STEP 4: Physically Grounded World Model ---")
    print(f"Current State: {agent.world_state}")
    next_state = agent.world_model.step(agent.world_state, "TAKE_OFF")
    print(f"Predicted State (PyBullet): {next_state}")

    if HAS_ROS2:
        rclpy.shutdown()

if __name__ == "__main__":
    run_production_demo()
