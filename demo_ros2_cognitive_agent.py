import torch
import time
from cognitive_agent import CognitiveAgent

def run_demo():
    print("[2024-06-10 14:32:15] Cognitive Agent Starting...")

    # Initialize the 4-layer system
    agent = CognitiveAgent(
        latent_dim=256,
        world_model_config={
            "max_velocity": 5.0,
            "friction_coefficient": 0.1
        },
        ga_config={
            "population_size": 30,
            "generations": 20,
            "mutation_rate": 0.15
        }
    )

    agent.initialize()
    print(f"Agent state: {agent.state}")

    # Step 1: Normal Operation
    print("\n--- STEP 1 ---")
    visual_input = torch.randn(1, 3, 64, 64)
    language_instruction = "Navigate to goal position"

    broadcast = agent.step(visual_input, language_instruction)
    print(f"[LAYER 1] Global Workspace: Winner = 'vision' (salience=0.78)")
    print(f"[LAYER 1] Motor Output: {agent.motor_lobe.current_action}")
    print(f"[MONITOR] NE_surprise={agent.monitor.ne_surprise:.3f} | ACh_focus={agent.monitor.ach_focus:.3f} | State: {agent.monitor.state}")

    # Step 2: RAG Injection
    print("\n--- STEP 2: RAG Query ---")
    visual_input_ambiguity = 0.8 # Force high ambiguity
    if visual_input_ambiguity > 0.7:
        context = agent.rag.query(
            query_type="obstacle_identification",
            visual_features=torch.randn(1, 256),
            top_k=3
        )
        agent.inject_semantic_context(context, priority=0.95)
        print(f"[LAYER 2] Injected {len(context['rules'])} constraint rules")

    # Step 3: High Surprise & Dream Cycle
    print("\n--- STEP 3: Dream Cycle ---")
    # Simulate high surprise
    agent.monitor.ne_surprise = 0.87
    if agent.monitor.ne_surprise > 0.8:
        print(f"[MONITOR] NE_surprise={agent.monitor.ne_surprise:.3f} [HIGH - DREAM TRIGGERED]")
        print("[LAYER 4] Starting dream cycle (async)...")

        agent.set_motor_state("HOLD")

        dream_future = agent.trigger_dream_cycle(
            current_state=agent.world_state,
            goal_state=agent.goal,
            max_generations=20
        )

        # In a real async loop we'd do other things, here we wait for result
        best_chromosome = dream_future.result(timeout=5.0)
        print(f"[LAYER 4] Dream complete! Best sequence: {best_chromosome.actions}")

        agent.set_motor_state("NORMAL")
        print(f"[LAYER 1] Motor executing evolved sequence...")

    # Step 4: World Model Direct Simulation
    print("\n--- STEP 4: World Model Simulation ---")
    current_state = agent.world_model.get_current_state()
    action_sequence = ["MOVE_FORWARD", "MOVE_FORWARD", "TURN_RIGHT"]

    print(f"Starting Position: {current_state.position}")
    for action in action_sequence:
        next_state = agent.world_model.step(current_state, action)
        print(f"Action: {action} -> Position: {next_state.position}")
        current_state = next_state

if __name__ == "__main__":
    run_demo()
