import torch
from demo_ros2_cognitive_agent import run_production_demo

def test_full_production_flow():
    # This basically runs the demo as a test
    try:
        run_production_demo()
    except Exception as e:
        assert False, f"Production demo failed with error: {e}"
