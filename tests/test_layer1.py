import torch
import numpy as np
from cognitive_agent.layer1_neuromorphic_core import (
    VisualLobe, MotorLobe, ThreadSafeGlobalWorkspace
)

def test_layer1_ros_inheritance():
    workspace = ThreadSafeGlobalWorkspace()
    visual = VisualLobe(workspace, 256)
    motor = MotorLobe()
    # Check if they have Node methods (even if mocked)
    assert hasattr(visual, 'get_logger')
    assert hasattr(motor, 'create_publisher')

def test_global_workspace_priority():
    workspace = ThreadSafeGlobalWorkspace()
    workspace.write("low", 1, salience=0.1)
    workspace.write("high", 2, salience=0.9)

    # Priority queue order (lower priority value = higher salience)
    p1, t1, k1, v1 = workspace._priority_queue.get_nowait()
    p2, t2, k2, v2 = workspace._priority_queue.get_nowait()

    assert k1 == "high"
    assert k2 == "low"
    assert p1 < p2
