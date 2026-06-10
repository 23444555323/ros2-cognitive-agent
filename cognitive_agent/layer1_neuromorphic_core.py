import torch
import threading
import time
import queue
from typing import Dict, Any, List, Optional
from .utils import log_info, log_error

# Mock ROS 2 if not available
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    class Node:
        def __init__(self, name): self.name = name
        def create_subscription(self, *args, **kwargs): return None
        def create_publisher(self, *args, **kwargs): return None
        def get_logger(self):
            class Logger:
                def info(self, msg): print(f"[INFO] {msg}")
                def error(self, msg): print(f"[ERROR] {msg}")
            return Logger()

class VisualLobe(Node):
    """Processes environmental state via ROS 2 Image subscriber."""
    def __init__(self, workspace, latent_dim: int = 256):
        super().__init__('visual_lobe')
        self.workspace = workspace
        self.latent_dim = latent_dim
        self.ambiguity = 0.0

        if HAS_ROS2:
            self.subscription = self.create_subscription(
                Image,
                '/camera/image_raw',
                self.image_callback,
                10)

    def image_callback(self, msg):
        """Native ROS 2 callback for async perception."""
        # Parsing raw pixels into environmental state (Vectorized Perception)
        # Mocking the torch tensor conversion from ROS msg
        visual_input = torch.randn(1, 3, 64, 64)
        encoding = self.process(visual_input)

        # Write to Global Workspace immediately
        self.workspace.write("vision", encoding, salience=0.8)
        self.get_logger().info("Visual Lobe updated Workspace via ROS 2 topic.")

    def process(self, visual_input: torch.Tensor) -> torch.Tensor:
        # Standard object detection backbone placeholder
        encoding = torch.randn(1, self.latent_dim)
        self.ambiguity = float(torch.rand(1).item())
        return encoding

class LanguageLobe:
    """Processes text/semantic context."""
    def __init__(self, latent_dim: int = 256):
        self.latent_dim = latent_dim
        self.context_tokens = 0

    def process(self, language_input: str, semantic_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        self.context_tokens = len(language_input.split())
        return torch.randn(1, self.latent_dim)

class MotorLobe(Node):
    """Outputs action commands to geometry_msgs/Twist."""
    def __init__(self):
        super().__init__('motor_lobe')
        self.state = "NORMAL" # NORMAL, HOLD, etc.
        self.current_action = "NONE"

        if HAS_ROS2:
            self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

    def execute(self, motor_command: torch.Tensor) -> str:
        if self.state == "HOLD":
            return "HOVER"

        actions = ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "TAKE_OFF", "LAND"]
        self.current_action = actions[torch.argmax(motor_command).item() % len(actions)]

        if HAS_ROS2:
            twist = Twist()
            if self.current_action == "MOVE_FORWARD": twist.linear.x = 1.0
            elif self.current_action == "MOVE_BACKWARD": twist.linear.x = -1.0
            self.publisher_.publish(twist)

        return self.current_action

class ThreadSafeGlobalWorkspace:
    """High-performance priority queue for cross-modal attention routing."""
    def __init__(self, max_size: int = 100):
        self._priority_queue = queue.PriorityQueue(maxsize=max_size)
        self._memory_pool: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def write(self, key: str, value: Any, salience: float = 0.5):
        try:
            priority = 1.0 - salience
            timestamp = time.time()
            item = (priority, timestamp, key, value)
            self._priority_queue.put_nowait(item)

            with self._lock:
                self._memory_pool[key] = {"value": value, "timestamp": timestamp}
        except queue.Full:
            pass

    def read(self, key: str) -> Optional[Any]:
        with self._lock:
            data = self._memory_pool.get(key)
            return data["value"] if data else None

    def get_broadcast(self) -> Dict[str, Any]:
        with self._lock:
            return {k: v["value"] for k, v in self._memory_pool.items()}

class MetacognitiveMonitor:
    """Surprise (NE) and focus (ACh) tracking."""
    def __init__(self):
        self.ne_surprise = 0.0
        self.ach_focus = 0.5
        self.state = "NORMAL"

    def update(self, prediction_error: float, task_demand: float):
        self.ne_surprise = prediction_error
        self.ach_focus = task_demand

        if self.ne_surprise > 0.8:
            self.state = "HIGH_SURPRISE"
        else:
            self.state = "NORMAL"
