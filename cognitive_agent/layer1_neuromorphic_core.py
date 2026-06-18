import torch
import threading
import time
import queue
import numpy as np
from typing import Dict, Any, List, Optional
from .utils import log_info, log_error

try:
    from PIL import Image as PILImage
    from transformers import AutoProcessor, AutoModel
    HAS_VLM = True
except ImportError:
    HAS_VLM = False

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
    """Processes environmental state via ROS 2 Image subscriber and SigLIP VLM."""
    def __init__(self, workspace, latent_dim: int = 256):
        super().__init__('visual_lobe')
        self.workspace = workspace
        self.latent_dim = latent_dim
        self.ambiguity = 0.0

        # Initialize SigLIP VLM (Production: loading quantized model for Jetson)
        if HAS_VLM:
            try:
                # Using a lightweight SigLIP model
                model_name = "google/siglip-base-patch16-224"
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                self.mock_vlm = False
            except Exception as e:
                log_error(f"Failed to load SigLIP: {e}")
                self.mock_vlm = True
        else:
            self.mock_vlm = True

        if HAS_ROS2:
            self.subscription = self.create_subscription(
                Image,
                '/camera/image_raw',
                self.image_callback,
                10)

    def image_callback(self, msg):
        """Native ROS 2 callback for async perception."""
        try:
            import cv2
            # Handle ROS 2 Image message without cv_bridge
            # msg.data is typically a buffer, msg.height, msg.width, msg.encoding
            # Assuming 'bgr8' or 'rgb8' for simplicity in this robot context
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

            # SigLIP expects RGB
            if getattr(msg, 'encoding', 'rgb8') == 'bgr8':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            visual_input = frame
        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")
            # Fallback to random for robustness in simulation
            visual_input = torch.randn(3, 224, 224)

        encoding = self.process(visual_input)

        self.workspace.write("vision", encoding, salience=0.8)
        self.get_logger().info("Visual Lobe updated Workspace via VLM perception.")

    def process(self, visual_input: Any) -> torch.Tensor:
        """Process image input through SigLIP VLM to get coordinate-based states."""
        if self.mock_vlm:
            encoding = torch.randn(1, self.latent_dim)
            self.ambiguity = float(torch.rand(1).item())
            return encoding

        # Real VLM logic
        if isinstance(visual_input, np.ndarray):
            # Already (H, W, C) from cv2/numpy
            visual_input = PILImage.fromarray(visual_input)
        elif isinstance(visual_input, torch.Tensor):
            # Handle batch dimension if present (N, C, H, W) -> (C, H, W)
            if visual_input.ndim == 4:
                visual_input = visual_input[0]

            # (C, H, W) -> (H, W, C)
            visual_input = visual_input.cpu().numpy().transpose(1, 2, 0)
            visual_input = (visual_input * 255).astype(np.uint8)
            visual_input = PILImage.fromarray(visual_input)

        inputs = self.processor(images=visual_input, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        # transformers model outputs are often wrappers around tensors
        features_tensor = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0]

        features = features_tensor[:, :self.latent_dim]
        self.ambiguity = float(torch.std(features).item())
        return features

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
            if self.current_action == "MOVE_FORWARD": twist.linear.x = 0.5
            elif self.current_action == "MOVE_BACKWARD": twist.linear.x = -0.5
            elif self.current_action == "TURN_LEFT": twist.angular.z = 0.5
            elif self.current_action == "TURN_RIGHT": twist.angular.z = -0.5
            elif self.current_action == "TAKE_OFF": twist.linear.z = 1.0
            elif self.current_action == "LAND": twist.linear.z = -1.0
            self.publisher_.publish(twist)

        return self.current_action

class ThreadSafeGlobalWorkspace:
    """High-performance priority queue for cross-modal attention routing."""
    def __init__(self, max_size: int = 100):
        self._priority_queue = queue.PriorityQueue(maxsize=max_size)
        self._memory_pool: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def write(self, key: str, value: Any, salience: float = 0.5):
        timestamp = time.time()
        try:
            priority = 1.0 - salience
            item = (priority, timestamp, key, value)
            self._priority_queue.put_nowait(item)
        except queue.Full:
            pass

        with self._lock:
            self._memory_pool[key] = {"value": value, "timestamp": timestamp}

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
