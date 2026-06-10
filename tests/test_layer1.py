import torch
from cognitive_agent.layer1_neuromorphic_core import (
    VisualLobe, LanguageLobe, MotorLobe, ThreadSafeGlobalWorkspace, MetacognitiveMonitor
)

def test_layer1_instantiation():
    visual = VisualLobe(256)
    language = LanguageLobe(256)
    motor = MotorLobe()
    workspace = ThreadSafeGlobalWorkspace()
    monitor = MetacognitiveMonitor()

    assert visual.latent_dim == 256
    assert motor.state == "NORMAL"
    assert monitor.ne_surprise == 0.0

def test_layer1_methods():
    visual = VisualLobe(256)
    encoding = visual.process(torch.randn(1, 3, 64, 64))
    assert encoding.shape == (1, 256)

    language = LanguageLobe(256)
    l_encoding = language.process("test instruction")
    assert l_encoding.shape == (1, 256)

    motor = MotorLobe()
    action = motor.execute(torch.randn(1, 6))
    assert isinstance(action, str)

    workspace = ThreadSafeGlobalWorkspace()
    workspace.write("test", 123)
    assert workspace.read("test") == 123
