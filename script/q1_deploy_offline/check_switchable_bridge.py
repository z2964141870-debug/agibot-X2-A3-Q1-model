from pathlib import Path
import argparse
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys, time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('candidate', type=Path)
args = parser.parse_args()
candidate = args.candidate.resolve()
sys.path.insert(0, str(candidate / 'code'))
import q1_ros2_lowlevel_bridge as bridge

patches = (
    patch.object(bridge.Node, '__init__', return_value=None),
    patch.object(bridge.Node, 'get_logger', return_value=Mock()),
    patch.object(bridge.Node, 'create_subscription'),
    patch.object(bridge.Node, 'create_publisher', return_value=Mock()),
)
with patches[0], patches[1], patches[2], patches[3]:
    node = bridge.Q1LowLevelBridge(
        candidate,
        enable_recovery=False,
        human_smpl=True,
        fgp_url='http://127.0.0.1:1',
        switchable_teleop=True,
    )
    q = node.default_q.copy()
    qd = np.zeros(22, np.float32)
    R = np.eye(3, dtype=np.float32)
    w = np.zeros(3, np.float32)
    tick = SimpleNamespace(
        segment=3,
        root_rotation=R,
        root_xy=np.zeros(2, np.float32),
        feature=np.zeros(430, np.float32),
    )
    queued = SimpleNamespace(tick=tick, arrival_ns=time.monotonic_ns())
    node.fgp.status = Mock(return_value={'fresh': True})
    node.fgp.pop_paced = Mock(return_value=(queued, 0, False))
    node._base_step = Mock(return_value=(q.copy(), np.zeros(22, np.float32), {}))
    node._human_smpl_step = Mock(return_value=(q.copy() + 0.01, np.zeros(22, np.float32), {}))
    node._infer_once((R, w, q, qd, None))
    assert node.get_control_mode() == 'planner'
    assert node._base_step.call_count == 1
    assert node.request_control_mode('human_smpl')
    node._infer_once((R, w, q, qd, None))
    assert node.get_control_mode() == 'human_smpl'
    assert node._human_smpl_step.call_count == 1
    node.fgp.status = Mock(return_value={'fresh': False})
    assert not node.request_control_mode('human_smpl')
    node.fgp.status = Mock(return_value={'fresh': True})
    assert node.request_control_mode('planner')
    node._infer_once((R, w, q, qd, None))
    assert node.get_control_mode() == 'planner'
    assert node._base_step.call_count == 2
print('SWITCHABLE_BRIDGE_PASS')
