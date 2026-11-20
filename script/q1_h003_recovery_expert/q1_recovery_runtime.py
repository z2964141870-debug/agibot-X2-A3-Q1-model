"""50 Hz advisory joint-target controller for the frozen V2b tracker and recovery expert."""
from pathlib import Path
import numpy as np
from leg_height import LegHeight
from recovery import Recovery

class Q1RecoveryBranch:
    """Wrap an initialized Q1V2bOnnxController. All observations are instantaneous proprioception."""

    def __init__(self,tracker,expert_onnx,q1_xml):
        self.tracker=tracker
        self.legs=LegHeight(Path(q1_xml))
        self.expert=Recovery(tracker,expert_onnx)

    def reset(self):
        self.expert.reset()

    def step(self,R_imu,omega_body,q,qd,previous_sent_action,base_step,reset_anchor):
        """Return a safe joint target, estimated PD torque and recovery weight.

        base_step computes the current reference, runs the original tracker once,
        and returns its (target, torque, info) tuple. reset_anchor rebuilds CGFA
        and initializes the causal reference window from the measured posture.
        """
        up=np.asarray(R_imu,dtype=np.float32)[2]
        joints=np.asarray(q,dtype=np.float32)
        obs=np.concatenate((joints-self.tracker.default_q,qd,omega_body,-up,previous_sent_action)).astype(np.float32)
        height,valid=self.legs.measure(up,joints)
        width=self.legs.width(up,joints)
        return self.expert.step_measured(obs,height,valid,width,base_step,reset_anchor)

