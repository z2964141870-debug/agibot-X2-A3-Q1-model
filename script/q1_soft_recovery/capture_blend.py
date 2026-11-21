"""Interpolate the first private-standing commands from the last issued recovery command."""
import argparse
import math
from pathlib import Path
import numpy as np

import evaluate
import eval_core
from low_catch import MidCrouchCatch


def controller(ticks):
    class BlendedCapture(MidCrouchCatch):
        def reset(self):
            super().reset()
            self.origin_target = None
            self.blend_index = ticks
            self.transitions = []

        def step(self,R,w,q,qd,previous,base,anchor,*,sample_time_s,teleop_stamp_s):
            before = self.rise.expert.gate.mode
            target,tau,owned = super().step(
                R,w,q,qd,previous,base,anchor,sample_time_s=sample_time_s,teleop_stamp_s=teleop_stamp_s)
            after = self.rise.expert.gate.mode
            if before == "recover" and after in ("track","handoff") and owned:
                self.origin_target = self.tracker.default_q + self.tracker.action_scale*previous
                self.blend_index = 0
                self.transitions.append(sample_time_s)
            if not owned or after == "recover":
                self.blend_index = ticks
            if self.blend_index >= ticks:
                return target,tau,owned
            self.blend_index += 1
            weight = .5-.5*math.cos(math.pi*self.blend_index/ticks)
            raw = self.origin_target + weight*(target-self.origin_target)
            c=self.tracker
            torque=np.clip(c.kp*(raw-q)-c.kd*qd,-c.eff,c.eff)
            return q+(torque+c.kd*qd)/c.kp,torque,owned
    return BlendedCapture


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("expert",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--ticks",type=int,required=True);p.add_argument("--seed",type=int,default=92017)
    p.add_argument("--group",choices=["static","push","flat","offline","double"],default="static")
    a=p.parse_args();eval_core.Q1AutonomousRecovery=controller(a.ticks)
    evaluate.main(argparse.Namespace(group=a.group,legacy=False,catch=False,out=a.out,seed=a.seed,expert=a.expert,delay=11))
