"""Evaluate directional damping while the autonomous recovery owns the output."""
import argparse
import numpy as np
from pathlib import Path

import evaluate
import eval_core
from low_catch import MidCrouchCatch


def controller(gain, ankle_threshold, soften_shoulders):
    class SoftOwner(MidCrouchCatch):
        def step(self, R, w, q, qd, prev, base, anchor, *, sample_time_s, teleop_stamp_s):
            target, torque, owned = super().step(
                R,w,q,qd,prev,base,anchor,sample_time_s=sample_time_s,teleop_stamp_s=teleop_stamp_s)
            if not owned:
                return target,torque,owned
            c=self.tracker
            t=target.copy()
            # Remove torque driving an already fast ankle, with a proportional
            # counter-torque above threshold. Leave an opposing torque intact.
            idx=[4,10]
            speed=qd[idx]
            direction=np.sign(speed)
            amount=gain*np.maximum(0.,np.abs(speed)-ankle_threshold)
            desired_tau=np.clip(torque[idx]-direction*amount,-c.eff[idx],c.eff[idx])
            t[idx]=q[idx]+(desired_tau+c.kd[idx]*qd[idx])/c.kp[idx]
            if soften_shoulders and sample_time_s-self.entered < .30:
                idx=[16,20]
                weight=np.clip((sample_time_s-self.entered)/.30,0.,1.)
                t[idx]=q[idx]+weight*(t[idx]-q[idx])
            effort=np.clip(c.kp*(t-q)-c.kd*qd,-c.eff,c.eff)
            return q+(effort+c.kd*qd)/c.kp,effort,owned
    return SoftOwner


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("expert",type=Path);p.add_argument("out",type=Path);p.add_argument("--seed",type=int,default=92017)
    p.add_argument("--group",choices=["static","push","offline","double","flat"],default="static")
    p.add_argument("--gain",type=float,default=1.4);p.add_argument("--threshold",type=float,default=6.)
    p.add_argument("--soften-shoulders",action="store_true");a=p.parse_args()
    eval_core.Q1AutonomousRecovery=controller(a.gain,a.threshold,a.soften_shoulders)
    evaluate.main(argparse.Namespace(group=a.group,legacy=False,catch=False,out=a.out,seed=a.seed,expert=a.expert,delay=11))
