"""Limit arm motion toward the legs using distances in the Q1 collision model."""
import argparse
from pathlib import Path

import numpy as np
import evaluate
import eval_core
from common import mj
from capture_blend import controller


class Clearance:
    def __init__(self, xml, clearance):
        self.model = mj.MjModel.from_xml_path(str(xml))
        self.data = mj.MjData(self.model)
        self.margin = clearance
        self.pairs = list(zip(self.model.pair_geom1, self.model.pair_geom2, strict=True))
        assert len(self.pairs) == 96
        self.segment = np.zeros(6)
        self.left = np.zeros((3,self.model.nv))
        self.right = self.left.copy()

    def project(self, q, qd, target, tracker):
        m, d = self.model, self.data
        d.qpos[7:] = q
        mj.mj_kinematics(m,d)
        mj.mj_comPos(m,d)
        step = target-q
        # Short command horizon includes sensing delay. Both endpoints use
        # joint proprioception; relative clearance is invariant to base pose.
        step = np.clip(step,-.13,.13)
        lookahead = .025*qd
        correction = np.zeros_like(step)
        active = False
        for first, second in self.pairs:
            if np.linalg.norm(d.geom_xpos[first]-d.geom_xpos[second]) > (
                m.geom_rbound[first]+m.geom_rbound[second]+.055
            ):
                continue
            distance = mj.mj_geomDistance(m,d,first,second,.055,self.segment)
            if distance >= .055 or abs(distance) < 1e-7:
                continue
            direction = (self.segment[3:]-self.segment[:3])/distance
            mj.mj_jac(m,d,self.left,None,self.segment[:3],m.geom_bodyid[first])
            mj.mj_jac(m,d,self.right,None,self.segment[3:],m.geom_bodyid[second])
            grad = direction @ (self.right-self.left)[:,6:]
            residual = self.margin - distance - grad @ (step+lookahead+correction)
            if residual <= 0:
                continue
            arm = grad[14:]
            correction[14:] += arm * min(residual/(arm@arm+1e-5),20.)
            active = True
        if not active:
            return None
        # Attenuate only the arm commands while clearing an obstacle.
        raw = target.copy()
        raw[14:] = q[14:] + step[14:] + np.clip(correction[14:],-.22,.22)
        tau = np.clip(tracker.kp*(raw-q)-tracker.kd*qd,-tracker.eff,tracker.eff)
        return q+(tau+tracker.kd*qd)/tracker.kp, tau


def run(args):
    guard = Clearance(args.xml,args.margin)
    parent = controller(8)
    class Guarded(parent):
        def step(self,R,w,q,qd,previous,base,anchor,*,sample_time_s,teleop_stamp_s):
            was_teleop = self.owner == "teleop"
            target,tau,owner = super().step(R,w,q,qd,previous,base,anchor,
                sample_time_s=sample_time_s,teleop_stamp_s=teleop_stamp_s)
            if owner:
                if was_teleop:
                    self.side_origin = q.copy() if abs(R[2,1])>.65 else None
                    self.side_ticks = 0
                if self.side_origin is not None and self.side_ticks < args.side_ramp:
                    self.side_ticks += 1
                    weight = .5-.5*np.cos(np.pi*self.side_ticks/args.side_ramp)
                    target = target.copy()
                    joints = slice(14,None) if args.arms_only else slice(None)
                    target[joints] = self.side_origin[joints] + weight*(target[joints]-self.side_origin[joints])
                    c = self.tracker
                    tau = np.clip(c.kp*(target-q)-c.kd*qd,-c.eff,c.eff)
                    target = q+(tau+c.kd*qd)/c.kp
                limited = guard.project(q,qd,target,self.tracker)
                if limited is not None:
                    target,tau = limited
            return target,tau,owner
    eval_core.Q1AutonomousRecovery=Guarded
    evaluate.main(argparse.Namespace(group=args.group,legacy=False,catch=False,out=args.out,
                                     seed=args.seed,expert=args.expert,delay=11))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("expert",type=Path)
    p.add_argument("xml",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--seed",type=int,default=92017)
    p.add_argument("--margin",type=float,default=.008)
    p.add_argument("--side-ramp",type=int,default=0)
    p.add_argument("--arms-only",action="store_true")
    p.add_argument("--group",choices=["static","push"],default="static")
    run(p.parse_args())
