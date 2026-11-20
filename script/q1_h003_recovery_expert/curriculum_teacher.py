"""Evaluate learned movement sequences as curriculum teachers on disjoint starts."""
import sys
from pathlib import Path
import numpy as np
from bench import HERE,evaluate
from recovery import Recovery,Gate

class Sequence:
    def __init__(self,ctrl,roll_path,stand_path,feedback=None,settle=0):
        self.settle=settle
        self.feedback=feedback
        self.ctrl=ctrl
        self.roll={j:np.load(roll_path/f"keys_{j}.npy") for j in (0,2,3)}
        self.stand=np.load(stand_path/"keys_1.npy")
        self.reset()

    def reset(self):
        self.mode=None
        self.tick=0
        self.start=None
        self.kind=None
        self.wait=0

    def run(self,outputs,inputs):
        x=inputs["rec_obs_hist"][0,-1]
        q=x[:22]+self.ctrl.default_q
        up=-x[47:50]
        if self.mode is None and self.wait<self.settle:
            self.wait+=1
            return [self.ctrl.default_q[None].copy()]
        if self.mode is None:
            self.kind=0 if -up[0]>abs(up[1]) else (1 if up[0]>abs(up[1]) else (2 if up[1]>0 else 3))
            self.mode="rise" if self.kind==1 else "roll"
            self.start=q.copy()
        if self.mode=="roll" and self.tick>=93 and (up[0]>.65 or self.tick>=122):
            self.mode="rise"
            self.tick=0
            self.start=q.copy()
        keys=self.stand if self.mode=="rise" else self.roll[self.kind]
        phase=min(6,self.tick//14)
        a=self.start if phase==0 else keys[phase-1] if phase<=5 else self.ctrl.default_q
        b=keys[phase] if phase<5 else self.ctrl.default_q
        blend=.5-.5*np.cos(np.pi*(1+self.tick%14)/14.)
        raw=a+(b-a)*blend
        if self.mode=="rise" and self.feedback is not None and self.tick>68:
            g=self.feedback
            pitch=np.arcsin(np.clip(-up[0],-1.,1.))
            roll=np.arctan2(up[1],up[2])
            wx,wy=x[44:46]*4
            ramp=min(1.,(self.tick-68)/13.)
            for joints,term in (([0,6],g[0]+g[1]*pitch+g[2]*wy),([4,10],g[3]+g[4]*pitch+g[5]*wy),
                               ([3,9],g[6]+g[7]*abs(pitch)),([14,18],g[8]*pitch),
                               ([5,11],g[9]*roll+g[10]*wx),([1,7],g[11]*roll)):
                raw[joints]+=ramp*term
            raw=np.clip(raw,self.ctrl.q_lo,self.ctrl.q_hi)
        self.tick+=1
        return [raw[None].astype(np.float32)]

class Teacher(Recovery):
    def __init__(self,ctrl,path):
        self.ctrl=ctrl
        self.gate=Gate()
        self.policy=Sequence(ctrl,HERE/"rolls_01",HERE/"teachers_01",np.load(path) if path!="teacher" else None,int(sys.argv[4]) if len(sys.argv)>4 else 0)
        self.reset()
    def reset(self):
        super().reset()
        self.policy.reset()

if __name__=="__main__":
    evaluate(Path(sys.argv[1]),int(sys.argv[2]),sys.argv[3] if len(sys.argv)>3 else "teacher",Teacher)

