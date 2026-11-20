"""Stateful recovery wrapper around the frozen Q1 deployment controller."""
from math import cos, pi
import numpy as np
import onnxruntime as ort
from q1_v2b_runtime import apply_torque_safe_pd_np

def frame(obs, soles, feet):
    lowest=int(np.argmin(soles[:,2]))
    valid=float(feet[lowest,2,2]>.65)
    height=float(np.clip(-soles[lowest,2],0,.65))*valid
    return sensor_frame(obs,height,valid)

def sensor_frame(obs,height,valid):
    return np.concatenate((obs[:22],obs[22:44]*.1,obs[44:47]*.25,
                           obs[47:50],obs[50:72]*.25,[height,valid])).astype(np.float32)

class Gate:
    def reset(self):
        self.mode="track"
        self.tilt_ticks=self.stand_ticks=self.blend_ticks=0

    def advance(self,x,width):
        cosine=-float(x[49])
        self.tilt_ticks=self.tilt_ticks+1 if cosine<.65 or (x[-1] and x[-2]<.19) else 0
        reset_anchor=False
        if self.mode in ("track","handoff") and self.tilt_ticks>=3:
            self.mode="recover"
            self.stand_ticks=self.blend_ticks=0
        if self.mode=="recover":
            stable=cosine>.92 and np.sqrt(np.mean(x[:22]**2))<.32 and x[-1] and x[-2]>.35
            self.stand_ticks=self.stand_ticks+1 if stable else 0
            dwell=40 if width>.35 else 25
            if self.stand_ticks>=dwell:
                self.mode="handoff"
                reset_anchor=True
        if self.mode=="handoff":
            self.blend_ticks+=1
            if self.blend_ticks>=30:
                self.mode="track"
                return 0.,reset_anchor
            return .5*(1+cos(pi*self.blend_ticks/30)),reset_anchor
        return float(self.mode=="recover"),reset_anchor

class Recovery:
    def __init__(self,ctrl,path):
        self.ctrl=ctrl
        options=ort.SessionOptions()
        options.intra_op_num_threads=options.inter_op_num_threads=1
        self.policy=ort.InferenceSession(str(path),options,providers=["CPUExecutionProvider"])
        self.stateful=any(x.name=="rec_state" for x in self.policy.get_inputs())
        self.gate=Gate()
        self.reset()

    def reset(self):
        self.state=np.zeros((1,26),dtype=np.float32)
        self.history=None
        self.gate.reset()

    def step_measured(self,obs,height,valid,width,base,reset_anchor):
        return self._step(obs,sensor_frame(obs,height,valid),width,base,reset_anchor)

    def step(self,obs,soles,feet,base,reset_anchor):
        up=-obs[47:50]
        # Heading-frame y in body coordinates is perpendicular to gravity and forward.
        # Callers with world-coordinate soles supply their heading-local width explicitly.
        width=float(np.linalg.norm(soles[0,:2]-soles[1,:2]))
        return self._step(obs,frame(obs,soles,feet),width,base,reset_anchor)

    def _step(self,obs,x,width,base,reset_anchor):
        if self.history is None:
            self.history=np.repeat(x[None,None],10,axis=1)
        else:
            self.history[0,:-1]=self.history[0,1:]
            self.history[0,-1]=x
        prior_mode=self.gate.mode
        weight,reanchor=self.gate.advance(x,width)
        if self.gate.mode=="recover" and prior_mode!="recover": self.state.fill(0.)
        c=self.ctrl
        q=obs[:22]+c.default_q
        qd=obs[22:44]
        if reanchor:
            c.reset(q)
            reset_anchor()
        if weight==0:
            target,tau,_=base()
            return target,tau,0.
        if getattr(self,"stateful",False):
            result,self.state=self.policy.run(["joint_target","next_state"],{"rec_obs_hist":self.history,"rec_state":self.state})
            raw=result[0]
        else:
            raw=self.policy.run(["joint_target"],{"rec_obs_hist":self.history})[0][0]
        if weight<1:
            flat,_,_=base()
            raw=flat+weight*(raw-flat)
        target=apply_torque_safe_pd_np(raw,q,qd,c.q_lo,c.q_hi,c.kp,c.kd,c.eff,engage_steps=0)
        return target,c.kp*(target-q)-c.kd*qd,weight

