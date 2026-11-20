"""Exportable causal movement and balance policy fitted on Q1 physics."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from bench import BUNDLE,HERE

class SequenceExpert(nn.Module):
    def __init__(self,directory):
        super().__init__()
        cfg=json.loads((BUNDLE/"config/q1_deploy_config.json").read_text())
        for name,key in (("stand","default_q"),("lo","q_lo"),("hi","q_hi")):
            self.register_buffer(name,torch.tensor(cfg[key],dtype=torch.float32))
        rise=np.load(directory/"keys_1.npy")
        bank=np.stack([np.load(HERE/f"rolls_01/keys_{k}.npy") if k!=1 else rise for k in range(4)]+[rise])
        self.register_buffer("keys",torch.tensor(bank,dtype=torch.float32))
        self.register_buffer("gain",torch.tensor(np.load(directory/"feedback.npy"),dtype=torch.float32))
        self.residual_scale=.04
        self.residual=nn.Sequential(nn.Linear(74*3,96),nn.Tanh(),nn.Linear(96,22))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def baseline(self,history,state):
        x=history[:,-1]
        q=x[:,:22]+self.stand
        up=-x[:,47:50]
        age,mode,kind,tick=state[:,:4].unbind(1)
        start=state[:,4:]
        calm=(x[:,44:47].square().sum(1)<(.25*1.2)**2)&(x[:,22:44].abs().mean(1)<.25)
        retry=(mode==2)&(tick>146)&((up[:,2]<.65)|((x[:,-1]>.5)&(x[:,-2]<.23)))
        mode=torch.where(retry,0.,mode)
        age=torch.where(retry,0.,age)
        enter=(mode==0)&(age>=15)&(calm|(age>=60))
        chosen=torch.where(-up[:,0]>up[:,1].abs(),0,torch.where(up[:,0]>up[:,1].abs(),1,torch.where(up[:,1]>0,2,3)))
        kind=torch.where(enter,chosen.float(),kind)
        to_rise=(mode==1)&(tick>=93)&((up[:,0]>.65)|(tick>=122))
        mode=torch.where(enter,torch.where(kind==1,2.,1.),mode)
        mode=torch.where(to_rise,2.,mode)
        start=torch.where((enter|to_rise)[:,None],q,start)
        tick=torch.where(enter|to_rise,0.,tick)
        bank=torch.where(mode==2,4.,kind).long()
        segment=torch.div(tick,14,rounding_mode="floor").clamp(0,6)
        a=self.keys[bank,(segment-1).clamp(0,4).long()]
        a=torch.where((segment==0)[:,None],start,torch.where((segment>=6)[:,None],self.stand,a))
        b=torch.where((segment>=5)[:,None],self.stand,self.keys[bank,segment.clamp(0,4).long()])
        blend=.5-.5*torch.cos(torch.pi*(1+torch.remainder(tick,14))/14)
        raw=a+(b-a)*blend[:,None]
        pitch=torch.asin((-up[:,0]).clamp(-1.,1.))
        roll=torch.atan2(up[:,1],up[:,2])
        wx=x[:,44]*4
        wy=x[:,45]*4
        g=self.gain
        hip=g[0]+g[1]*pitch+g[2]*wy
        ankle=g[3]+g[4]*pitch+g[5]*wy
        knee=g[6]+g[7]*pitch.abs()
        arm=g[8]*pitch
        ankle_roll=g[9]*roll+g[10]*wx
        hip_roll=g[11]*roll
        zero=torch.zeros_like(pitch)
        leg=[hip,hip_roll,zero,knee,ankle,ankle_roll]
        delta=torch.stack(leg+leg+[zero,zero,arm,zero,zero,zero,arm,zero,zero,zero],1)
        ramp=torch.where(mode==2,((tick-68)/13).clamp(0,1),0.)
        target=torch.where((mode==0)[:,None],self.stand,(raw+ramp[:,None]*delta).clamp(self.lo,self.hi))
        new_state=torch.cat((torch.stack(((age+1).clamp(max=500),mode,kind,torch.where(mode==0,0.,(tick+1).clamp(max=500))),1),start),1)
        return target,new_state

    def mean(self,hist):
        return self.residual(hist[:,[0,5,9]].flatten(1))

    def forward(self,rec_obs_hist,rec_state):
        base,state=self.baseline(rec_obs_hist,rec_state)
        return (base+self.residual_scale*torch.tanh(self.mean(rec_obs_hist))).clamp(self.lo,self.hi),state

def export(model,path):
    sample=torch.zeros(1,10,74,device=model.stand.device)
    sample[:,:,49]=-1.
    torch.onnx.export(model,(sample,torch.zeros(1,26,device=sample.device)),str(path),input_names=["rec_obs_hist","rec_state"],
                      output_names=["joint_target","next_state"],opset_version=17,dynamo=False)
if __name__=="__main__":
    import sys
    export(SequenceExpert(Path(sys.argv[1])),Path(sys.argv[2]))

