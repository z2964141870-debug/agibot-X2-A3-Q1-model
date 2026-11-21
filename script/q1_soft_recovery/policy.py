"""Causal recovery teacher and stable ONNX-compatible recurrent interface."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import STAND,LO,HI


def load_actor(model,weights,network_only=False):
    if network_only:
        selected={k:v for k,v in weights.items() if k.startswith(("encoder.","actor."))}
        incompatible=model.load_state_dict(selected,strict=False)
        assert not incompatible.unexpected_keys
        assert not any(k.startswith(("encoder.","actor.")) for k in incompatible.missing_keys)
    else:
        for name,value in model.named_buffers():
            if name in weights and not torch.equal(value.detach().cpu(),weights[name].detach().cpu()):
                raise ValueError(f"Checkpoint buffer {name} differs from the requested dataset/config; use explicit network-only transfer")
        model.load_state_dict(weights)


class Teacher(nn.Module):
    def __init__(self,reference,correction=.85):
        super().__init__()
        self.correction=correction
        self.phase_increment=float(reference["phase_increment"]) if "phase_increment" in reference else .75
        self.register_buffer("stand",torch.tensor(STAND,dtype=torch.float32))
        self.register_buffer("lo",torch.tensor(LO,dtype=torch.float32))
        self.register_buffer("hi",torch.tensor(HI,dtype=torch.float32))
        self.register_buffer("reference_q",torch.tensor(reference["q"],dtype=torch.float32))
        self.register_buffer("reference_up",torch.tensor(reference["up"],dtype=torch.float32))
        self.encoder=nn.Sequential(nn.Linear(74,48),nn.ELU())
        self.actor=nn.Sequential(nn.Linear(506,192),nn.ELU(),nn.Linear(192,128),nn.ELU(),nn.Linear(128,22))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)

    def context(self,history,state):
        x=history[:,-1]
        initial=state[:,0]==0
        kind=torch.where(initial,(-x[:,47]>0).float(),state[:,2]).clamp(0,1)
        phase=torch.where(initial,0.,state[:,3]).clamp(0,self.reference_q.shape[1]-1)
        indices=phase.long()
        q=self.reference_q[kind.long(),indices]
        up=self.reference_up[kind.long(),indices]
        ref=torch.cat((q-self.stand,up,phase[:,None]/self.reference_q.shape[1]),1)
        return ref,q,up,kind,phase

    def mean(self,history,state):
        reference,*_=self.context(history,state)
        feature=self.encoder(history.clamp(-12,12)).flatten(1)
        return self.actor(torch.cat((feature,reference),1))

    def target(self,action,history,state):
        _,reference,_,_,_=self.context(history,state)
        x=history[:,-1]
        initial=state[:,0]==0
        age=torch.where(initial,0.,state[:,0])
        origin=torch.where(initial[:,None],x[:,:22]+self.stand,state[:,4:])
        blend=.5-.5*torch.cos(torch.pi*((age+1)/35.).clamp(0,1))
        raw=reference+self.correction*action.tanh()
        return (origin+(raw-origin)*blend[:,None]).clamp(self.lo,self.hi)

    def advance(self,history,state):
        _,_,_,kind,phase=self.context(history,state)
        initial=state[:,0]==0
        age=torch.where(initial,0.,state[:,0])
        origin=torch.where(initial[:,None],history[:,-1,:22]+self.stand,state[:,4:])
        phase=(phase+torch.where(age<35,0.,self.phase_increment)).clamp(0,self.reference_q.shape[1]-1)
        return torch.cat((torch.stack(((age+1).clamp(max=10000),age*0+4,kind,phase),1),origin),1)

    def forward(self,rec_obs_hist,rec_state):
        action=self.mean(rec_obs_hist,rec_state)
        return self.target(action,rec_obs_hist,rec_state),self.advance(rec_obs_hist,rec_state)


def export(model,path):
    from copy import deepcopy
    model=deepcopy(model).cpu().eval()
    sample=torch.zeros((1,10,74));sample[:,:,49]=-1.
    torch.onnx.export(model,(sample,torch.zeros(1,26)),str(path),
                      input_names=["rec_obs_hist","rec_state"],output_names=["joint_target","next_state"],
                      opset_version=17,dynamo=False)
