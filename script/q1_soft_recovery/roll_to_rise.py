"""Join a learned prone roll to the recorded supine rise using sensed gravity."""
import argparse
from pathlib import Path

import numpy as np
import torch
from floor_command_policy import FloorCommands
from policy import export


class RollToRise(torch.nn.Module):
    def __init__(self, fit, settle_ticks):
        super().__init__()
        self.rise = FloorCommands(use_crouch=False)
        self.register_buffer("keys", torch.tensor(fit["keys"],dtype=torch.float32))
        self.duration = float(fit["duration"])
        self.settle_ticks = settle_ticks

    def forward(self,rec_obs_hist,rec_state):
        x = rec_obs_hist[:, -1]
        q = x[:, :22] + self.rise.crouch.stand
        up = -x[:, 47:50]
        mode = rec_state[:, 1]
        start = (mode == 0) & (up[:, 0] < -.67)
        active = (mode == 8) | start
        tick = torch.where(start, 0., rec_state[:, 3])
        initial = torch.where(start[:,None], q, rec_state[:,4:])
        phase = ((tick + 1) / self.duration).clamp(0,5)
        idx = phase.long().clamp(0,4)
        origin = torch.where((idx == 0)[:,None],initial,self.keys[(idx-1).clamp(0,4)])
        blend = .5-.5*torch.cos(torch.pi*(phase-idx))
        roll_target = origin + blend[:,None]*(self.keys[idx]-origin)
        quiet = (up[:,0] > .8) & (torch.linalg.vector_norm(x[:,44:47]*4,dim=1)<1.7)
        count = torch.where(quiet,torch.where(start,0.,rec_state[:,2])+1,0.)
        release = active & (count >= self.settle_ticks)
        z=tick*0
        roll_state=torch.cat((torch.stack((tick+1,z+8,count,tick+1),1),initial),1)
        target,state=self.rise(rec_obs_hist,torch.where(active[:,None],0.,rec_state))
        rolling=(active & ~release)[:,None]
        return torch.where(rolling,roll_target,target), torch.where(rolling,roll_state,state)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("fit",type=Path);p.add_argument("out",type=Path);p.add_argument("--settle-ticks",type=int,default=10)
    a=p.parse_args();a.out.parent.mkdir(exist_ok=True,parents=True)
    export(RollToRise(np.load(a.fit),a.settle_ticks).eval(),a.out)
