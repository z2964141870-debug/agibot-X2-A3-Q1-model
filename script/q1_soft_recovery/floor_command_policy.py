"""Compose recorded approach commands, sensed crouch, and the external owner."""
import argparse
from pathlib import Path
import numpy as np
import torch
from common import HERE, LO, HI
from command_support import sample_record
from crouch_handoff import SensedStart
from policy import export
from importlib.util import module_from_spec, spec_from_file_location
from export import FiniteRoll


class FloorCommands(torch.nn.Module):
    def __init__(self, use_crouch=True, soft=False, rate=.75):
        super().__init__()
        self.use_crouch = use_crouch
        self.soft = soft
        self.rate = rate
        self.crouch = SensedStart(np.load(HERE/"prepared_balanced_02/references.npz"))
        records = [sample_record(s)["command_position"] for s in
                   ("q1_official_standup_from_mcap","q1_official_standup_faceup_20261120")]
        length = max(map(len, records))
        padded = [np.pad(x,((0,length-len(x)),(0,0)),mode="edge") for x in records]
        self.register_buffer("commands",torch.tensor(np.stack(padded),dtype=torch.float32))
        legacy_root = HERE.parent / "h003_recovery_expert"
        spec = spec_from_file_location("q1_frozen_sequence", legacy_root/"sequence_model.py")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        module.HERE = legacy_root
        legacy = module.SequenceExpert(legacy_root/"candidate_seq01")
        legacy.load_state_dict(torch.load(legacy_root/"ppo_residual_01/actor_00200.pt", map_location="cpu"))
        self.legacy = FiniteRoll(legacy)
        self.register_buffer("lo",torch.tensor(LO,dtype=torch.float32))
        self.register_buffer("hi",torch.tensor(HI,dtype=torch.float32))

    def forward(self,rec_obs_hist,rec_state):
        frame=rec_obs_hist[:,-1]
        joints=frame[:,:22]+self.crouch.stand
        up=-frame[:,47:50]
        mode=rec_state[:,1]
        legacy_target,legacy_state=self.legacy(rec_obs_hist,torch.where((mode>=6)[:,None],0.,rec_state))
        aligned=(up[:,0].abs()>.70)&(up[:,0].abs()>up[:,1].abs())
        enter=((mode==0)|((mode==1)&(rec_state[:,3]>15)))&aligned
        approach=enter|(mode==6)
        kind=torch.where(enter,(up[:,0]>0).float(),rec_state[:,2]).clamp(0,1)
        tick=torch.where(enter,0.,rec_state[:,3])
        origin=torch.where(enter[:,None],joints,rec_state[:,4:])
        sample=((tick-25).clamp(min=0)*self.rate).clamp(0,self.commands.shape[1]-1)
        index=sample.long()
        waypoint=self.commands[kind.long(),index]
        if self.soft:
            upper=(index+1).clamp(max=self.commands.shape[1]-1)
            waypoint=waypoint+(sample-index)[:,None]*(self.commands[kind.long(),upper]-waypoint)
            legacy_blend=(rec_state[:,0]/23).clamp(0,1)
            legacy_target=joints+legacy_blend[:,None]*(legacy_target-joints)
        blend=.5-.5*torch.cos(torch.pi*((tick+1)/25).clamp(0,1))
        target=origin+blend[:,None]*(waypoint-origin)
        next_approach=torch.cat((torch.stack((rec_state[:,0]+1,mode*0+6,kind,tick+1),1),origin),1)
        capture=approach&(tick>40)&(up[:,2]>.75)&(frame[:,73]>.5)&(frame[:,72]>.25)
        crouching=(capture|(mode==7)) if self.use_crouch else torch.zeros_like(capture)
        inner=torch.where(capture[:,None],0.,rec_state)
        balance,balance_state=self.crouch(rec_obs_hist,inner)
        balance_state=torch.cat((balance_state[:,:1],balance_state[:,1:2]*0+7,balance_state[:,2:]),1)
        raw=torch.where(crouching[:,None],balance,torch.where(approach[:,None],target,legacy_target))
        state=torch.where(crouching[:,None],balance_state,torch.where(approach[:,None],next_approach,legacy_state))
        return raw.clamp(self.lo,self.hi),state


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("path",type=Path);p.add_argument("--direct", action="store_true");p.add_argument("--soft", action="store_true");p.add_argument("--rate",type=float,default=.75);args=p.parse_args()
    args.path.parent.mkdir(parents=True,exist_ok=True)
    export(FloorCommands(use_crouch=not args.direct,soft=args.soft,rate=args.rate).eval(),args.path)
