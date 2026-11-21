"""Identify phase-dependent command responses along a confirmed supine rise."""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import Plant, STAND
from command_support import sample_record
from policy import export
from roll_to_rise import RollToRise
from support_transfer import basis_matrix


START, STOP = 205, 349


def measured_features(states, basis):
    # Measured states use the same delayed quaternion and velocities as the owner.
    quat = states[:, 3:7].copy()
    quat /= np.linalg.norm(quat,axis=1,keepdims=True)
    w,x,y,z = quat.T
    up_x = 2*(x*z-w*y)
    up_z = 1-2*(x*x+y*y)
    return np.column_stack((np.arctan2(-up_x,up_z)*3, states[:,33]*.22,
                            (states[:,7:29]-STAND) @ basis.T*.32,
                            states[:,35:57] @ basis.T*.018))


def obs_features(obs, basis):
    return np.column_stack((np.arctan2(obs[:,47],-obs[:,49])*3,obs[:,45]*.88,
                            obs[:,:22] @ basis.T*.32,obs[:,22:44] @ basis.T*.18))


def identify(run,out,trial):
    out.mkdir(parents=True)
    plant=Plant(11,out/"scene")
    record=np.load(run/("control_state_%02d.npz"%trial))["states"]
    basis=basis_matrix()
    # First row uses the unmodified commands, each other pair is a centered
    # finite difference on one set of bilateral joints.
    actions=np.vstack((np.zeros((1,5)),np.eye(5)*.035,-np.eye(5)*.035)) @ basis
    command=sample_record("q1_official_standup_faceup_20261120")["command_position"]
    reference=[]
    jacobians=[]
    for age in range(START,STOP+1):
        ids=np.flatnonzero((record[:,-3]==6)&(record[:,-2]==1)&(record[:,-1]==age))
        if not len(ids):
            raise ValueError(f"trial {trial} did not execute supine age {age}")
        row=record[ids[0]]
        reference.append(measured_features(row[None,79:136],basis)[0])
        plant.reset(np.tile(row[:79],(11,1)))
        for t in range(6):
            sample=int((age+t-25)*.75)
            obs,_=plant.step(command[sample]+actions)
        features=obs_features(obs,basis)
        jacobians.append(((features[1:6]-features[6:])/.07).T)
    plant.close()
    jacobians=np.stack(jacobians)
    response=[]
    for i in range(len(reference)):
        # Smooth over neighboring contact samples before solving for a bounded
        # least-squares correction over the next 120 milliseconds.
        B=np.median(jacobians[max(0,i-2):i+3],axis=0)
        response.append(np.linalg.solve(B.T@B+.5*np.eye(5),B.T))
    np.savez(out/"response.npz",basis=basis,reference=reference,response=response,jacobians=jacobians)


class ResponseFeedback(torch.nn.Module):
    def __init__(self,roll,fit,strength):
        super().__init__()
        self.inner=RollToRise(roll,10)
        self.strength=strength
        for k in ("reference","response","basis"):
            self.register_buffer(k,torch.tensor(fit[k],dtype=torch.float32))

    def forward(self,rec_obs_hist,rec_state):
        target,state=self.inner(rec_obs_hist,rec_state)
        x=rec_obs_hist[:,-1]
        phase=rec_state[:,3]
        i=(phase-START).long().clamp(0,STOP-START)
        features=torch.cat((torch.atan2(x[:,47],-x[:,49])[:,None]*3,x[:,45:46]*.88,
                            x[:,:22]@self.basis.T*.32,x[:,22:44]@self.basis.T*.18),1)
        error=self.reference[i]-features
        raw=(self.response[i]@error[:,:,None]).squeeze(-1)
        weight=((phase-START)/12).clamp(0,1)*((STOP-phase)/8).clamp(0,1)
        active=(rec_state[:,1]==6)&(rec_state[:,2]==1)
        correction=(self.strength*raw).clamp(-.22,.22)@self.basis
        return (target+(active*weight)[:,None]*correction).clamp(self.inner.rise.lo,self.inner.rise.hi),state


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("roll",type=Path)
    p.add_argument("--trial",type=int,default=14)
    a=p.parse_args()
    identify(a.run,a.out,a.trial)
    for gain in (.25,.6,1.):
        export(ResponseFeedback(np.load(a.roll),np.load(a.out/"response.npz"),gain),a.out/f"gain_{gain}.onnx")
