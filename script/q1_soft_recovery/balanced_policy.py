"""Causal residual policy bootstrapped from mesh-grounded poses and balance feedback."""
import numpy as np
import torch
from policy import Teacher


class BalancedTeacher(Teacher):
    def __init__(self,reference,correction=.3):
        super().__init__(reference,correction)
        self.register_buffer("reference_command",torch.tensor(reference["command"],dtype=torch.float32))
        self.register_buffer("balance_gains",torch.tensor(reference["gains"],dtype=torch.float32))

    def target(self,action,history,state):
        _,_,up,kind,phase=self.context(history,state)
        x=history[:,-1];base=self.reference_command[kind.long(),phase.long()]
        error=x[:,47:50]+up
        pitch=torch.stack((error[:,0],x[:,45]*4),1)
        roll=torch.stack((error[:,1],x[:,44]*4),1)
        offsets=torch.zeros_like(base)
        for indices,features,g in (([4,10],pitch,self.balance_gains[:2]),([0,6],pitch,self.balance_gains[2:4]),
                                   ([5,11],roll,self.balance_gains[4:6]),([1,7],roll,self.balance_gains[6:8])):
            value=(features*g).sum(1).clamp(-.5,.5)
            for index in indices:offsets[:,index]=value
        initial=state[:,0]==0;age=torch.where(initial,0.,state[:,0])
        origin=torch.where(initial[:,None],x[:,:22]+self.stand,state[:,4:])
        blend=.5-.5*torch.cos(torch.pi*((age+1)/35.).clamp(0,1))
        raw=base+offsets+self.correction*action.tanh()
        return (origin+(raw-origin)*blend[:,None]).clamp(self.lo,self.hi)
