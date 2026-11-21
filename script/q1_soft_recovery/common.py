"""Shared frozen configuration and exact CPU vector plant."""
import ctypes
import json
import os
from pathlib import Path
import sys
import numpy as np
from numpy.ctypeslib import ndpointer

HERE=Path(__file__).resolve().parent
EXPERIMENTS=HERE.parent
sys.path.insert(0,str(EXPERIMENTS/"q1_supported_rise_20261121"))
from limbs import add_limb_contacts, mj, replay
BUNDLE=replay.BUNDLE
CONFIG=json.loads((BUNDLE/"config/q1_deploy_config.json").read_text())
KEYS=("default_q","q_lo","q_hi","action_scale","kp","kd","effort_limit_nm")
CONSTANTS=np.ascontiguousarray([CONFIG[k] for k in KEYS],dtype=np.float64)
STAND,LO,HI,SCALE,KP,KD,EFF=CONSTANTS


class Plant:
    def __init__(self,n,scene,delay=11):
        self.n=n
        self.model=add_limb_contacts(scene.resolve())
        self.model.opt.timestep=.002
        self.lib=ctypes.CDLL(str(HERE/"plant.so"))
        ptr=ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
        integer=ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
        self.lib.create.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ptr]
        self.lib.create.restype=ctypes.c_void_p
        self.lib.destroy.argtypes=[ctypes.c_void_p]
        self.lib.reset.argtypes=[ctypes.c_void_p,integer,ptr,ptr,ptr,ptr]
        self.lib.step.argtypes=[ctypes.c_void_p,ptr,ptr,ptr,ptr]
        self.handle=self.lib.create(str((scene.resolve()/"fall_hard.xml")).encode(),n,delay,CONSTANTS)
        if not self.handle:raise RuntimeError("Cannot construct collision-enabled vector plant")
        self.obs=np.zeros((n,74))
        self.state=np.zeros((n,57))
        self.metrics=np.zeros((n,13))

    def reset(self,rows,mask=None):
        if mask is None:mask=np.ones(self.n,np.int32)
        self.lib.reset(self.handle,np.ascontiguousarray(mask,dtype=np.int32),np.ascontiguousarray(rows),self.obs,self.state,self.metrics)
        return self.obs.copy()

    def step(self,targets):
        if not np.isfinite(targets).all():raise ValueError("nonfinite target")
        self.lib.step(self.handle,np.ascontiguousarray(targets,dtype=np.float64),self.obs,self.state,self.metrics)
        if not np.isfinite(self.state).all():raise ValueError("nonfinite physics")
        return self.obs.copy(),self.metrics.copy()

    def close(self):
        if self.handle:self.lib.destroy(self.handle);self.handle=None
