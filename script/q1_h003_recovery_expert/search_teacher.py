"""Search physically replayable curriculum examples on disjoint training starts."""
from pathlib import Path
from ctypes import CDLL,c_char_p,c_int
import json
import numpy as np
from numpy.ctypeslib import ndpointer

HERE=Path(__file__).resolve().parent
cfg=json.loads((HERE.parent/"q1_v2b_gantry_deploy_bundle/config/q1_deploy_config.json").read_text())
stand,lo,hi=(np.array(cfg[k]) for k in ("default_q","q_lo","q_hi"))
lib=CDLL(str(HERE/"trajectory_search.so"))
ptr=ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
lib.evaluate.argtypes=[c_char_p,c_int,ptr,ptr,ptr,ptr,c_int]
xml=str(HERE/"scene/fall_hard.xml").encode()
pool=np.load(HERE/"pool.npy")
def keys(parameters):
    # Sagittal/symmetric targets, with a controlled left-right roll component.
    key=np.broadcast_to(stand,(*parameters.shape[:-1],22)).copy()
    for j,k,p,sign in [(0,6,0,1),(3,9,1,1),(4,10,2,1),(14,18,3,1),(17,21,4,1),(1,7,5,-1),(15,19,6,-1)]:
        key[...,j]=parameters[...,p]
        key[...,k]=sign*parameters[...,p]
    key[...,1]+=parameters[...,7]
    key[...,7]+=parameters[...,7]
    key[...,12]=.35*parameters[...,7]
    if parameters.shape[-1]>8:
        for j,k,p in ((0,6,8),(3,9,9),(14,18,10),(17,21,11)):
            key[...,j]+=parameters[...,p]
            key[...,k]-=parameters[...,p]
        key[...,13]=parameters[...,12]
    return np.ascontiguousarray(key.clip(lo,hi))

def evaluate(params,starts,objective=0):
    moves=keys(params)
    rows=np.ascontiguousarray(starts[:,1:58])
    score=np.empty(len(rows))
    if lib.evaluate(xml,len(rows),rows,moves,stand,score,objective): raise RuntimeError("model load failed")
    return score

def fit():
    out=HERE/"teachers_01"
    out.mkdir(exist_ok=True)
    rng=np.random.default_rng(617233)
    means=[]
    for kind in range(4):
        means.append(np.array([[-1.8,2.1,-.65,-1.6,.2,.15,.6,0],[-2.1,2.3,-.7,.65,.5,.15,.75,0],
                     [-1.5,2.2,-.78,.4,1.,.1,.45,0],[-1.,1.6,-.7,-.8,.8,.08,.25,0],[-.5,.8,-.4,-.4,1.2,.05,.2,0]]))
    sigma0=np.tile([.85,.58,.31,1.05,.67,.25,.55,.26],(5,1))
    sigmas=[sigma0.copy() for _ in means]
    best=[-np.inf]*4
    for epoch in range(55):
        for kind in range(4):
            starts=pool[pool[:,0]==kind]
            size=56
            params=means[kind]+rng.normal(size=(size,5,8))*sigmas[kind]
            params[0]=means[kind]
            # Use common starts within each generation and validate the elite on fresh rows.
            init=np.repeat(starts[[epoch%len(starts)]],size,axis=0)
            scores=evaluate(params,init)
            elite=params[np.argsort(scores)[-9:]]
            means[kind]=.25*means[kind]+.75*elite.mean(0)
            sigmas[kind]=(.25*sigmas[kind]+.75*elite.std(0)).clip(sigma0*.12,sigma0)
            winner=params[scores.argmax()]
            valid=evaluate(np.repeat(winner[None],8,axis=0),starts[::16][:8])
            if valid.mean()>best[kind]:
                best[kind]=valid.mean()
                np.save(out/f"keys_{kind}.npy",keys(winner))
            print(epoch,kind,round(scores.max(),3),round(valid.mean(),3),round(best[kind],3),flush=True)
if __name__=="__main__":fit()

