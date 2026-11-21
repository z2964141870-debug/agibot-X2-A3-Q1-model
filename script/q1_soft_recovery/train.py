"""PPO using exact parallel CPU MuJoCo and proprioceptive observations."""
import argparse
import fcntl
import json
import os
import hashlib
import shutil
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from common import HERE,Plant,STAND,SCALE
from policy import Teacher,export,load_actor


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    rng=np.random.default_rng(args.seed)
    out=HERE/args.run;out.mkdir(exist_ok=False)
    frozen=out/"frozen_src";frozen.mkdir()
    sources=[*HERE.glob("*.py"),HERE/"plant.cc",HERE/"plant.so"]
    manifest={}
    for source in sources:
        shutil.copy2(source,frozen/source.name);manifest[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
    data=HERE/args.data
    for name in ("references.npz","starts.npz"):
        source=data/name;shutil.copy2(source,frozen/name);manifest[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (frozen/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (out/"configuration.json").write_text(json.dumps(vars(args),default=str,indent=2))
    reference=np.load(data/"references.npz")
    pool=np.load(data/"starts.npz")
    rows,meta=pool["train"],pool["train_meta"]
    env=Plant(args.envs,out/"scene")
    if args.balanced:
        from balanced_policy import BalancedTeacher
        model=BalancedTeacher(reference).to(device)
    else:model=Teacher(reference).to(device)
    critic=nn.Sequential(nn.Linear(113,192),nn.ELU(),nn.Linear(192,128),nn.ELU(),nn.Linear(128,1)).to(device)
    logstd=nn.Parameter(torch.full((22,),-2.5 if args.balanced else -2.0,device=device))
    if args.parent:
        prior=torch.load(args.parent,map_location=device,weights_only=False)
        load_actor(model,prior["actor"],args.transfer_network_only)
        critic.load_state_dict(prior["critic"])
        logstd.data.copy_(prior["logstd"])
    params=[*model.parameters(),*critic.parameters(),logstd]
    optimizer=torch.optim.Adam(params,lr=args.lr)
    states=torch.zeros(args.envs,26,device=device)
    ages=np.zeros(args.envs,dtype=int)
    histories=np.zeros((args.envs,10,74),np.float32)
    minimum=args.minimum_phase
    fractions=meta[:,1]/np.array([reference["length"][k] for k in meta[:,0]])
    easy=np.flatnonzero(fractions>=minimum)
    if not len(easy):raise ValueError("empty training curriculum")
    def reset(mask):
        chosen=rng.choice(easy,len(rows),replace=True)
        if args.floor_fraction:
            low=np.flatnonzero(meta[:,1]<35)
            use=rng.random(len(chosen))<args.floor_fraction
            chosen[use]=rng.choice(low,int(use.sum()))
        indices=chosen[:args.envs]
        frames=env.reset(rows[indices],mask.astype(np.int32))
        histories[mask]=frames[mask,None]
        states[mask]=0
        states[mask,0]=60
        states[mask,1]=4
        states[mask,2]=torch.tensor(meta[indices[mask],0],dtype=torch.float32,device=device)
        states[mask,3]=torch.tensor(meta[indices[mask],1],dtype=torch.float32,device=device)
        states[mask,4:]=torch.tensor(rows[indices[mask],7:29],dtype=torch.float32,device=device)
        ages[mask]=0
    reset(np.ones(args.envs,dtype=bool))
    def features(h,s,metrics):
        context=model.context(h,s)[0]
        privileged=torch.tensor(metrics,dtype=torch.float32,device=device)
        privileged=privileged*torch.tensor([1,1,.01,1,.1,.1,.1,1,.2,1,1,1,1],device=device)
        return torch.cat((h[:,-1],context,privileged),1)
    start=time.monotonic()
    for update in range(1,args.updates+1):
        hs=[];ms=[];fs=[];actions=[];oldlp=[];values=[];rewards=[];done_rows=[];stats=[]
        with torch.no_grad():
            for step in range(32):
                h=torch.from_numpy(histories.copy()).to(device)
                f=features(h,states,env.metrics)
                mean=model.mean(h,states)
                dist=Normal(mean,logstd.clamp(-2.8,-.3).exp())
                action=dist.sample()
                target=model.target(action,h,states)
                ref,qref,upref,kind,phase=model.context(h,states)
                hs.append(h);ms.append(states.clone());fs.append(f)
                actions.append(action);oldlp.append(dist.log_prob(action).sum(1));values.append(critic(f).squeeze(1))
                old=env.metrics.copy()
                old_up=env.obs[:,47:50].copy()
                obs,met=env.step(target.cpu().numpy())
                states=model.advance(h,states)
                ages+=1
                joint=np.mean((env.state[:,7:29]-qref.cpu().numpy())**2,axis=1)
                true_up=np.column_stack((-2*(env.state[:,4]*env.state[:,6]-env.state[:,3]*env.state[:,5]),
                                        -2*(env.state[:,5]*env.state[:,6]+env.state[:,3]*env.state[:,4]),
                                        -(1-2*(env.state[:,4]**2+env.state[:,5]**2))))
                tilt=np.sum((true_up+upref.cpu().numpy())**2,axis=1)
                progress=met[:,0]*np.maximum(met[:,1],0)-old[:,0]*np.maximum(old[:,1],0)
                late=(phase.cpu().numpy()>reference["length"][kind.cpu().numpy().astype(int)]*.72)
                upright=(met[:,0]>.35)&(met[:,1]>.92)&(met[:,2]>10)&(met[:,3]==0)
                quiet=upright&(met[:,7]<.2)&(met[:,8]<.8)
                reward=1.8*np.exp(-3*joint)+2.2*np.exp(-2.5*tilt)+12*progress
                reward+=late*(1.1*upright+1.5*quiet+.8*np.maximum(met[:,1],0))
                if args.balanced:reward+=late*(2*upright+4*quiet)
                reward-=.015*np.maximum(met[:,4]-10,0)**2+.06*np.maximum(met[:,5]-5,0)+.2*np.maximum(met[:,6]-1,0)
                reward-=.015*np.mean((target.cpu().numpy()-STAND-SCALE*4*h[:,-1,50:72].cpu().numpy())**2,axis=1)
                done=(ages>=args.horizon)|(met[:,0]<-.02)|(met[:,4]>55)|(met[:,6]>45)
                if args.balanced:done|=(ages>20)&(met[:,0]<.13)
                reward=np.clip(reward,-20,10)
                if args.balanced:
                    reward-=.05*np.maximum(met[:,4]-10,0)**2+.2*np.maximum(met[:,5]-8,0)+.6*np.maximum(met[:,6]-1,0)
                    reward-=8*(done&(ages<args.horizon))
                histories=np.concatenate((histories[:,1:],obs[:,None].astype(np.float32)),axis=1)
                rewards.append(torch.tensor(reward*.1,dtype=torch.float32,device=device));done_rows.append(torch.tensor(done,device=device))
                stats.append([float(reward.mean()),float(quiet.mean()),float(met[:,4].max()),float(met[:,5].max()),float(met[:,6].max()),float(met[:,0].mean())])
                if done.any():reset(done)
            next_features=features(torch.from_numpy(histories.copy()).to(device),states,env.metrics)
            following=critic(next_features).squeeze(1)
            gae=torch.zeros_like(following);advs=[]
            for i in reversed(range(32)):
                alive=(~done_rows[i]).float()
                gae=rewards[i]+.995*alive*following-values[i]+.995*.95*alive*gae
                advs.append(gae);following=values[i]
            advantage=torch.cat(advs[::-1]);returns=advantage+torch.cat(values)
            advantage=(advantage-advantage.mean())/(advantage.std()+1e-6)
        h,s,f,a,lp=map(torch.cat,(hs,ms,fs,actions,oldlp))
        for epoch in range(4):
            for ix in torch.randperm(len(a),device=device).split(1024):
                dist=Normal(model.mean(h[ix],s[ix]),logstd.clamp(-2.8,-.3).exp())
                ratio=(dist.log_prob(a[ix]).sum(1)-lp[ix]).exp()
                pg=torch.minimum(ratio*advantage[ix],ratio.clamp(.8,1.2)*advantage[ix])
                loss=-pg.mean()+.3*(critic(f[ix]).squeeze(1)-returns[ix]).square().mean()-.001*dist.entropy().sum(1).mean()
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);nn.utils.clip_grad_norm_(critic.parameters(),1.);optimizer.step()
        v=np.array(stats)
        metrics=dict(update=update,steps=update*args.envs*32,elapsed_s=time.monotonic()-start,
                     reward=float(v[:,0].mean()),quiet_fraction=float(v[:,1].mean()),peak_qd=float(v[:,2].max()),
                     ground_mm=float(v[:,3].max()),self_mm=float(v[:,4].max()),mean_z=float(v[:,5].mean()),loss=float(loss.detach()),minimum_phase=minimum)
        with (out/"metrics.jsonl").open("a") as fp:fp.write(json.dumps(metrics)+"\n")
        print(json.dumps(metrics),flush=True)
        if update%20==0 or update==args.updates or time.monotonic()-start>=args.seconds:
            checkpoint=dict(actor=model.state_dict(),critic=critic.state_dict(),logstd=logstd.detach(),update=update,args=vars(args))
            torch.save(checkpoint,out/"latest.pt.tmp");(out/"latest.pt.tmp").replace(out/"latest.pt")
            if update%60==0 or update==args.updates or time.monotonic()-start>=args.seconds:
                torch.save(checkpoint,out/f"checkpoint_{update:05d}.pt")
                export(model,out/f"expert_{update:05d}.onnx")
        if time.monotonic()-start>=args.seconds:break
    env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",required=True);p.add_argument("--parent",type=Path)
    p.add_argument("--data",type=Path,default=Path("prepared_v3"));p.add_argument("--balanced",action="store_true")
    p.add_argument("--transfer-network-only",action="store_true",help="retain the new dataset reference buffers when initializing neural weights")
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--envs",type=int,default=64);p.add_argument("--updates",type=int,default=240)
    p.add_argument("--minimum-phase",type=float,default=.65);p.add_argument("--floor-fraction",type=float,default=0.)
    p.add_argument("--horizon",type=int,default=450);p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--seconds",type=int,default=1800);p.add_argument("--seed",type=int,default=91771)
    args=p.parse_args()
    with (HERE/"cpu_training.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
