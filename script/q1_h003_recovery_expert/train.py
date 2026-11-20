"""Batched mjwarp recovery PPO. The frozen tracking policy is outside this optimizer."""
import argparse
from pathlib import Path
import fcntl
import os
import subprocess
import time
import torch
from torch import nn
from torch.distributions import Normal
import warp as wp
import mujoco_warp as mw
from bench import BUNDLE,HERE,scene
import numpy as np
import mujoco
import json

class Actor(nn.Module):
    def __init__(self,stand,lo,hi,scale,alpha):
        super().__init__()
        self.register_buffer("stand",stand)
        self.register_buffer("lo",lo)
        self.register_buffer("hi",hi)
        self.register_buffer("scale",scale)
        self.register_buffer("alpha",torch.tensor(alpha,device=stand.device))
        self.encoder=nn.Sequential(nn.Linear(74,64),nn.ELU())
        self.net=nn.Sequential(nn.Linear(640,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,22))
        nn.init.orthogonal_(self.net[-1].weight,gain=.03)
        nn.init.zeros_(self.net[-1].bias)

    def mean(self,history):
        return self.net(self.encoder(history.clamp(-12,12)).flatten(1))

    def targets(self,action,history):
        raw=(self.stand+action.tanh()*2.4).clamp(self.lo,self.hi)
        previous=self.stand+history[:,-1,50:72]*4*self.scale
        return (1-self.alpha)*previous+self.alpha*raw

    def forward(self,rec_obs_hist):
        return self.targets(self.mean(rec_obs_hist),rec_obs_hist)

class World:
    def __init__(self,n,reward_mode="height"):
        self.reward_mode=reward_mode
        self.n=n
        self.model=scene(HERE/"training_scene")
        self.m=mw.put_model(self.model)
        self.d=mw.put_data(self.model,mujoco.MjData(self.model),nworld=n,nconmax=96,njmax=320)
        for name in ("qpos","qvel","ctrl","xpos","xmat","qacc_warmstart"):
            setattr(self,name,wp.to_torch(getattr(self.d,name)))
        config=json.loads((BUNDLE/"config/q1_deploy_config.json").read_text())
        for attr,key in (("lo","q_lo"),("hi","q_hi"),("stand","default_q"),("scale","action_scale"),("kp","kp"),("kd","kd"),("eff","effort_limit_nm")):
            setattr(self,attr,torch.tensor(config[key],device="cuda"))
        self.external=wp.to_torch(self.d.xfrc_applied)
        self.pull_body=self.model.body("head_link").id
        self.weight=float(self.model.body_mass.sum()*9.81)
        self.assist=0.
        self.feet=[self.model.body(s+"_ankle_roll_link").id for s in ("left","right")]
        self.pelvis=self.model.body("pelvis").id
        self.sole=torch.tensor([.02,0.,-.022],device="cuda")
        self.pool=torch.tensor(np.load(HERE/"pool.npy"),device="cuda",dtype=torch.float32)
        self.ids=torch.arange(n,device="cuda")
        self.age=torch.zeros_like(self.ids)
        self.prev=torch.zeros((n,22),device="cuda")
        self.reset(torch.ones(n,device="cuda",dtype=torch.bool),0)
        self.advance()
        with wp.ScopedCapture() as capture:
            self.advance()
        self.graph=capture.graph
        self.reset(torch.ones(n,device="cuda",dtype=torch.bool),0)

    def advance(self):
        for _ in range(4): mw.step(self.m,self.d)
        mw.kinematics(self.m,self.d)

    def reset(self,mask,progress):
        # Interleave broad fall starts with a shrinking fraction of near-standing examples.
        indices=torch.randint(len(self.pool),(self.n,),device="cuda")
        easy=torch.rand(self.n,device="cuda") < max(.05,.6-progress*.55)
        indices=torch.where(easy,indices//8*8+4+indices%3,indices)
        rows=self.pool[indices[mask]]
        self.qpos[mask]=rows[:,1:30]
        self.qvel[mask]=rows[:,30:58]
        self.ctrl[mask]=rows[:,58:]
        self.qacc_warmstart[mask]=0.
        self.prev[mask]=(self.ctrl[mask]-self.stand)/self.scale
        self.age[mask]=0
        mw.kinematics(self.m,self.d)

    def frame(self):
        r=self.xmat[:,self.feet]
        soles=self.xpos[:,self.feet]-self.qpos[:,None,:3]+(r@self.sole)
        height,lowest=(-soles[:,:,2]).max(1)
        valid=(r[self.ids,lowest,2,2]>.65).float()
        return torch.cat((self.qpos[:,7:]-self.stand,self.qvel[:,6:]*.1,self.qvel[:,3:6]*.25,
                          -self.xmat[:,self.pelvis,2],self.prev*.25,(height.clamp(0,.65)*valid)[:,None],valid[:,None]),1)

    def step(self,raw):
        q=self.qpos[:,7:]
        qd=self.qvel[:,6:]
        old_qd=qd.clone()
        old_z=self.qpos[:,2].clone()
        old_up=self.xmat[:,self.pelvis,2,2].clone()
        dq_margin=.5*self.eff/self.kp
        tau=(self.kp*(raw.clamp(self.lo-dq_margin,self.hi+dq_margin)-q)-self.kd*qd).clamp(-self.eff,self.eff)
        self.ctrl.copy_(q+(tau+self.kd*qd)/self.kp)
        act=(self.ctrl-self.stand)/self.scale
        change=(act-self.prev).abs().mean(1)
        self.prev.copy_(act)
        if self.assist:
            self.external[:,self.pull_body,2]=self.assist*self.weight*((.43-old_z)/.09).clamp(0,1)
        else:
            self.external[:,self.pull_body,2].zero_()
        wp.capture_launch(self.graph)
        z=self.qpos[:,2]
        up=self.xmat[:,self.pelvis,2,2]
        height=((z-.07)/.33).clamp(0,1)
        feet=self.xpos[:,self.feet]
        dual=((feet[:,:,2]-.035).abs().max(1).values<.045).float()
        stance=(self.qpos[:,7:]-self.stand).square().mean(1)
        good=(z>=.39)&(up>=.85)
        reward=(12*(z-old_z)+2*(up-old_up)+.12*height+.07*(1+up)*height+
                .10*height*dual+.24*good.float()+.1*height*torch.exp(-4*stance)
                -.018*change-.002*(tau/self.eff).square().mean(1)-.001*(qd-old_qd).abs().mean(1))
        if self.reward_mode=="supported":
            # Couple lift to orientation and feet below pelvis so inverted support earns no lift.
            support=((z-feet[:,:,2].mean(1))/.34).clamp(0,1)
            upright=((up+.15)/1.15).clamp(0,1)
            signed_progress=10*(z*upright-old_z*((old_up+.15)/1.15).clamp(0,1))+2*(up-old_up)
            reward=(signed_progress+.16*up+.42*height*upright*support+.12*dual*upright+
                    .65*good.float()+.12*upright*height*torch.exp(-3*stance)
                    -.018*change-.002*(tau/self.eff).square().mean(1)-.001*(qd-old_qd).abs().mean(1))
        self.age+=1
        failed=(z<-.035)|(self.qvel.abs().amax(1)>160)|~torch.isfinite(reward)
        done=(self.age>=300)|failed
        return torch.where(failed,-torch.ones_like(reward),reward),done,good

def run(args):
    active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True)
    for line in active.splitlines():
        pid,name=line.split(",",1)
        if Path(name.strip()).name=="ffmpeg":
            cmd=Path("/proc",pid.strip(),"cmdline").read_bytes()
            if b"x11grab" in cmd and b"nvenc" in cmd: continue
        raise SystemExit("GPU occupied: "+line)
    wp.init()
    wp.set_device("cuda:0")
    # Run Warp and torch on the same CUDA stream, including state writes at resets.
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream().cuda_stream, device="cuda:0"))
    torch.manual_seed(args.seed)
    env=World(args.envs,args.reward)
    if args.template:
        from sequence_model import SequenceExpert,export
        actor=SequenceExpert(args.template).cuda()
    else:
        actor=Actor(env.stand,env.lo,env.hi,env.scale,args.alpha).cuda()
    memory=torch.zeros(args.envs,26,device="cuda")
    critic=nn.Sequential(nn.Linear(75,192),nn.ELU(),nn.Linear(192,128),nn.ELU(),nn.Linear(128,1)).cuda()
    logstd=nn.Parameter(torch.full((22,),-.75,device="cuda"))
    params=[*actor.parameters(),*critic.parameters(),logstd]
    optim=torch.optim.Adam(params,lr=3e-4)
    def value(history,z):
        return critic(torch.cat((history[:,-1],z[:,None]),1)).squeeze(1)
    out=HERE/args.run
    out.mkdir(exist_ok=True)
    resume=out/"latest.pt"
    start=0
    if resume.exists():
        state=torch.load(resume)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        logstd.data.copy_(state["logstd"])
        optim.load_state_dict(state["optim"])
        start=state["update"]
    hist=env.frame()[:,None].expand(-1,10,-1).clone()
    trial_start=time.monotonic()
    for update in range(start+1,args.updates+1):
        env.assist=args.assist*max(0.,min(1.,(1100-update)/800))
        hs,acts,logps,vals,rews,dones,zs=[],[],[],[],[],[],[]
        standing=[]
        with torch.no_grad():
            for _ in range(32):
                z=env.qpos[:,2].clone()
                dist=Normal(actor.mean(hist),logstd.clamp(-2,.2).exp())
                sample=dist.sample()
                vals.append(value(hist,z))
                logps.append(dist.log_prob(sample).sum(1))
                hs.append(hist)
                acts.append(sample)
                zs.append(z)
                if args.template:
                    prior,memory=actor.baseline(hist,memory)
                    raw=(prior+actor.residual_scale*sample.tanh()).clamp(env.lo,env.hi)
                else:
                    raw=actor.targets(sample,hist)
                reward,done,good=env.step(raw)
                rews.append(reward)
                dones.append(done)
                standing.append(good)
                if done.any():
                    env.reset(done,min(1.,update/180))
                    memory[done]=0.
                frame=env.frame()
                hist=torch.cat((hist[:,1:],frame[:,None]),1)
                hist[done]=frame[done,None]
            following=value(hist,env.qpos[:,2])
            gae=torch.zeros_like(following)
            adv=[]
            for i in reversed(range(32)):
                alive=(~dones[i]).float()
                gae=rews[i]+.995*alive*following-vals[i]+.995*.95*alive*gae
                adv.append(gae)
                following=vals[i]
            advantage=torch.cat(adv[::-1])
            returns=advantage+torch.cat(vals)
            advantage=(advantage-advantage.mean())/(advantage.std()+1e-6)
        h,a,lp,z=map(torch.cat,(hs,acts,logps,zs))
        for _ in range(4):
            for ix in torch.randperm(len(a),device="cuda").split(min(2048,len(a))):
                dist=Normal(actor.mean(h[ix]),logstd.clamp(-2,.2).exp())
                ratio=(dist.log_prob(a[ix]).sum(1)-lp[ix]).exp()
                pg=torch.minimum(ratio*advantage[ix],ratio.clamp(.8,1.2)*advantage[ix])
                loss=-pg.mean()+.5*(value(h[ix],z[ix])-returns[ix]).square().mean()-.002*dist.entropy().sum(1).mean()
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params,1.)
                optim.step()
        metrics=dict(pull_fraction=env.assist,update=update,steps=update*args.envs*32,elapsed=time.monotonic()-trial_start,
                     reward=torch.stack(rews).mean().item(),height=z.mean().item(),upright_fraction=torch.stack(standing).float().mean().item(),
                     loss=loss.item(),exploration=logstd.exp().mean().item())
        print(json.dumps(metrics),flush=True)
        with (out/"metrics.jsonl").open("a") as fp: fp.write(json.dumps(metrics)+"\n")
        if update%20==0 or update==args.updates:
            torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),logstd=logstd.detach(),optim=optim.state_dict(),update=update),out/"latest.tmp")
            os.replace(out/"latest.tmp",resume)
            if update%100==0 or update==args.updates:
                torch.save(actor.state_dict(),out/f"actor_{update:05}.pt")
                if args.template:
                    export(actor,out/f"expert_{update:05}.onnx")
                else:
                    torch.onnx.export(actor,(hist[:1],),str(out/f"expert_{update:05}.onnx"),input_names=["rec_obs_hist"],output_names=["joint_target"],opset_version=17,dynamo=False)

if __name__=="__main__":
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument("--template",type=Path)
    parser.add_argument("--alpha",type=float,default=1.)
    parser.add_argument("--assist",type=float,default=0.)
    parser.add_argument("--reward",choices=["height","supported"],default="height")
    parser.add_argument("--envs",type=int,default=512)
    parser.add_argument("--updates",type=int,default=400)
    parser.add_argument("--seed",type=int,default=69117)
    parser.add_argument("--run",default="ppo_01")
    args=parser.parse_args()
    with (HERE/"gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)

