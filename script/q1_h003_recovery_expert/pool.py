"""Generate settled curriculum starts on CPU with seeds independent of E1."""
from bench import *
from q1_kinematics_np import Q1_JOINT_NAMES
from q1_v2b_runtime import apply_torque_safe_pd_np

def build(path):
    rng=np.random.default_rng(51177)
    model=scene(HERE/"pool_scene")
    c=Q1V2bOnnxController(BUNDLE,())
    assert [model.actuator(i).name for i in range(model.nu)]==list(Q1_JOINT_NAMES)
    rows=[]
    for i in range(1024):
        group=i%8
        d=mujoco.MjData(model)
        q=c.default_q.copy()
        d.qpos[:7]=[0,0,.425,1,0,0,0]
        if group<4:
            axis=(0,1,0) if group<2 else (1,0,0)
            angle=(1 if group%2==0 else -1)*rng.uniform(1.35,1.85)
            d.qpos[2]=.16
            d.qpos[3:7]=np.r_[np.cos(angle/2),np.sin(angle/2)*np.asarray(axis)]
            q+=rng.uniform(-.25,.25,22)
        elif group<7:
            bend=rng.uniform(.25,2.05)
            q[[3,9]]=bend
            q[[0,6]]=-.58*bend
            q[[4,10]]=-.41*bend
            d.qpos[2]=.42
        q=np.clip(q,c.q_lo,c.q_hi)
        d.qpos[7:]=q
        d.ctrl[:]=q
        mujoco.mj_forward(model,d)
        if group in (4,5,6):
            floor=model.geom("floor").id
            ankle=[model.body(s+"_ankle_roll_link").id for s in ("left","right")]
            d.qpos[2]-=min(d.xpos[j,2]-.022 for j in ankle)
            mujoco.mj_forward(model,d)
        if group==7:
            d.qvel[:2]=rng.normal(size=2)*1.6
        for _ in range(25 if group==7 else 5):
            d.ctrl[:]=apply_torque_safe_pd_np(q,d.qpos[7:],d.qvel[6:],c.q_lo,c.q_hi,c.kp,c.kd,c.eff,engage_steps=0)
            mujoco.mj_step(model,d,nstep=4)
        rows.append(np.r_[group,d.qpos,d.qvel,d.ctrl])
    np.save(path,np.asarray(rows))
if __name__=="__main__":
    build(HERE/"pool.npy")

