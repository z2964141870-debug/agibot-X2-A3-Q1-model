// Exact CPU MuJoCo vector environment with the archived 50 Hz sender contract.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <omp.h>

constexpr int NQ=29, NV=28, NU=22, OBS=74, MET=13;
struct World {
    mjData* data;
    mjData* measured;
    double history[7][NQ+NV]{};
    double sent[NU]{};
    int slot=0;
    double peak_speed=0., ground_depth=0., self_depth=0., max_command=0., issued_excess=0., actual_excess=0., vzmax=0.;
};
struct Plant {
    mjModel* model;
    std::vector<World> worlds;
    std::array<double,NU> stand, lo, hi, scale, kp, kd, eff;
    int floor, pelvis, feet[2], geomfeet[2], delay;
};

static void save(World& w) {
    std::copy_n(w.data->qpos,NQ,w.history[w.slot]);
    std::copy_n(w.data->qvel,NV,w.history[w.slot]+NQ);
}
static void observe(Plant* p, World& w, double* output, double* state, double* metrics) {
    double sample[NQ+NV];
    if(p->delay==0) std::copy_n(w.history[w.slot],NQ+NV,sample);
    else {
        const double* a=w.history[(w.slot+1)%7];
        const double* b=w.history[(w.slot+2)%7];
        for(int j=0;j<NQ+NV;++j)sample[j]=.5*(a[j]+b[j]);
        double dot=0.;for(int j=3;j<7;++j)dot+=a[j]*b[j];
        for(int j=3;j<7;++j) sample[j]=.5*(a[j]+(dot>=0.?b[j]:-b[j]));
        mj_normalizeQuat(p->model,sample);
    }
    std::copy_n(sample,NQ,w.measured->qpos);
    std::copy_n(sample+NQ,NV,w.measured->qvel);
    mj_kinematics(p->model,w.measured);
    mjData* d=w.measured;
    for(int j=0;j<NU;++j){
        output[j]=sample[j+7]-p->stand[j];
        output[j+22]=sample[NQ+6+j]*.1;
        output[j+50]=.25*(w.sent[j]-p->stand[j])/p->scale[j];
    }
    for(int j=0;j<3;++j){output[44+j]=sample[NQ+3+j]*.25;output[47+j]=-d->xmat[p->pelvis*9+6+j];}
    double lowest=1e9,normal=0.;
    for(int f=0;f<2;++f){
        const int body=p->feet[f];
        const double* r=d->xmat+body*9;
        double z=d->xpos[3*body+2]-sample[2]+.020*r[6]-.022*r[8];
        if(z<lowest){lowest=z;normal=r[8];}
    }
    output[73]=normal>.65;
    output[72]=std::clamp(-lowest,0.,.65)*output[73];
    std::copy_n(w.data->qpos,NQ,state);
    std::copy_n(w.data->qvel,NV,state+NQ);
    if(!metrics)return;
    mjData* true_data=w.data;
    double foot=0.,body=0.;
    for(int c=0;c<true_data->ncon;++c){
        const auto& ct=true_data->contact[c];
        if(ct.geom1!=p->floor && ct.geom2!=p->floor)continue;
        int other=ct.geom1==p->floor?ct.geom2:ct.geom1;
        if(other==p->geomfeet[0] || other==p->geomfeet[1]){
            mjtNum force[6];mj_contactForce(p->model,true_data,c,force);foot+=std::max(0.,force[0]);
        }else body=1.;
    }
    const double* v=true_data->qvel;
    double angle=std::sqrt(v[3]*v[3]+v[4]*v[4]+v[5]*v[5]);
    double values[MET]={true_data->qpos[2],true_data->xmat[p->pelvis*9+8],foot,body,
        w.peak_speed,1000*w.ground_depth,1000*w.self_depth,std::hypot(v[0],v[1]),angle,
        w.issued_excess,w.actual_excess,w.max_command,w.vzmax};
    std::copy_n(values,MET,metrics);
}

extern "C" void* create(const char* xml,int n,int delay,const double* config){
    char error[2048]{};
    mjModel* model=mj_loadXML(xml,nullptr,error,sizeof(error));
    if(!model || model->nq!=NQ || model->nv!=NV || model->nu!=NU || model->npair!=96)return nullptr;
    model->opt.timestep=.002;
    auto* p=new Plant; p->model=model;p->delay=delay;p->worlds.resize(n);
    auto arrays={&p->stand,&p->lo,&p->hi,&p->scale,&p->kp,&p->kd,&p->eff};
    int row=0;for(auto* a:arrays){std::copy_n(config+NU*row,NU,a->begin());++row;}
    p->floor=mj_name2id(model,mjOBJ_GEOM,"floor");p->pelvis=mj_name2id(model,mjOBJ_BODY,"pelvis");
    for(int i=0;i<2;++i){std::string side=i==0?"left":"right";
        p->feet[i]=mj_name2id(model,mjOBJ_BODY,(side+"_ankle_roll_link").c_str());
        p->geomfeet[i]=mj_name2id(model,mjOBJ_GEOM,(side+"_ankle_roll_link_collision_1").c_str());}
    for(auto& w:p->worlds){w.data=mj_makeData(model);w.measured=mj_makeData(model);}
    return p;
}
extern "C" void destroy(void* handle){auto* p=(Plant*)handle;if(!p)return;
    for(auto& w:p->worlds){mj_deleteData(w.data);mj_deleteData(w.measured);}mj_deleteModel(p->model);delete p;}
extern "C" void reset(void* handle,const int* mask,const double* states,double* obs,double* output,double* metrics){
    auto* p=(Plant*)handle;
    #pragma omp parallel for num_threads(7)
    for(int i=0;i<(int)p->worlds.size();++i){
        auto& w=p->worlds[i];
        if(mask[i]){
            mj_resetData(p->model,w.data);
            std::copy_n(states+79*i,NQ,w.data->qpos);std::copy_n(states+79*i+NQ,NV,w.data->qvel);
            for(int j=0;j<NU;++j)w.sent[j]=(float)states[79*i+NQ+NV+j];std::copy_n(w.sent,NU,w.data->ctrl);
            mj_forward(p->model,w.data);w.slot=0;save(w);
            for(int k=1;k<7;++k)std::copy_n(w.history[0],NQ+NV,w.history[k]);
            w.peak_speed=w.ground_depth=w.self_depth=w.max_command=w.issued_excess=w.actual_excess=w.vzmax=0.;
        }
        observe(p,w,obs+i*OBS,output+i*(NQ+NV),metrics+i*MET);
    }
}
extern "C" void step(void* handle,const double* target,double* obs,double* output,double* metrics){
    auto* p=(Plant*)handle;
    #pragma omp parallel for num_threads(7)
    for(int i=0;i<(int)p->worlds.size();++i){
        auto& w=p->worlds[i];double measured[OBS],state[NQ+NV],desired[NU];
        observe(p,w,measured,state,nullptr);
        w.peak_speed=w.ground_depth=w.self_depth=w.max_command=w.issued_excess=w.actual_excess=w.vzmax=0.;
        for(int j=0;j<NU;++j){
            const float q=(float)w.measured->qpos[7+j], qd=(float)w.measured->qvel[6+j];
            const float kp=(float)p->kp[j],kd=(float)p->kd[j],eff=(float)p->eff[j];
            const float margin=.5f*eff/kp;
            const float lo=(float)p->lo[j],hi=(float)p->hi[j],stand=(float)p->stand[j];
            const float raw=std::clamp((float)target[NU*i+j],lo-margin,hi+margin);
            const float tau=std::clamp(kp*(raw-q)-kd*qd,-eff,eff);
            desired[j]=std::clamp(q+(tau+kd*qd)/kp,stand-2.f,stand+2.f);
        }
        for(int sub=0;sub<10;++sub){
            for(int j=0;j<NU;++j){
                float change=std::clamp((float)desired[j]-(float)w.sent[j],-.2f,.2f);w.sent[j]=(float)w.sent[j]+change;
                w.max_command=std::max(w.max_command,(double)std::abs(change));w.data->ctrl[j]=w.sent[j];
            }
            mj_step(p->model,w.data);w.slot=(w.slot+1)%7;save(w);
            w.vzmax=std::max(w.vzmax,w.data->qvel[2]);
            for(int j=0;j<NU;++j){
                w.peak_speed=std::max(w.peak_speed,std::abs(w.data->qvel[6+j]));
                w.actual_excess=std::max(w.actual_excess,std::abs(w.data->actuator_force[j])-p->eff[j]);
            }
            for(int c=0;c<w.data->ncon;++c){const auto& ct=w.data->contact[c];
                double depth=std::max(0.,-ct.dist);
                if(ct.geom1==p->floor || ct.geom2==p->floor)w.ground_depth=std::max(w.ground_depth,depth);
                else w.self_depth=std::max(w.self_depth,depth);
            }
        }
        observe(p,w,obs+i*OBS,output+i*(NQ+NV),metrics+i*MET);
    }
}
