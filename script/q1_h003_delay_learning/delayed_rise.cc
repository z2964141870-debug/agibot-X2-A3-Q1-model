#include <algorithm>
#include <array>
#include <cmath>
#include <mujoco/mujoco.h>
#include <omp.h>

// Each candidate specifies five full-body targets followed by a return to stand.
static int simulate(const char* xml, int count, const double* initial,
                        const double* motion, const double* stand, double* scores, int objective,const double* feedback) {
    char error[1024]{};
    mjModel* m=mj_loadXML(xml,nullptr,error,sizeof(error));
    if (!m || m->nu!=22 || m->nq!=29) return -1;
    const int pelvis=mj_name2id(m,mjOBJ_BODY,"pelvis");
    const int left=mj_name2id(m,mjOBJ_BODY,"left_ankle_roll_link");
    const int right=mj_name2id(m,mjOBJ_BODY,"right_ankle_roll_link");
    #pragma omp parallel num_threads(6)
    {
        mjData* d=mj_makeData(m);
        #pragma omp for schedule(dynamic,2)
        for(int i=0;i<count;++i){
            mj_resetData(m,d);
            const double* state=initial+79*i;
            std::copy_n(state,29,d->qpos);
            std::copy_n(state+29,28,d->qvel);
            std::array<double,22> pending;
            std::copy_n(state+57,22,pending.begin());
            std::copy(pending.begin(),pending.end(),d->ctrl);
            mj_forward(m,d);
            double score=0.,peak=0.;
            const double* keys=motion+5*22*i;
            const int horizon=feedback?170:122;
            for(int tick=0;tick<horizon;++tick){
                const double* gains=feedback?feedback+i*12:nullptr;
                const double ramp=gains?std::clamp((tick-68.)/13.,0.,1.):0.;
                const double pitch=std::asin(std::clamp(-d->xmat[9*pelvis+6],-1.,1.));
                const double roll=std::atan2(d->xmat[9*pelvis+7],d->xmat[9*pelvis+8]);
                const double wy=d->qvel[4],wx=d->qvel[3];
                const int phase=std::min(6,tick/14);
                const double blend=.5-.5*std::cos(3.141592653589793*(1+tick%14)/14.);
                const double* from=phase==0?state+7:phase<=5?keys+(phase-1)*22:stand;
                const double* to=phase<5?keys+phase*22:stand;
                for(int j=0;j<22;++j){
                    const double kp=m->actuator_gainprm[mjNGAIN*j];
                    const double kd=-m->actuator_biasprm[mjNBIAS*j+2];
                    const double limit=m->actuator_forcerange[2*j+1];
                    double target=from[j]+blend*(to[j]-from[j]);
                    if(gains && ramp>0.){
                        double adjustment=0.;
                        if(j==0||j==6) adjustment=gains[0]+gains[1]*pitch+gains[2]*wy;
                        if(j==4||j==10) adjustment=gains[3]+gains[4]*pitch+gains[5]*wy;
                        if(j==3||j==9) adjustment=gains[6]+gains[7]*std::abs(pitch);
                        if(j==14||j==18) adjustment=gains[8]*pitch;
                        if(j==5||j==11) adjustment=gains[9]*roll+gains[10]*wx;
                        if(j==1||j==7) adjustment=gains[11]*roll;
                        int joint=m->actuator_trnid[2*j];
                        target=std::clamp(target+ramp*adjustment,m->jnt_range[2*joint],m->jnt_range[2*joint+1]);
                    }
                    const double tau=std::clamp(kp*(target-d->qpos[7+j])-kd*d->qvel[6+j],-limit,limit);
                    const double next=d->qpos[7+j]+(tau+kd*d->qvel[6+j])/kp;
                    d->ctrl[j]=pending[j];
                    pending[j]=next;
                }
                for(int sub=0;sub<4;++sub) mj_step(m,d);
                const double z=d->qpos[2],up=d->xmat[9*pelvis+8];
                const double feet=.5*(d->xpos[3*left+2]+d->xpos[3*right+2]);
                const double support=std::clamp((z-feet)/.34,0.,1.);
                double quality=.35*up+1.2*std::max(0.,up)*z*support+1.5*(z>.39 && up>.85);
                if(objective==1) {
                    const double face_up=d->xmat[9*pelvis+6];
                    quality=1.5*face_up+.5*std::exp(-18*(z-.105)*(z-.105));
                }
                if(feedback) {
                    const double speed=std::hypot(d->qvel[3],d->qvel[4]);
                    quality=std::min(.42,z)*std::max(0.,up)+.55*up+
                        (z>=.39 && up>=.85?2.0:0.)+.3*std::exp(-speed)*support;
                    if(objective==2) {
                        const double* R=d->xmat+9*pelvis;
                        const double nx=R[0],ny=R[3];
                        const double dx=d->xpos[3*left]-d->xpos[3*right];
                        const double dy=d->xpos[3*left+1]-d->xpos[3*right+1];
                        const double width=std::abs(-ny*dx+nx*dy)/std::max(.01,std::hypot(nx,ny));
                        quality-=7.*std::max(0.,width-.22);
                        if(z>=.39 && up>=.85 && width<.25) quality+=1.;
                    }
                    if(tick<90) quality*=.04;
                }
                peak=std::max(peak,quality);
                score+=(tick<72?.1:1.)*quality;
            }
            scores[i]=score/60.+peak*.25;
        }
        mj_deleteData(d);
    }
    mj_deleteModel(m);
    return 0;
}


extern "C" int evaluate_delayed(const char* xml,int n,const double* initial,
                         const double* motion,const double* stand,double* scores,const double* feedback) {
    return simulate(xml,n,initial,motion,stand,scores,2,feedback);
}
