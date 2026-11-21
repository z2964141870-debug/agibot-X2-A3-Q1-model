// Short 50 Hz trajectory optimizations with delayed PD observations and 2 ms physics.
#include <algorithm>
#include <array>
#include <cmath>
#include <mujoco/mujoco.h>
#include <omp.h>

constexpr int joints=22, points=8, ticks=355;


extern "C" int evaluate(const char* xml,int count,const double* initial,const double* keys,
                        const double* gains,const double* stand,const int* duration,int discovery,const double* crouch,double* results) {
    char error[1024]{};
    mjModel* model=mj_loadXML(xml,nullptr,error,sizeof(error));
    if(!model || model->nq!=29 || model->nu!=22) return 1;
    model->opt.timestep=.002;
    int landing=0;for(int j=0;j<5;++j) landing+=duration[j];
    const int pelvis=mj_name2id(model,mjOBJ_BODY,"pelvis");
    const int floor=mj_name2id(model,mjOBJ_GEOM,"floor");
    const int lf=mj_name2id(model,mjOBJ_GEOM,"left_ankle_roll_link_collision_1");
    const int rf=mj_name2id(model,mjOBJ_GEOM,"right_ankle_roll_link_collision_1");
    #pragma omp parallel num_threads(7)
    {
        mjData* d=mj_makeData(model);
        #pragma omp for schedule(dynamic)
        for(int row=0;row<count;++row){
            mj_resetData(model,d);
            const double* state=initial+57*row;
            std::copy_n(state,29,d->qpos);
            std::copy_n(state+29,28,d->qvel);
            mj_forward(model,d);
            double recent_q[7][22], recent_v[7][22], pitch[7], wy[7], roll[7], wx[7];
            double desired[22], sent[22], start[22];
            int slot=0;
            auto save=[&](int k){
                std::copy_n(d->qpos+7,22,recent_q[k]);
                std::copy_n(d->qvel+6,22,recent_v[k]);
                const double *quat=d->qpos+3;
                const double x=2*(quat[1]*quat[3]-quat[0]*quat[2]);
                const double y=2*(quat[2]*quat[3]+quat[0]*quat[1]);
                const double z=1-2*(quat[1]*quat[1]+quat[2]*quat[2]);
                pitch[k]=std::asin(std::clamp(-x,-1.,1.)); roll[k]=std::atan2(y,z);
                wy[k]=d->qvel[4]; wx[k]=d->qvel[3];
            };
            for(int k=0;k<7;++k) save(k);
            std::copy_n(state+7,22,sent);
            std::copy_n(sent,22,start);
            const double* trajectory=keys+row*points*joints;
            const double* g=gains+row*6;
            double approach=0.,best_approach=0.;
            double crouch_sum=0., crouch_best=0., filtered=0.;
            double support_sum=0., capture_best=0.; int run=0, longest=0;
            double score=0., max_speed=0., max_depth=0., quiet=0., highest=0.;
            for(int t=0;t<ticks;++t){
                int local=t,stage=0;
                while(stage<points && local>=duration[stage]) local-=duration[stage++];
                const double weight=stage<points?.5-.5*std::cos(3.141592653589793*(local+1)/duration[stage]):1.;
                const double* from=stage==0?start:trajectory+(stage-1)*joints;
                const double* to=stage==points?(discovery==3?trajectory+(points-1)*joints:stand):trajectory+stage*joints;
                const int a=(slot+1)%7,b=(slot+2)%7;
                const double p=(pitch[a]+pitch[b])*.5, v=(wy[a]+wy[b])*.5;
                const double r=(roll[a]+roll[b])*.5, w=(wx[a]+wx[b])*.5;
                const double blend=std::clamp((t-landing+4.)/(duration[5]+duration[6]),0.,1.);
                for(int j=0;j<22;++j){
                    double raw=from[j]+(to[j]-from[j])*weight;
                    if(j==0 || j==6) raw+=blend*(g[0]*p+g[1]*v);
                    if(j==4 || j==10) raw+=blend*(g[2]*p+g[3]*v);
                    if(j==5 || j==11) raw+=blend*(g[4]*r+g[5]*w);
                    const double kp=model->actuator_gainprm[j*mjNGAIN];
                    const double kd=-model->actuator_biasprm[j*mjNBIAS+2];
                    const double eff=model->actuator_forcerange[2*j+1];
                    const int ji=model->actuator_trnid[2*j];
                    raw=std::clamp(raw,model->jnt_range[2*ji],model->jnt_range[2*ji+1]);
                    const double q=(recent_q[a][j]+recent_q[b][j])*.5;
                    const double qd=(recent_v[a][j]+recent_v[b][j])*.5;
                    const double torque=std::clamp(kp*(raw-q)-kd*qd,-eff,eff);
                    desired[j]=std::clamp(q+(torque+kd*qd)/kp,stand[j]-2.,stand[j]+2.);
                }
                for(int sub=0;sub<10;++sub){
                    for(int j=0;j<22;++j){
                        sent[j]+=std::clamp(desired[j]-sent[j],-.2,.2);
                        d->ctrl[j]=sent[j];
                    }
                    mj_step(model,d); slot=(slot+1)%7; save(slot);
                    for(int j=0;j<22;++j) max_speed=std::max(max_speed,std::abs(d->qvel[j+6]));
                    for(int k=0;k<d->ncon;++k) max_depth=std::max(max_depth,-d->contact[k].dist);
                }
                bool foot=false,body=false;
                for(int k=0;k<d->ncon;++k){
                    const mjContact& c=d->contact[k];
                    if(c.geom1!=floor && c.geom2!=floor) continue;
                    const int other=c.geom1==floor?c.geom2:c.geom1;
                    if(other==lf || other==rf) foot=true;
                    else body=true;
                }
                const double up=d->xmat[9*pelvis+8], z=d->qpos[2];
                const double support=foot && !body;
                const double calm=1./(1.+std::hypot(d->qvel[3],d->qvel[4]));
                const double stable=support && z>.36 && up>.92;
                const double quality=1.5*std::clamp(z/.41,0.,1.)+up+.6*support+stable*(1.+calm);
                if(t>landing-27) {score+=quality*(t>landing+58?1.:.25);highest=std::max(highest,quality);}
                if(t>=duration[0]){
                    const double balance=(std::max(0.,up)+.1)*std::clamp(z/.40,0.,1.);
                    const double progress=3.*balance+1.7*stable+.35*support*balance;
                    best_approach=std::max(best_approach,progress);
                    approach+=progress;
                }
                if((discovery==2 || discovery==3) && t>landing-10) {
                    double force=0.;
                    for(int k=0;k<d->ncon;++k) {
                        const auto& c=d->contact[k];
                        if((c.geom1==floor && (c.geom2==lf || c.geom2==rf)) ||
                           (c.geom2==floor && (c.geom1==lf || c.geom1==rf))) {
                            mjtNum wrench[6]; mj_contactForce(model,d,k,wrench);
                            force+=std::max(0.,wrench[0]);
                        }
                    }
                    const double vertical=std::clamp((z-.16)/.22,0.,1.);
                    const double tilt=std::clamp((up-.3)/.65,0.,1.);
                    const double grounded=std::min(1.,force/60.) * !body;
                    const double quality=vertical*tilt*grounded;
                    const double slow=1./(1.+.3*std::hypot(d->qvel[3],d->qvel[4])
                                           +.8*std::hypot(d->qvel[0],d->qvel[1])+std::abs(d->qvel[2]));
                    support_sum+=quality*(.4+slow);
                    capture_best=std::max(capture_best,quality*(.4+slow));
                    run=(quality>.55)?run+1:0;
                    longest=std::max(longest,run);
                    if(discovery==3) {
                        double mse=0.;
                        for(int j=0;j<joints;++j) {
                            double error=d->qpos[j+7]-crouch[j]; mse+=error*error;
                        }
                        const double* rot=d->xmat+9*pelvis;
                        double alignment=0.;
                        for(int j=0;j<3;++j)alignment+=rot[6+j]*crouch[22+j];
                        const double torso=std::exp(-3.*std::max(0.,1.-alignment));
                        const double height=std::exp(-9.*std::abs(z-.25));
                        const double posture=std::exp(-2.*mse/joints);
                        const double load=std::clamp(force/65.,0.,1.);
                        const double stage_score=torso*height*posture*(.25+.75*load)*(.5+.5*slow);
                        filtered=.85*filtered+.15*stage_score;
                        crouch_best=std::max(crouch_best,filtered);
                        crouch_sum+=filtered;
                    }
                }
                if(t>=ticks-50)quiet+=stable;
            }
            double* result=results+5*row;
            result[0]=score/145.+.5*highest+.035*quiet-.10*std::max(0.,max_speed-10.)
                        -.07*std::max(0.,1000*max_depth-8.);
            if(discovery) result[0]=3.*best_approach+approach/100.+.12*quiet
                  -.012*std::max(0.,max_speed-13.)-.012*std::max(0.,1000*max_depth-9.);
            if(discovery==2) result[0]=3.*capture_best+.12*support_sum+.08*longest+.05*quiet
                    -.045*std::max(0.,max_speed-13.)-.04*std::max(0.,1000*max_depth-9.);
            if(discovery==3) result[0]=10.*crouch_best+.025*crouch_sum
                    -.10*std::max(0.,max_speed-10.)-.08*std::max(0.,1000*max_depth-7.);
            result[1]=quiet;result[2]=max_speed;result[3]=1000*max_depth;result[4]=highest;
        }
        mj_deleteData(d);
    }
    mj_deleteModel(model);return 0;
}
