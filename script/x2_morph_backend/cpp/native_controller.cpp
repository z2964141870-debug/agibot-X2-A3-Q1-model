#include "native_adapter.hpp"
#include "writer_gate.hpp"
#include "aimdk_io.hpp"
#include <aimdk_msgs/msg/joint_command_array.hpp>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>

using namespace x2_morph_backend;
using Clock = std::chrono::steady_clock;
static volatile std::sig_atomic_t stopping = 0;
static void Signal(int) { stopping = 1; }
static double Now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
static Json ReadJson(const std::filesystem::path& p) {
  std::ifstream f(p); if (!f) throw std::runtime_error("cannot read " + p.string());
  Json j; f >> j; return j;
}

struct RuntimeThreads {
  std::atomic<bool>& done;
  rclcpp::executors::MultiThreadedExecutor& executor;
  std::thread spin, inference;
  void Join() {
    done=true; executor.cancel();
    if (inference.joinable()) inference.join();
    if (spin.joinable()) spin.join();
  }
  ~RuntimeThreads() { Join(); }
};

struct Args {
  std::string profile, output, endpoint="tcp://127.0.0.1:5765", mode="observe";
  std::string prefix="/aima/hal/joint", imu="/aima/hal/imu/torso/state", run_id;
  double seconds=10.; bool confirmed=false;
};
static Args Parse(int argc, char** argv) {
  Args a;
  for (int i=1; i<argc; ++i) {
    std::string k=argv[i];
    auto val=[&] { if (++i>=argc) throw std::runtime_error("missing "+k); return std::string(argv[i]); };
    if (k=="--profile") a.profile=val(); else if (k=="--output") a.output=val();
    else if (k=="--endpoint") a.endpoint=val(); else if (k=="--mode") a.mode=val();
    else if (k=="--prefix") a.prefix=val(); else if (k=="--imu-topic") a.imu=val();
    else if (k=="--run-id") a.run_id=val(); else if (k=="--seconds") a.seconds=std::stod(val());
    else if (k=="--confirm-live") a.confirmed=val()=="X2_ULTRA_MORPH_CONTROL";
    else throw std::runtime_error("unknown argument "+k);
  }
  if (a.profile.empty() || a.output.empty() || a.run_id.empty() || !std::isfinite(a.seconds)
      || a.seconds<=0 || a.seconds>600.) throw std::runtime_error("profile/output/run-id and 0..600s required");
  if (a.mode!="observe" && a.mode!="isolated" && a.mode!="hardware") throw std::runtime_error("invalid mode");
  if (a.mode=="isolated" && (a.prefix.rfind("/x2_morph_test/",0)!=0
      || a.imu.rfind("/x2_morph_test/",0)!=0)) throw std::runtime_error("isolated mode requires private test topics");
  if (a.mode=="hardware" && (!a.confirmed || a.prefix!="/aima/hal/joint"
      || a.imu!="/aima/hal/imu/torso/state")) throw std::runtime_error("explicit hardware confirmation required");
  if (a.mode=="observe" && (a.prefix!="/aima/hal/joint" || a.imu!="/aima/hal/imu/torso/state"))
    throw std::runtime_error("observe mode uses real feedback with output disabled");
  return a;
}

int main(int argc, char** argv) {
  try {
    const auto a=Parse(argc,argv); const auto cfg=ReadJson(a.profile);
    int authority_lock=-1;
    if (a.mode=="hardware") {
      authority_lock=open("/tmp/x2_morph_native_hardware.lock",O_CREAT|O_RDWR,0600);
      if (authority_lock<0 || flock(authority_lock,LOCK_EX|LOCK_NB)!=0)
        throw std::runtime_error("another Morph hardware writer owns the lock");
    }
    auto names=cfg.at("joint_names").get<std::vector<std::string>>();
    if (names.size()!=31) throw std::runtime_error("31 native joint names required");
    for (std::size_t i=0;i<31;++i)
      if (names[i]!=mujoco_joint_names[i]) throw std::runtime_error("Sonic/Morph native joint order mismatch");
    WriterConfig wc;
    wc.kp=cfg.at("kp").get<Joints>(); wc.kd=cfg.at("kd").get<Joints>();
    wc.fault_kd=cfg.at("fault_kd").get<Joints>();
    wc.lower=cfg.at("lower").get<Joints>(); wc.upper=cfg.at("upper").get<Joints>();
    wc.feedback_age_s=cfg.at("feedback_age_s"); wc.target_age_s=cfg.at("target_age_s");
    wc.engage_s=cfg.at("engage_s"); wc.return_s=cfg.at("return_s");
    wc.max_velocity=cfg.at("max_velocity"); wc.fall_cos=cfg.at("fall_cos");
    const int hz=cfg.at("writer_hz");
    if (hz<50 || hz>1000 || cfg.at("policy_hz")!=50) throw std::runtime_error("invalid loop rates");
    const std::filesystem::path out=a.output;
    if (!std::filesystem::create_directory(out)) throw std::runtime_error("output must be a new directory");
    std::ofstream tape(out/"native.jsonl");
    std::mutex mutex, log_mutex;
    WriterGate gate(wc);
    MotorCommand captured_mc;
    std::array<bool,4> mc_seen{};
    bool mc_capture_frozen=false;
    std::atomic<double> external_last{0.};
    std::atomic<bool> done{false}, transport_failed{false};
    std::atomic<std::uint64_t> requests{0}, candidates{0}, rejected{0}, writer_ticks{0};
    std::signal(SIGINT,Signal); std::signal(SIGTERM,Signal);
    rclcpp::init(0,nullptr);
    auto node=std::make_shared<rclcpp::Node>("x2_morph_native_"+a.run_id,
        rclcpp::NodeOptions().enable_rosout(false).start_parameter_services(false));
    agi_x2::AimdkIo io(node,a.prefix+"/leg",a.prefix+"/waist",a.prefix+"/arm",a.prefix+"/head",a.imu);
    std::vector<rclcpp::Subscription<aimdk_msgs::msg::JointCommandArray>::SharedPtr> watchers;
    rclcpp::SubscriptionOptions options; options.ignore_local_publications=true;
    const std::array<std::string,4> groups={"leg","waist","arm","head"};
    const std::array<int,5> offsets={0,12,15,29,31};
    for (std::size_t g=0;g<4;++g) watchers.push_back(
      node->create_subscription<aimdk_msgs::msg::JointCommandArray>(a.prefix+"/"+groups[g]+"/command",
        rclcpp::SensorDataQoS(),[&,g](const aimdk_msgs::msg::JointCommandArray::SharedPtr msg) {
          external_last.store(Now());
          std::lock_guard<std::mutex> lock(mutex);
          if (gate.stage()!=Stage::Standby) { gate.ExternalPublisher(); return; }
          if (mc_capture_frozen) return;
          if (msg->joints.size()!=static_cast<std::size_t>(offsets[g+1]-offsets[g])) return;
          for (int i=offsets[g];i<offsets[g+1];++i) {
            const auto& j=msg->joints[i-offsets[g]];
            if (j.name!=names[i] || !std::isfinite(j.position) || !std::isfinite(j.stiffness)
                || !std::isfinite(j.damping) || j.stiffness<0 || j.damping<0) return;
          }
          for (int i=offsets[g];i<offsets[g+1];++i) {
            const auto& j=msg->joints[i-offsets[g]];
            captured_mc.q[i]=j.position; captured_mc.kp[i]=j.stiffness; captured_mc.kd[i]=j.damping;
          }
          mc_seen[g]=true;
        },options));
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),3);
    executor.add_node(node);
    auto feedback=[](const agi_x2::RobotState& state) {
      WriterFeedback f; f.complete=state.leg_seen && state.waist_seen && state.arm_seen
          && state.head_seen && state.imu_seen;
      f.q=state.joint_pos_mj; f.dq=state.joint_vel_mj;
      f.oldest_receive_s=std::min({state.leg_stamp_s,state.waist_stamp_s,state.arm_stamp_s,
                                 state.head_stamp_s,state.imu_stamp_s});
      const auto& q=state.base_quat_wxyz;
      const double n=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
      if (!std::isfinite(n) || std::abs(n-1.)>1e-3) f.complete=false;
      f.gravity_z=-(1.-2.*(q[1]*q[1]+q[2]*q[2]));
      return f;
    };
    RuntimeThreads threads{done,executor,{},{}};
    threads.spin=std::thread([&] {
      try { executor.spin(); }
      catch (...) { std::lock_guard<std::mutex> lock(mutex); gate.EStop("feedback_executor_failure"); }
    });
    threads.inference=std::thread([&] {
      try {
        x2_morph::Client client(a.endpoint,100);
        std::int64_t seq=0; auto next=Clock::now();
        while (!done && rclcpp::ok()) {
          next+=std::chrono::milliseconds(20);
          agi_x2::RobotState state; io.SnapshotState(state);
          auto f=feedback(state); const double sent=Now();
          if (!f.complete || sent-f.oldest_receive_s>wc.feedback_age_s) {
            { std::lock_guard<std::mutex> lock(mutex); gate.Fault("feedback_unavailable"); }
            std::this_thread::sleep_until(next); continue;
          }
          Json measurement={{"kind","x2_measurement"},{"version",1},{"sequence",seq},
            {"joint_names",names},{"q_rad",f.q},{"dq_rad_s",f.dq},
            {"imu_quat_wxyz",state.base_quat_wxyz},{"imu_gyro_rad_s",state.base_ang_vel},
            {"imu_body",cfg.at("imu_body")},{"body_from_sensor_wxyz",cfg.at("body_from_sensor_wxyz")},
            {"source_receive_ages_s",{{"leg",sent-state.leg_stamp_s},{"waist",sent-state.waist_stamp_s},
              {"arm",sent-state.arm_stamp_s},{"head",sent-state.head_stamp_s},{"imu",sent-state.imu_stamp_s}}}};
          const auto reply=client.request(measurement); const double received=Now(); ++requests;
          const auto native=DecodeNative(reply,seq,names,wc.kp,wc.kd);
          bool accepted=false; std::string rejection_reason;
          std::vector<std::string> outside_position_ranges;
          if (native) for (std::size_t i=0;i<31;++i)
            if (native->target_pos_mj[i]<wc.lower[i] || native->target_pos_mj[i]>wc.upper[i])
              outside_position_ranges.push_back(names[i]);
          {
            std::lock_guard<std::mutex> lock(mutex);
            if (received-sent>.020 || received-f.oldest_receive_s>wc.feedback_age_s)
              gate.Fault("rpc_or_feedback_deadline");
            else if (native) accepted=gate.Offer(native->target_pos_mj,seq,sent,received);
            else gate.Fault("human_reference_unavailable");
            if (!accepted) rejection_reason=!native?"human_reference_unavailable":
                (received-sent>.020?"rpc_deadline":
                 (received-f.oldest_receive_s>wc.feedback_age_s?"feedback_deadline":gate.reason()));
          }
          if (accepted) ++candidates; else ++rejected;
          { std::lock_guard<std::mutex> lock(log_mutex);
            tape << Json({{"kind","policy_sample"},{"state",measurement},{"reply",reply},
              {"accepted",accepted},{"rejection_reason",rejection_reason},
              {"pd_targets_outside_qpos_ranges",outside_position_ranges},
              {"rpc_ms",(received-sent)*1000.}}).dump() << '\n'; }
          ++seq;
          if (Clock::now()>next) next=Clock::now();
          std::this_thread::sleep_until(next);
        }
      } catch (const std::exception& e) {
        transport_failed=true;
        { std::lock_guard<std::mutex> lock(mutex); gate.Fault("policy_transport_or_contract_failure"); }
        std::lock_guard<std::mutex> lock(log_mutex);
        tape << Json({{"kind","policy_failure"},{"error",e.what()}}).dump() << '\n';
      }
    });
    Json status;
    auto write_status=[&] {
      std::lock_guard<std::mutex> lock(mutex);
      status={{"run_id",a.run_id},{"stage",StageName(gate.stage())},{"reason",gate.reason()},
        {"mode",a.mode},{"ready_target",gate.has_target()},{"requests",requests.load()},
        {"candidates",candidates.load()},{"rejected",rejected.load()},
        {"writer_ticks",writer_ticks.load()},{"sdk_publish_calls",io.commands_published()},
        {"hardware_commands",a.mode=="hardware"?io.commands_published():0},
        {"transport_failed",transport_failed.load()},{"mc_capture_complete",std::all_of(mc_seen.begin(),mc_seen.end(),[](bool x){return x;})},
        {"mc_capture_frozen",mc_capture_frozen},
        {"external_age_s",external_last>0?Now()-external_last.load():-1.},
        {"captured_mc",{{"q",captured_mc.q},{"kp",captured_mc.kp},{"kd",captured_mc.kd}}}};
      { std::ofstream f(out/"status.json.tmp"); f << status.dump(2) << '\n'; }
      std::filesystem::rename(out/"status.json.tmp",out/"status.json");
    };
    const int original_flags=fcntl(STDIN_FILENO,F_GETFL,0);
    fcntl(STDIN_FILENO,F_SETFL,original_flags|O_NONBLOCK);
    std::string input;
    const double start=Now(); double last_status=0.; auto tick=Clock::now();
    agi_x2::RobotState writer_state;
    while (!done && rclcpp::ok()) {
      // The high-rate writer never waits for feedback callbacks' mutex.
      // Retained snapshots still expire against their original receive stamps.
      io.TrySnapshotState(writer_state);
      const auto f=feedback(writer_state); const double now=Now();
      char bytes[4096]; const auto n=read(STDIN_FILENO,bytes,sizeof(bytes));
      if (n>0) input.append(bytes,static_cast<std::size_t>(n));
      if (n==0 && a.mode=="hardware") {
        std::lock_guard<std::mutex> lock(mutex); gate.EStop("control_transport_closed");
      }
      for (std::size_t end;(end=input.find('\n'))!=std::string::npos;) {
        const auto line=input.substr(0,end); input.erase(0,end+1);
        try {
          const auto command=Json::parse(line);
          if (command.at("run_id")!=a.run_id) throw std::runtime_error("control run identity mismatch");
          std::lock_guard<std::mutex> lock(mutex);
          const auto op=command.at("op").get<std::string>();
          if (op=="capture_return") {
            if (gate.stage()!=Stage::Standby || !std::all_of(mc_seen.begin(),mc_seen.end(),[](bool x){return x;}))
              throw std::runtime_error("complete MC standing command required before capture");
            mc_capture_frozen=true;
          } else if (op=="arm") {
            const bool captured=std::all_of(mc_seen.begin(),mc_seen.end(),[](bool x){return x;});
            const bool silent=external_last>0 && now-external_last.load()>=.3;
            if (a.mode=="observe" || !captured || !mc_capture_frozen || !silent || !command.value("mc_stopped",false)
                || !gate.Arm(f,now,captured_mc,true)) throw std::runtime_error("arm preconditions failed");
          } else if (op=="return") gate.RequestReturn(now);
          else if (op=="fault") gate.EStop("operator_estop");
          else throw std::runtime_error("unknown control operation");
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> lock(log_mutex);
          tape << Json({{"kind","control_rejected"},{"error",e.what()}}).dump() << '\n';
        }
      }
      std::optional<MotorCommand> command; Stage stage;
      {
        std::lock_guard<std::mutex> lock(mutex);
        if ((stopping || now-start>=a.seconds) && gate.stage()==Stage::Control) gate.RequestReturn(now);
        command=gate.Tick(f,now); stage=gate.stage();
        // The same mutex serialises foreign-publisher suppression and this
        // publish, so no queued policy command follows a detected MC takeover.
        if (command && a.mode!="observe") {
          const Joints zero{}; io.PublishCommand(command->q,zero,command->kp,command->kd);
        }
      }
      ++writer_ticks;
      if (now-last_status>=.1) { write_status(); last_status=now; }
      if (stage==Stage::Released || ((stopping || now-start>=a.seconds)
          && (a.mode!="hardware" || stage==Stage::Standby))) break;
      tick+=std::chrono::nanoseconds(1'000'000'000/hz);
      if (Clock::now()>tick) tick=Clock::now();
      std::this_thread::sleep_until(tick);
    }
    threads.Join();
    write_status(); tape.flush(); fcntl(STDIN_FILENO,F_SETFL,original_flags);
    rclcpp::shutdown(); std::cout << status.dump() << '\n';
    if (authority_lock>=0) close(authority_lock);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "native backend: " << e.what() << '\n'; return 1;
  }
}
