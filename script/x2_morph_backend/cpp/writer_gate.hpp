#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace x2_morph_backend {
using Joints = std::array<double, 31>;
enum class Stage { Standby, Control, Fault, Return, HoldForMc, Released };
inline const char* StageName(Stage s) {
  switch (s) {
    case Stage::Standby: return "STANDBY";
    case Stage::Control: return "CONTROL";
    case Stage::Fault: return "FAULT_DAMPING";
    case Stage::Return: return "RETURN_TO_MC_POSE";
    case Stage::HoldForMc: return "HOLD_FOR_MC";
    case Stage::Released: return "RELEASED";
  }
  return "INVALID";
}
inline bool Finite(const Joints& a) {
  return std::all_of(a.begin(), a.end(), [](double x) { return std::isfinite(x); });
}
struct MotorCommand { Joints q{}, kp{}, kd{}; };
struct WriterFeedback {
  Joints q{}, dq{};
  double oldest_receive_s = 0., gravity_z = 0.;
  bool complete = false;
};
struct WriterConfig {
  Joints kp{}, kd{}, fault_kd{}, lower{}, upper{};
  double feedback_age_s = .12, target_age_s = .08;
  double engage_s = 2., return_s = 2., max_velocity = 35., fall_cos = -.3;
};

// Called under the controller's mailbox mutex. No RPC, ROS, or allocation on
// the high-rate writer path; a fault never re-arms itself on a fresh target.
class WriterGate {
 public:
  explicit WriterGate(WriterConfig cfg) : cfg_(cfg) {
    if (!Finite(cfg.kp) || !Finite(cfg.kd) || !Finite(cfg.fault_kd) || !Finite(cfg.lower) || !Finite(cfg.upper)
        || !std::isfinite(cfg.feedback_age_s) || cfg.feedback_age_s <= 0
        || !std::isfinite(cfg.target_age_s) || cfg.target_age_s <= 0
        || !std::isfinite(cfg.engage_s) || cfg.engage_s <= 0
        || !std::isfinite(cfg.return_s) || cfg.return_s <= 0
        || !std::isfinite(cfg.max_velocity) || cfg.max_velocity <= 0
        || !std::isfinite(cfg.fall_cos) || cfg.fall_cos < -1 || cfg.fall_cos > 1)
      throw std::runtime_error("invalid writer config");
    for (std::size_t i = 0; i < 31; ++i)
      if (cfg.kp[i] < 0 || cfg.kd[i] < 0 || cfg.fault_kd[i] < 0 || cfg.lower[i] > cfg.upper[i])
        throw std::runtime_error("invalid gain or joint range");
  }
  Stage stage() const { return stage_; }
  const std::string& reason() const { return reason_; }
  bool has_target() const { return target_.has_value(); }
  bool Healthy(const WriterFeedback& f, double now) const {
    return f.complete && Finite(f.q) && Finite(f.dq) && std::isfinite(now)
        && std::isfinite(f.oldest_receive_s) && f.oldest_receive_s > 0
        && f.oldest_receive_s <= now && now-f.oldest_receive_s <= cfg_.feedback_age_s
        && std::isfinite(f.gravity_z) && f.gravity_z <= cfg_.fall_cos
        && std::all_of(f.dq.begin(), f.dq.end(), [this](double x) { return std::abs(x) <= cfg_.max_velocity; });
  }
  bool Offer(const Joints& q, std::int64_t sequence, double stamp, double now) {
    bool valid = Finite(q) && sequence > sequence_ && std::isfinite(stamp)
        && stamp <= now && now-stamp <= cfg_.target_age_s;
    // These are virtual PD equilibrium targets, not commanded measured poses.
    // Morph's DC motor plant bounds torque, not these targets to qpos ranges.
    if (!valid) { Fault("invalid_target"); return false; }
    sequence_ = sequence;
    if (stage_ == Stage::Fault || stage_ == Stage::Return
        || stage_ == Stage::HoldForMc || stage_ == Stage::Released) return false;
    target_ = q; target_stamp_ = stamp;
    return true;
  }
  bool Arm(const WriterFeedback& f, double now, const MotorCommand& mc_pose,
           bool mc_stopped_and_bus_silent) {
    if (stage_ != Stage::Standby || !mc_stopped_and_bus_silent || !Healthy(f, now)
        || !FreshTarget(now) || !Finite(mc_pose.q) || !Finite(mc_pose.kp) || !Finite(mc_pose.kd))
      return false;
    for (std::size_t i = 0; i < 31; ++i)
      if (mc_pose.kp[i] < 0 || mc_pose.kd[i] < 0
          || mc_pose.q[i] < cfg_.lower[i] || mc_pose.q[i] > cfg_.upper[i]) return false;
    from_ = f.q; return_pose_ = mc_pose; entered_ = now;
    last_ = MotorCommand{f.q, cfg_.kp, cfg_.kd};
    stage_ = Stage::Control; reason_ = "explicit_mc_handoff";
    return true;
  }
  void Fault(const std::string& why) {
    if (stage_ == Stage::Control) { stage_ = Stage::Fault; reason_ = why; }
    target_.reset();
  }
  void EStop(const std::string& why) {
    if (stage_ == Stage::Control || stage_ == Stage::Return || stage_ == Stage::HoldForMc) {
      stage_ = Stage::Fault; reason_ = why; target_.reset();
    }
  }
  bool RequestReturn(double now) {
    if (stage_ != Stage::Control) return false;
    from_ = last_.q; entered_ = now; stage_ = Stage::Return;
    reason_ = "operator_return"; return true;
  }
  void ExternalPublisher() {
    if (stage_ != Stage::Standby && stage_ != Stage::Released) {
      reason_ = stage_ == Stage::HoldForMc ? "mc_bus_takeover" : "unexpected_external_writer";
      stage_ = Stage::Released; target_.reset();
    }
  }
  std::optional<MotorCommand> Tick(const WriterFeedback& f, double now) {
    if (stage_ == Stage::Standby || stage_ == Stage::Released) return std::nullopt;
    if (!Healthy(f, now) && stage_ != Stage::Fault) {
      stage_ = Stage::Fault; reason_ = "feedback_velocity_or_tilt_fault"; target_.reset();
    }
    if (stage_ == Stage::Control && !FreshTarget(now)) Fault("policy_target_expired");
    MotorCommand out;
    if (stage_ == Stage::Fault) {
      out.q = last_.q; out.kp.fill(0.); out.kd = cfg_.fault_kd;
    } else if (stage_ == Stage::Control) {
      const double alpha = std::clamp((now-entered_)/cfg_.engage_s, 0., 1.);
      for (std::size_t i = 0; i < 31; ++i) out.q[i] = from_[i] + alpha*((*target_)[i]-from_[i]);
      out.kp = cfg_.kp; out.kd = cfg_.kd;
    } else if (stage_ == Stage::Return) {
      const double alpha = std::clamp((now-entered_)/cfg_.return_s, 0., 1.);
      for (std::size_t i = 0; i < 31; ++i) out.q[i] = from_[i] + alpha*(return_pose_.q[i]-from_[i]);
      out.kp = cfg_.kp; out.kd = cfg_.kd;
      if (alpha >= 1.) stage_ = Stage::HoldForMc;
    } else {
      out = return_pose_;
    }
    last_ = out; return out;
  }
 private:
  bool FreshTarget(double now) const {
    return target_ && now >= target_stamp_ && now-target_stamp_ <= cfg_.target_age_s;
  }
  WriterConfig cfg_;
  Stage stage_ = Stage::Standby;
  std::string reason_ = "awaiting_explicit_authority";
  std::optional<Joints> target_;
  std::int64_t sequence_ = -1;
  double target_stamp_ = 0., entered_ = 0.;
  Joints from_{};
  MotorCommand last_{}, return_pose_{};
};
}  // namespace x2_morph_backend
