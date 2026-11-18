// Offline harness for the pinned native SONIC observation builders.
#include "math_utils.hpp"
#include "proprioception_buffer.hpp"
#include "smpl_obs.hpp"
#include "tokenizer_obs.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string_view>

namespace {
class RecordedWindow final : public agi_x2::ReferenceMotion {
 public:
  std::array<agi_x2::ReferenceFrame, NUM_FUTURE_FRAMES> frames{};
  agi_x2::ReferenceFrame Sample(double t) const override {
    const long slot = std::clamp(std::lround(t / DT_FUTURE_REF), 0L,
                                 static_cast<long>(frames.size() - 1));
    return frames[static_cast<std::size_t>(slot)];
  }
  std::string Name() const override { return "recorded_window"; }
  void Anchor(const std::array<double, 4>&) override {}
};

template <std::size_t N>
bool Read(std::array<double, N>& value) {
  std::cin.read(reinterpret_cast<char*>(value.data()), N * sizeof(double));
  return static_cast<bool>(std::cin) &&
         std::all_of(value.begin(), value.end(), [](double v) { return std::isfinite(v); });
}

int BuildSmplBatch() {
  for (int frames = 0; frames < 1000; ++frames) {
    if (std::cin.peek() == EOF) return std::cout ? 0 : 3;
    std::array<double, 4> base, anchor_base, anchor_human;
    std::array<double, 6> wrists;
    if (!Read(base) || !Read(anchor_base) || !Read(anchor_human) || !Read(wrists)) return 2;
    std::array<agi_x2::SmplFrame, agi_x2::SMPL_WINDOW> window{};
    for (auto& frame : window) {
      std::array<double, agi_x2::SMPL_JOINTS> joints;
      if (!Read(joints) || !Read(frame.quat_wxyz)) return 2;
      std::copy(joints.begin(), joints.end(), frame.joints.begin());
    }
    const auto align = agi_x2::SmplYawAlign(anchor_base, anchor_human);
    const auto features = agi_x2::BuildSmplObs(window, base, wrists, align, "full");
    std::cout.write(reinterpret_cast<const char*>(features.data()), features.size() * sizeof(float));
  }
  return std::cin.peek() == EOF && std::cout ? 0 : 3;
}
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "smpl") return BuildSmplBatch();
  if (argc != 1) return 2;
  agi_x2::ProprioceptionBuffer history;
  for (int frames = 0; frames < 1000; ++frames) {
    if (std::cin.peek() == EOF) return std::cout ? 0 : 3;
    std::array<double, 4> quat;
    std::array<double, 3> gyro;
    std::array<double, NUM_DOFS> q, dq, last_action, relative_il, velocity_il;
    if (!Read(quat) || !Read(gyro) || !Read(q) || !Read(dq) || !Read(last_action)) return 2;
    RecordedWindow reference;
    for (auto& frame : reference.frames) {
      if (!Read(frame.joint_pos_mj) || !Read(frame.joint_vel_mj) || !Read(frame.root_quat_xyzw)) return 2;
    }
    for (std::size_t mj = 0; mj < NUM_DOFS; ++mj) {
      const auto il = mujoco_to_isaaclab[mj];
      relative_il[il] = q[mj] - default_angles[mj];
      velocity_il[il] = dq[mj];
    }
    history.Append(gyro, relative_il, velocity_il, last_action,
                   agi_x2::body_frame_gravity_from_quat_wxyz(quat));
    const auto token = agi_x2::BuildTokenizerObs(reference, 0., quat);
    const auto prop = history.GetFlat();
    std::cout.write(reinterpret_cast<const char*>(token.data()), token.size() * sizeof(float));
    std::cout.write(reinterpret_cast<const char*>(prop.data()), prop.size() * sizeof(float));
  }
  return std::cin.peek() == EOF && std::cout ? 0 : 3;
}
