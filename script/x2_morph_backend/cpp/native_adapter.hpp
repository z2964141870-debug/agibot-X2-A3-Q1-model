#pragma once
// Native candidate data only: no ROS/AimDK publishers or MC orchestration.
#include "../../x2_morph_human/cpp/morph_advisory_client.hpp"

namespace x2_morph_backend {
using Json = x2_morph::Json;

struct NativeCandidate {
  std::array<double, 31> target_pos_mj;
  std::array<double, 31> model_kp;
  std::array<double, 31> model_kd;
  std::int64_t state_sequence;
};

inline std::optional<NativeCandidate> DecodeNative(
    const Json& reply, std::int64_t sequence,
    const std::vector<std::string>& native_names,
    const std::array<double, 31>& kp, const std::array<double, 31>& kd) {
  if (!reply.is_object() || !reply.at("ready").is_boolean()
      || !reply.at("publish_allowed").is_boolean()
      || reply.at("publish_allowed").get<bool>())
    throw std::runtime_error("advisory-only boolean contract required");
  if (!reply.at("ready").get<bool>()) return std::nullopt;
  if (!reply.at("version").is_number_integer()
      || !reply.at("state_sequence").is_number_integer()
      || !reply.at("canonical_action").is_array()
      || reply.at("canonical_action").size() != 29)
    throw std::runtime_error("policy sequence/action contract mismatch");
  for (const auto& value : reply.at("canonical_action")) {
    if (!value.is_number() || !std::isfinite(value.get<double>()))
      throw std::runtime_error("nonfinite canonical action");
  }
  const auto decoded = x2_morph::Client::decode(reply, sequence, native_names);
  if (!decoded) return std::nullopt;
  for (std::size_t i = 0; i < 31; ++i) {
    if (!std::isfinite(kp[i]) || !std::isfinite(kd[i]) || kp[i] < 0 || kd[i] < 0)
      throw std::runtime_error("invalid model gain metadata");
  }
  // Compatible with the existing writer's 31-slot MJ/native position layout.
  // No Sonic normalisation, action_scale, default pose, wrist sign or LPF here.
  return NativeCandidate{decoded->position_rad, kp, kd, sequence};
}
}  // namespace x2_morph_backend
