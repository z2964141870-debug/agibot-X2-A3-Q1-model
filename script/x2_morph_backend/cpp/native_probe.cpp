#include "native_adapter.hpp"
#include <iostream>

int main() {
  using namespace x2_morph_backend;
  try {
    std::string line;
    while (std::getline(std::cin, line)) {
      const auto row = Json::parse(line);
      const auto names = row.at("joint_names").get<std::vector<std::string>>();
      const auto kp = row.at("kp").get<std::array<double, 31>>();
      const auto kd = row.at("kd").get<std::array<double, 31>>();
      const auto candidate = DecodeNative(row.at("reply"), row.at("sequence").get<std::int64_t>(), names, kp, kd);
      if (!candidate) {
        std::cout << Json{{"ready", false}, {"publish_allowed", false}}.dump() << std::endl;
        continue;
      }
      std::cout << Json{{"ready", true}, {"publish_allowed", false},
                         {"q_target_rad", candidate->target_pos_mj},
                         {"model_kp", candidate->model_kp}, {"model_kd", candidate->model_kd},
                         {"state_sequence", candidate->state_sequence}}.dump() << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
