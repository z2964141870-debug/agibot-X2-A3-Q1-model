#pragma once
// Local policy IPC only. This component has NO AimDK/ROS/HAL command publisher.
#include <zmq.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace x2_morph {
using Json = nlohmann::json;
struct AdvisoryTarget {
  std::vector<std::string> joint_names;
  std::array<double, 31> position_rad;
  std::int64_t state_sequence;
};

class Client {
 public:
  explicit Client(const std::string& endpoint, int io_timeout_ms = 1000) {
    if (endpoint.rfind("tcp://127.0.0.1:", 0) != 0)
      throw std::runtime_error("development client permits localhost only");
    context_ = zmq_ctx_new();
    if (!context_) throw std::runtime_error("zmq context failed");
    socket_ = zmq_socket(context_, ZMQ_REQ);
    if (!socket_) { close(); throw std::runtime_error("zmq socket failed"); }
    int linger=0;
    zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger));
    zmq_setsockopt(socket_, ZMQ_SNDTIMEO, &io_timeout_ms, sizeof(io_timeout_ms));
    zmq_setsockopt(socket_, ZMQ_RCVTIMEO, &io_timeout_ms, sizeof(io_timeout_ms));
    if (zmq_connect(socket_,endpoint.c_str())!=0) { close(); throw std::runtime_error("zmq connect failed"); }
  }
  Client(const Client&)=delete;
  Client& operator=(const Client&)=delete;
  ~Client(){close();}
  Json request(const Json& message) {
    const auto bytes=message.dump();
    if (zmq_send(socket_,bytes.data(),bytes.size(),0)<0)
      throw std::runtime_error("policy request failed; recreate client before retry");
    zmq_msg_t reply;zmq_msg_init(&reply);
    if (zmq_msg_recv(&reply,socket_,0)<0) {
      zmq_msg_close(&reply);
      throw std::runtime_error("policy reply unavailable; no target; recreate client");
    }
    std::string body(static_cast<char*>(zmq_msg_data(&reply)),zmq_msg_size(&reply));
    zmq_msg_close(&reply);
    return Json::parse(body);
  }
  static std::optional<AdvisoryTarget> decode(const Json& reply, std::int64_t expected_sequence,
                                              const std::vector<std::string>& expected_joint_names) {
    if (!reply.value("ready",false)) return std::nullopt;
    if (reply.at("kind")!="x2_morph_advisory_target" || reply.at("version")!=1 ||
        reply.at("publish_allowed")!=false || reply.at("robot_profile")!="morphmimic_v01_x2_ultra" ||
        reply.at("policy_sha256")!="cf95194793e4732447f66432fc2381d82dd7c3d82c94da15ad074f2cfe295512" ||
        reply.at("state_sequence").get<std::int64_t>()!=expected_sequence)
      throw std::runtime_error("advisory identity/sequence mismatch");
    const auto names=reply.at("joint_names").get<std::vector<std::string>>();
    const auto q=reply.at("q_target_rad").get<std::vector<double>>();
    if (names!=expected_joint_names || names.size()!=31 || q.size()!=31)
      throw std::runtime_error("X2 model joint order mismatch");
    AdvisoryTarget result;result.joint_names=names;result.state_sequence=expected_sequence;
    for (std::size_t i=0;i<31;++i) {
      if (!std::isfinite(q[i]))throw std::runtime_error("nonfinite advisory target");
      result.position_rad[i]=q[i];
    }
    return result;
  }
 private:
  void close(){if(socket_){zmq_close(socket_);socket_=nullptr;}if(context_){zmq_ctx_term(context_);context_=nullptr;}}
  void* context_=nullptr;
  void* socket_=nullptr;
};
}  // namespace x2_morph
