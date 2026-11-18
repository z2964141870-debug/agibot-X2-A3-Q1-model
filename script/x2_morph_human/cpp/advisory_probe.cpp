// Reads JSONL from stdin; writes JSONL replies. No robot SDK is linked.
#include "morph_advisory_client.hpp"
#include <iostream>

int main(int argc,char** argv) {
  if(argc!=2){std::cerr<<"usage: advisory_probe tcp://127.0.0.1:5765 < requests.jsonl\n";return 2;}
  try {
    x2_morph::Client client(argv[1]);
    std::string line;
    while(std::getline(std::cin,line)) {
      const auto request=x2_morph::Json::parse(line);
      const auto reply=client.request(request);
      if(request.value("kind","")=="x2_measurement" && reply.value("ready",false)) {
        // Decode in the exported model order. A HAL adapter must explicitly map
        // this order to its feedback/writer order; it cannot use this probe to act.
        auto names=request.at("expected_model_joint_names").get<std::vector<std::string>>();
        x2_morph::Client::decode(reply,request.at("sequence").get<std::int64_t>(),names);
      }
      std::cout<<reply.dump()<<std::endl;
    }
  } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
