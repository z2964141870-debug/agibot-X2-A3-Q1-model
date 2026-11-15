"""Generate an isolated native source and profiles from the pinned R6 source."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA='ca3f80b4b8a77613b243e1c903ba856475e1fb0b4618aeb0d3cbb7a46a18ee47'


def transform(source):
    if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Original native controller changed; audit before generating')
    marker='    const auto a=Parse(argc,argv); const auto cfg=ReadJson(a.profile);'
    assert source.count(marker)==1 and source.count('received-sent>.020')==2
    result=source.replace(marker,marker+'''
    const double rpc_budget_ms=cfg.value("rpc_budget_ms",20.);
    if (!std::isfinite(rpc_budget_ms) || rpc_budget_ms<20. || rpc_budget_ms>30.)
      throw std::runtime_error("RPC experiment budget must be 20..30ms");''')
    result=result.replace('received-sent>.020','received-sent>rpc_budget_ms/1000.')
    old = """            if (received-sent>rpc_budget_ms/1000. || received-f.oldest_receive_s>wc.feedback_age_s)
              gate.Fault("rpc_or_feedback_deadline");
            else if (native) accepted=gate.Offer(native->target_pos_mj,seq,sent,received);
            else gate.Fault("human_reference_unavailable");"""
    new = """            if (received-f.oldest_receive_s>wc.feedback_age_s)
              gate.Fault("feedback_deadline");
            else if (!native && reply.value("ready",false))
              gate.Fault("policy_contract_invalid");
            else if (native && received-sent<=rpc_budget_ms/1000.)
              accepted=gate.Offer(native->target_pos_mj,seq,sent,received);
            // Late or unavailable replies are discarded. The unchanged writer
            // gate enforces the original target age on its independent clock."""
    assert result.count(old)==1
    result=result.replace(old,new)
    marker='{"mode",a.mode},{"ready_target",gate.has_target()}'
    assert result.count(marker)==1
    return result.replace(marker,'{"rpc_budget_ms",rpc_budget_ms},'+marker)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--project',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    a=p.parse_args();root=a.project.resolve();out=a.output.resolve()
    if out.exists():p.error('Preserve existing experiment output')
    cpp=root/'script/x2_morph_backend/cpp'
    source=transform((cpp/'native_controller.cpp').read_text())
    for name in ('native_adapter.hpp','writer_gate.hpp'):
        source=source.replace('#include "'+name+'"','#include "'+str(cpp/name)+'"')
    cmake=(cpp/'CMakeLists.txt').read_text().replace('add_executable(test_morph_writer test_writer_gate.cpp)',
      'add_executable(test_morph_writer "'+str(cpp/'test_writer_gate.cpp')+'")')
    # Native SDK adapter and writer gate remain the original source files.
    profile=json.loads(a.profile.read_text())
    assert profile['feedback_age_s']==.12 and profile['target_age_s']==.08
    assert profile['policy_hz']==50 and profile['writer_hz']==500
    out.mkdir(parents=True);(out/'source').mkdir()
    (out/'source/native_controller.cpp').write_text(source)
    (out/'source/CMakeLists.txt').write_text(cmake)
    for budget in (20,25,30):
        (out/f'profile_{budget}ms.json').write_text(json.dumps(dict(profile,rpc_budget_ms=budget),indent=2)+'\n')
    receipt=dict(status='ISOLATED_TOLERANT_SOURCE_PREPARED',original_source_sha256=SOURCE_SHA,
      generated_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
      profiles_ms=[20,25,30],scheduler_period_ms=20,target_max_age_ms=80,
      feedback_max_age_ms=120,writer_hz=500,hardware_commands=0)
    (out/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
