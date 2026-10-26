"""Exercise the final reporting wrapper with real GPU transitions and no optimizer."""

import argparse
import json
from pathlib import Path

from worker import prepare, learner_class, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    env=None
    try:
        _,env,cfg=prepare(args,num_envs=16)
        BoundedLearner,_=learner_class()
        learner=BoundedLearner(env,cfg,args.output,device="cuda:0",backend="mjwarp",budget_seconds=3500)
        learner.collect(64)
        learner.collect(8)
        learner.iteration=1
        learner.flush_reporting()
        rows=[json.loads(s) for s in (args.output/"training_evidence.jsonl").read_text().splitlines()]
        assert len(rows)==1 and rows[0]["transitions"]==128
        assert 0<=rows[0]["hard_terminations"]<=rows[0]["completed_episodes"]
        assert rows[0]["wm_err8"] is None and rows[0]["imag_survival_H"] is None
        assert learner.wm_updates==learner.actor_updates==0
        save(args.output/"receipt.json",{"status":"REPORTING_REAL_TRANSITIONS_PASSED","num_envs":16,"control_steps":72,"physical_transitions":1152,"optimizer_updates":0,"synthetic_iteration_counter_for_reporting_test_only":1,"recorded_window":rows[0]})
        print(json.dumps({"status":"REPORTING_REAL_TRANSITIONS_PASSED","optimizer_updates":0}))
    finally:
        if env is not None:
            env.close()
            env.backend.close()


if __name__=="__main__":
    main()
