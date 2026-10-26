"""Create the requested six-panel figure from actual HP-Q1 episode sums."""

import argparse
import csv
import json
from pathlib import Path
import sys

from worker import identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"project_tools"))
    from plot_training_dashboard import create_dashboard
    rows=[json.loads(line) for line in (args.run/"training_evidence.jsonl").read_text().splitlines()]
    points={}
    for key,field in (("episode_return","normalized_return_sum"),("episode_length","length_steps_sum"),("episode_termination","hard_terminations")):
        points[key]=[(r["step"],float(r[field]),float(r["completed_episodes"])) for r in rows]
    for key,field,scale in (("step_reward","reward_mean",.1),("world_model","wm_err8",1),("survival","imag_survival_H",1)):
        points[key]=[(r["step"],float(r[field])/scale,1.) for r in rows if r[field] is not None]
    receipt=json.loads((args.run/"receipt.json").read_text())
    run={"label":"RF08 HP-Q1 E4096 / WM1024 / actor1024-H32 / max1h","path":str(args.run),"points":points,"sources":[identity(args.run/"training_evidence.jsonl"),identity(args.run/"receipt.json"),identity(args.run/"config.yaml")],"last_logged_step":rows[-1]["step"],"requested_steps":320000,"num_envs":4096,"actor_batch":1024,"wm_batch":1024,"wm_window":8,"horizon":32,"status":receipt["status"],"ignored_incomplete_tail_bytes":0,"reward_max":.1,"reward_max_source":"Pinned HP MAX_STEP_REWARD=(4*1+2*.5)*.02"}
    dashboard=create_dashboard([run],args.output,window=500,title="Q1 SuperMimic RF08: SOMA / HP configuration / GPU Warp")
    # HP collects 2/3 steps per update at UTD=.4, plus the initial warmup.
    actual_steps={r["step"]:r["sim_steps"] for r in rows}
    csv_path=args.output/"training_dashboard.csv"
    with csv_path.open(newline="") as stream:
        reader=csv.DictReader(stream)
        fields=reader.fieldnames
        exported=list(reader)
    for row in exported:
        row["env_steps"]=actual_steps[int(row["iteration"])]
    with csv_path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(exported)
    dashboard["artifacts"]=[identity(item["path"]) for item in dashboard["artifacts"]]
    dashboard["env_steps_source"]="Actual collected sim_steps, including warmup; not learning iteration times num_envs"
    (args.output/"receipt.json").write_text(json.dumps(dashboard,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
