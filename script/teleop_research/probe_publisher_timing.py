"""Execute ONLY the supplied publisher's derivative method with synthetic inputs.
No demo module import, hardware, model, socket or robot execution.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def probe(source):
    text=Path(source).read_text()
    tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Gr00TZmqPosePublisher')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_joint_vel')
    # Compile only this inspected arithmetic function; omit every other AST node.
    module=ast.Module(body=[method],type_ignores=[])
    scope={'np':np};exec(compile(ast.fix_missing_locations(module),'<publisher-method-only>','exec'),scope)
    fn=scope['_compute_joint_vel']
    out={}
    # Synthetic trajectory q(t_source)=t_source radians; four 30Hz samples.
    positions=np.arange(4,dtype=np.float32)/30
    for label,period in [('source_paced',1/30),('slow_publish',1/15),('fast_publish',1/60)]:
        state=SimpleNamespace(prev_joint_pos_il=None,prev_time=None,prev_joint_vel_il=None,vel_alpha=1.0)
        vel=[]
        for i,q in enumerate(positions):
            vel.append(float(fn(state,np.array([q],dtype=np.float32),i*period)[0]))
        out[label]={'publication_period_s':period,'velocities_rad_s':vel}
    return dict(source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                extracted_method_first_line=method.lineno,alpha=1.0,results=out,
                synthetic_only=True,robot_effect_verified=False,
                interpretation='Same source positions, different publication clock -> different velocity reference. Not proof of a deployment bug: timestamp semantics must match use case.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True)
    args=p.parse_args();r=probe(args.source)
    with open(args.output,'x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r,indent=2))
