"""Independent f0 worker: change only CPU conversion; reuse saved action math."""

import argparse
from contextlib import ExitStack, redirect_stdout
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'q1_five_stage'), str(HERE.parent/'primebot_q1')]
from common import require
from native_state_action_math import isolated_converter
from paused_runtime import assert_owner
from policy_process import MAX_MESSAGE
from student_same_state_worker import SameStateSession


class IsolatedStateSession(SameStateSession):
    def __init__(self, args):
        super().__init__(args)
        from native_state_cpu import KinematicState
        self.state_converter = KinematicState(self.robot, self.spec)

    def ready(self):
        return dict(super().ready(), cpu_converter='isolated_mujoco_kinematics',
                    original_worker_source_modified=False)

    def action(self, request):
        import mujoco
        import q1_native_review
        started = time.perf_counter_ns()
        with ExitStack() as stack:
            stack.enter_context(isolated_converter(q1_native_review, self.robot, self.spec, self.state_converter))
            for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse'):
                stack.enter_context(patch.object(mujoco, name, side_effect=AssertionError('No dynamics in action parity')))
            result = super().action(request)
        result['profile']['converter_dispatch_worker_wall_ns'] = time.perf_counter_ns()-started
        result['profile']['cpu_converter'] = 'isolated_mujoco_kinematics'
        return result


def serve(args):
    output = sys.stdout
    def reply(value):
        raw = json.dumps(value, allow_nan=False)
        require(len(raw.encode()) < MAX_MESSAGE, 'CPU-action RPC bound exceeded')
        output.write(raw+'\n'); output.flush()
    with redirect_stdout(sys.stderr):
        try:
            import mujoco
            with ExitStack() as stack:
                for name in ('mj_step', 'mj_step1', 'mj_step2'):
                    stack.enter_context(patch.object(mujoco, name, side_effect=AssertionError('No physical integration')))
                session = IsolatedStateSession(args)
                reply(dict(ok=True, value=dict(session.ready(), canonical_present=session.actor.present.tolist())))
                while True:
                    raw = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                    require(raw and raw.endswith(b'\n') and len(raw) <= MAX_MESSAGE, 'Invalid bounded action request')
                    request = json.loads(raw)
                    if request.get('op') == 'quit':
                        reply(dict(ok=True, value=dict(closed=True))); return
                    require(request.get('op') in ('start', 'action'), 'Unknown action-check operation')
                    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                    reply(dict(ok=True, value=getattr(session, request['op'])(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f'{type(error).__name__}: {error}'))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('plan', 'checkpoint', 'frozen', 'primebot-checkpoint', 'morph-source', 'primebot-source', 'student-checkpoint'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--event-kind', choices=('nominal',), required=True)
    p.add_argument('--expected-gpu', required=True)
    p.add_argument('--gpu-owner-pid', type=int, required=True)
    p.add_argument('--gpu-owner-ticks', required=True)
    p.add_argument('--preview-frames', type=int, choices=(0,), required=True)
    p.add_argument('--tail-method', choices=('masked',), required=True)
    serve(p.parse_args())
