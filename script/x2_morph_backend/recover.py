"""Explicit operator recovery of one existing faulted Morph writer."""
import argparse
import json
from pathlib import Path
import time
from .mc_handoff import RosMc, HandoffError, stopped


def recover(mc, status, *, clock=time.monotonic, sleep=time.sleep, emit=lambda event: None):
    initial = status()
    if initial.get('stage') != 'FAULT_DAMPING' or initial.get('mode') != 'hardware':
        raise HandoffError('Recovery requires the existing FAULT_DAMPING writer')
    run_id = initial['run_id']
    def current():
        row = status()
        if row.get('run_id') != run_id:
            raise HandoffError('Writer identity changed')
        return row
    def wait(fn, seconds, reason):
        deadline = clock() + seconds
        while clock() < deadline:
            value = fn()
            if value:
                return value
            sleep(.1)
        raise HandoffError(reason)
    runtime = mc.runtime()
    if not runtime.get('status', '').startswith('Running('):
        if not stopped(runtime):
            raise HandoffError('MC runtime is unknown; no restart issued')
        emit('START_MC')
        mc.app('start_app')
    wait(lambda: mc.runtime().get('status', '').startswith('Running('), 30., 'MC runtime did not return')
    def mode_available():
        try:
            return mc.mode()
        except (HandoffError, OSError, TimeoutError):
            return None
    mode = wait(mode_available, 30., 'MC services did not return; retain supported posture and writer')
    if mode == 'PASSIVE_DEFAULT':
        emit('REQUEST_JOINT')
        mc.set_mode('JOINT_DEFAULT')
        wait(lambda: mode_available() == 'JOINT_DEFAULT', 8., 'MC JOINT mode not confirmed')
        mode = 'JOINT_DEFAULT'
    if mode not in ('JOINT_DEFAULT', 'STAND_DEFAULT', 'DAMPING_DEFAULT'):
        raise HandoffError('Unexpected MC mode; no further mode command issued: '+str(mode))
    emit('WAIT_WRITER_RELEASE')
    wait(lambda: current().get('stage') == 'RELEASED', 10., 'MC command bus takeover not observed; no STAND command issued')
    if mode != 'STAND_DEFAULT':
        emit('REQUEST_STAND_AFTER_RELEASE')
        mc.set_mode('STAND_DEFAULT')
    wait(lambda: mode_available() == 'STAND_DEFAULT', 8., 'MC STAND mode not confirmed')
    result = dict(status='MC_RECOVERY_CONFIRMED', run_id=run_id, native_stage=current()['stage'], mc_mode='STAND_DEFAULT')
    emit(result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--native-output', type=Path, required=True)
    p.add_argument('--native-pid', type=int, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--confirm-supported-recovery', choices=('X2_SUPPORTED_MC_RECOVERY',), required=True)
    a = p.parse_args()
    command = Path('/proc') / str(a.native_pid) / 'cmdline'
    argv = command.read_bytes().split(b'\0')
    if not any(x.endswith(b'/x2_morph_native') for x in argv):
        p.error('PID is not the existing native writer')
    if b'--run-id' not in argv or argv[argv.index(b'--run-id')+1].decode() != a.run_id:
        p.error('PID run identity does not match')
    def status():
        row = json.loads((a.native_output/'status.json').read_text())
        if row.get('run_id') != a.run_id:
            raise HandoffError('Native status run identity mismatch')
        return row
    def emit(event):
        entry = dict(monotonic_s=time.monotonic(), event=event)
        print(json.dumps(entry), flush=True)
        with (a.native_output.parent/'operator_recovery.jsonl').open('a') as f:
            f.write(json.dumps(entry)+'\n')
    mc = RosMc()
    try:
        recover(mc, status, emit=emit)
    except Exception as error:
        emit(dict(status='RECOVERY_NOT_CONFIRMED', error=repr(error)))
        raise
    finally:
        mc.close()


if __name__ == '__main__':
    main()
