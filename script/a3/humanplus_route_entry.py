"""Independent D01 storage and native-close binding; no HP12 trial calls."""

import copy
import json
import os
from pathlib import Path

from script.a3.humanplus_training_entry import validate_training_receipts
from script.a3.humanplus_kit_log import binding, logged_tree
from script.a3.humanplus_head_protocol import digest, transactions
from script.a3.humanplus_replay_protocol import forward_accounting, validate_receipt
from script.a3.humanplus_route_protocol import validate_config, validate_routes


def validate_head_receipts(directory):
    read = lambda name: json.loads((directory / name).read_text())
    head = read('head_contract.json')
    rows = [json.loads(line) for line in (directory / 'finite_candidates.jsonl').read_text().splitlines()]
    ledger = transactions(rows, digest(head['ownership']))
    if (head['owner_sha256'] != digest(head['ownership']) or ledger['rejected'] is not None
            or len(ledger['commits']) != 32 or len(head['steps']) != 32):
        raise ValueError('Complete exact-head transactions required for native finish')
    from script.a3 import humanplus_route_trial as trial
    replay = read('replay_loss_contract.json')
    if replay['design_sha256'] != trial.sha256(trial.DESIGN):
        raise ValueError('Replay design identity changed')
    evidence = validate_receipt(replay, trial.read(trial.DESIGN), 32, rows)
    routes = read('reference_route_coverage.json')['rollouts']
    return dict(**evidence, native_route=validate_routes(routes, 8), owner_sha256=head['owner_sha256'],
                derived_outer_actor_decoder_calls=forward_accounting(8, 32, 32, ledger['evaluations']))


def environment(original, config, device, args_cli):
    validate_config(config)
    from script.a3 import humanplus_route_trial as trial
    from carb.settings import get_settings
    from script.a3.checkpoint_store import atomic_json
    atomic_json(trial.LOG / 'train/kit_binding.json', binding(trial.LOG / 'train', get_settings(), os.getpid()))
    trial.event('kit_log_bound')
    from isaaclab.sim.converters import UrdfConverter
    original_init = UrdfConverter.__init__
    conversions = []
    expected = trial.VENDOR / config.manager_env.config.robot.asset_path

    def initialize(instance, cfg):
        if Path(cfg.asset_path).resolve() != expected.resolve():
            raise ValueError('Unregistered URDF conversion')
        local = copy.deepcopy(cfg)
        before = local.to_dict()
        local.usd_dir = str(trial.OUT / 'usd')
        local.usd_file_name = 'a3_035.usd'
        after = local.to_dict()
        for name in ('usd_dir', 'usd_file_name'):
            before.pop(name, None)
            after.pop(name, None)
        if before != after:
            raise ValueError('USD storage adapter changed physics')
        trial.event('urdf_conversion_begin', asset_path=str(expected), usd_dir=local.usd_dir)
        original_init(instance, local)
        conversions.append(str(instance.usd_path))
        trial.event('urdf_conversion_returned', usd_path=str(instance.usd_path))

    UrdfConverter.__init__ = initialize
    try:
        env = original(config, device, args_cli)
    finally:
        UrdfConverter.__init__ = original_init
    if len(conversions) != 1 or env.num_envs != 16:
        raise ValueError('Exactly one robot conversion and 16 environments required')
    trial.event('training_environment_created', num_envs=16, device=str(device), usd=conversions)
    return env


def finish(app, trainer):
    import torch
    from isaaclab.sim import SimulationContext
    from script.a3 import humanplus_route_trial as trial
    from script.a3.checkpoint_store import atomic_json
    if int(trainer.state.global_step) != 8 or trainer._optimizer_step_ordinal != 32:
        raise ValueError('Training returned before fixed budget')
    receipt = validate_training_receipts(trial.LOG / 'train')
    receipt.update(validate_head_receipts(trial.LOG / 'train'))
    for step in (0, 2, 4, 8):
        path = trial.checkpoint(step)
        record = json.loads(path.with_suffix('.json').read_text())
        if path.stat().st_size != record['bytes']:
            raise ValueError('Missing durable checkpoint')
    torch.cuda.synchronize()
    atomic_json(trial.LOG / 'train/native_training.json', dict(status='TRAINING_COMPLETE',
        global_step=8, cuda_synchronized=True, **receipt))
    trial.event('training_complete', **receipt)
    SimulationContext.clear_instance()
    trial.event('context_clear_returned')
    trial.event('native_close_begin')
    app.close()
    trial.event('native_close_returned')


def execute(path):
    namespace = dict(__name__='__main__', __file__=str(path),
                     humanplus_environment=environment, humanplus_finish=finish)
    from script.a3 import humanplus_route_trial as trial
    exec(compile(logged_tree(path.read_text(), str(path), trial.LOG / 'train'), str(path), 'exec'), namespace)


