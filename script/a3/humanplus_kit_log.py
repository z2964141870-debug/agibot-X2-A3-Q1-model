"""Bind the native Kit shutdown evidence to one fresh, trial-owned log."""

import ast
import json
from pathlib import Path
import stat

from script.a3.checkpoint_store import sha256
from script.a3.humanplus_training_entry import adapted_tree

SHUTDOWN = ('SimulationApp.close: Closing application',
            'SimulationApp.close: Stage closed',
            'SimulationApp.close: shutting down app and releasing framework')


def log_path(directory):
    directory = Path(directory)
    if not directory.is_absolute() or any(c.isspace() for c in str(directory)):
        raise ValueError('Absolute whitespace-free trial directory required')
    target = directory / 'kit.log'
    if target.is_symlink() or directory.resolve() != directory:
        raise ValueError('Kit log and parent must not be aliases')
    return target


def require_fresh(directory):
    target = log_path(directory)
    if target.exists():
        raise FileExistsError('Native log must not predate this launch')
    return target


def logged_tree(source, filename, directory):
    target = log_path(directory)
    tree = adapted_tree(source, filename)
    count = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and ast.unparse(node.targets[0]) == 'args_cli.kit_args'):
            if (not isinstance(node.value, ast.Constant) or node.value.value !=
                    '--/log/level=error --/log/fileLogLevel=info --/log/outputStreamLevel=error'):
                raise ValueError('Pinned vendor logging shape changed')
            node.value = ast.copy_location(ast.Constant(value=(
                '--/log/level=info --/log/fileLogLevel=info --/log/outputStreamLevel=error '
                '--/log/file=' + str(target))), node.value)
            count += 1
    if count != 1:
        raise ValueError('Exactly one explicit Kit logging assignment required')
    return ast.fix_missing_locations(tree)


def binding(directory, settings, pid):
    target = log_path(directory)
    values = {key: settings.get('/log/' + key) for key in ('file', 'level', 'fileLogLevel')}
    if (values['file'] != str(target)
            or any(str(values[key]).lower() != 'info' for key in ('level', 'fileLogLevel'))):
        raise ValueError('Live Kit settings differ from the registered log contract')
    return dict(path=str(target), native_pid=pid, settings=values)


def verify_log(directory, native_pid):
    target = log_path(directory)
    record = json.loads((Path(directory) / 'kit_binding.json').read_text())
    expected = dict(file=str(target), level='info', fileLogLevel='info')
    if (record['native_pid'] != native_pid or record['path'] != str(target)
            or {k: str(v).lower() if k != 'file' else v for k, v in record['settings'].items()} != expected):
        raise ValueError('Native log runtime identity mismatch')
    if not stat.S_ISREG(target.lstat().st_mode):
        raise ValueError('Native log is not an ordinary file')
    text = target.read_text(errors='replace')
    positions = [text.find(marker) for marker in SHUTDOWN]
    if any(p < 0 for p in positions) or positions != sorted(positions):
        raise ValueError('Missing or unordered native shutdown evidence')
    return dict(kit_log=str(target), kit_log_bytes=target.stat().st_size,
                kit_log_sha256=sha256(target), kit_binding=record)
