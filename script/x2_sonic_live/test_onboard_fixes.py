#!/usr/bin/env python3
"""Offline verification suite for the three X2 live handoff onboard fixes.

Runs without BLE hardware, ONNX files, or live robot topics (Mac & X2 safe):
  1. garment_intent_adapter: SMPLXConverter Y-up -> Z-up mapping & canonical passthrough.
  2. x2_smpl_token_service.sync_and_tick: monotonic clock progression & stream-stop disengage.
  3. garment_udp_v2_reconnect_safe: syntax check, signal/atexit wiring check, and
     isolated execution of shutdown_garment_session / _disconnect_ble_fallback.
"""
from __future__ import annotations

import ast
import logging
import py_compile
import sys
import unittest
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PATCHED_ROOT = HERE / "onboard_patched"
X2_SW = PATCHED_ROOT / "x2_smpl_wholebody"
ORIG_X2_SW = (
    REPO_ROOT
    / "data"
    / "analysis"
    / "X2_SONIC_LIVE_HANDOFF_20261116"
    / "onboard_originals"
    / "x2_smpl_wholebody"
)
LIVE_X2_SW = Path("/home/agi/projects/project_tools/x2_smpl_wholebody")
SMARTWEAR = PATCHED_ROOT / "smartwear_v2"

for candidate in (X2_SW, ORIG_X2_SW, LIVE_X2_SW):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.append(str(candidate))
# Ensure patched takes precedence over originals or live copy
if str(X2_SW) in sys.path:
    sys.path.remove(str(X2_SW))
    sys.path.insert(0, str(X2_SW))

import garment_intent_adapter as gia  # noqa: E402
from mock_pico_intent_sender import frame as mock_smpl_frame  # noqa: E402
from token_service_core import CONTROL_DT, INTENT_FRESH_S, NUM_X2, TokenServiceCore  # noqa: E402
from x2_smpl_token_service import sync_and_tick  # noqa: E402

_SMPL_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_hand", "right_hand",
]


def _make_human_dict(joints24x3: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ident_q = np.array([1.0, 0.0, 0.0, 0.0], np.float64)
    return {name: (joints24x3[i].astype(np.float64), ident_q) for i, name in enumerate(_SMPL_NAMES)}


class TestGarmentIntentAdapterFix(unittest.TestCase):
    def test_smpl_fk_yup_converts_to_zup_upright(self) -> None:
        # SMPLXConverter fast FK emits Y-up (+Z forward, +X left, +Y up)
        zup_j, _q, _w = mock_smpl_frame("stand", 0.0, np.random.default_rng(0))
        zup_local = zup_j - zup_j[0]
        # Rotate Z-up -> Y-up by applying _SMPL_TO_ZUP.T (x=fwd->z, y=left->x, z=up->y)
        yup_j = zup_j @ gia._SMPL_TO_ZUP
        yup_j[:, 1] += 1.0  # elevate pelvis to y=+1.0
        human_yup = _make_human_dict(yup_j)
        pos, quats = gia.joints_from_human_data(human_yup)
        joints, _quat = gia.root_local_joints(pos, quats[0], source_frame="smpl_fk")
        self.assertEqual(joints.shape, (24, 3))
        # Z-up: Head > Hips > Knees
        self.assertGreater(joints[15, 2], joints[1, 2])
        self.assertGreater(joints[1, 2], joints[4, 2])
        self.assertGreater(joints[2, 2], joints[5, 2])
        self.assertTrue(np.allclose(joints, zup_local, atol=1e-4))

    def test_canonical_passthrough_preserved(self) -> None:
        zup_j, _q, _w = mock_smpl_frame("stand", 0.0, np.random.default_rng(0))
        zup_local = zup_j - zup_j[0]
        human_zup = _make_human_dict(zup_j + np.array([0.0, 0.0, 1.0], np.float32))
        pos, quats = gia.joints_from_human_data(human_zup)
        joints, _ = gia.root_local_joints(pos, quats[0], source_frame="canonical")
        self.assertTrue(np.allclose(joints, zup_local, atol=1e-4))


class TestTokenServiceClockFix(unittest.TestCase):
    def test_sync_and_tick_advances_window_and_drops_on_stop(self) -> None:
        seen_obs: list[np.ndarray] = []

        def fake_tok(obs: np.ndarray) -> np.ndarray:
            seen_obs.append(obs.copy())
            return np.full((64,), 0.25, dtype=np.float32)

        t0 = 4200.0
        core = TokenServiceCore(tokenizer_run=fake_tok, ori_mode="heading", now=t0)
        rng = np.random.default_rng(123)
        base_q = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
        body_q = np.zeros(NUM_X2, np.float32)

        for i in range(120):
            t = t0 + i * CONTROL_DT
            j, _q, _w = mock_smpl_frame("stand", float(i) * CONTROL_DT, rng)
            j[20, 2] += 0.12 * np.sin(i * 0.25)
            j[21, 2] += 0.12 * np.cos(i * 0.25)
            core.feed_intent(t, j, base_q, 1.0)
            core.feed_debug(t + 0.001, base_q, body_q)
            sync_and_tick(core, t)

        self.assertGreaterEqual(len(seen_obs), 20)
        self.assertGreater(float(np.max(np.abs(seen_obs[-1] - seen_obs[0]))), 1e-3)

        t_last = t0 + 119 * CONTROL_DT
        stopped = False
        for k in range(1, 50):
            t = t_last + k * CONTROL_DT
            core.feed_debug(t, base_q, body_q)
            res = sync_and_tick(core, t)
            if not res.published and not res.intent_fresh:
                self.assertLessEqual(t - t_last, INTENT_FRESH_S + 2 * CONTROL_DT)
                stopped = True
                break
        self.assertTrue(stopped, "token service did not release within INTENT_FRESH_S")


class TestGarmentBleCleanupFix(unittest.TestCase):
    def test_syntax_wiring_and_shutdown_cleanup(self) -> None:
        garment_py = SMARTWEAR / "garment_udp_v2_reconnect_safe.py"
        py_compile.compile(str(garment_py), doraise=True)

        source = garment_py.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(garment_py))

        fn_nodes = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_disconnect_ble_fallback", "shutdown_garment_session"}
        }
        self.assertEqual(
            set(fn_nodes),
            {"_disconnect_ble_fallback", "shutdown_garment_session"},
        )
        self.assertIn("signal.signal(signal.SIGINT, _handle_signal)", source)
        self.assertIn("signal.signal(signal.SIGTERM, _handle_signal)", source)
        self.assertIn("atexit.register(_cleanup_once)", source)

        # Execute the exact patched cleanup functions in an isolated mock namespace
        mock_subproc = MagicMock()
        stop_event = Event()
        ns: dict[str, object] = {
            "RUNNING": True,
            "_STOP_EVENT": stop_event,
            "logging": logging,
            "subprocess": mock_subproc,
        }
        mod_ast = ast.Module(
            body=[fn_nodes["_disconnect_ble_fallback"], fn_nodes["shutdown_garment_session"]],
            type_ignores=[],
        )
        ast.fix_missing_locations(mod_ast)
        exec(compile(mod_ast, str(garment_py), "exec"), ns)

        manager = SimpleNamespace(close_all=MagicMock())
        pico_pub = SimpleNamespace(close=MagicMock())
        worker = Thread(target=lambda: stop_event.wait(2.0), daemon=True)
        worker.start()

        addrs = ("F5:C6:03:E6:4B:30", "DA:1B:78:C1:56:1F")
        ns["shutdown_garment_session"](
            manager,
            pico_pub=pico_pub,
            threads=[worker],
            ble_addresses=addrs,
        )

        self.assertFalse(ns["RUNNING"])
        self.assertTrue(stop_event.is_set())
        manager.close_all.assert_called_once()
        pico_pub.close.assert_called_once_with(0)
        self.assertFalse(worker.is_alive())
        called_cmds = [c.args[0] for c in mock_subproc.run.call_args_list]
        self.assertEqual(
            called_cmds,
            [
                ["bluetoothctl", "disconnect", "F5:C6:03:E6:4B:30"],
                ["bluetoothctl", "disconnect", "DA:1B:78:C1:56:1F"],
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
