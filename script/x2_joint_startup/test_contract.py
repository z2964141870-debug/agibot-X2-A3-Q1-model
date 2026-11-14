import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

from .native_exec import validate
from .run import decode_ready, read_ready, select_worker


class StartupContract(unittest.TestCase):
    def test_native_and_python_diagnostics_do_not_enter_control_pipe(self):
        code = """
import ctypes,json,os,sys
from script.x2_joint_startup.worker import diagnostic_channel
with diagnostic_channel():
    os.write(1,b'[11/14] native diagnostic\\n')
    libc=ctypes.CDLL(None)
    libc.printf(b'buffered native diagnostic\\n')
    print('python diagnostic')
    print(json.dumps(dict(status='FGP_READY')),file=sys.__stdout__,flush=True)
"""
        p=subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(p.stdout),dict(status="FGP_READY"))
        for text in ("native diagnostic","buffered native diagnostic","python diagnostic"):
            self.assertIn(text,p.stderr)

    def test_policy_python_protocol_and_exception_restore(self):
        code = """
import os
from script.x2_joint_startup.worker import diagnostic_channel
try:
    with diagnostic_channel(python_protocol=True):
        os.write(1,b'native diagnostic\\n')
        print('policy_json')
        raise ValueError('expected')
except ValueError:
    print('restored')
"""
        p=subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True,check=True)
        self.assertEqual(p.stdout.splitlines(),["policy_json","restored"])
        self.assertEqual(p.stderr,"native diagnostic\n")

    def test_only_registered_workers_and_observe_binary(self):
        c=["python","-m","script.x2_joint_alloc.worker","fgp"]
        self.assertEqual(select_worker(c,Path("out"))[2],"script.x2_joint_startup.worker")
        with self.assertRaises(ValueError):select_worker(c[:-1]+["armed"],Path("out"))
        c=["taskset","--cpu-list","0,1","/x/x2_morph_native","--mode","observe"]
        fixed=select_worker(c,Path("out"))
        self.assertEqual(fixed[:3],c[:3])
        self.assertEqual(fixed[-3:],c[-3:])
        with self.assertRaises(ValueError):select_worker(c[:-1]+["armed"],Path("out"))

    def test_native_mask_is_measured_and_arm_rejected(self):
        validate(["/x/x2_morph_native","--mode","observe"],{0,1})
        for command,cpus in ((["/x/x2_morph_native","--mode","armed"],{0,1}),
                             (["/x/x2_morph_native","--mode","observe"],{0,1,2})):
            with self.assertRaises(ValueError):validate(command,cpus)

    def test_malformed_readiness_is_saved_then_rejected(self):
        with subprocess.Popen([sys.executable,"-B","-c","print('[11/14] malformed')"],
                              stdout=subprocess.PIPE,text=True) as p:
            tape=io.StringIO()
            with self.assertRaises(json.JSONDecodeError):read_ready(p,tape)
            self.assertEqual(json.loads(tape.getvalue())["raw"],"[11/14] malformed\n")
        with self.assertRaises(ValueError):decode_ready("[]")


if __name__ == "__main__":
    unittest.main()
