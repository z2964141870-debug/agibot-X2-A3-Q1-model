import multiprocessing as mp
import tempfile
from pathlib import Path
import unittest

import numpy as np

from .cpu import CpuClient, decode
from .run import validate


def finish_on_close(connection):
    if connection.recv() != "close":
        raise RuntimeError("Unexpected close packet")


class Contracts(unittest.TestCase):
    def packet(self):
        return dict(sequence=1, frame=29, calibrated=[0.]*120,
                    acc=np.zeros((10, 3), dtype=np.float32), rot=np.tile(np.eye(3, dtype=np.float32), (10, 1, 1)))

    def test_order_and_bound(self):
        value = self.packet()
        self.assertEqual(decode(value, (0, -1), 64), (1, 29))
        for previous, limit in (((1, 29), 64), ((0, 30), 64), ((0, -1), 0)):
            with self.assertRaises(ValueError):
                decode(value, previous, limit)

    def test_nonfinite_and_shape(self):
        for field in ("acc", "rot", "calibrated"):
            value = self.packet()
            value[field] = np.zeros((1,), dtype=np.float32)
            with self.assertRaises(ValueError):
                decode(value, (0, -1), 64)
        value = self.packet()
        value["acc"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            decode(value, (0, -1), 64)

    def test_protocol_rejects_extra_data(self):
        value = self.packet()
        value["future"] = 1
        with self.assertRaises(ValueError):
            decode(value, (0, -1), 64)

    def test_child_close_is_reaped(self):
        ctx = mp.get_context("spawn")
        local, remote = ctx.Pipe()
        client = CpuClient.__new__(CpuClient)
        client.closed = False
        client.connection = local
        client.process = ctx.Process(target=finish_on_close, args=(remote,))
        client.process.start()
        remote.close()
        client.close()
        self.assertFalse(client.process.is_alive())
        self.assertEqual(client.process.exitcode, 0)
        client.close()

    def test_old_prefix_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(FileNotFoundError):
                validate(Path(path), "0"*40)


if __name__ == "__main__":
    unittest.main()
