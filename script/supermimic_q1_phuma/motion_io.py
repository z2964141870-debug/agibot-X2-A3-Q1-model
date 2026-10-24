"""Strict PHUMA G1 decoding and source-level grouping without arbitrary pickle globals."""

import io
import importlib
from pathlib import PurePosixPath
import pickle
import re

import numpy as np


DATA_COLUMNS_23 = tuple(range(19)) + tuple(range(22, 26))
WRIST_COLUMNS = (19, 20, 21, 26, 27, 28)
JOINT_NAMES_23 = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
)


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        prefix = "numpy._core" if int(np.__version__.split(".")[0]) >= 2 else "numpy.core"
        multiarray = importlib.import_module(prefix + ".multiarray")
        allowed = {
            (source, kind): getattr(multiarray, kind)
            for source in ("numpy.core.multiarray", "numpy._core.multiarray")
            for kind in ("_reconstruct", "scalar")
        }
        allowed.update({
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        })
        if (module, name) not in allowed:
            raise ValueError(f"Unsupported PHUMA pickle global: {module}.{name}")
        return allowed[(module, name)]


def decode_g1(raw):
    if not 128 <= len(raw) <= 4 * 1024 ** 2:
        raise ValueError("Unexpected PHUMA chunk size")
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise ValueError("Unreviewed NPY format version")
    shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    if shape != () or fortran or dtype != np.dtype("O"):
        raise ValueError("PHUMA requires a scalar object container")
    outer = NumpyOnlyUnpickler(stream).load()
    if not isinstance(outer, np.ndarray) or outer.shape != () or stream.read(1):
        raise ValueError("Unexpected object container or trailing data")
    motion = outer.item()
    if not isinstance(motion, dict) or set(motion) != {"root_trans", "root_ori", "dof_pos", "fps"}:
        raise ValueError("Unexpected PHUMA fields")
    if not np.isscalar(motion["fps"]) or float(motion["fps"]) != 30.0:
        raise ValueError("Pinned PHUMA source must explicitly contain 30 Hz")
    arrays = {}
    frames = None
    for key, width in (("root_trans", 3), ("root_ori", 4), ("dof_pos", 29)):
        value = motion[key]
        if (not isinstance(value, np.ndarray) or value.dtype.kind != "f"
                or value.ndim != 2 or value.shape[1] != width or not np.isfinite(value).all()):
            raise ValueError(f"Invalid {key} matrix")
        if frames is not None and len(value) != frames:
            raise ValueError("Different source frame counts")
        frames = len(value)
        arrays[key] = np.asarray(value, dtype=np.float64)
    if frames < 2 or frames > 1000:
        raise ValueError("Unexpected PHUMA chunk length")
    if np.max(np.abs(np.linalg.norm(arrays["root_ori"], axis=1) - 1)) > 1e-5:
        raise ValueError("Source xyzw quaternions must be unit length")
    if np.max(np.abs(arrays["dof_pos"][:, WRIST_COLUMNS])) > 1e-12:
        raise ValueError("Expected the six officially padded wrist columns to be zero")
    return {**arrays, "fps": 30.0,
            "dof_pos_23": arrays["dof_pos"][:, DATA_COLUMNS_23]}


def member_key(name):
    path = PurePosixPath(name)
    if (path.parts[:2] != ("data", "g1") or path.suffix != ".npy"
            or ".." in path.parts or "\\" in name):
        raise ValueError("Expected a scoped PHUMA G1 member")
    return str(PurePosixPath(*path.parts[2:]).with_suffix(""))


def source_group(key):
    path = PurePosixPath(key)
    stem = re.sub(r"_chunk_\d+$", "", path.name)
    stem = re.sub(r"_clip_?\d+$", "", stem)
    return str(path.parent / stem)


def describe(motion):
    root, joints = motion["root_trans"], motion["dof_pos_23"]
    speed = np.abs(np.diff(joints, axis=0) * motion["fps"])
    tilt_cos = 1 - 2 * (motion["root_ori"][:, :2] ** 2).sum(axis=1)
    return {
        "source_frames": len(root), "source_hz": motion["fps"],
        "source_duration_s": (len(root) - 1) / motion["fps"],
        "max_horizontal_displacement_m": float(np.linalg.norm(root[:, :2] - root[:1, :2], axis=1).max()),
        "horizontal_path_m": float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum()),
        "root_height_min_m": float(root[:, 2].min()),
        "max_tilt_rad": float(np.arccos(np.clip(tilt_cos, -1, 1)).max()),
        "source_joint_speed_p95_rad_s": float(np.quantile(speed, .95)),
        "source_joint_speed_max_rad_s": float(speed.max()),
    }
