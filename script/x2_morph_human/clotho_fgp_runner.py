"""Headless Latest-FGP (CLOTHO-mocap) 11-IMU -> SMPL24 [24,3,3] + metric root runner.

Runs Stage-1 FlowDiT IMUCanonicalization (`U100hL7.07h_residual.pth`) + either:
  - Stage-2A GlobalPose (`Full_GR_OV_10(..., enable_physics=False)` via `forward_frame_fast0`), or
  - Stage-2B LIP (`BiPoser` 2-stage causal `EasyLSTM` + SMPL foot-lock `stabilize_root_translation`).

Deliberately bypasses Unity TCP:8888 / UDP:23338 sockets, Windows-only `carticulate.pyd`,
and Unity display hacks (`tran[2]*=0.7; tran[1]+=1` and neck/head zeroing).
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import types

import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_clotho_environment(clotho_root: Path):
    """Register Python 3.11 chumpy/scipy shims and import CLOTHO modules safely."""
    clotho_root = clotho_root.resolve()
    if not clotho_root.is_dir():
        raise FileNotFoundError(f"CLOTHO-mocap root not found: {clotho_root}")

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": np.object_,
        "unicode": str,
        "str": np.str_,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    class _ChStub:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
                self.__dict__.update(state[1])

        def __array__(self, dtype=None):
            for key in ("x", "r", "a"):
                if key in self.__dict__:
                    return np.asarray(self.__dict__[key], dtype=dtype)
            raise AttributeError(f"Unsupported chumpy stub keys: {list(self.__dict__.keys())}")

    for mod_name in ("chumpy", "chumpy.ch", "chumpy.reordering"):
        mod = types.ModuleType(mod_name)
        mod.Ch = _ChStub
        mod.Select = _ChStub
        sys.modules.setdefault(mod_name, mod)

    # Inference-only stubs so articulate.utils.torch.train does not require tensorboard/thop.
    tb_mod = types.ModuleType("torch.utils.tensorboard")
    tb_mod.SummaryWriter = object
    sys.modules.setdefault("torch.utils.tensorboard", tb_mod)
    thop_mod = types.ModuleType("thop")
    thop_mod.clever_format = lambda *a, **kw: a
    sys.modules.setdefault("thop", thop_mod)

    from scipy.spatial.transform import Rotation as _SciRot
    cv2_mod = types.ModuleType("cv2")
    def _rodrigues(src):
        arr = np.asarray(src, dtype=np.float64)
        if arr.shape == (3, 3):
            return _SciRot.from_matrix(arr).as_rotvec().reshape(3, 1), None
        return _SciRot.from_rotvec(arr.reshape(3)).as_matrix(), None
    cv2_mod.Rodrigues = _rodrigues
    sys.modules.setdefault("cv2", cv2_mod)

    import scipy.sparse
    import scipy.sparse.csc

    if not hasattr(scipy.sparse.csc, "csc_matrix"):
        scipy.sparse.csc.csc_matrix = scipy.sparse.csc_matrix

    root_str = str(clotho_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Avoid PoseReconstruction/GlobalPose_nets/__init__.py importing Windows-only carticulate via full_G_OV.
    pkg_dir = clotho_root / "PoseReconstruction" / "GlobalPose_nets"
    if "PoseReconstruction" not in sys.modules:
        pr_pkg = types.ModuleType("PoseReconstruction")
        pr_pkg.__path__ = [str(clotho_root / "PoseReconstruction")]
        sys.modules["PoseReconstruction"] = pr_pkg
    if "PoseReconstruction.GlobalPose_nets" not in sys.modules:
        gp_pkg = types.ModuleType("PoseReconstruction.GlobalPose_nets")
        gp_pkg.__path__ = [str(pkg_dir)]
        sys.modules["PoseReconstruction.GlobalPose_nets"] = gp_pkg


class ClothoCalibrator:
    """Socket-free exact replica of DataProcessServer_FullBody T-pose calibration (10 active IMUs)."""

    def __init__(self, torch_mod, art_mod, bulid_rot_fn, t_pose_ori):
        self.torch = torch_mod
        self.art = art_mod
        self.imu_num = 10
        self.root_2_smpl = bulid_rot_fn(theta=180, rotation_axis=[0, 1, 0]).squeeze(0).float()
        self.cali_pose_ori = t_pose_ori[[18, 16, 19, 17, 12, 4, 5, 1, 2, 0]].float()
        self.acc_offset = torch_mod.tensor([0.0, 0.0, 9.8], dtype=torch_mod.float32).view(1, 3, 1)
        self.smpl2imu = None
        self.device2bone = None

    def set_calibration_120(self, cal_120) -> None:
        t = self.torch.as_tensor(cal_120, dtype=self.torch.float32).reshape(-1)
        if t.numel() != 120:
            raise ValueError(f"Expected 120-float calibration_data (10*3 + 10*9), got {t.numel()}")
        init_oris = t[30:].view(self.imu_num, 3, 3)
        imu2smpl = init_oris[-1].matmul(self.root_2_smpl).view(3, 3)
        col_y = self.torch.tensor([0.0, 0.0, 1.0], dtype=self.torch.float32)
        z_raw = imu2smpl[:, 2]
        col_z = z_raw - self.torch.dot(col_y, z_raw) * col_y
        col_z = col_z / (self.torch.linalg.norm(col_z) + 1e-8)
        col_x = self.torch.linalg.cross(col_y, col_z)
        imu2smpl_ortho = self.torch.stack((col_x, col_y, col_z), dim=-1)
        self.smpl2imu = imu2smpl_ortho.t().contiguous()
        self.device2bone = (
            self.smpl2imu.matmul(init_oris).transpose(1, 2).matmul(self.cali_pose_ori).contiguous()
        )

    def calibrate_ordered_10x7(self, ordered_10x7: np.ndarray):
        """Input: [10, 7] float32 [acc_x, acc_y, acc_z, qw, qx, qy, qz].
        Returns (acc_smpl_m_s2 [10,3], rot_smpl_bone [10,3,3]) matching UDP.py + online_fgp*30.
        """
        if self.smpl2imu is None or self.device2bone is None:
            raise RuntimeError("Calibrator not initialized")
        arr = self.torch.as_tensor(ordered_10x7, dtype=self.torch.float32)
        if arr.shape != (10, 7):
            raise ValueError(f"Expected (10, 7) ordered IMU array, got {tuple(arr.shape)}")
        accs = arr[:, :3]
        quats = arr[:, 3:7]
        oris = self.art.math.quaternion_to_rotation_matrix(quats).view(10, 3, 3)
        accs_global = oris.matmul(accs.unsqueeze(-1)) - self.acc_offset
        rot_cal = self.smpl2imu.matmul(oris).matmul(self.device2bone)
        acc_cal = self.smpl2imu.matmul(accs_global).squeeze(-1)
        return acc_cal, rot_cal


def so3_max_error(mats: np.ndarray) -> float:
    eye = np.eye(3, dtype=np.float64)
    m = np.asarray(mats, dtype=np.float64).reshape(-1, 3, 3)
    ortho = np.max(np.abs(np.matmul(np.swapaxes(m, -1, -2), m) - eye))
    det = np.max(np.abs(np.linalg.det(m) - 1.0))
    return float(max(ortho, det))


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle in degrees between two [..., 3, 3] rotation arrays."""
    rel = np.matmul(np.swapaxes(a, -1, -2), b)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_theta))


def run_worn_recording(
    *,
    clotho_root: Path,
    recording_dir: Path,
    backend: str,
    start_frame: int,
    frame_count: int,
    warmup_frames: int,
    output_npz: Path,
    fps_nominal: float = 30.0,
) -> dict:
    prepare_clotho_environment(clotho_root)
    import torch
    import articulate as art
    from Aplus.models import EasyLSTM
    from Aplus.tools.smpl_light import SMPLPose
    from PoseReconstruction.GlobalPose_nets.full_GR_OV import Full_GR_OV_10
    from PoseReconstruction.LIP_model import BiPoser
    from Utils.config import bulid_rot
    from neural_backbone import IMUCanonicalization

    torch.set_num_threads(4)
    device = torch.device("cpu")

    denoise_ckpt = clotho_root / "checkpoints" / "canonicalization" / "U100hL7.07h_residual.pth"
    smpl_pkl = clotho_root / "models" / "SMPL_male.pkl"
    gp_ckpt = clotho_root / "checkpoints" / "globalpose" / "weights.pt"
    lip_ckpt = clotho_root / "checkpoints" / "LIP" / "LIP_10_real_denoised_7.07_10imu_10.pth"

    cal_doc = json.loads((recording_dir / "tpose_calibration.json").read_text())
    cal_120 = cal_doc["data"]["calibration_data"]
    calibrator = ClothoCalibrator(torch, art, bulid_rot, SMPLPose.t_pose_ori)
    calibrator.set_calibration_120(cal_120)

    flow_model = IMUCanonicalization(
        pretrained_ae=None,
        cin_state=9,
        cin_cond=9,
        cout=9,
        depth=6,
        nhead=4,
        ffn=512,
        dropout=0.1,
        max_T=512,
        t_embed_dim=256,
        enc_d_model=384,
        enc_layers=4,
        enc_nhead=8,
    ).to(device)
    flow_state = torch.load(denoise_ckpt, map_location="cpu", weights_only=False)
    flow_state = flow_state["model"] if isinstance(flow_state, dict) and "model" in flow_state else flow_state
    flow_model.load_state_dict(flow_state, strict=True)
    flow_model.eval()

    window_size = 30
    ema_alpha = 0.08
    acc_scale = 30.0
    acc_buf: deque = deque(maxlen=window_size)
    rot_buf: deque = deque(maxlen=window_size)
    x_noisy = torch.zeros(1, window_size, 10, 9, device=device, dtype=torch.float32)
    mem_pad_mask = torch.zeros(1, window_size * 10, dtype=torch.bool, device=device)
    vis_ids_full = torch.arange(10, dtype=torch.long, device=device).unsqueeze(0)
    prev_out_acc = None
    prev_out_rot = None

    @torch.inference_mode()
    def denoise_step(acc_in: torch.Tensor, rot_in: torch.Tensor):
        nonlocal prev_out_acc, prev_out_rot
        acc_c = acc_in.detach().to(device=device, dtype=torch.float32)
        rot_c = art.math.normalize_rotation_matrix(rot_in.detach().to(device=device, dtype=torch.float32))
        acc_buf.append(acc_c)
        rot_buf.append(rot_c)
        if len(acc_buf) < window_size:
            return acc_c, rot_c, False
        acc_seq = torch.stack(list(acc_buf), dim=0)
        rot_seq = torch.stack(list(rot_buf), dim=0)
        r6d = art.math.rotation_matrix_to_r6d(rot_seq.reshape(-1, 3, 3)).reshape(window_size, 10, 6)
        x_noisy[0, :, :, :3].copy_(acc_seq / acc_scale)
        x_noisy[0, :, :, 3:9].copy_(r6d)
        v = flow_model(x_t_9=x_noisy, x_n_9=x_noisy, vis_ids=vis_ids_full, mem_pad_mask=mem_pad_mask)
        clean_last = (x_noisy + v)[0, -1]
        den_acc = clean_last[:, :3] * acc_scale
        den_rot = art.math.r6d_to_rotation_matrix(clean_last[:, 3:9].reshape(-1, 6)).reshape(10, 3, 3)
        if prev_out_acc is None or prev_out_rot is None:
            prev_out_acc, prev_out_rot = den_acc, den_rot
        else:
            prev_out_acc = ema_alpha * prev_out_acc + (1.0 - ema_alpha) * den_acc
            prev_out_rot = art.math.normalize_rotation_matrix(
                ema_alpha * prev_out_rot + (1.0 - ema_alpha) * den_rot
            )
        return prev_out_acc, prev_out_rot, True

    smpl_body = art.ParametricModel(str(smpl_pkl), device=device)
    gp_net = None
    lip_net = None
    h_s1 = c_s1 = h_s2 = c_s2 = None
    R_prev_cache = None
    foot_lock_state = {"prev_global_feet": None, "locked_feet": None, "initialized": False}

    if backend == "globalpose":
        gp_net = Full_GR_OV_10(
            weights_path=str(gp_ckpt),
            body_model_path=str(smpl_pkl),
            enable_physics=False,
        ).to(device)
        gp_net.eval()
    elif backend == "lip":
        model_s1 = EasyLSTM(
            n_input=10 * 9,
            n_hidden=256,
            n_output=24 * 3,
            n_lstm_layer=2,
            bidirectional=False,
            output_type="seq",
            act_func="tanh",
            dropout=0.2,
        ).to(device)
        model_s2 = EasyLSTM(
            n_input=10 * 9 + 24 * 3,
            n_hidden=256,
            n_output=24 * 6,
            n_lstm_layer=2,
            bidirectional=False,
            output_type="seq",
            act_func="tanh",
            dropout=0.2,
        ).to(device)
        lip_net = BiPoser(net_s1=model_s1, net_s2=model_s2, export_mode=True).to(device)
        lip_state = torch.load(lip_ckpt, map_location="cpu", weights_only=False)
        lip_net.load_state_dict(lip_state["model"], strict=True)
        lip_net.eval()
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    @torch.no_grad()
    def stabilize_root_translation(pose_mat_t: torch.Tensor, tran_t: torch.Tensor) -> torch.Tensor:
        joint_pos = smpl_body.forward_kinematics(
            pose_mat_t.unsqueeze(0), tran=tran_t.unsqueeze(0)
        )[1][0].detach().cpu()
        local_feet = joint_pos[[10, 11]]
        tran_out = tran_t.detach().cpu().clone()
        global_feet = local_feet + tran_out.view(1, 3)
        min_foot_y = float(global_feet[:, 1].min().item())
        floor_target_y = -0.96
        if min_foot_y < floor_target_y:
            delta_y = floor_target_y - min_foot_y
            tran_out[1] += delta_y
            global_feet[:, 1] += delta_y
        if not foot_lock_state["initialized"]:
            foot_lock_state["prev_global_feet"] = global_feet.clone()
            foot_lock_state["locked_feet"] = global_feet.clone()
            foot_lock_state["initialized"] = True
            return tran_out
        prev_feet = foot_lock_state["prev_global_feet"]
        locked_feet = foot_lock_state["locked_feet"]
        foot_speed = torch.norm(global_feet - prev_feet, dim=1)
        foot_height = global_feet[:, 1] - floor_target_y
        contact_mask = (foot_height < 0.06) & (foot_speed < 0.045)
        if contact_mask.any():
            drift = global_feet[contact_mask][:, [0, 2]] - locked_feet[contact_mask][:, [0, 2]]
            tran_out[[0, 2]] -= 0.75 * drift.mean(dim=0)
            global_feet = local_feet + tran_out.view(1, 3)
        locked_feet[contact_mask] = global_feet[contact_mask]
        locked_feet[~contact_mask] = global_feet[~contact_mask]
        foot_lock_state["prev_global_feet"] = global_feet.clone()
        foot_lock_state["locked_feet"] = locked_feet
        return tran_out

    read_start = max(0, start_frame - warmup_frames)
    read_end = start_frame + frame_count
    raw_rows = []
    with (recording_dir / "frames.jsonl").open() as f:
        for idx, line in enumerate(f):
            if idx < read_start:
                continue
            if idx >= read_end:
                break
            raw_rows.append((idx, json.loads(line)))

    if len(raw_rows) < (start_frame - read_start) + frame_count:
        raise ValueError(f"Insufficient frames in {recording_dir}: got {len(raw_rows)}")

    poses_out = []
    trans_out = []
    unity_hack_trans_out = []
    legacy_poses_out = []
    legacy_trans_out = []
    capture_ns_out = []
    frame_indices_out = []
    denoise_ms_out = []
    pose_ms_out = []
    total_ms_out = []

    dt_gyro = 1.0 / fps_nominal
    for idx, row in raw_rows:
        t_frame0 = time.perf_counter()
        imu = row["raw_imu"]
        ordered_10x7 = np.concatenate(
            [
                np.asarray(imu["upper_ordered_6x7"], dtype=np.float32),
                np.asarray(imu["lower_ordered_5x7"], dtype=np.float32),
            ],
            axis=0,
        )
        acc_cal, rot_cal = calibrator.calibrate_ordered_10x7(ordered_10x7)
        t_den0 = time.perf_counter()
        acc_den, rot_den, window_ready = denoise_step(acc_cal, rot_cal)
        t_den1 = time.perf_counter()
        if not window_ready:
            continue

        t_pose0 = time.perf_counter()
        with torch.inference_mode():
            if backend == "globalpose":
                RMB = rot_den.view(10, 3, 3)
                aM = RMB.matmul(acc_den.view(10, 3, 1)).squeeze(-1).clamp(-30.0, 30.0)
                if R_prev_cache is None:
                    w_raw = torch.zeros_like(aM)
                else:
                    R_delta = R_prev_cache.transpose(-1, -2).matmul(RMB)
                    from scipy.spatial.transform import Rotation as _Rot
                    aa = torch.from_numpy(
                        _Rot.from_matrix(R_delta.detach().cpu().numpy().reshape(-1, 3, 3)).as_rotvec()
                    ).to(device=R_delta.device, dtype=R_delta.dtype)
                    aa = torch.where(aa.abs() < 1e-4, torch.zeros_like(aa), aa)
                    w_raw = torch.nan_to_num(aa / dt_gyro, nan=0.0, posinf=0.0, neginf=0.0).clamp(-60.0, 60.0)
                wM = RMB.matmul(w_raw.unsqueeze(-1)).squeeze(-1)
                R_prev_cache = RMB.clone()
                pose_mat, tran = gp_net.forward_frame_fast0(aM, wM, RMB)
                pose_mat = pose_mat.detach().cpu()
                tran = tran.detach().cpu()
            else:
                acc_clamped = acc_den.float().clamp(-60.0, 60.0)
                rot_norm = art.math.normalize_rotation_matrix(rot_den.float())
                acc_rooted = torch.cat([acc_clamped[:9] - acc_clamped[9:], acc_clamped[9:]], dim=0)
                acc_rooted = acc_rooted @ rot_norm[9]
                rot_rooted = torch.cat(
                    [torch.matmul(rot_norm[9].transpose(0, 1), rot_norm[:9]), rot_norm[9:]],
                    dim=0,
                )
                x_lip = torch.cat(
                    [(acc_rooted / 30.0).reshape(-1), art.math.rotation_matrix_to_r6d(rot_rooted).reshape(-1)],
                    dim=-1,
                ).view(1, 1, 90)
                if h_s1 is None:
                    h0_1 = lip_net.net_s1.h_0.repeat(1, 1, 1)
                    c0_1 = lip_net.net_s1.c_0.repeat(1, 1, 1)
                    out_1, h_s1, c_s1 = lip_net.net_s1(x_lip, h0_1, c0_1)
                    h0_2 = lip_net.net_s2.h_0.repeat(1, 1, 1)
                    c0_2 = lip_net.net_s2.c_0.repeat(1, 1, 1)
                    out_2, h_s2, c_s2 = lip_net.net_s2(torch.cat([x_lip, out_1], dim=-1), h0_2, c0_2)
                else:
                    out_1, h_s1, c_s1 = lip_net.net_s1(x_lip, h_s1, c_s1)
                    out_2, h_s2, c_s2 = lip_net.net_s2(torch.cat([x_lip, out_1], dim=-1), h_s2, c_s2)
                pred_r6d = out_2.squeeze(0).squeeze(0).view(24, 6)
                pose_mat = art.math.r6d_to_rotation_matrix(pred_r6d).detach().cpu()
                tran = stabilize_root_translation(pose_mat, torch.zeros(3))
        t_pose1 = time.perf_counter()

        if idx < start_frame:
            continue

        unity_tran = tran.clone()
        unity_tran[2] *= 0.7
        unity_tran[1] += 1.0

        poses_out.append(pose_mat.numpy().astype(np.float64))
        trans_out.append(tran.numpy().astype(np.float64))
        unity_hack_trans_out.append(unity_tran.numpy().astype(np.float64))
        legacy_poses_out.append(np.asarray(row["fgp_pose_matrix"], dtype=np.float64).reshape(24, 3, 3))
        legacy_trans_out.append(np.asarray(row["translation_m"], dtype=np.float64).reshape(3))
        capture_ns_out.append(int(round(float(row["monotonic_s"]) * 1e9)))
        frame_indices_out.append(int(row["frame_index"]))
        denoise_ms_out.append((t_den1 - t_den0) * 1000.0)
        pose_ms_out.append((t_pose1 - t_pose0) * 1000.0)
        total_ms_out.append((t_pose1 - t_frame0) * 1000.0)

    pose_arr = np.stack(poses_out, axis=0)
    tran_arr = np.stack(trans_out, axis=0)
    unity_tran_arr = np.stack(unity_hack_trans_out, axis=0)
    legacy_pose_arr = np.stack(legacy_poses_out, axis=0)
    legacy_tran_arr = np.stack(legacy_trans_out, axis=0)
    capture_ns_arr = np.asarray(capture_ns_out, dtype=np.int64)
    for k in range(1, len(capture_ns_arr)):
        if capture_ns_arr[k] <= capture_ns_arr[k - 1]:
            capture_ns_arr[k] = capture_ns_arr[k - 1] + 1_000
    den_arr = np.asarray(denoise_ms_out, dtype=np.float64)
    pos_arr = np.asarray(pose_ms_out, dtype=np.float64)
    tot_arr = np.asarray(total_ms_out, dtype=np.float64)

    diff_deg = geodesic_deg(pose_arr, legacy_pose_arr)
    active_joint_mask = np.ones(24, dtype=bool)
    active_joint_mask[[10, 11, 20, 21, 22, 23]] = False

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    capture_clock_id = f"worn_ble_{recording_dir.name}_monotonic"
    np.savez_compressed(
        output_npz,
        pose_mat=pose_arr,
        root_translation_m=tran_arr,
        unity_hack_translation_m=unity_tran_arr,
        legacy_pose_mat=legacy_pose_arr,
        legacy_translation_m=legacy_tran_arr,
        capture_ns=capture_ns_arr,
        capture_clock=np.array(capture_clock_id),
        frame_index=np.asarray(frame_indices_out, dtype=np.int64),
        denoise_ms=den_arr,
        pose_ms=pos_arr,
        total_fgp_ms=tot_arr,
    )

    dt_ms = np.diff(capture_ns_arr) / 1e6
    steady = slice(2, None) if len(tot_arr) > 3 else slice(None)
    summary = {
        "status": "LATEST_FGP_HEADLESS_EXPORT_OK",
        "recording": recording_dir.name,
        "backend": backend,
        "start_frame": start_frame,
        "warmup_frames": warmup_frames,
        "exported_frames": int(len(pose_arr)),
        "duration_s": float((capture_ns_arr[-1] - capture_ns_arr[0]) / 1e9),
        "source_dt_ms": {
            "median": float(np.median(dt_ms)),
            "p95": float(np.percentile(dt_ms, 95)),
            "max": float(np.max(dt_ms)),
        },
        "checkpoints": {
            "canonicalizer": {
                "path": str(denoise_ckpt.relative_to(clotho_root)),
                "bytes": denoise_ckpt.stat().st_size,
                "sha256": _sha256(denoise_ckpt),
            },
            "stage2": {
                "path": str((gp_ckpt if backend == "globalpose" else lip_ckpt).relative_to(clotho_root)),
                "bytes": (gp_ckpt if backend == "globalpose" else lip_ckpt).stat().st_size,
                "sha256": _sha256(gp_ckpt if backend == "globalpose" else lip_ckpt),
            },
            "smpl_model": {
                "path": str(smpl_pkl.relative_to(clotho_root)),
                "bytes": smpl_pkl.stat().st_size,
                "sha256": _sha256(smpl_pkl),
            },
        },
        "timing_ms_steady": {
            "denoise_median": float(np.median(den_arr[steady])),
            "denoise_p95": float(np.percentile(den_arr[steady], 95)),
            "pose_median": float(np.median(pos_arr[steady])),
            "pose_p95": float(np.percentile(pos_arr[steady], 95)),
            "total_median": float(np.median(tot_arr[steady])),
            "total_p95": float(np.percentile(tot_arr[steady], 95)),
            "total_max_all": float(np.max(tot_arr)),
        },
        "so3_max_error": so3_max_error(pose_arr),
        "root_translation_span_smpl_yup_m": (np.ptp(tran_arr, axis=0)).tolist(),
        "root_translation_first_smpl_yup_m": tran_arr[0].tolist(),
        "legacy_translation_span_m": (np.ptp(legacy_tran_arr, axis=0)).tolist(),
        "vs_legacy_fgp_geodesic_deg": {
            "all24_mean": float(np.mean(diff_deg)),
            "all24_p95": float(np.percentile(diff_deg, 95)),
            "active18_mean": float(np.mean(diff_deg[:, active_joint_mask])),
            "pelvis_mean": float(np.mean(diff_deg[:, 0])),
            "shoulders_elbows_mean": float(np.mean(diff_deg[:, [16, 17, 18, 19]])),
            "hips_knees_ankles_mean": float(np.mean(diff_deg[:, [1, 2, 4, 5, 7, 8]])),
        },
        "output_npz": str(output_npz),
        "output_sha256": _sha256(output_npz),
    }
    summary_path = output_npz.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clotho-root", type=Path, required=True)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("globalpose", "lip"), default="globalpose")
    parser.add_argument("--start-frame", type=int, default=4100)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--warmup-frames", type=int, default=29)
    parser.add_argument("--output-npz", type=Path, required=True)
    args = parser.parse_args()
    summary = run_worn_recording(
        clotho_root=args.clotho_root,
        recording_dir=args.recording_dir,
        backend=args.backend,
        start_frame=args.start_frame,
        frame_count=args.frames,
        warmup_frames=args.warmup_frames,
        output_npz=args.output_npz,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
