"""Stateful latest CLOTHO CPU inference, matching the audited offline runner."""
from collections import deque
import hashlib
from pathlib import Path
import time

import numpy as np


WEIGHTS = (
    ("checkpoints/canonicalization/U100hL7.07h_residual.pth", "b613eb8226b26b9dc82904c22f4aec78773992503337c3cbde674c998f61ce12"),
    ("checkpoints/globalpose/weights.pt", "b63f62b846713cfae1f7f3119633f36d41befff8b9b25003c29fdd946777feef"),
    ("models/SMPL_male.pkl", "0e8c0bbbbc635dcb166ed29c303fb4bef16ea5f623e5a89263495a9e403575bd"),
)


class LatestFgp:
    def __init__(self, root: Path):
        self.model_profile = dict(
            selection="EgoLocate default FGP adapter, resolved with user-supplied CLOTHO assets",
            canonicalizer="IMUCanonicalization B", pose_model="Full_GR_OV_10",
            denoise_enabled=True, denoise_window_frames=30, device="cpu",
            weights=[dict(path=rel, sha256=expected) for rel, expected in WEIGHTS],
            external_egolocate_ble_web_pose_source_available=False,
        )
        for relative, expected in WEIGHTS:
            digest = hashlib.sha256()
            with (root / relative).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"EgoLocate-profile asset mismatch: {relative}")
        from script.x2_morph_human.clotho_fgp_runner import prepare_clotho_environment, ClothoCalibrator
        prepare_clotho_environment(root)
        import torch
        import articulate as art
        from Aplus.tools.smpl_light import SMPLPose
        from Utils.config import bulid_rot
        from neural_backbone import IMUCanonicalization
        from PoseReconstruction.GlobalPose_nets.full_GR_OV import Full_GR_OV_10

        self.torch, self.art = torch, art
        torch.set_num_threads(4)
        self.calibrator = ClothoCalibrator(torch, art, bulid_rot, SMPLPose.t_pose_ori)
        self.flow = IMUCanonicalization(pretrained_ae=None, cin_state=9, cin_cond=9, cout=9,
            depth=6, nhead=4, ffn=512, dropout=.1, max_T=512, t_embed_dim=256,
            enc_d_model=384, enc_layers=4, enc_nhead=8)
        state = torch.load(root / "checkpoints/canonicalization/U100hL7.07h_residual.pth",
                           map_location="cpu", weights_only=True)
        self.flow.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state, strict=True)
        self.flow.eval()
        self.gp = Full_GR_OV_10(weights_path=str(root / "checkpoints/globalpose/weights.pt"),
                              body_model_path=str(root / "models/SMPL_male.pkl"), enable_physics=False).eval()
        self.x = torch.zeros(1, 30, 10, 9)
        self.pad = torch.zeros(1, 300, dtype=torch.bool)
        self.ids = torch.arange(10, dtype=torch.long).unsqueeze(0)
        self.reset()

    def reset(self, calibration=None):
        if calibration is not None:
            self.calibrator.set_calibration_120(calibration)
        self.acc, self.rot = deque(maxlen=30), deque(maxlen=30)
        self.previous_acc = self.previous_rot = self.previous_R = None
        self.gp.rnn_initialize()
        self.gp.fast_tran = self.torch.zeros(3)
        self.gp.fast_initialized = False

    def step(self, values):
        torch, art = self.torch, self.art
        started = time.perf_counter()
        with torch.inference_mode():
            a, R = self.calibrator.calibrate_ordered_10x7(values)
            self.acc.append(a.float())
            self.rot.append(art.math.normalize_rotation_matrix(R.float()))
            if len(self.acc) < 30:
                return None
            acc, rot = torch.stack(list(self.acc)), torch.stack(list(self.rot))
            self.x[0, :, :, :3].copy_(acc/30)
            self.x[0, :, :, 3:].copy_(art.math.rotation_matrix_to_r6d(rot.reshape(-1, 3, 3)).reshape(30, 10, 6))
            clean = (self.x + self.flow(x_t_9=self.x, x_n_9=self.x, vis_ids=self.ids, mem_pad_mask=self.pad))[0, -1]
            a = clean[:, :3]*30
            R = art.math.r6d_to_rotation_matrix(clean[:, 3:].reshape(-1, 6)).reshape(10, 3, 3)
            if self.previous_acc is not None:
                a = .08*self.previous_acc+.92*a
                R = art.math.normalize_rotation_matrix(.08*self.previous_rot+.92*R)
            self.previous_acc, self.previous_rot = a, R
            a_world = R.matmul(a.view(10, 3, 1)).squeeze(-1).clamp(-30, 30)
            if self.previous_R is None:
                w = torch.zeros_like(a_world)
            else:
                from scipy.spatial.transform import Rotation
                delta = self.previous_R.transpose(-1, -2).matmul(R)
                aa = torch.from_numpy(Rotation.from_matrix(delta.numpy().reshape(-1, 3, 3)).as_rotvec()).to(dtype=delta.dtype)
                aa = torch.where(aa.abs() < 1e-4, torch.zeros_like(aa), aa)
                w = torch.nan_to_num(aa*30, nan=0, posinf=0, neginf=0).clamp(-60, 60)
            self.previous_R = R.clone()
            pose, tran = self.gp.forward_frame_fast0(a_world, R.matmul(w.unsqueeze(-1)).squeeze(-1), R)
        result = dict(pose_mat=pose.numpy().astype(np.float32), root_translation_m=tran.numpy().astype(np.float32),
                      inference_ms=(time.perf_counter()-started)*1000)
        if not np.isfinite(result["pose_mat"]).all() or not np.isfinite(result["root_translation_m"]).all():
            raise ValueError("non-finite FGP output")
        return result
