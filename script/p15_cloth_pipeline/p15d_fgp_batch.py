#!/usr/bin/env python3
# p15d_fgp_batch.py — 91-scene FGP batch pipeline (lead recipe 2026-10-20: SEQ feed + -3 frame align)
# Stage A: smb raw (imu_samples_*.jsonl + calib_*.npz) -> ble10_30fps.npz on GT grid
# Stage B: ble10_30fps.npz -> CLOTHO -> LIP -> poses_fgp.npz (+ MPJPE audit vs GT when present)
# Resume-safe: per-scene STATUS_fgp.txt marker. Run with a3-sonic python.
import sys, os, json, glob, time
import numpy as np

sys.path.insert(0, "/opt/data/private/yuanqi_ops/script")
sys.path.insert(0, "/opt/data/private/yuanqi_ops/script/tools/FGP_Full/FGP-main")
sys.path.insert(0, "/opt/data/private/yuanqi_ops/script/tools/FGP_Full/FGP-main/Aplus/tools")

PREVIEW_ROOT = "/opt/data/private/yuanqi_ops/data/humanplus_preview"
RAW_ROOT = "/opt/data/private/yuanqi_ops/data/smb_bulk"
RAW_PILOT = "/opt/data/private/yuanqi_ops/data/smb_orig_pilot"
SYNC_SHIFT_FRAMES = 3  # BLE lags GT by 3 frames @30fps (xcorr -3, ironclad 0.643)

def stage_a(scene_dir, raw_dir=None):
    """Parse raw BLE jsonl (rows: [ch,?,tick,utc_ns,acc3,quat4 wxyz]) -> ble10_30fps.npz on GT grid, -3 frame aligned."""
    import h5py
    from scipy.spatial.transform import Rotation as R
    if raw_dir is None:
        rd = os.path.join(RAW_ROOT, scene_dir)
        cands = sorted(glob.glob(os.path.join(rd, "imu_samples*.jsonl")))
        if not cands:
            return "no_raw"
        jf = cands[0]
    else:
        jf = os.path.join(raw_dir, "imu_samples.jsonl")
    by_ch = {}
    with open(jf) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate corrupted lines (path-string pollution residuals)
            if not isinstance(d, dict) or d.get("k") != "i" or "i" not in d:
                continue
            rslot = d.get("r", "u")
            for row in d["i"]:
                if len(row) < 11 or row[0] is None or row[3] is None:
                    continue
                try:
                    ch = (rslot, int(row[0])); utc = float(row[3])
                    acc = [float(v) for v in row[4:7]]
                    q = [float(v) for v in row[7:11]]  # wxyz
                except (TypeError, ValueError):
                    continue
                if sum(abs(x) for x in q) < 1e-6:
                    continue  # dead-wire placeholder rows
                by_ch.setdefault(ch, ([], [], []))
                by_ch[ch][0].append(utc); by_ch[ch][1].append(acc); by_ch[ch][2].append(q)
    # Adjudicated feed order (p15f_grid.py, WIRE_GRID.json: drop_u4_l1st MPJPE 0.2513 best;
    # head u4 vs l5 indistinguishable at 0.4mm -> drop u4 per GT head-corr top1)
    FEED = [("l", i) for i in (0, 1, 2, 4, 5)] + [("u", i) for i in (0, 1, 2, 3, 5)]
    missing = [c for c in FEED if c not in by_ch]
    if missing:
        return "ch_missing_%s" % ",".join("%s%d" % c for c in missing)
    chs = FEED
    mp = os.path.join(PREVIEW_ROOT, scene_dir, "metadata.json")
    if os.path.exists(mp):
        try:
            n_meta = json.load(open(mp)).get("capture", {}).get("num_body_imus")
        except json.JSONDecodeError:
            n_meta = None  # placeholder note files (upstream had no metadata)
        if n_meta is not None and int(n_meta) != 10:
            return "ch_meta_mismatch_%s" % n_meta
    d2b = np.eye(3)  # adjudicated: no per-wire d2b, raw device frame
    with h5py.File(os.path.join(PREVIEW_ROOT, scene_dir, "annotation.hdf5"), "r") as f:
        T_gt = f["body_motion/body_keypoints"].shape[0]
    T_out = T_gt
    rot = np.zeros((T_out, 10, 3, 3), np.float32); acc = np.zeros((T_out, 10, 3), np.float32)
    for s, ch in enumerate(chs):
        uts, accs, qs = by_ch[ch]
        order = np.argsort(uts); uts = np.asarray(uts)[order]
        accs = np.asarray(accs)[order]; qs = np.asarray(qs)[order]
        rm = R.from_quat(qs[:, [1, 2, 3, 0]]).as_matrix()  # stored wxyz -> xyzw
        rmd = rm  # no d2b: raw device frame, production-isomorphic (grid verified)
        t0 = uts[0]
        idx = np.round((uts - t0) / 1e9 * 30.0).astype(int) - SYNC_SHIFT_FRAMES
        m = (idx >= 0) & (idx < T_out)
        rot[idx[m], s] = rmd[m].astype(np.float32)
        acc[idx[m], s] = accs[m].astype(np.float32)
    np.savez_compressed(os.path.join(PREVIEW_ROOT, scene_dir, "ble10_30fps.npz"),
                        rot=rot, acc=acc, d2b=d2b, sync_shift=SYNC_SHIFT_FRAMES, channels=len(chs))
    return "ok_T%d_ch%d" % (T_out, len(chs))

def stage_b(scene_dir):
    """SEQ feed + CLOTHO + LIP; save fgp poses; MPJPE audit when GT present."""
    import torch, h5py
    import p15b_fgp_pipeline as P
    P.CK = "/opt/data/private/yuanqi_ops/script/tools/FGP/checkpoints"
    p = os.path.join(PREVIEW_ROOT, scene_dir, "ble10_30fps.npz")
    if not os.path.exists(p):
        return "no_ble"
    import hashlib
    D = np.load(p)
    acc = torch.from_numpy(D["acc"].astype(np.float32))
    rot = torch.from_numpy(D["rot"].astype(np.float32))
    in_sha = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    n_ch = int(D["channels"]) if "channels" in D.files else int(D["rot"].shape[1])
    with torch.no_grad():
        acc_d, rot_d = P.run_cano(acc, rot, imu_ids=list(range(10)), chunk_len=512)
        poses = P.lip_batch(acc_d, rot_d)
    out = os.path.join(PREVIEW_ROOT, scene_dir, "poses_fgp.npz")
    np.savez_compressed(out, poses=poses.astype(np.float16), acc_d=acc_d.numpy().astype(np.float16),
                        rot_d=rot_d.numpy().astype(np.float16), feed="SEQ", recipe="SEQ+shift3")
    # audit
    audit = {"feed": "SEQ", "frames": int(poses.shape[0]), "channels": n_ch, "input_ble_sha16": in_sha, "recipe": "SEQ+shift3"}
    ap = os.path.join(PREVIEW_ROOT, scene_dir, "annotation.hdf5")
    if os.path.exists(ap):
        with h5py.File(ap, "r") as f:
            gt = f["body_motion/body_keypoints"][:].astype(np.float32)[:len(poses)]
        gr = gt - gt[:, 0:1]
        jf = np.zeros((len(poses), 24, 3), np.float32)
        from smpl_light import SMPLight
        sm = SMPLight()
        with torch.no_grad():
            for s in range(0, len(poses), 4096):
                Rb = torch.from_numpy(poses[s:s+4096].astype(np.float32))
                _, j = sm.forward_kinematics(Rb, calc_joint=True)
                jf[s:s+len(j)] = j.numpy()
        jr = jf - jf[:, 0:1]
        V, W = jr[::20].reshape(-1, 3), gr[::20].reshape(-1, 3)
        H = V.T @ W; U, S2, Vt = np.linalg.svd(H)
        dd = np.sign(np.linalg.det(Vt.T @ U.T)); Rw = U @ np.diag([1, 1, dd]) @ Vt
        jra = jr @ Rw.T
        sc = (jra[::20] * gr[::20]).sum() / (jra[::20] ** 2).sum(); jra = jra * sc
        audit["mpjpe_mean_m"] = float(np.linalg.norm(jra - gr, axis=2).mean())
        audit["scale"] = float(sc)
    json.dump(audit, open(os.path.join(PREVIEW_ROOT, scene_dir, "fgp_audit.json"), "w"))
    return "ok_%s" % json.dumps(audit)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    if which.upper() == "B":
        import p15b_fgp_pipeline as P
        P.cano = P.load_cano(); P.lip = P.load_lip()
        P.s1, P.s2 = P.lip.net_s1, P.lip.net_s2
        print("models loaded", flush=True)
    scenes = sorted(os.path.basename(p) for p in glob.glob(os.path.join(PREVIEW_ROOT, "session_*")))
    t0 = time.time(); done = 0
    for i, s in enumerate(scenes):
        mark = os.path.join(PREVIEW_ROOT, s, "STATUS_fgp_%s.txt" % which.upper())
        if os.path.exists(mark):
            continue
        try:
            st = stage_a(s) if which.upper() == "A" else stage_b(s)
        except Exception as e:
            st = "ERR_%s_%s" % (type(e).__name__, str(e)[:60])
        open(mark, "w").write(st)
        done += 1
        print("[%d/%d] %s %s" % (i + 1, len(scenes), s, st), flush=True)
    print("DONE stage=%s processed=%d/%d in %.0fs" % (which, done, len(scenes), time.time() - t0), flush=True)
