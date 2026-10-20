"""p15c_grid_rebuild.py — rebuild weave10 grid from live imu_samples.jsonl."""
import json, numpy as np
P = "/opt/data/private/yuanqi_ops/data/smb_weave_staging/session_20261018_152655"
SLOT = {3: 8, 2: 9, 0: 4, 4: 6, 1: 5, 5: 7}  # imu_idx -> grid slot
buf = {k: [[], [], [], []] for k in range(6)}
with open(P + "/imu_samples.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("k") != "i":
            continue
        for e in d["i"]:
            k = e[0]; t = e[3]
            buf[k][0].append(t); buf[k][1].append(e[4:7]); buf[k][2].append(e[7:11])
quat_ord = {}
for k in range(6):
    t = np.array(buf[k][0]); a = np.array(buf[k][1]); q = np.array(buf[k][2])
    m = (t > t.min() + 5e9)
    g = np.array([0, 0, -9.81])
    best = None
    for order in ["xyzw", "wxyz"]:
        qq = q if order == "xyzw" else q[:, [3, 0, 1, 2]]
        x, y, z, w = qq[:, 0], qq[:, 1], qq[:, 2], qq[:, 3]
        R = np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                      2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                      2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(-1, 3, 3)
        e1 = np.abs(np.einsum("tni,i->tn", R[:, :].transpose(0, 2, 1), g) - a[m][:len(R)]).mean()
        e2 = np.abs(np.einsum("tni,i->tn", R, g) - a[m][:len(R)]).mean()
        cand = min(e1, e2)
        if best is None or cand < best[0]:
            best = (cand, order, R if order == "xyzw" else R)
    quat_ord[k] = best
    print("imu%d n=%d err=%.3f order=%s" % (k, len(t), best[0], best[1]), flush=True)
t0 = min(np.array(buf[k][0]).min() for k in range(6))
t1 = max(np.array(buf[k][0]).max() for k in range(6))
G = np.arange(t0, t1, 1e9 / 30.0)
T = len(G)
acc = np.zeros((T, 10, 3), np.float32); acc[:] = [0, 0, 9.81]
rot = np.tile(np.eye(3, dtype=np.float32), (T, 10, 1))
for k in range(6):
    t = np.array(buf[k][0]); a = np.array(buf[k][1], np.float32); q = np.array(buf[k][2])
    order = quat_ord[k][1]
    qq = q if order == "xyzw" else q[:, [3, 0, 1, 2]]
    x, y, z, w = qq[:, 0], qq[:, 1], qq[:, 2], qq[:, 3]
    R = np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                  2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                  2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(-1, 3, 3).astype(np.float32)
    idx = np.searchsorted(t, G).clip(1, len(t) - 1)
    pick = t[idx] - G > G - t[idx - 1]
    ii = np.where(pick, idx, idx - 1)
    acc[:, SLOT[k]] = a[ii]; rot[:, SLOT[k]] = R[ii]
np.savez_compressed(P + "/weave10_grid_rebuilt.npz", acc=acc, rot=rot,
                    slots=json.dumps(sorted(SLOT.values())), provenance="rebuilt_from_jsonl_6live")
print("REBUILT T=%d slots=%s" % (T, sorted(SLOT.values())), flush=True)
for s in sorted(SLOT.values()):
    print("slot%d tstd=%.3f" % (s, float(acc[:, s].std(0).mean())), flush=True)
