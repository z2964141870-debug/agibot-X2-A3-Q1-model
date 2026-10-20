import json, collections
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
# P1.5b: parse SMB raw imu_samples.jsonl, resample 6 wire channels to 30fps grid via Slerp.
# Output: resampled6_30fps.npz (rot[T,6,3,3], acc[T,6,3]) + quat-order verdict.
W = '/opt/data/private/yuanqi_ops/data/smb_weave_staging/session_20261018_152655/'
per = collections.defaultdict(lambda: ([], []))
with open(W + 'imu_samples.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for s in d.get('i', []):
            if s[3] is not None:
                per[s[0]][0].append(s[3]); per[s[0]][1].append(s[4:])
FPS, T = 30.049435, 21040
t_min = min(min(v[0]) for v in per.values())
tgrid = t_min + np.arange(T) / FPS * 1e9
c0t, c0p = per[0]
c0t = np.array(c0t); c0p = np.array(c0p, dtype=np.float64)
o = np.argsort(c0t); c0t = c0t[o]; c0p = c0p[o]
zc1 = R.from_quat(c0p[:200, 3:7]).as_matrix()[:, :, 2].mean(axis=0)[2]
zc2 = R.from_quat(c0p[:200, [6, 3, 4, 5]]).as_matrix()[:, :, 2].mean(axis=0)[2]
QORD = [3, 4, 5, 6] if zc1 >= zc2 else [6, 3, 4, 5]
rots = np.zeros((T, 6, 3, 3)); accs = np.zeros((T, 6, 3))
for cid in range(6):
    ts, pl = per[cid]
    ts = np.array(ts); pl = np.array(pl, dtype=np.float64)
    o = np.argsort(ts); ts = ts[o]; pl = pl[o]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    ts = ts[keep]; pl = pl[keep]
    qu = pl[:, QORD]
    nrm = np.linalg.norm(qu, axis=1)
    m = nrm > 0.5
    tt = ts[m]; qu = qu[m] / nrm[m, None]; ac = pl[m, :3]
    tg = np.clip(tgrid, tt[0] + 1, tt[-1] - 1)
    rots[:, cid] = Slerp(tt, R.from_quat(qu))(tg).as_matrix()
    accs[:, cid] = np.stack([np.interp(tg, tt, ac[:, k]) for k in range(3)], axis=1)
np.savez_compressed(W + 'resampled6_30fps.npz', rot=rots.astype(np.float32), acc=accs.astype(np.float32), tgrid0=int(t_min), fps=FPS)
print('resampled6 saved, quat order last-w =', zc1 >= zc2)
