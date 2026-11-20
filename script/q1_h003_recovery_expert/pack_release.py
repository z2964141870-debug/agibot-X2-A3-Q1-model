"""Assemble the source, inference model and reviewed evidence into the Card 02 release."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

SOURCE = """bench.py bench_inference.py check_base.py curriculum_teacher.py delivery.py fit_balance.py
fit_rise.py flat_audit.py freeze_candidate.py pack_release.py leg_height.py parity.py pool.py probe_session.py
profile_closedloop.py q1_recovery_runtime.py recovery.py reference.py relay.py search_roll.py
search_teacher.py sequence_model.py train.py trajectory_search.cc verify_export.py
dependencies/q1_cgfa_obs_np.py dependencies/q1_kinematics_np.py README.md""".split()
EVIDENCE = """formal_03/q1_recovery_expert.onnx formal_03/export_parity.json formal_03/pure_v2b_hashes.json formal_03/runtime_contract.json
formal_03/public_e1/trials.json formal_03/public_relay/relay.json formal_03/public_relay/after_release.json formal_03/extra_e1/trials.json
formal_03/pinned_relay/relay.json formal_03/pinned_e1/trials.json formal_03/pinned_e1/sources.json
formal_03/flat/results.json formal_03/box50/results.json formal_03/latency_loaded.json formal_03/server_final.json
formal_03/server_50hz.json formal_03/server_50hz_poll.json formal_03/mac_final.json formal_03/mac_50hz.json
formal_03/mac_50hz_poll.json formal_03/session_comparison.json formal_03/profiled_relay/timing.json
formal_03/media_final/recovery_all_poses_1x.mp4 formal_03/media_final/fall_recover_then_walk_1x.mp4
formal_03/media_final/training.png formal_03/media_final/recovery_all_poses_1x.png
formal_03/media_final/fall_recover_then_walk_1x.png formal_03/input_tracks/loop_figure8_walk.csv
formal_03/input_tracks/straight_walk_50hz.csv formal_03/e1/trials.json formal_03/relay/relay.json
baseline_01/trials.json baseline_01/sources.json ppo_residual_01/metrics.jsonl ppo_residual_01/actor_00200.pt
candidate_seq01/feedback.npy candidate_seq01/keys_1.npy rolls_01/keys_0.npy rolls_01/keys_2.npy
rolls_01/keys_3.npy parity.json""".split()


def assemble(project, evidence, target):
    files = {key: project / key for key in SOURCE}
    files.update({key: evidence / key for key in EVIDENCE})
    for pattern in ("formal_03/e1/trial_*.npz", "formal_03/public_relay/relay_*.npz"):
        for path in evidence.glob(pattern):
            files[str(path.relative_to(evidence))] = path
    assert sum(name.endswith(".npz") for name in files) == 29
    manifest = []
    with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=5) as out:
        for name, source in sorted(files.items()):
            data = source.read_bytes()
            out.writestr(name, data)
            manifest.append(hashlib.sha256(data).hexdigest() + "  " + name)
        out.writestr("SHA256SUMS", "\n".join(manifest) + "\n")
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        for line in archive.read("SHA256SUMS").decode().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    print(target, len(files)+1, target.stat().st_size, hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("evidence", type=Path)
    p.add_argument("target", type=Path)
    a = p.parse_args()
    assemble(Path(__file__).resolve().parent, a.evidence, a.target)

