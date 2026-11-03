"""Counterexamples for irreversible stage transitions and capacity fallback."""

import copy
import json
from pathlib import Path
import unittest

from contracts import (COUNTS, FIXED, allow_capacity_fallback, build_plans,
                       check_profile, disk_budget, expected_transitions, primebot_gate,
                       required_delivery_paths)


def evidence():
    pairs = [dict(clip=c, seed=s, final_complete=True, initial_complete=False, common_steps=10,
        initial_joint_rmse_rad=.5, final_joint_rmse_rad=.3,
        initial_body_relative_rmse_m=.05, final_body_relative_rmse_m=.03)
        for c in FIXED for s in (0, 1, 2)]
    cases = [dict(clip=c, seed=s, video=dict(status="PASSED")) for c in FIXED for s in (0, 1, 2)]
    fixed = {s: dict(status="PASSED", cases=copy.deepcopy(cases)) for s in ("initial", "final")}
    libraries = {s: dict(status="PASSED", cases=[dict(clip=c, seed=0) for c in FIXED])
                 for s in ("initial", "final")}
    artifacts = [dict(relative=path, bytes=100, sha256="a" * 64)
                 for path in sorted(required_delivery_paths(list(FIXED)))]
    manifest = dict(status="READY_FOR_MAC_VERIFICATION", identity=dict(path="/manifest.json", bytes=30, sha256="b" * 64),
                    artifacts=artifacts)
    mac = dict(status="VERIFIED", artifacts=copy.deepcopy(artifacts), delivery_manifest=manifest["identity"],
        media_checks=[dict(relative=r["relative"], status="PASSED", visual_reviewed=True, decoded_frames=50)
                      for r in artifacts if r["relative"].endswith(".mp4")])
    return dict(server=dict(status="COMPLETED", local_mac_delivery_required_before_morph=True,
                            library_count=3, final_iteration=10000, passed=True),
        pipeline=dict(status="SERVER_EVALUATION_COMPLETED_PENDING_MAC_DELIVERY"), pairs=pairs, fixed=fixed,
        libraries=libraries, expected_names=list(FIXED), audit=dict(status="PASSED", initial=dict(iteration=0),
        final=dict(iteration=10000)), mac=mac, manifest=manifest)


class GateTests(unittest.TestCase):
    def test_only_complete_final_evidence_advances(self):
        self.assertTrue(primebot_gate(**evidence())["authorizes_morph_training"])

    def test_provisional_observer_cannot_advance(self):
        e = evidence()
        e["pipeline"]["status"] = "TRAINING"
        e["server"] = dict(status="PASSED", gate=dict(passed=True, provisional_only=True))
        self.assertFalse(primebot_gate(**e)["passed"])

    def test_missing_or_duplicate_fixed_pairs_fail(self):
        for change in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(0, rows[1])):
            e = evidence()
            change(e["pairs"])
            self.assertFalse(primebot_gate(**e)["passed"])

    def test_early_failure_and_empty_prefix_fail(self):
        for key, value in (("final_complete", False), ("common_steps", 0),
                           ("final_joint_rmse_rad", float("nan")), ("final_body_relative_rmse_m", None)):
            e = evidence()
            e["pairs"][0][key] = value
            self.assertFalse(primebot_gate(**e)["passed"])

    def test_incomplete_full_library_blocks(self):
        e = evidence()
        e["libraries"]["final"]["cases"].pop()
        self.assertFalse(primebot_gate(**e)["passed"])

    def test_missing_local_media_audit_or_hash_blocks(self):
        for change in (lambda e: e["mac"].update(status="PENDING_VISUAL_REVIEW"),
                       lambda e: e["mac"]["media_checks"][0].update(visual_reviewed=False),
                       lambda e: e["mac"]["artifacts"][0].update(sha256="c" * 64),
                       lambda e: (e["manifest"]["artifacts"].pop(0), e["mac"]["artifacts"].pop(0)),
                       lambda e: e["audit"]["final"].update(iteration=5000)):
            e = evidence()
            change(e)
            self.assertFalse(primebot_gate(**e)["passed"])


class CapacityTests(unittest.TestCase):
    def test_only_explicit_capacity_failure_falls_back(self):
        phase = dict(status="FAILED", inputs_unchanged=True, cgroup_oom_kill_delta=0, gpu_recovery_action="None",
            guard=dict(status="WORKER_FAILED", cleanup=dict(worker_reaped=True, worker_return_code=1)))
        self.assertTrue(allow_capacity_fallback(phase, "torch.OutOfMemoryError: CUDA out of memory"))
        for message in ("NCCL timeout", "CPU allocation failed", "CUDA out of memory; unspecified launch failure",
                        "CUDA out of memory; memory reserve reached", "CUDA out of memory; requires reset"):
            self.assertFalse(allow_capacity_fallback(phase, message))
        for key, value in (("inputs_unchanged", False), ("cgroup_oom_kill_delta", 1), ("gpu_recovery_action", "Reset")):
            self.assertFalse(allow_capacity_fallback(dict(phase, **{key: value}), "CUDA out of memory"))
        for guard in ({}, dict(status="FAILED"), dict(status="STOPPED_STORAGE"),
                      dict(status="WORKER_FAILED", cleanup=dict(worker_reaped=True, worker_return_code=-15))):
            self.assertFalse(allow_capacity_fallback(dict(phase, guard=guard), "CUDA out of memory"))

    def test_both_tiers_preserve_exposure_and_networks(self):
        template = json.loads((Path(__file__).resolve().parents[1] /
            "morphmimic_q1/full1h_shared_profile_e2048_handoff_v04_20261103.json").read_text())
        for envs, iterations in ((1024, 160000), (512, 320000)):
            profile, formal = build_plans(template, "/root/new", ["/g1.yaml", "/q1.yaml"], "/human", "a" * 40, envs)
            self.assertEqual(formal["cfg"]["iterations"], iterations)
            self.assertEqual(expected_transitions(formal["cfg"]), 410009600)
            self.assertEqual(expected_transitions(profile["cfg"]), (400 * (1024 // envs) + 20) * envs)
            self.assertEqual(profile["cfg"]["wm_only_iters"], 0)
            self.assertNotEqual(profile["output"], formal["output"])
            self.assertFalse(formal["reuse_profile_weights_for_formal_training"])
            budget = disk_budget(dict(g1=300000000, q1=300000000), formal["cfg"])
            self.assertEqual(budget["numbered_checkpoints_per_robot"], 11)
            self.assertGreater(budget["maximum_experiment_bytes"], 16 * 1024**3)

    def test_incomplete_eight_update_profile_blocks(self):
        with self.assertRaises(RuntimeError):
            check_profile([], {}, list(FIXED))

    def test_real_profile_evidence_requires_both_ranks_and_all_counters(self):
        template = json.loads((Path(__file__).resolve().parents[1] /
            "morphmimic_q1/full1h_shared_profile_e2048_handoff_v04_20261103.json").read_text())
        profile, _ = build_plans(template, "/root/new", ["/g1.yaml", "/q1.yaml"], "/human", "a" * 40, 1024)
        workers = [dict(status="COMPLETED", robot=robot, code_commit="a" * 40, formal_training=False,
            iterations_completed=8, reference_clips=3, collected_environment_transitions=430080,
            optimizer_updates=copy.deepcopy(COUNTS), optimizer_adam_steps={k: [v] for k, v in COUNTS.items()},
            native_rank_checks_required=True, native_rank_checks_passed=True, peer_audit=[{}, {}],
            resolved_cfg=copy.deepcopy(profile["cfg"]), common_clip_names=list(FIXED),
            nonfinite_updates={"wm/nonfinite_steps": 0}, iteration_seconds=[2.] * 8,
            latent_draws=[dict(it=0, n=512, clip_sha="a", offset_sha="b")],
            last_latent_draw=dict(it=7, n=512, clip_sha="c", offset_sha="d"),
            shared_parameter_sha=dict(wm="a", policy="b", critic="c", critic_kp="d"),
            initial_buffers=dict(joint_presence=robot)) for robot in ("g1", "q1")]
        check_profile(workers, profile, list(FIXED))
        for change in (lambda w: w.update(iterations_completed=7),
                       lambda w: w.update(collected_environment_transitions=429056),
                       lambda w: w["optimizer_updates"].update(kp_opt=7),
                       lambda w: w["optimizer_adam_steps"].update(critic_kp_opt=[140]),
                       lambda w: w["shared_parameter_sha"].update(policy="different"),
                       lambda w: w["last_latent_draw"].update(offset_sha="different"),
                       lambda w: w["initial_buffers"].update(joint_presence="g1")):
            wrong = copy.deepcopy(workers)
            change(wrong[1])
            with self.assertRaises(RuntimeError):
                check_profile(wrong, profile, list(FIXED))


if __name__ == "__main__":
    unittest.main()
