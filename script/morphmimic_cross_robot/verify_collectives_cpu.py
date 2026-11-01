"""Verify original multi-robot collective semantics using three CPU Gloo ranks."""

import argparse
import ast
import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
from types import SimpleNamespace


UPSTREAM_HASHES = {
    "distributed.py": "30423357374f3fb6e543a7ab227477df93f319ca031a8d3b5ba848b4850e97a7",
    "multi_robot.py": "5c1b5b07eda6feed580a14094326ddcc81312192b5c9a0c6d3b5f8c16bb65552",
    "normalizers.py": "402c4693102c1573309466fff882da80ff424ed5ffaa7da7465b290b9a337dd6",
    "trainer.py": "82519fcb992471eee910dcacf7f962a1e02174e4f7b32cd9d763105220993c67",
}
GIB = 1024 ** 3
ROOT = Path("/opt/data/private/ZHY")
BOOT = "4d859008-4c83-4a6d-814d-cbb9dfe885c9"
GPU = "GPU-26028785-9fb9-45af-e614-4cc53af15d08"


def configure_environment(root=ROOT):
    executable = root / "data/environments/morphmimic_git_tools/usr/bin/git"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError("Existing MorphMimic Git executable is required for upstream imports")
    runtime = dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo", MUJOCO_GL="egl",
                   GIT_PYTHON_GIT_EXECUTABLE=str(executable))
    os.environ.update(runtime)
    return runtime


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_contract(upstream):
    allowed = {"is_available", "is_initialized", "get_rank", "get_world_size",
               "all_reduce", "broadcast", "all_gather", "all_gather_object"}
    rows = []
    for name, expected in UPSTREAM_HASHES.items():
        path = upstream / "src/morphmimic/train" / name
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError("Pinned upstream changed: " + name)
        calls = sorted({node.func.attr for node in ast.walk(ast.parse(data))
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "dist"})
        if set(calls) - allowed:
            raise ValueError("Unaudited upstream collective: " + repr(calls))
        rows.append(dict(path=str(path), sha256=actual, direct_dist_calls=calls))
    return rows


def rank_worker(rank, upstream, output):
    import sys

    record = dict(rank=rank, status="RUNNING", checks=[], surrogate_Adam_steps=0,
                  native_simulation_steps=0, native_policy_inference=0,
                  native_training_optimizer_updates=0)
    dist = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.path.insert(0, str(upstream))
        sys.path.insert(0, str(upstream / "src"))
        import torch
        import torch.distributed as dist
        from morphmimic.train import distributed, multi_robot, normalizers
        from collective_transport import installed_transport

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        dist.init_process_group("gloo", init_method=(output / "rendezvous").as_uri(),
                                rank=rank, world_size=3,
                                timeout=datetime.timedelta(seconds=45))
        modules = (distributed, multi_robot, normalizers)
        with installed_transport(modules) as bridge:
            def close(actual, expected, **kwargs):
                torch.testing.assert_close(actual, expected, **kwargs)

            class Robot(torch.nn.Module):
                def __init__(self, robot):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.tensor([[.2, -.4], [.1, .7]]) + robot)
                    self.unused = torch.nn.Parameter(torch.tensor([.3]))
                    self.register_buffer("morphology", torch.full((125,), float(robot)))
                    self.register_buffer("limits", torch.tensor([-robot - 1., robot + 2.]))

            names = ("wm", "motion", "critic", "kp", "critic_kp")
            models = {name: Robot(rank) for name in names}
            distributed.broadcast_params_(*models.values())
            for model in models.values():
                close(model.weight, torch.tensor([[.2, -.4], [.1, .7]]))
                close(model.morphology, torch.full((125,), float(rank)))
                close(model.limits, torch.tensor([-rank - 1., rank + 2.]))
            record["checks"].append("parameters_broadcast_buffers_robot_specific")

            # The oracle differentiates the mean of all robots' losses locally.
            # It never calls the transport or upstream gradient-reduction helper.
            refs = {name: Robot(0) for name in names}
            opts = {name: torch.optim.Adam(m.parameters(), lr=.02) for name, m in models.items()}
            ref_opts = {name: torch.optim.Adam(m.parameters(), lr=.02) for name, m in refs.items()}
            wrong_clip_detected = False
            maximum_error = 0.
            def loss(model, robot, batch):
                x = torch.tensor([[1. + robot, -.3], [.2, 1. + batch]])
                target = torch.tensor([[.7, -.2], [-.1, .8]]) * (robot + 1)
                return ((x @ model.weight - target).square().mean()
                        + (.07 * (robot - 1)) * model.weight.sum())
            for iteration in range(3):
                for name, batches in (("wm", 2), ("motion", 1), ("critic", 2),
                                      ("kp", 1), ("critic_kp", 2)):
                    model, ref = models[name], refs[name]
                    for batch in range(batches):
                        counter = iteration * 2 + batch
                        opts[name].zero_grad(set_to_none=True)
                        ref_opts[name].zero_grad(set_to_none=True)
                        loss(model, rank, counter).backward()
                        oracle = sum(loss(ref, robot, counter) for robot in range(3)) / 3
                        oracle.backward()
                        distributed.all_reduce_grads_(model.parameters())
                        close(model.weight.grad, ref.weight.grad, atol=2e-6, rtol=2e-6)
                        assert model.unused.grad is None
                        local_clipped = []
                        for robot in range(3):
                            g = torch.autograd.grad(loss(ref, robot, counter), ref.weight)[0]
                            local_clipped.append(g * (.15 / (g.norm() + 1e-6)).clamp(max=1))
                        torch.nn.utils.clip_grad_norm_(model.parameters(), .15)
                        torch.nn.utils.clip_grad_norm_(ref.parameters(), .15)
                        wrong_clip_detected |= not torch.allclose(
                            torch.stack(local_clipped).mean(0), ref.weight.grad,
                            atol=1e-5, rtol=1e-5)
                        opts[name].step()
                        ref_opts[name].step()
                        record["surrogate_Adam_steps"] += 1
                        close(model.weight, ref.weight, atol=2e-6, rtol=2e-6)
                        maximum_error = max(maximum_error, float((model.weight - ref.weight).abs().max()))
                        actual_state, expected_state = opts[name].state[model.weight], ref_opts[name].state[ref.weight]
                        for key in ("step", "exp_avg", "exp_avg_sq"):
                            close(actual_state[key], expected_state[key], atol=2e-6, rtol=2e-6)
            assert wrong_clip_detected
            record["maximum_parameter_error_vs_centralized_oracle"] = maximum_error
            record["checks"].extend(("all_five_paths_mean_then_clip_then_Adam_and_moments",
                                      "clip_before_mean_counterexample_detected"))

            top, bottom = distributed.param_checksums(*models.values(), buffers=False)
            close(top, bottom, atol=2e-6, rtol=2e-6)
            top_buffers, bottom_buffers = distributed.param_checksums(*models.values(), buffers=True)
            assert bool((top_buffers > bottom_buffers).all())
            record["checks"].append("shared_parameter_checksums_exclude_distinct_buffers")

            pooled = normalizers.RunningNormalizer(2, sync=True)
            central = normalizers.RunningNormalizer(2)
            for step in range(2):
                batches = [torch.arange((r + step) * 2, dtype=torch.float32).reshape(-1, 2) + r
                           for r in range(3)]
                pooled.update(batches[rank])
                central.update(torch.cat(batches))
                close(pooled.mean, central.mean)
                close(pooled.var, central.var)
                close(pooled.count, central.count)
            record["checks"].append("pooled_normalizer_matches_union_including_empty_rank")

            z = (torch.arange(12, dtype=torch.float32).reshape(4, 3) * .1 + rank).requires_grad_()
            align, distance = distributed.latent_alignment_loss(z)
            center = torch.arange(12, dtype=torch.float32).reshape(4, 3) * .1 + 1
            close(align, (z.detach() - center).square().sum(-1).mean())
            align.backward()
            close(z.grad, 2 * (z.detach() - center) / 4)
            expected_distance = torch.tensor((1 + 2 + 1) * 3 ** .5 / 3)
            close(distance, expected_distance)
            record["checks"].append("original_latent_loss_detached_center_and_local_gradient")

            orders = (("walk", "raise", "extra"), ("extra", "raise", "walk"),
                      ("raise", "walk", "extra"))
            lengths = {"walk": (11, 8, 9), "raise": (7, 10, 6), "extra": (3, 3, 3)}
            order = orders[rank]
            lens = torch.tensor([lengths[name][rank] for name in order])
            starts = torch.cat((torch.zeros(1, dtype=torch.long), lens.cumsum(0)[:-1]))
            library = SimpleNamespace(clip_names=list(order), clip_len=lens, clip_start=starts)
            sampler = multi_robot.CommonClipSampler(["raise", "walk"], library, 17)
            close(sampler.min_len, torch.tensor([6, 8]))
            clips, offsets = sampler.draw(123, 64)
            pairs = torch.stack((clips, offsets))
            gathered = [torch.empty_like(pairs) for _ in range(3)]
            bridge.all_gather(gathered, pairs)
            for other in gathered:
                close(pairs, other)
            assert bool((offsets >= 0).all())
            assert bool((offsets < sampler.min_len[clips]).all())
            assert not torch.equal(pairs, torch.stack(sampler.draw(124, 64)))
            close(sampler.frames(123, 64), starts[sampler.clips[clips]] + offsets)
            record["checks"].append("common_sampler_rank_permutations_min_frames_same_draws")

            values = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
            assert not values.is_contiguous()
            values.add_(rank)
            destinations = [torch.empty(3, 4).t() for _ in range(3)]
            bridge.all_gather(destinations, values)
            for robot, tensor in enumerate(destinations):
                close(tensor, torch.arange(12, dtype=torch.float32).reshape(3, 4).t() + robot)
            bridge.broadcast(values, src=0)
            close(values, torch.arange(12, dtype=torch.float32).reshape(3, 4).t())
            objects = [None] * 3
            bridge.all_gather_object(objects, {"rank": rank, "rng_fixture": [rank, 17]})
            assert objects == [{"rank": r, "rng_fixture": [r, 17]} for r in range(3)]
            record["checks"].append("noncontiguous_gather_broadcast_and_object_checkpoint_state")
            try:
                bridge.all_reduce(torch.ones(2), async_op=True)
            except TypeError:
                pass
            else:
                raise AssertionError("Unaudited asynchronous call was accepted")
            try:
                bridge.barrier()
            except AttributeError:
                pass
            else:
                raise AssertionError("Unaudited collective was accepted")
            try:
                bridge.all_reduce(torch.ones(2), op=dist.ReduceOp.PRODUCT)
            except ValueError:
                pass
            else:
                raise AssertionError("Unaudited reduction operation was accepted")
            record["checks"].append("unaudited_async_and_collective_requests_rejected")
            record["transport_calls"] = bridge.calls
            record["staged_input_bytes"] = bridge.staged_input_bytes
        assert all(module.dist is dist for module in modules)
        record["checks"].append("module_aliases_restored_global_torch_dist_unchanged")
        try:
            with installed_transport(modules):
                raise LookupError("deliberate context-exit fixture")
        except LookupError:
            pass
        assert all(module.dist is dist for module in modules)
        record["checks"].append("module_aliases_restored_after_exception")
        dist.destroy_process_group()
        single = torch.ones(2, 3, requires_grad=True)
        zero, distance = distributed.latent_alignment_loss(single)
        zero.backward()
        close(zero.detach(), torch.tensor(0.))
        close(distance, torch.tensor(0.))
        close(single.grad, torch.zeros_like(single))
        record["checks"].append("single_rank_latent_alignment_is_zero")
        assert not torch.cuda.is_initialized()
        record.update(status="PASSED", gpu_context_created=False, torch_version=torch.__version__)
    except BaseException:
        record.update(status="FAILED", traceback=traceback.format_exc())
        raise
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()
        save(output / ("rank_%s.json" % rank), record)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.upstream = args.upstream.resolve(strict=True)
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        parser.error("Output must be a new independent Q1 data root")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != BOOT:
        raise RuntimeError("Server boot identity changed")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                  text=True).strip()
    if gpu != GPU:
        raise RuntimeError("Server GPU identity changed")
    before = source_contract(args.upstream)
    runtime_environment = configure_environment()
    os.nice(10)
    cgroup = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if int(cgroup.read_text()) > 24 * GIB:
        raise RuntimeError("CPU contract test requires the original 24 GiB admission margin")
    if shutil.disk_usage(ROOT).free < 100 * GIB:
        raise RuntimeError("Original 100 GiB disk reserve would be violated")
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", started_utc=stamp(), original_sources=before,
                  world_size=3, backend="gloo", devices="CPU_ONLY",
                  native_simulation_steps=0, native_policy_inference=0,
                  native_training_optimizer_updates=0, automatic_retry=False,
                  CUDA_transport_tested=False, native_Trainer_executed=False)
    result["runtime_environment"] = runtime_environment
    save(args.output / "status.json", result)
    context = multiprocessing.get_context("spawn")
    children = []
    def interrupted(signum, _frame):
        raise KeyboardInterrupt("CPU verification interrupted by signal %s" % signum)

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for rank in range(3):
            child = context.Process(target=rank_worker, args=(rank, args.upstream, args.output))
            child.start()
            children.append(child)
        result["children"] = [child.pid for child in children]
        save(args.output / "status.json", result)
        deadline = time.monotonic() + 180
        while any(child.is_alive() for child in children):
            if time.monotonic() > deadline or int(cgroup.read_text()) > 28 * GIB:
                raise RuntimeError("Stop only CPU-test children: resource/time guard")
            if any(child.exitcode not in (None, 0) for child in children):
                raise RuntimeError("A CPU contract rank failed")
            for child in children:
                child.join(.1)
        if any(child.exitcode != 0 for child in children):
            raise RuntimeError("A CPU contract rank failed")
        result["ranks"] = [json.loads((args.output / ("rank_%s.json" % rank)).read_text())
                           for rank in range(3)]
        if not all(row["status"] == "PASSED" for row in result["ranks"]):
            raise RuntimeError("CPU test did not produce three passing rank receipts")
        after = source_contract(args.upstream)
        if before != after:
            raise RuntimeError("Pinned upstream changed during CPU test")
        result.update(status="COMPLETED", original_sources_after=after,
                      all_sources_unchanged=True,
                      checks_per_rank=len(result["ranks"][0]["checks"]),
                      total_surrogate_Adam_steps=sum(row["surrogate_Adam_steps"] for row in result["ranks"]))
    except BaseException:
        result.update(status="FAILED", traceback=traceback.format_exc())
        raise
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(5)
            if child.is_alive():
                child.kill()
                child.join(5)
        result["child_exitcodes"] = [child.exitcode for child in children]
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        result["ended_utc"] = stamp()
        save(args.output / "status.json", result)
    print(json.dumps({"status": result["status"], "checks_per_rank": result["checks_per_rank"],
                      "surrogate_Adam_steps": result["total_surrogate_Adam_steps"]}))


if __name__ == "__main__":
    main()
