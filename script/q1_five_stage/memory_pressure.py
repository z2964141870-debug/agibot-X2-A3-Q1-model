"""Reclaim project file pages only when the owned Q1 phase reaches30GiB."""

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import stat
import sys
import time

from guard_phuma_batch import MemoryBoundBudget, MemoryLimit
from memory_watch import process, subtree
from runtime_guard import save

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "project_tools"))
import reclaim_q1_file_cache as cache

GIB = 1024**3
TRIGGER = 30 * GIB
TARGET = 28 * GIB
MAXIMUM_SECONDS = 15
MAXIMUM_ADVICE = 8 * GIB
MINIMUM_INTERVAL = 30


def accounting(cgroup):
    total = int((cgroup / "memory.usage_in_bytes").read_text())
    limit = int((cgroup / "memory.limit_in_bytes").read_text())
    fields = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    oom = dict(line.split() for line in (cgroup / "memory.oom_control").read_text().splitlines())
    if total < 0 or limit <= TRIGGER:
        raise MemoryLimit("Require measured container capacity above the original30GiB reserve")
    return dict(total_bytes=total, container_limit_bytes=limit,
        immediate_headroom_bytes=max(0, limit - total), guard_headroom_bytes=max(0, TRIGGER - total),
        anonymous_rss_bytes=int(fields["total_rss"]), cache_bytes=int(fields["total_cache"]),
        shmem_bytes=int(fields["total_shmem"]), dirty_bytes=int(fields["total_dirty"]),
        writeback_bytes=int(fields["total_writeback"]),
        oom_kill=int(oom["oom_kill"]), under_oom=int(oom["under_oom"]),
        headroom_is_immediate_not_a_guaranteed_training_allocation=True)


@contextmanager
def paused_descendants(proc, owner_pid, deadline):
    """Hold pidfds so pause/resume can never address a reused process number."""
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise MemoryLimit("Runtime cache reclaim requires Linux pidfd pause/resume")
    owner = process(proc, owner_pid)
    if owner is None:
        raise MemoryLimit("Memory pressure guard handle disappeared")
    handles, paused = {}, []
    try:
        while True:
            current = process(proc, owner_pid)
            if current is None or current["start_ticks"] != owner["start_ticks"]:
                raise MemoryLimit("Memory pressure guard PID identity changed")
            rows = [row for row in subtree(proc, owner) if row["pid"] != owner_pid and row["state"] != "Z"]
            for row in rows:
                key = row["pid"], row["start_ticks"]
                if key in handles:
                    continue
                descriptor = os.pidfd_open(row["pid"], 0)
                try:
                    current = process(proc, row["pid"])
                except BaseException:
                    os.close(descriptor)
                    raise
                if current is None or current["start_ticks"] != row["start_ticks"]:
                    os.close(descriptor)
                    raise MemoryLimit("Owned worker PID changed before pause")
                was_stopped = current["state"] in {"T", "t"}
                handles[key] = descriptor, was_stopped
                if not was_stopped:
                    signal.pidfd_send_signal(descriptor, signal.SIGSTOP)
                    paused.append(dict(pid=row["pid"], start_ticks=row["start_ticks"]))
            if all(row["state"] in {"T", "t"} for row in rows):
                break
            if time.monotonic() >= deadline:
                raise MemoryLimit("Owned worker pause did not settle within the bounded deadline")
            time.sleep(.01)
        yield paused
    finally:
        errors = []
        for descriptor, was_stopped in reversed(list(handles.values())):
            try:
                if not was_stopped:
                    signal.pidfd_send_signal(descriptor, signal.SIGCONT)
            except ProcessLookupError:
                pass
            except BaseException as error:
                errors.append(str(error))
            finally:
                os.close(descriptor)
        if errors:
            raise MemoryLimit("Could not resume every owned worker: " + "; ".join(errors))


def eligible_files(root, preferred, protected, deadline):
    seen, priority = set(), []
    for value in preferred:
        if time.monotonic() >= deadline:
            return
        path = Path(value)
        if path.is_symlink() or not cache.in_scope(path, root):
            continue
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        key = info.st_dev, info.st_ino
        if stat.S_ISREG(info.st_mode) and info.st_size and key not in protected and key not in seen:
            priority.append((path, info))
            seen.add(key)
    yield from sorted(priority, key=lambda row: (-row[1].st_size, str(row[0])))
    for path, info in cache.iter_candidates(root, protected, deadline):
        key = info.st_dev, info.st_ino
        if key not in seen:
            seen.add(key)
            yield path, info


def supervisor_ancestors(proc, owner_pid, root):
    """Recognize the waiting controller without admitting unrelated project compute."""
    owner, result = process(proc, owner_pid), []
    allowed = {root / "script/q1_five_stage" / name for name in ("controller.py", "shared_review.py")}
    while owner is not None and owner["ppid"] > 1:
        parent = process(proc, owner["ppid"])
        if parent is None:
            raise MemoryLimit("Waiting controller disappeared during pressure scan")
        directory = proc / str(parent["pid"])
        arguments = (directory / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        index = 2 if len(arguments) > 1 and arguments[1] == "-u" else 1
        if len(arguments) <= index:
            break
        script = Path(arguments[index])
        script = script if script.is_absolute() else root / script
        if script not in allowed or Path(os.readlink(directory / "cwd")) != root:
            break
        if script.name == "controller.py" and arguments[index + 1:index + 2] != ["run"]:
            break
        current = process(proc, parent["pid"])
        if current is None or current["start_ticks"] != parent["start_ticks"]:
            raise MemoryLimit("Waiting controller PID identity changed")
        result.append(dict(pid=parent["pid"], start_ticks=parent["start_ticks"], script=str(script)))
        owner = parent
    return result


class PressureReclaimer:
    def __init__(self, root, cgroup, output, preferred=(), proc=Path("/proc")):
        self.root, self.cgroup, self.output, self.proc = root, cgroup, output, proc
        self.preferred = tuple(preferred)
        self.last_attempt = None
        self.events = 0
        self.history = []
        if output.exists():
            raise FileExistsError("Preserve previous Q1 memory pressure evidence")

    def reclaim(self):
        started = time.monotonic()
        before = accounting(self.cgroup)
        if before["total_bytes"] < TRIGGER:
            return before
        if before["oom_kill"] or before["under_oom"]:
            raise MemoryLimit("OOM state changed; pressure maintenance is not a recovery attempt")
        if self.last_attempt is not None and started - self.last_attempt < MINIMUM_INTERVAL:
            raise MemoryLimit("Memory returned to30GiB within30s of bounded reclaim; stop without a reclaim loop")
        if self.root.resolve() != self.root or Path.cwd().resolve() != self.root:
            raise MemoryLimit("Runtime reclaim must stay in the unique physical Q1 root")
        self.last_attempt = started
        row = dict(status="RECLAIMING", started_unix=time.time(), before=before,
            trigger_total_bytes=TRIGGER, target_total_bytes=TARGET,
            maximum_advice_bytes=MAXIMUM_ADVICE, maximum_seconds=MAXIMUM_SECONDS,
            advice_range_bytes=0, observed_resident_bytes=0, files_advised=0,
            files_deleted=0, file_contents_read=False, global_drop_caches=False,
            workers_restarted=False, paused_processes=[])
        deadline = started + MAXIMUM_SECONDS
        try:
            with paused_descendants(self.proc, os.getpid(), min(deadline, started + 2)) as paused:
                row["paused_processes"] = paused
                supervisors = supervisor_ancestors(self.proc, os.getpid(), self.root)
                protected, busy, gaps = cache.active_files(self.root, deadline=deadline)
                owned = {item["pid"] for item in subtree(self.proc, process(self.proc, os.getpid()))}
                allowed = owned | {item["pid"] for item in supervisors}
                for item in supervisors:
                    current = process(self.proc, item["pid"])
                    if current is None or current["start_ticks"] != item["start_ticks"]:
                        raise MemoryLimit("Waiting controller PID changed during visibility scan")
                if any(item["pid"] not in allowed for item in busy):
                    raise MemoryLimit("Unowned project compute is active; stop before file advice")
                row["waiting_supervisors"] = supervisors
                row["protected_inodes"] = len(protected)
                row["visibility_gaps"] = gaps
                row["reason"] = "ELIGIBLE_PROJECT_FILES_EXHAUSTED"
                for path, info in eligible_files(self.root, self.preferred, protected, deadline):
                    offset = 0
                    file_advised = False
                    while offset < info.st_size:
                        current = accounting(self.cgroup)
                        if current["oom_kill"] or current["under_oom"]:
                            raise MemoryLimit("OOM state changed during pressure reclaim")
                        if current["total_bytes"] <= TARGET:
                            row["reason"] = "TOTAL_TARGET_REACHED"
                            break
                        if time.monotonic() >= deadline:
                            row["reason"] = "BOUNDED_WALL_TIME_REACHED"
                            break
                        remaining = MAXIMUM_ADVICE - row["advice_range_bytes"]
                        page = os.sysconf("SC_PAGE_SIZE")
                        if remaining < page:
                            row["reason"] = "ADVICE_RANGE_BOUND_REACHED"
                            break
                        length = min(cache.CHUNK, info.st_size - offset, remaining // page * page)
                        resident, advised = cache.release(path, info, protected, offset, length, self.root)
                        if advised:
                            row["advice_range_bytes"] += length
                            row["observed_resident_bytes"] += resident
                            row["files_advised"] += int(not file_advised)
                            file_advised = True
                        offset += length
                    if row["reason"] != "ELIGIBLE_PROJECT_FILES_EXHAUSTED":
                        break
                row["after"] = accounting(self.cgroup)
                if (row["after"]["total_bytes"] >= TRIGGER or row["after"]["oom_kill"]
                        or row["after"]["under_oom"]):
                    raise MemoryLimit("Total memory remains at30GiB or OOM changed after bounded reclaim")
            row["status"] = "RECLAIMED_BELOW_ORIGINAL_LIMIT"
            return row["after"]
        except BaseException as error:
            row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
            if "after" not in row:
                try:
                    row["after"] = accounting(self.cgroup)
                except Exception as error:
                    row["after_measurement_error"] = f"{type(error).__name__}: {error}"
            if "after" in row:
                row["observed_total_reduction_bytes"] = before["total_bytes"] - row["after"]["total_bytes"]
                row["observed_cache_reduction_bytes"] = before["cache_bytes"] - row["after"]["cache_bytes"]
            row["deltas_are_not_exclusive_attribution"] = True
            self.events += 1
            self.history = [*self.history[-31:], row]
            save(self.output, dict(event_count=self.events, maximum_retained_events=32,
                policy="TOTAL30_RECLAIM_THEN_ENFORCE_ORIGINAL_LIMIT", events=self.history))


class PressureMemoryBudget(MemoryBoundBudget):
    def __init__(self, storage, maximum_bytes, expected_oom, *, root, output, preferred=(),
                 cgroup=Path("/sys/fs/cgroup/memory"), observer=None):
        if maximum_bytes != TRIGGER or expected_oom != 0:
            raise ValueError("Pressure reclaim must preserve the30GiB and zero-OOM guards")
        super().__init__(storage, maximum_bytes, expected_oom, cgroup)
        self.reclaimer = PressureReclaimer(root, cgroup, output, preferred)
        self.observer = observer

    def memory(self):
        before = accounting(self.cgroup)
        if before["oom_kill"] or before["under_oom"]:
            raise MemoryLimit("OOM state changed; no pressure cleanup or retry")
        if before["total_bytes"] >= self.maximum_bytes:
            if self.observer is not None:
                self.observer.record()
            self.reclaimer.reclaim()
        evidence = super().memory()
        current = accounting(self.cgroup)
        if current["total_bytes"] >= self.maximum_bytes or current["oom_kill"] or current["under_oom"]:
            raise MemoryLimit("Memory reserve reached again after pressure check")
        evidence.update(accounting=current, pressure_reclaim_events=self.reclaimer.events)
        return evidence
