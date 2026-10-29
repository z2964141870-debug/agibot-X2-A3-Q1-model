"""Release clean cached pages of idle Q1 project files without reading their data."""

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import time

ROOT = Path("/opt/data/private/ZHY")
GIB = 1024**3
CHUNK = 256 * 1024**2
BOOT = "4d859008-4c83-4a6d-814d-cbb9dfe885c9"
GPU = "GPU-26028785-9fb9-45af-e614-4cc53af15d08"


def snapshot():
    cgroup = Path("/sys/fs/cgroup/memory")
    fields = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    return dict(total_bytes=int((cgroup / "memory.usage_in_bytes").read_text()),
                cache_bytes=int(fields["total_cache"]),
                anonymous_rss_bytes=int(fields["total_rss"]), shmem_bytes=int(fields["total_shmem"]),
                oom=(cgroup / "memory.oom_control").read_text().strip())


def in_scope(path, root):
    return any(path.is_relative_to(root / directory) for directory in ("data", "logs", "script", "reports"))


def active_files(root):
    """Protect open/mapped inodes, and skip when project compute is running."""
    protected, busy, inaccessible_init = set(), [], []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            name = (process / "comm").read_text().strip()
            cwd = Path(os.readlink(process / "cwd"))
            if cwd.is_relative_to(root) and name.startswith(("python", "ffmpeg", "torchrun", "train", "render")):
                busy.append(dict(pid=int(process.name), name=name))
            for fd in (process / "fd").iterdir():
                try:
                    info = fd.stat()
                    protected.add((info.st_dev, info.st_ino))
                except FileNotFoundError:
                    pass
            for line in (process / "maps").read_text().splitlines():
                values = line.split(maxsplit=5)
                if len(values) == 6 and values[5].startswith("/"):
                    try:
                        info = Path(values[5]).stat()
                        protected.add((info.st_dev, info.st_ino))
                    except FileNotFoundError:
                        pass
        except FileNotFoundError:
            continue
        except PermissionError:
            if process.name != "1" or name != "pod":
                raise
            inaccessible_init.append(dict(pid=1, name=name, reason="Infrastructure init is not ptrace-readable"))
    return protected, busy, inaccessible_init


def candidates(root, protected):
    result = []
    for directory in ("data", "logs", "script", "reports"):
        for parent, directories, files in os.walk(root / directory, followlinks=False):
            directories[:] = [name for name in directories if name != ".git" and
                               not (Path(parent) / name).is_symlink()]
            for name in files:
                path = Path(parent) / name
                info = path.lstat()
                if (stat.S_ISREG(info.st_mode) and info.st_size and in_scope(path, root)
                        and (info.st_dev, info.st_ino) not in protected):
                    result.append((path, info))
    return sorted(result, key=lambda item: (-item[1].st_size, str(item[0])))


def resident_bytes(fd, offset, length):
    """mincore observes residency; PROT_NONE never faults in file contents."""
    library = ctypes.CDLL(None, use_errno=True)
    library.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int, ctypes.c_longlong)
    library.mmap.restype = ctypes.c_void_p
    library.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    library.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    pointer = library.mmap(None, length, 0, 1, fd, offset)
    if pointer == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), "mmap residency check")
    page = os.sysconf("SC_PAGE_SIZE")
    pages = (length + page - 1) // page
    vector = (ctypes.c_ubyte * pages)()
    try:
        if library.mincore(pointer, length, vector):
            raise OSError(ctypes.get_errno(), "mincore")
        return min(length, sum(value & 1 for value in vector) * page)
    finally:
        library.munmap(pointer, length)


def release(path, expected, protected, offset, length, root):
    if path.is_symlink() or not in_scope(path.resolve(strict=True), root):
        raise ValueError("Cache target escaped the project")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        current = os.fstat(fd)
        key = (current.st_dev, current.st_ino)
        if key != (expected.st_dev, expected.st_ino) or key in protected:
            return 0, False
        if not stat.S_ISREG(current.st_mode) or current.st_size != expected.st_size:
            return 0, False
        resident = resident_bytes(fd, offset, length)
        if resident:
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        after = os.fstat(fd)
        if (after.st_size, after.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
            raise RuntimeError("File changed concurrently; stop maintenance")
        return resident, bool(resident)
    finally:
        os.close(fd)


def maintain(args):
    start = time.monotonic()
    if ROOT.resolve() != ROOT or Path.cwd().resolve() != ROOT:
        raise ValueError("Run from the unique physical Q1 server root")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != BOOT:
        raise ValueError("Container boot identity changed")
    uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if uuid != GPU:
        raise ValueError("GPU identity changed")
    before = snapshot()
    receipt = dict(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   before=before, mode="once" if args.once else "periodic",
                   trigger_cache_bytes=args.trigger_gib * GIB, target_cache_bytes=args.target_gib * GIB,
                   maximum_advice_bytes=int(args.batch_gib * GIB), files_advised=0,
                   advice_range_bytes=0, observed_resident_bytes_in_advised_ranges=0,
                   file_contents_read=False, files_deleted=0, global_drop_caches=False,
                   agents_started=False, new_training_seconds=0, skipped_open_or_mapped_inodes=0)
    state_path = ROOT / "logs/q1_cache_maintenance_cursor.json"
    cursor = {} if args.once or not state_path.exists() else json.loads(state_path.read_text())
    if not args.once and time.time() - cursor.get("completed_unix", 0) < 1800:
        receipt["status"] = "SKIPPED_30_MINUTE_COOLDOWN"
        return receipt
    if before["cache_bytes"] <= args.trigger_gib * GIB:
        receipt["status"] = "SKIPPED_BELOW_CACHE_THRESHOLD"
        return receipt
    protected, busy, inaccessible_init = active_files(ROOT)
    receipt["unreadable_infrastructure_init"] = inaccessible_init
    gpu_busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy or gpu_busy:
        receipt.update(status="SKIPPED_ACTIVE_WORK", active_project_processes=busy, gpu_busy=bool(gpu_busy))
        return receipt
    receipt["skipped_open_or_mapped_inodes"] = len(protected)
    files = candidates(ROOT, protected)
    receipt["candidate_files"] = len(files)
    index = next((i for i, (p, _) in enumerate(files) if str(p) == cursor.get("path")), 0)
    files = files[index:] + files[:index]
    seen_files = set()
    scanned = 0
    next_cursor = {}
    reason = "PROJECT_SCAN_COMPLETED"
    last_active_check = time.monotonic()
    for path, info in files:
        offset = min(cursor.get("offset", 0), info.st_size) if str(path) == cursor.get("path") else 0
        while offset < info.st_size:
            if time.monotonic() - start >= args.maximum_seconds:
                reason = "BOUNDED_WALL_TIME_REACHED"
                break
            remaining = receipt["maximum_advice_bytes"] - receipt["advice_range_bytes"]
            page = os.sysconf("SC_PAGE_SIZE")
            if remaining < page:
                reason = "BATCH_LIMIT_REACHED"
                break
            length = min(CHUNK, info.st_size - offset, remaining // page * page)
            resident, advised = release(path, info, protected, offset, length, ROOT)
            scanned += length
            if advised:
                receipt["advice_range_bytes"] += length
                receipt["observed_resident_bytes_in_advised_ranges"] += resident
                seen_files.add(str(path))
            offset += length
            next_cursor = dict(path=str(path), offset=offset)
            if snapshot()["cache_bytes"] <= args.target_gib * GIB:
                reason = "CACHE_TARGET_REACHED"
                break
        if reason != "PROJECT_SCAN_COMPLETED":
            break
        # Recheck active work between files; the GAIT_GOAL lock also excludes new runs.
        if time.monotonic() - last_active_check >= 10:
            current_protected, busy, _ = active_files(ROOT)
            protected |= current_protected
            last_active_check = time.monotonic()
            if busy:
                reason = "ACTIVE_WORK_STARTED_STOPPED"
                break
    next_cursor["completed_unix"] = time.time()
    state_path.write_text(json.dumps(next_cursor) + "\n")
    after = snapshot()
    receipt.update(status=reason, after=after, files_advised=len(seen_files),
                   scanned_file_range_bytes=scanned, elapsed_seconds=time.monotonic() - start,
                   observed_total_reduction_bytes=before["total_bytes"] - after["total_bytes"],
                   observed_cache_reduction_bytes=before["cache_bytes"] - after["cache_bytes"],
                   caveat="Before/after cgroup differences are observed deltas, not proof of exclusive attribution")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--trigger-gib", type=float, default=8.)
    parser.add_argument("--target-gib", type=float, default=4.)
    parser.add_argument("--batch-gib", type=int, default=8)
    parser.add_argument("--maximum-seconds", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.target_gib <= args.trigger_gib or not 1 <= args.batch_gib <= 256:
        parser.error("Use nonnegative target<=trigger and a1..256GiB advice bound")
    if not 1 <= args.maximum_seconds <= 180:
        parser.error("Wall bound must be1..180s")
    if not args.output.resolve().is_relative_to(ROOT / "logs") or args.output.exists():
        parser.error("Use a fresh small receipt under the server logs directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "data/supermimic_for_q1/.gait_goal01.gpu.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = maintain(args)
    except BlockingIOError:
        result = dict(status="SKIPPED_ACTIVE_GOAL_LOCK", files_deleted=0)
    except Exception as error:
        result = dict(status="FAILED", error=f"{type(error).__name__}: {error}", files_deleted=0)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    lock.close()
    return int(result["status"] == "FAILED")


if __name__ == "__main__":
    raise SystemExit(main())
