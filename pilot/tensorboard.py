"""Torch-free TensorBoard logging helpers of the trainers: the --tensorboard path guard, the fixed tag names, the
lazy event-file writer and a None-skipping scalar write.

--tensorboard DIR is an opt-in output: rank 0 writes live scalars into DIR/RUN_ID/ through tensorboardX (already
hash-locked through ray[train]; no new dependency). It never changes what a run trains or what any record says: the
flag is NOT_IDENTITY, and the logdir is kept out of every checkpoint config and run record.

Design notes:
- the path guard (check_logdir) judges a DIR by its realpath string and by file identity (os.path.samestat on every
  existing ancestor), so no symlink, case variant or path through a file reaches results/, data/, ui/, protocol.json
  or the run's own output dirs; it runs on DIR and again on DIR/RUN_ID, and it never creates anything. Inside the
  repo a DIR is accepted only where git ignores it (git check-ignore; tensorboard/ is gitignored), because an event
  file's name carries the host name; without git, nothing can be committed and an in-repo DIR is accepted.
- nothing here imports tensorboardX or torch at import: tensorboardX imports torch, and berturk_train.py must set the
  MPS env before torch loads (torch reads PYTORCH_ENABLE_MPS_FALLBACK once, at import). open_writer imports it.
- every writer gets a unique filename_suffix: tensorboardX opens an event file 'wb' under a name built from a 10-digit
  second and the hostname, so two writers opened in one second in one dir would truncate each other's events.
- the purge step hides the steps a killed partial epoch logged: a writer opened with purge_step P writes a session_log
  START at P, and TensorBoard hides every earlier file's event at a step >= P.
- event files hold only the fixed TAGS and numbers (write refuses any other tag), never message text, a uid or a label.
- comet logging is explicitly disabled (comet_config {"disabled": True}): no telemetry leaves the machine.

usage: from pilot import tensorboard
       root = tensorboard.check_logdir(a.tensorboard, {"--storage": a.storage, "--out": a.out})   # before any load
       tb = tensorboard.open_writer(os.path.join(root, run_id), purge_step, tensorboard.segment_suffix(0))
       tensorboard.write(tb, step, {"train/loss": 0.25})
       tb.close()
"""
import os
import stat
import subprocess
from typing import Any

from pilot.data import REPO_ROOT

FORBIDDEN = ("results", "data", "ui", "protocol.json")     # repo-relative; no event file may be written under them
FLUSH_SECS = 10                                             # the flush timer: a live view without a per-step fsync
MAX_QUEUE = 100                                             # pending events before an add forces a flush
GIB = 2**30
MIB = 2**20
# per optimizer step: the loss, the LRs, the pre-clip gradient norm, the throughput (Laya items or BERTurk messages per
# second), the RLCD sigma (Laya) and the memory gauges
STEP_TAGS = ("train/loss", "train/lr_encoder", "train/lr_head", "train/grad_norm", "train/items_per_s",
             "train/messages_per_s", "train/sigma", "memory/device_current_gib", "memory/mps_driver_gib",
             "memory/rss_gib", "memory/swap_used_gib")
# once per epoch, at the step that closes the epoch
EPOCH_TAGS = ("epoch/loss", "epoch/train_s", "epoch/wall_s", "epoch/save_s", "epoch/peak_total_gib",
              "epoch/swap_delta_mib")
RATE_TAGS = ("train/items_per_s", "train/messages_per_s")
TAGS = STEP_TAGS + EPOCH_TAGS


def _under(real: str, top: str) -> bool:
    """True when the real path `real` is `top` or lies under it, by the path strings."""
    return real == top or real.startswith(top.rstrip(os.sep) + os.sep)


def _existing(real: str) -> list[tuple[str, os.stat_result]]:
    """(path, os.stat) of every existing ancestor of `real`, `real` included, deepest first. A path that cannot be
    stat'ed (missing, through a file, unreadable) is left out."""
    out = []
    p = real
    while True:
        try:
            out.append((p, os.stat(p)))
        except OSError:
            pass
        parent = os.path.dirname(p)
        if parent == p:
            return out
        p = parent


def _within(real: str, root: str) -> bool:
    """True when the real path `real` is `root` or lies under it: by the realpath strings, or, for a root that exists,
    by file identity (os.path.samestat of the root against every existing ancestor of `real`, `real` included).

    The identity test catches what the strings miss: a case variant on a case-insensitive filesystem (realpath keeps
    the typed case, so RESULTS/tb reads as another path) and a bind mount or hard link of the root.
    """
    top = os.path.realpath(root)
    if _under(real, top):
        return True
    try:
        st = os.stat(top)
    except OSError:
        return False
    return any(os.path.samestat(s, st) for _, s in _existing(real))


def _repo_rel(real: str) -> str | None:
    """The path of the real path `real` relative to REPO_ROOT ("" for the root itself), found by walking up from `real`
    to the first ancestor that is REPO_ROOT by file identity (os.path.samestat), so a case variant is found too; None
    when no ancestor is (outside the repo)."""
    try:
        top = os.stat(REPO_ROOT)
    except OSError:
        return None
    parts: list[str] = []
    p = real
    while True:
        try:
            if os.path.samestat(os.stat(p), top):
                return os.path.join(*reversed(parts)) if parts else ""
        except OSError:
            pass
        parent = os.path.dirname(p)
        if parent == p:
            return None
        parts.append(os.path.basename(p))
        p = parent


def check_logdir(path: str, outputs: dict[str, str | None]) -> str:
    """The absolute real path (symlinks resolved, os.path.realpath) of a --tensorboard DIR; creates nothing.

    SystemExit, with a message starting "--tensorboard " that names the fix, in this order, when the path:
    1. is or lies under results/, data/, ui/ or protocol.json of this repo (FORBIDDEN), judged by _within: by the
       realpath string and by file identity, so a symlink into them, a case variant on a case-insensitive filesystem
       and a path through protocol.json are all refused;
    2. is or lies under any non-None `outputs` value ({flag: path}: the run's own output dirs, e.g. --storage and
       --out), judged the same way;
    3. exists and is not a directory, or lies inside a file (its deepest existing ancestor is not a directory);
    4. lies inside this repo (an ancestor is REPO_ROOT by file identity) where git does not ignore it: `git
       check-ignore -q --no-index` on DIR/run/events.out.tfevents.probe exits 1. An event file's name carries the host
       name, so inside the repo it may only land where `git add` never picks it up (tensorboard/ is gitignored). Exit 0
       accepts; any other exit (no git work tree), a missing git binary or a timeout accepts too: nothing can be
       committed then.
    The trainers call it on DIR before any split read, fetch, load or ray.init, and on DIR/RUN_ID again once the run
    id is known, so a DIR/RUN_ID that is itself a symlink into a guarded place is refused too.
    """
    real = os.path.realpath(path)
    for name in FORBIDDEN:
        if _within(real, os.path.join(REPO_ROOT, name)):
            raise SystemExit("--tensorboard %s is under %s of this repo, which holds frozen records or tracked files; "
                             "pass tensorboard/ (gitignored) or a dir outside the repo" % (path, name))
    for flag, value in outputs.items():
        if value is not None and _within(real, value):
            raise SystemExit("--tensorboard %s is %s %s or lies under it (the run's own output dir); pass tensorboard/ "
                             "(gitignored) or another dir" % (path, flag, value))
    found = _existing(real)
    if found and not stat.S_ISDIR(found[0][1].st_mode):
        if found[0][0] == real:
            raise SystemExit("--tensorboard %s exists and is not a directory; pass a directory" % path)
        raise SystemExit("--tensorboard %s lies inside the file %s; pass a directory" % (path, found[0][0]))
    rel = _repo_rel(real)
    if rel is not None:
        probe = os.path.join(rel, "run", "events.out.tfevents.probe")
        try:
            r = subprocess.run(["git", "check-ignore", "-q", "--no-index", "--", probe], cwd=REPO_ROOT,
                               capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            r = None                                        # no git binary: nothing can be committed
        if r is not None and r.returncode == 1:
            raise SystemExit("--tensorboard %s is inside the repo but not gitignored, so git add would pick up its "
                             "event files (their names carry the host name); pass tensorboard/ (gitignored) or a dir "
                             "outside the repo" % path)
    return real


def segment_suffix(segment: int) -> str:
    """The event-file suffix of a ray_train.py run segment (one file per segment: a rerun never truncates the last)."""
    return ".segment-%03d" % segment


def pid_suffix(pid: int) -> str:
    """The event-file suffix of a berturk_train.py process (one file per process)."""
    return ".pid%d" % pid


def open_writer(logdir: str, purge_step: int, suffix: str) -> Any:
    """A tensorboardX SummaryWriter on logdir (created), with the purge marker (session_log START) at purge_step, the
    unique file suffix, FLUSH_SECS and MAX_QUEUE, and comet logging disabled; SystemExit when logdir is not writable.

    tensorboardX imports torch, so it is imported here only, after the caller has set any device env.
    """
    from tensorboardX import SummaryWriter
    try:
        return SummaryWriter(logdir=logdir, purge_step=purge_step, flush_secs=FLUSH_SECS, max_queue=MAX_QUEUE,
                             filename_suffix=suffix, comet_config={"disabled": True})
    except OSError as e:
        raise SystemExit("--tensorboard: cannot write event files under %s (%s); pass a writable directory"
                         % (logdir, e))


def write(writer: Any, step: int, scalars: dict[str, float | int | None]) -> None:
    """add_scalar(tag, float(value), step) for every tag of `scalars` whose value is not None, in sorted tag order.

    ValueError naming every tag outside TAGS, before anything is written (an event file holds the fixed tags only).
    Never flushes: the writer's FLUSH_SECS timer and close() flush (a flush fsyncs).
    """
    unknown = sorted(t for t in scalars if t not in TAGS)
    if unknown:
        raise ValueError("unknown TensorBoard tag(s) %s; use pilot.tensorboard.TAGS" % ", ".join(unknown))
    for tag in sorted(scalars):
        if scalars[tag] is not None:
            writer.add_scalar(tag, float(scalars[tag]), step)


def _gib(b: int | None) -> float | None:
    """Bytes in GiB, None for None."""
    return None if b is None else b / GIB


def memory_scalars(current: int | None, driver: int | None, rss: int | None,
                   swap: int | None) -> dict[str, float | None]:
    """The memory/* gauges of one optimizer step, in GiB (bytes / GIB), None where a value is None (skipped by write):
    memory/device_current_gib (the device's current allocation: mps current_allocated_memory, cuda memory_allocated;
    None on cpu), memory/mps_driver_gib (mps driver_allocated_memory; None elsewhere), memory/rss_gib (the process's
    resident set) and memory/swap_used_gib (the system's used swap)."""
    return {"memory/device_current_gib": _gib(current), "memory/mps_driver_gib": _gib(driver),
            "memory/rss_gib": _gib(rss), "memory/swap_used_gib": _gib(swap)}


def step_scalars(loss: float, lr_encoder: float | None, lr_head: float, grad_norm: float, sigma: float | None,
                 rate_tag: str, items: int, seconds: float, memory: dict[str, float | None]) -> dict[str, float | None]:
    """The per-optimizer-step scalars: train/loss (the mean micro-batch loss of the step), train/lr_encoder (None with
    a frozen encoder) and train/lr_head (the LRs the step used), train/grad_norm (the pre-clip total norm), train/sigma
    (the RLCD sigma; None for another loss), the rate under rate_tag (one of RATE_TAGS: Laya items or BERTurk messages
    per second, items / seconds, None unless seconds > 0; the other rate tag is never set) and the memory gauges
    (memory_scalars). ValueError for a rate_tag outside RATE_TAGS."""
    if rate_tag not in RATE_TAGS:
        raise ValueError("rate tag %s is not one of %s" % (rate_tag, ", ".join(RATE_TAGS)))
    return dict({"train/loss": loss, "train/lr_encoder": lr_encoder, "train/lr_head": lr_head,
                 "train/grad_norm": grad_norm, "train/sigma": sigma,
                 rate_tag: items / seconds if seconds > 0 else None}, **memory)


def epoch_scalars(loss: float, train_s: float, wall_s: float, save_s: float | None, peak_total_bytes: int,
                  swap_delta_bytes: int) -> dict[str, float | None]:
    """The per-epoch scalars (exactly EPOCH_TAGS): epoch/loss (the epoch's mean loss), epoch/train_s (seconds of the
    training steps), epoch/wall_s (the epoch's wall clock up to the write), epoch/save_s (the checkpoint save time;
    None when unknown), epoch/peak_total_gib (the peak memory, bytes / GIB) and epoch/swap_delta_mib (the system swap
    change over the epoch, bytes / MIB)."""
    return {"epoch/loss": loss, "epoch/train_s": train_s, "epoch/wall_s": wall_s, "epoch/save_s": save_s,
            "epoch/peak_total_gib": peak_total_bytes / GIB, "epoch/swap_delta_mib": swap_delta_bytes / MIB}
