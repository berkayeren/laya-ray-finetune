"""Torch-free TensorBoard logging helpers of the trainers: the --tensorboard path guard, the fixed tag names, the
lazy event-file writer and a None-skipping scalar write.

--tensorboard DIR is an opt-in output: rank 0 writes live scalars into DIR/RUN_ID/ through tensorboardX (already
hash-locked through ray[train]; no new dependency). It never changes what a run trains or what any record says: the
flag is NOT_IDENTITY, and the logdir is kept out of every checkpoint config and run record.

Design notes:
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
    """True when the real path `real` is `top` or lies under it."""
    return real == top or real.startswith(top.rstrip(os.sep) + os.sep)


def check_logdir(path: str, outputs: dict[str, str | None]) -> str:
    """The absolute real path (symlinks resolved, os.path.realpath) of a --tensorboard DIR; creates nothing.

    SystemExit, with a message starting "--tensorboard ", when the path is or lies under results/, data/, ui/ or
    protocol.json of this repo (FORBIDDEN), when it is or lies under any non-None `outputs` value ({flag: path}: the
    run's own output dirs, e.g. --storage and --out), or when it exists and is not a directory. The trainers call it on
    DIR before any split read, fetch, load or ray.init, and on DIR/RUN_ID again once the run id is known.
    """
    real = os.path.realpath(path)
    for name in FORBIDDEN:
        if _under(real, os.path.realpath(os.path.join(REPO_ROOT, name))):
            raise SystemExit("--tensorboard %s is under %s of this repo, which holds frozen records or tracked files; "
                             "pass tensorboard/ (gitignored) or a dir outside the repo" % (path, name))
    for flag, value in outputs.items():
        if value is not None and _under(real, os.path.realpath(value)):
            raise SystemExit("--tensorboard %s is %s %s or lies under it (the run's own output dir); pass tensorboard/ "
                             "(gitignored) or another dir" % (path, flag, value))
    if os.path.exists(real) and not os.path.isdir(real):
        raise SystemExit("--tensorboard %s exists and is not a directory; pass a directory" % path)
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
