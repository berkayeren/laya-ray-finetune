"""Distributed fine-tuning of a Laya checkpoint with Ray Train (PyTorch DDP).

`laya.train.finetune` / `laya-train` run on one device. This script keeps Laya's
own building blocks -- item encoding, RLCD / soft-CE loss and the checkpoint
layout `laya.load` reads, all reached through
pilot/laya_adapter.py (the only private-API module, pinned to one laya version)
-- and runs the training loop under Ray's TorchTrainer, so it scales from 1 CPU worker to N GPUs on one
node with only --workers / --device changed. It is single-node: workers read the
base weights and the split from driver-local paths, so it refuses a multi-node
Ray cluster.

Per worker:  load base -> build items -> shard by rank -> DDP train
Rank 0:      save the final, uncalibrated checkpoint (T = 1) and report it to Ray; postfit.py fits the temperature
             on message-level val data after training (LAYA-05). A --postfit final seed serves that val-fitted T.

Devices (--device cpu|cuda|mps, default cpu; --gpu is an alias for --device cuda): cpu and cuda take the device from
Ray (use_gpu on cuda, DDP across --workers). Ray has no MPS device manager, so --device mps runs one Ray worker with
use_gpu False on an explicit torch.device("mps"), with the Phase 2 spike's worker env (strict fallback 0, HIGH and LOW
MPS watermarks 1.0) passed through ray.init(runtime_env) and checked in the worker. On every device the worker
refuses to train unless every model parameter landed on the requested device (no silent CPU run). Gradient
checkpointing (encoder and head) is on for every full-encoder run (--no-grad-checkpointing turns it off); AMP is a
separate --amp choice, on by default only on cuda.

Run record (LAYA-06): the checkpoint config holds training.ray_train.record: device, parameter device, torch threads,
worker env, precision and checkpointing choices, load and final-save seconds (calib_s is None: training no longer
calibrates, LAYA-05), and one row per completed
epoch with the pilot.runs.EPOCH_FIELDS (device-synchronised wall clock of the training steps, save time of the train
state, RNG and export plus Ray's persist of the epoch checkpoint (save_s_includes_persist; the epoch a rerun resumes
from keeps the time up to the persist, from save.json), optimizer steps, items, peak memory sampled after every
micro-batch backward and optimizer step, system swap delta, and the segment that trained it). Across a kill and resume
the record keeps exactly one row per completed epoch (the partial epoch leaves none) and one pilot.runs.SEGMENT_FIELDS
entry per run segment (start epoch, git sha, dirty flag and uncommitted-diff digest, device, resume load seconds and
memory), plus code_changed_on_resume (another HEAD or another uncommitted diff between segments, or no git; flagged,
never refused), the relative export names, train_state_bytes and export_bytes.

Items (data_neg8 by default): the split files label all 48 tags. Every message the --max-messages subsample selects
(all of them without it) trains: there is no calibration holdout (LAYA-05: the temperature comes only from val). Every
message keeps all its positives plus --neg K random negatives (pilot.spike.neg8_rows, one random.Random(--neg-seed)
draw per run, --neg-seed defaulting to --seed); --neg 0 trains on all 48 labels (data_full). The record carries the
items laya built and the torch-free count pilot.runs.neg_items gives, so the compute projection can count any split
without a tokenizer. Every export and the final checkpoint carry temperature UNCALIBRATED_TEMPERATURE and an empty
temperature_by_options, reset explicitly, never inherited from the base (a base with a fitted T must not leak it).

Fallback ladder (LAYA-03): every run starts from the Phase 2 spike's measured config (pilot.runs.SPIKE_START: micro-batch
8, max_len 128, grad-accum 4, full encoder). The fallbacks are flags, never an in-process retry (the MPS allocator keeps a
process-wide high-water mark): --freeze-embeddings (rung embeddings_frozen), then also --micro-batch 4 --grad-accum 8
(rung embeddings_frozen+smaller_batch, effective batch 32 kept), then --freeze-encoder (rung encoder_frozen, head-only,
weaker). pilot.runs.rung_label derives the rung from the resolved flags; the record carries rung, fallback, weaker,
spike_departures and the trainable/total parameter counts. An MPS out-of-memory error ends the run with one line naming
the epoch and micro-batch (or the stage outside the training step: model move, resume, checkpoint save), the rung and
the flags of the next rung; the driver gives a raw MPS out-of-memory error from anywhere else the same rung line, never
"rerun to resume" (the rerun would resume into the same allocation). After the last epoch (the final save;
pilot.runs.RESUME_STAGES) the line says to rerun the same command instead: every epoch is saved, writing the final
checkpoint needs less memory than training (the optimizer state is freed and the MPS cache emptied first), and a rung
flag would change the run_id and retrain from epoch 0. That rerun resumes a train state that already holds the last
epoch and writes the final checkpoint again: it never loads the optimizer, scheduler or scaler state
(restore_optimizer), and an OOM while it moves the model is a resume stage too (pilot.runs.STAGE_FINALIZE_MOVE).

Run identity (LAYA-04): the run_id is pilot.runs.run_id of the command's canonical config (pilot.runs.IDENTITY: split,
data and base digests (a local base's tokenizer/ and encoder/ files too), laya version, the resolved device and precision
choices, every hyperparameter and the seeds; never a path, a resource or the time), e.g.
laya-selection_train-neg8-s0-1a2b3c4d. It is the Ray experiment name, so
storage/<run_id>/ is the run's only directory and the same command always lands in it; --print-run-id prints it after
the split, base and pin checks and exits without starting Ray.

Resume (LAYA-04): at the end of every epoch every rank reports Ray checkpoint storage/<run_id>/epoch-NNN: rank 0's
train_state.pt (fp32 model, AdamW, cosine scheduler, GradScaler on cuda, epoch, record rows and segments, run_id and
canonical config) and each rank's rng-rank<r>.pt (torch CPU RNG, MPS or CUDA RNG, the option-order random.Random). Ray
keeps one (num_to_keep 1) and never retries in-process (max_failures 0). Rerunning the same command after a kill
resumes from the last one: Ray's get_checkpoint hands it back on every rank, every load is weights_only, every rank
refuses a state of another kind, run_id or canonical config (pilot.runs.check_state), and on cpu the result is
byte-identical to an uninterrupted run. The fp16 laya export is never resume state. A finished run's final
checkpoint carries final.json, so its rerun copies that checkpoint to the new --out without training. Rank 0 also writes
each epoch's uncalibrated laya export to storage/<run_id>/exports/epoch-NNN, outside the rotation (--no-export-epochs
skips them), for Phase 5 to pick the epoch on val. Before ray.init the driver refuses when the free disk under --storage
is below pilot.runs.disk_need_bytes. A failed run ends in one line: the MPS OOM rung message, the snapshot dir to delete
when Ray's snapshot points at deleted checkpoints, a worker's refusal (SystemExit) as is, or "rerun the same command to
resume".

Final seeds (Phase 5, LAYA-08): --stop-after-epoch K trains epochs 1..K of the --epochs schedule and publishes epoch K
as the final checkpoint; the cosine T_max and the sigma anneal stay sized by --epochs, so epoch K is the selection run's
epoch K (RESEARCH A3; on cpu its weights are byte-identical to that run's exports/epoch-NNN). K is 1..--epochs - 1:
K = --epochs has one spelling, the omitted flag. --postfit RECORD binds a final seed to the val selection by refusal,
not transcription: the record is read (schema, sha256 of its bytes) before any fetch, and once the base identity is
known, before --print-run-id and ray.init, pilot.postfit.check_final refuses unless --data is the record's final split
(final_train), --seed is one of its final seeds, --stop-after-epoch is the chosen one and every other canonical value
equals the chosen selection run's (naming the first differing key). The final config then serves the record's
val-fitted T (temperature [1, 1, T] bit for bit, temperature_by_options {}, training.calibrated True, training.postfit
with the record's sha256, chosen settings, T and θ) and the run record carries postfit {sha256, chosen, temperature}.
stop_after_epoch and postfit_sha256 are run-identity keys left out when None (every selection run id is unchanged);
postfit.py --print-final-commands prints the commands.

Few-shot (Phase 9, FEW-01, FEW-02): two modes run the frozen protocol.json few-shot block unchanged, each bound to it
by refusal before --print-run-id and ray.init. pilot.protocol.require_frozen() and the D-08 decision record
(pilot.fewshot.read_mode) come first, before the split is read; pilot.fewshot.check_source then proves --data is the
frozen source (selection_train) and that every frozen draw regenerates from its verified bytes.
- --fewshot-base trains a label-masked base: the protocol's held-out tags (--mask-tags is optional and refused unless
  it names exactly them) are deleted from every row's expected before neg sampling (pilot.fewshot.masked_rows; every
  message is kept); pilot.fewshot.check_base binds the command to rules.base_runs.laya and to the chosen selection run's
  canonical config, read through the protocol-pinned postfit record. The worker counts the items per question
  (pilot.laya_adapter.items_per_question) and refuses to train when any held-out question got an item.
- --fewshot-draw D --fewshot-k K --base <masked base> continues a masked base on the frozen support set of (D, K) plus
  the draw's replay set, by the frozen Laya item rule (pilot.fewshot.laya_rows; laya_item_count gives the count). The
  base's run id and canonical config are read from its rl_agent_config.json bytes, bound to its identity digest;
  pilot.fewshot.check_continued binds the epochs, LRs, seed (the draw seed), draw digest and base pairing (the recorded
  mode). The worker refuses unless every held-out question has exactly one item per support row (none from replay
  rows) and the item count equals laya_item_count.
mask_tags and fewshot (pilot.fewshot.binding) are run-identity keys left out when None, so every committed run id is
unchanged. The frozen uid lists reach the worker in the config key fewshot_rows, which the run record and every
checkpoint config leave out; the record's fewshot entry holds only counts, tags, the protocol sha256, the draw digest
and the base's run id and canonical config (the masking proof, D-04).

--data must be a split registered in data/splits.json with role train and a matching sha256; anything else
(tampered, unregistered, val/test/probe) is refused before Ray starts, and so is a train split under any manifest that
holds committed held-out content (pilot.data.refuse_held_out: committed test content, or committed val content in
selection_train; 06-REVIEW WR-02). --base multilingual fetches the pinned,
sha256-verified Laya weights (pilot.weights) before Ray starts; any other value must be a local checkpoint dir. A base
whose tokenizer/tokenizer_config.json laya would have to rewrite on load in a read-only tokenizer/ (or that is not a
JSON object) is refused before Ray starts too (pilot.weights.check_tokenizer_config_rewritable, 04-REVIEW CR-01); the
worker's post-load check_tokenizer_config_rewritten stays the backstop.
--manifest selects the prepare-written splits.json to verify against (default: the committed data/splits.json; use
OTHER_DIR/splits.json for a `prepare_data.py --out OTHER_DIR` variant). Verification itself cannot be skipped, a
manifest cannot change any split's role, path or folds, and the checkpoint provenance records manifest_is_committed.

usage:
  python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --out ckpt \
                      --workers 4 --device cuda    # 4 GPUs on one node (--gpu is the same)
  python ray_train.py ... --device mps --workers 1  # the full encoder on the Apple GPU
  python ray_train.py ... --device mps --workers 1 --freeze-embeddings   # fallback rung 1 after an MPS OOM
  python ray_train.py ... --workers 2 --freeze-encoder --max-messages 100   # CPU smoke test
  python ray_train.py ... --print-run-id            # the run id (storage/<run_id>/); rerun a killed command to resume
  python ray_train.py --data data/splits/final_train.jsonl ... --epochs 4 --stop-after-epoch 2 --seed 1 \
                      --postfit results/postfit/laya.json --no-export-epochs   # a final seed (postfit.py prints it)
  python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --out ckpt-fewshot-base-s0 \
                      --device mps --workers 1 --epochs 4 --stop-after-epoch 3 --encoder-lr 2e-05 --seed 0 \
                      --fewshot-base                        # a Phase 9 masked base (the frozen recipe)
  python ray_train.py --data data/splits/selection_train.jsonl --base ckpt-fewshot-base-s0 --out ckpt-fewshot-d0-k10 \
                      --device mps --workers 1 --epochs 10 --encoder-lr 2e-05 --seed 0 \
                      --fewshot-draw 0 --fewshot-k 10       # a Phase 9 continued run from it
"""
import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager, nullcontext

import psutil
import ray
import ray.train
import torch
from ray.train import Checkpoint, CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from pilot import laya_adapter as la
from pilot import dumps, fewshot, pins, postfit, protocol, runs, spike
from pilot.data import (DEFAULT_MANIFEST, REPO_ROOT, manifest_is_committed, parse_jsonl, refuse_held_out,
                        repo_relpath, sha256_bytes, sha256_file, verify_split_file)
from pilot.weights import (check_tokenizer_config_rewritable, check_tokenizer_config_rewritten, checkpoint_digests,
                           fetch, identity, verify_files)

WEIGHT_DECAY = 0.01                                         # AdamW weight decay (part of the run identity)
CLIP_GRAD_NORM = 1.0                                        # gradient-norm clip per optimizer step (same)
# laya's QTYPES order (choice, score, noul): T = 1 until postfit.py fits T on val (LAYA-05). Every writer copies it
# (list(UNCALIBRATED_TEMPERATURE)), never shares the list object.
UNCALIBRATED_TEMPERATURE = [1.0, 1.0, 1.0]
FINAL_MARK = "final.json"                                   # in the final checkpoint: the run is finished
SAVE_META = "save.json"                                     # in an epoch checkpoint: that epoch's save time


def read_verified_rows(path, sha256):
    """Rows of a JSONL split whose bytes must hash to `sha256` (the driver-verified digest); parses those same bytes.

    Workers re-check because they read the file minutes after the driver verified it (fetch, ray.init).
    Rows match laya.train.read_jsonl (pilot.data.parse_jsonl splits on "\n" only).
    """
    with open(path, "rb") as f:
        raw = f.read()
    have = hashlib.sha256(raw).hexdigest()
    if have != sha256:
        raise SystemExit("%s sha256 %s != %s verified on the driver; regenerate with `make data`"
                         % (path, have, sha256))
    return parse_jsonl(raw)


def require_single_node(nodes):
    """Refuse a multi-node cluster: base_dir and the split path are the driver's local paths (no copy elsewhere)."""
    alive = [n for n in nodes if n.get("Alive")]
    if len(alive) > 1:
        raise SystemExit("ray_train.py is single-node: workers read the driver-local base weights and split, "
                         "but the Ray cluster has %d alive nodes" % len(alive))


def require_driver_node(node_id, driver_node_id):
    """Worker-side half of require_single_node: an autoscaler can add a node after the driver's one-time check."""
    if node_id != driver_node_id:
        raise SystemExit("ray_train.py is single-node: this worker runs on node %s, not the driver's node %s, so the "
                         "driver-local base weights and split are not here" % (node_id, driver_node_id))


def check_env(expected):
    """{name: value seen} for the expected env vars; refuses on the first one the worker did not get."""
    seen = {k: os.environ.get(k) for k in expected}
    for k in sorted(expected):
        if seen[k] != expected[k]:
            raise SystemExit("worker env %s=%r, expected %r; env vars must reach the worker through "
                             "ray.init(runtime_env)" % (k, seen[k], expected[k]))
    return seen


def require_param_device(model, device_type):
    """Refuse unless every model parameter sits on `device_type` (LAYA-02: a run asserts where the model landed)."""
    have = {p.device.type for p in model.parameters()}
    if have != {device_type}:
        raise SystemExit("model parameters on %s, expected %s; refusing to train on the wrong device (no silent CPU "
                         "run)" % (", ".join(sorted(have)), device_type))


def resolve(a):
    """The one resolved device and its precision, checkpointing, thread and optimizer choices for parsed args `a`.

    --gpu is an alias for --device cuda (ValueError next to any other --device). AMP defaults on only for cuda;
    checkpointing is on for every full-encoder run unless --no-grad-checkpointing. AdamW's foreach and fused are pinned
    off on mps (what the Phase 2 spike measured; torch's default there anyway) and left to torch on cpu and cuda.
    """
    if a.gpu and a.device not in (None, "cuda"):
        raise ValueError("--gpu is an alias for --device cuda; it conflicts with --device %s" % a.device)
    device = "cuda" if a.gpu else (a.device or "cpu")
    mps = device == "mps"
    return {"device": device,
            "amp": (device == "cuda") if a.amp is None else a.amp,
            "grad_checkpointing": bool(a.grad_checkpointing and not a.freeze_encoder),
            # 4 = the spike's timing basis; an explicit value is never replaced (main refuses below 1)
            "cpus_per_worker": a.cpus_per_worker if a.cpus_per_worker is not None else (4 if mps else 1),
            "neg_seed": a.seed if a.neg_seed is None else a.neg_seed,
            "adamw_foreach": False if mps else None,
            "adamw_fused": False if mps else None}


@contextmanager
def mps_oom_exit(epoch, step, rung, stage=None):
    """Turn an MPS out-of-memory RuntimeError into the one-line rung message (pilot.runs.oom_message); no retry, since
    the MPS allocator keeps a process-wide high-water mark. stage names an MPS allocation outside the training step
    (model move, resume, epoch save, final save). Any other error propagates unchanged."""
    try:
        yield
    except RuntimeError as e:
        if spike.is_mps_oom(str(e)):
            raise RuntimeError(runs.oom_message(epoch, step, rung, stage)) from e
        raise


def raw_mps_oom(text):
    """True when a (multi-line) failure text holds the MPS allocator's own out-of-memory line (`RuntimeError: MPS
    backend out of memory ...`), i.e. an MPS OOM that no mps_oom_exit turned into the rung message."""
    return any(spike.is_mps_oom(ln.strip().split(": ", 1)[-1]) or spike.is_mps_oom(ln) for ln in str(text).splitlines())


def _failure_text(e):
    """Type names and messages of `e` and every exception behind it (cause, context, Ray's controller and worker
    failures), so a wrapped CheckpointManagerInitializationError is still found."""
    seen, todo, parts = set(), [e], []
    while todo:
        x = todo.pop()
        if x is None or id(x) in seen:
            continue
        seen.add(id(x))
        parts.append("%s: %s" % (type(x).__name__, x))
        todo += [x.__cause__, x.__context__, getattr(x, "controller_failure", None)]
        todo += list((getattr(x, "worker_failures", None) or {}).values())
    return "\n".join(parts)


def worker_refusal(text):
    """The message of a worker's SystemExit in a (multi-line) Ray failure text (Ray Train reports it as a
    `SystemExit: <message>` line), or None."""
    for line in reversed(str(text).splitlines()):
        line = line.strip()
        if line.startswith("SystemExit: ") and line[len("SystemExit: "):].strip():
            return line[len("SystemExit: "):].strip()
    return None


def fit_or_exit(trainer, run_dir=None, rung="none"):
    """trainer.fit(), with Ray's failures turned into one console line (SystemExit):
    - a worker's MPS out-of-memory exit: the rung line (pilot.runs.oom_message), which wins over the others; a raw MPS
      out-of-memory error that no mps_oom_exit wrapped still gets the rung line of `rung` (rerunning would only resume
      into the same allocation), so the compute orchestrator walks the ladder;
    - a snapshot under run_dir whose checkpoints are gone (CheckpointManagerInitializationError): name the dir to delete;
    - a worker's refusal (a SystemExit in train_loop_per_worker, e.g. the tokenizer-config backstop or check_env):
      its own message, which names its own fix, never resume advice (04-REVIEW CR-01);
    - any other training failure: rerun the same command to resume from the last epoch checkpoint in run_dir.
    Exceptions that are not Ray training failures propagate unchanged."""
    try:
        return trainer.fit()
    except Exception as e:
        failed = isinstance(e, ray.train.TrainingFailedError)
        line = runs.failure_line(str(e)) if failed else None
        if line is not None:
            raise SystemExit(line) from None
        if failed and raw_mps_oom("%s\n%s" % (e, _failure_text(e))):
            raise SystemExit(runs.oom_message(None, None, rung, "outside the training step")) from None
        if "CheckpointManagerInitializationError" in _failure_text(e):
            raise SystemExit("%s holds a Ray snapshot whose checkpoints are gone; delete that dir to start the run "
                             "fresh" % run_dir) from None
        if not failed:
            raise
        refusal = worker_refusal(_failure_text(e))
        if refusal is not None:                             # it names its own fix; a rerun would refuse again
            raise SystemExit(refusal) from None
        raise SystemExit("training failed (%s); rerun the same command to resume from the last epoch checkpoint in %s"
                         % (runs.last_line(str(e)), run_dir)) from None


def code_state():
    """{"git_sha", "git_dirty", "git_diff_sha256"} of the repo this script runs from; all None without git. Both
    git_dirty and git_diff_sha256 read the same files (04-REVIEW IN-08): tracked changes outside .planning/ (planning
    docs are not code) plus untracked, unignored *.py files (a new module an import can pick up). git_diff_sha256 is
    the sha256 of `git diff HEAD --binary` over them, then each untracked .py file's name and bytes, so it is the
    sha256 of an empty diff exactly when git_dirty is False, and an uncommitted edit between a kill and its rerun
    changes it while HEAD stays put. Recorded per run segment, never part of the run_id (a code fix must not orphan a
    long epoch's checkpoint)."""
    def git(*args):
        r = subprocess.run(("git",) + args, cwd=REPO_ROOT, capture_output=True, timeout=60)
        if r.returncode:
            raise OSError(r.stderr.decode("utf-8", "replace"))
        return r.stdout
    code = ("--", ".", ":(exclude).planning")
    try:
        sha = git("rev-parse", "HEAD").decode("ascii").strip()
        tracked = git("status", "--porcelain", "--untracked-files=no", *code).strip()
        new = sorted(n for n in git("ls-files", "--others", "--exclude-standard", "-z", "--", "*.py",
                                    ":(exclude).planning").split(b"\0") if n)
        digest = hashlib.sha256(git("diff", "HEAD", "--binary", *code))
        for n in new:
            with open(os.path.join(REPO_ROOT, os.fsdecode(n)), "rb") as f:
                digest.update(b"\0" + n + b"\0" + f.read())
        return {"git_sha": sha, "git_dirty": bool(tracked or new), "git_diff_sha256": digest.hexdigest()}
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return {"git_sha": None, "git_dirty": None, "git_diff_sha256": None}


def code_changed(segments):
    """code_changed_on_resume of a run record: True when the segments ran different code (another HEAD or another
    uncommitted diff), or when a resumed run cannot tell (git unavailable in a segment). Flagged, never refused
    (RESEARCH A3). A segment of an older train state without git_diff_sha256 counts as unknown."""
    if len(segments) < 2:
        return False
    if any(s.get("git_sha") is None or s.get("git_diff_sha256") is None for s in segments):
        return True
    return len({(s["git_sha"], s["git_diff_sha256"]) for s in segments}) > 1


def provenance(entry, manifest_path, base, base_dir, ident=None):
    """Checkpoint provenance: split, manifest and base model by repo-relative path and sha256, never a local path."""
    return {"data": entry["path"], "data_role": entry["role"], "data_sha256": entry["sha256"],
            "manifest": repo_relpath(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "manifest_is_committed": manifest_is_committed(manifest_path),
            "base": identity("multilingual" if base == "multilingual" else base_dir) if ident is None else ident}


def epoch_config(cfg, max_len, head_max_len, ray_cfg, epoch):
    """Config of an intermediate epoch checkpoint: marked fine-tuned and uncalibrated, and traceable to its data
    (provenance, run record) like the final one, so a best-epoch pick from ray_results stays auditable. Every selected
    message trains (no calibration holdout); the temperature is reset to UNCALIBRATED_TEMPERATURE with no
    temperature_by_options, never inherited from the base (RESEARCH Pitfall 5): postfit.py fits T on val (LAYA-05)."""
    return dict(cfg, max_len=max_len, head_max_len=head_max_len, fine_tuned=True,
                temperature=list(UNCALIBRATED_TEMPERATURE), temperature_by_options={},
                training=dict(cfg.get("training") or {}, ray_train=ray_cfg, epoch=epoch, calibrated=False))


def identity_values(a, r, entry, base_name, base_ident, postfit_sha256=None, mask_tags=None, fewshot=None):
    """The pilot.runs.IDENTITY values of a command: parsed args `a`, resolve(a) `r`, the verified split `entry` and the
    base identity. Paths enter only as their verified digests, aliases (--gpu) only as their resolved value. A local
    base also enters by the digest of its tokenizer/ and encoder/ files (the tokenizer builds every training item;
    04-REVIEW WR-01); the pinned base's are fixed by the pin, so base_files_sha256 is None there and
    pilot.runs.canonical leaves it out, which keeps every pinned run id unchanged. stop_after_epoch and postfit_sha256
    (the sha256 of the --postfit record) are None and left out the same way unless a final seed sets them; mask_tags
    (a masked base's sorted held-out tags) and fewshot (the pilot.fewshot.binding of a Phase 9 run) likewise."""
    return {"split": os.path.splitext(os.path.basename(entry["path"]))[0],
            "data_sha256": entry["sha256"], "data_role": entry["role"],
            "base_model_sha256": base_ident["model_sha256"],
            "base_config_sha256": (pins.LAYA["sha256"]["rl_agent_config.json"] if base_name == "multilingual"
                                   else base_ident["config_sha256"]),
            "base_files_sha256": (None if base_name == "multilingual"
                                  else dumps.files_digest(base_ident["files_sha256"])),
            "laya_version": la.LAYA_VERSION,
            "device": r["device"], "amp": r["amp"], "grad_checkpointing": r["grad_checkpointing"],
            "neg_seed": r["neg_seed"], "adamw_foreach": r["adamw_foreach"], "adamw_fused": r["adamw_fused"],
            "weight_decay": WEIGHT_DECAY, "clip_grad_norm": CLIP_GRAD_NORM,
            "workers": a.workers, "epochs": a.epochs, "micro_batch": a.micro_batch, "grad_accum": a.grad_accum,
            "encoder_lr": a.encoder_lr, "head_lr": a.head_lr, "loss": a.loss, "neg": a.neg,
            "freeze_encoder": a.freeze_encoder, "freeze_embeddings": a.freeze_embeddings, "max_len": a.max_len,
            "head_max_len": a.head_max_len, "max_messages": a.max_messages,
            "shuffle_options": a.shuffle_options, "seed": a.seed,
            "stop_after_epoch": a.stop_after_epoch, "postfit_sha256": postfit_sha256,
            "mask_tags": mask_tags, "fewshot": fewshot}


def fewshot_base_of(base_dir, base_ident):
    """{"run_id", "canonical"} of a continued run's base, read from its rl_agent_config.json bytes, which must hash to
    the base identity's config_sha256 (the digest the run identity and the workers bind); SystemExit naming the fix for
    other bytes, a checkpoint without a ray_train.py run id and canonical config, or a run id that is not the one its
    canonical config names."""
    with open(os.path.join(base_dir, "rl_agent_config.json"), "rb") as f:
        raw = f.read()
    if sha256_bytes(raw) != base_ident.get("config_sha256"):
        raise SystemExit("--base rl_agent_config.json changed while it was read (sha256 %s, identity %s); rerun the "
                         "command" % (sha256_bytes(raw), base_ident.get("config_sha256")))
    try:
        rt = (json.loads(raw.decode("utf-8")).get("training") or {}).get("ray_train") or {}
    except (ValueError, AttributeError):
        rt = {}
    rid, canon = rt.get("run_id"), rt.get("canonical")
    if not isinstance(rid, str) or not isinstance(canon, dict):
        raise SystemExit("--base %s holds no ray_train.py run id and canonical config; a continued run starts from a "
                         "--fewshot-base checkpoint" % repo_relpath(base_dir))
    if runs.run_id(canon) != rid:
        raise SystemExit("--base %s names run %s, but its canonical config is run %s; a continued run starts from an "
                         "unedited --fewshot-base checkpoint" % (repo_relpath(base_dir), rid, runs.run_id(canon)))
    return {"run_id": rid, "canonical": canon}


def fewshot_proof(c, per_q, n_items, counted):
    """The run record's fewshot entry of a Phase 9 run (None otherwise), from the items laya built (per_q =
    la.items_per_question of them, n_items their number, counted the torch-free count); SystemExit, on every rank
    before training, when the masking proof fails (D-04, FEW-01, FEW-02).

    A masked base must hold no item of a held-out question: {protocol_sha256, stage base, tags, held_out_items (all
    0), items_per_question}. A continued run must hold exactly one item of each held-out question per support row (so
    replay rows contributed none) and the item count of pilot.fewshot.laya_item_count: {protocol_sha256, stage
    continued, draw, k, digest, tags, support_rows, replay_rows, new_tag_items, items_per_question, base {run_id,
    canonical}}. Counts, tags and digests only, never a uid."""
    b = c.get("fewshot")
    if not b:
        return None
    if b["stage"] == "base":
        tags = list(c["mask_tags"])
        held = {t: per_q.get(t, 0) for t in tags}
        leaked = ["%s %d" % (t, n) for t, n in held.items() if n]
        if leaked:
            raise SystemExit("the masked base built items of held-out question(s) (%s); refusing to train: masking "
                             "deletes the held-out keys before negative sampling (FEW-01)" % ", ".join(leaked))
        return {"protocol_sha256": b["protocol_sha256"], "stage": "base", "tags": tags, "held_out_items": held,
                "items_per_question": per_q}
    rows = c["fewshot_rows"]
    tags, n_sup = list(rows["tags"]), len(rows["support"])
    new = {t: per_q.get(t, 0) for t in tags}
    off = ["%s %d" % (t, n) for t, n in new.items() if n != n_sup]
    if off:
        raise SystemExit("the continued run built items of held-out question(s) (%s) for %d support rows; refusing to "
                         "train: exactly one per support row, none from a replay row (the frozen Laya item rule)"
                         % (", ".join(off), n_sup))
    if n_items != counted:
        raise SystemExit("the continued run built %d items, the frozen Laya item rule gives %d "
                         "(pilot.fewshot.laya_item_count); refusing to train" % (n_items, counted))
    return {"protocol_sha256": b["protocol_sha256"], "stage": "continued", "draw": b["draw"], "k": b["k"],
            "digest": b["digest"], "tags": tags, "support_rows": n_sup, "replay_rows": len(rows["replay"]),
            "new_tag_items": new, "items_per_question": per_q, "base": c["fewshot_base"]}


def worker_device(c):
    """(torch device, env seen) of this worker: explicit mps (Ray has no MPS device manager), else Ray's device."""
    # a local torch: cloudpickle cannot ship torch.backends.mps when it is reached through the module-global torch
    import torch

    if c["device"] == "mps":
        env_seen = check_env(c["expect_env"])               # the spike's env must have reached this process
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS is not available in this Ray worker; run --device mps on Apple silicon with the "
                             "locked torch build, or use --device cpu")
        return torch.device("mps"), env_seen
    device = ray.train.torch.get_device()                   # cpu, or cuda:<local rank> with use_gpu=True
    if device.type != c["device"]:
        raise SystemExit("Ray gave this worker device %s, expected %s; check the cluster's GPUs or use --device cpu"
                         % (device.type, c["device"]))
    return device, {}


def memory_sampler(device):
    """(reset, sync, sample, current, peak) for per-epoch memory peaks, sampled like the Phase 2 spike (torch.mps keeps
    no max_memory_allocated).

    peak["driver"]: torch.mps.driver_allocated_memory() on mps, torch.cuda.max_memory_allocated() on cuda, None on cpu
    (never 0); peak["rss"]: the worker's resident set; peak["total"]: driver + rss (the spike's peak_memory_bytes
    basis), so the peak RSS on cpu.
    """
    proc, kind, peak = psutil.Process(), device.type, {}

    def reset():
        peak.update(total=0, driver=None, rss=0)
        if kind == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def sync():
        if kind == "mps":
            torch.mps.synchronize()
        elif kind == "cuda":
            torch.cuda.synchronize(device)

    def sample():
        driver = (torch.mps.driver_allocated_memory() if kind == "mps"
                  else torch.cuda.max_memory_allocated(device) if kind == "cuda" else None)
        rss = proc.memory_info().rss
        peak["total"], peak["rss"] = max(peak["total"], (driver or 0) + rss), max(peak["rss"], rss)
        if driver is not None:
            peak["driver"] = driver if peak["driver"] is None else max(peak["driver"], driver)

    def current():
        return (torch.mps.current_allocated_memory() if kind == "mps"
                else torch.cuda.memory_allocated(device) if kind == "cuda" else None)

    reset()
    return reset, sync, sample, current, peak


def _rewrite_config(path, cfg):
    """Rewrite a just-saved checkpoint's config in laya's own layout (json indent 2), e.g. to add its save time."""
    tmp = os.path.join(path, "rl_agent_config.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, os.path.join(path, "rl_agent_config.json"))


def load_resume(c, rank):
    """(train state, this rank's RNG state, save.json of that epoch, train_state.pt bytes, load seconds), or "final"
    when the run already finished, or None when storage/<run_id> holds no checkpoint (a fresh start).

    Every rank reads the same train_state.pt and checks it (rank-symmetric: ray.train.report is a barrier, so a refusal
    on one rank only would leave the others waiting). Every load is weights_only (tensors and primitives only)."""
    ck = ray.train.get_checkpoint()
    if ck is None:
        return None
    with ck.as_directory() as d:
        if os.path.exists(os.path.join(d, FINAL_MARK)):
            return "final"
        t = time.perf_counter()
        path = os.path.join(d, "train_state.pt")
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        runs.check_state(state, c["run_id"], c["canonical"])
        rng_state = torch.load(os.path.join(d, "rng-rank%d.pt" % rank), map_location="cpu", weights_only=True)
        with open(os.path.join(d, SAVE_META), encoding="utf-8") as f:
            meta = json.load(f)
        return state, rng_state, meta, os.path.getsize(path), time.perf_counter() - t


def device_rng_state(device):
    """The device RNG state (mps or cuda), None on cpu."""
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return torch.cuda.get_rng_state(device) if device.type == "cuda" else None


def set_rng_states(rs, device, rng):
    """Restore torch's CPU RNG, the device RNG and the option-order random.Random from a saved rng-rank<r>.pt."""
    torch.set_rng_state(rs["torch"])
    if device.type == "mps":
        torch.mps.set_rng_state(rs["device"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state(rs["device"], device)
    version, internal, gauss = rs["option"]
    rng.setstate((version, tuple(internal), gauss))         # random.Random wants its (version, tuple, gauss) shape


def training_done(state, epochs):
    """True when a resumed train state already holds the last epoch (a rerun after an OOM in the final save): the
    epoch loop is empty and the rerun only writes the final checkpoint."""
    return state is not None and state["epoch"] >= epochs


def restore_optimizer(state, epochs, opt, sched, scaler, rung):
    """Restore AdamW, the cosine scheduler and the GradScaler of a resumed train state; returns True, or False without
    touching them when no epoch is left to train (04-REVIEW WR-01): the AdamW moments (about 2.6 GB of the measured
    3.86 GB full-encoder train state) would move to the device only to be freed before the final save, on the machine
    that just ran out of memory writing the final checkpoint."""
    if training_done(state, epochs):
        return False
    with mps_oom_exit(None, None, rung, "while restoring the optimizer state of epoch %d" % state["epoch"]):
        opt.load_state_dict(state["opt"])                   # moves the AdamW moments to the device
    sched.load_state_dict(state["sched"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    return True


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def prune_exports(exports_dir, start_epoch):
    """Delete every export a dead segment left beyond the epoch this segment resumes from (and any partial .tmp);
    returns the names removed. An export is written before its epoch's train state registers, so a kill in between
    leaves an export of weights the resumed run never continues from (MPS is not bitwise deterministic); the rerun
    retrains that epoch and writes it again. Rank 0 only, before training. A deletion that fails refuses the segment
    before training (04-REVIEW IN-09): a half-deleted stale export must never be kept as the retrained epoch's."""
    if not os.path.isdir(exports_dir):
        return []
    gone = []
    for n in sorted(os.listdir(exports_dir)):
        keep = len(n) == 9 and n.startswith("epoch-") and n[6:].isdigit() and int(n[6:]) <= start_epoch
        if n.startswith("epoch-") and not keep:
            try:
                shutil.rmtree(os.path.join(exports_dir, n))
            except OSError as e:
                raise SystemExit("cannot delete the stale export exports/%s of a killed segment (%s); delete it, then "
                                 "rerun the same command to resume" % (n, e)) from None
            gone.append(n)
    return gone


def export_record(record, name):
    """The run record an epoch export embeds: a pre-save snapshot (04-REVIEW IN-06). It is written before the epoch's
    train state, so the export already lists itself in exports, while what only the save measures (this epoch's save_s,
    train_state_bytes, export_bytes) is None, never the previous epoch's value. Phase 5 reads these exports."""
    rows = record["epochs"]
    return dict(record, exports=sorted(set(record["exports"]) | {"exports/" + name}), train_state_bytes=None,
                export_bytes=None, epochs=rows[:-1] + [dict(rows[-1], save_s=None)] if rows else [])


def write_export(inner, tok, ecfg, run_dir, name):
    """Rank 0's uncalibrated laya export of a completed epoch at run_dir/exports/<name>, outside Ray's checkpoint
    rotation (Phase 5 picks the epoch on val without retraining). The segment's start removed every export past the
    resumed epoch (prune_exports), so an existing one here is a stale export that survived: refused, never kept."""
    final = os.path.join(run_dir, "exports", name)
    if os.path.lexists(final):
        raise SystemExit("exports/%s already exists before its epoch was saved (a stale export of a killed segment); "
                         "delete it, then rerun the same command to resume" % name)
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)                   # a partial export of a killed save (pruned at the start)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    la.save_checkpoint(inner, tok, ecfg, tmp)
    os.rename(tmp, final)                                    # whole or absent, never half-written
    return final


def train_loop_per_worker(c):
    require_driver_node(ray.get_runtime_context().get_node_id(), c["driver_node_id"])
    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    if rank == 0:
        print("run_id %s" % c["run_id"], flush=True)
    device, env_seen = worker_device(c)
    torch.set_num_threads(c["cpus_per_worker"])
    torch.manual_seed(c["seed"])

    # Resume (LAYA-04): rerunning the same command hands this run's latest checkpoint back through Ray
    resume = load_resume(c, rank)
    if resume == "final":                                   # finished: Ray returns the final checkpoint to the driver
        if rank == 0:
            print("run %s already finished; its final checkpoint is returned without training" % c["run_id"],
                  flush=True)
        return
    state, rng_state, save_meta, state_bytes, resume_load_s = resume or (None, None, None, None, None)
    run_dir = os.path.join(c["storage"], c["run_id"])

    # the base must still be the bytes the driver recorded in provenance.base (load_checkpoint verifies nothing)
    t_load = time.perf_counter()
    verify_files(c["base_dir"], c["base_sha256"], "base checkpoint")
    model, tok, cfg = la.load_checkpoint(c["base_dir"])
    check_tokenizer_config_rewritten(c["base_dir"], "base checkpoint")    # laya only warns (04-REVIEW IN-01)
    load_s = time.perf_counter() - t_load
    max_len = c["max_len"] or cfg.get("max_len", 512)
    head_max_len = c["head_max_len"] or cfg.get("head_max_len", 192)
    parallel = la.uses_parallel_layout(cfg)

    # Every rank builds the same items (fixed seed), then takes its own shard of the training items. Every selected
    # message trains: there is no calibration holdout (LAYA-05: T is fitted on message-level val data by postfit.py).
    rows = read_verified_rows(c["data"], c["data_sha256"])
    if c["max_messages"]:
        random.Random(c["seed"]).shuffle(rows)
        rows = rows[:c["max_messages"]]
    fs_rows = c.get("fewshot_rows")
    if fs_rows:                                             # Phase 9 continued: the frozen support and replay rows
        sel = {"support": set(fs_rows["support"]), "replay": set(fs_rows["replay"])}
        train_rows = fewshot.laya_rows(rows, sel, fs_rows["tags"], c["neg_seed"], c["neg"])
        counted = fewshot.laya_item_count(rows, sel, fs_rows["tags"], c["neg"])
    else:
        # a Phase 9 masked base deletes the held-out keys BEFORE negative sampling (laya builds a negative item for a
        # False value, and sampling after the deletion can never keep a held-out label)
        train_rows = fewshot.masked_rows(rows, c["mask_tags"]) if c.get("mask_tags") else rows
        counted = runs.neg_items(train_rows, c["neg"])
        if c["neg"]:
            train_rows = spike.neg8_rows(train_rows, c["neg_seed"], c["neg"])
    train_items, skipped = la.items_from_rows(tok, train_rows, max_len, head_max_len)
    # the masking proof (D-04): counted from the items laya built, refused on every rank before training
    fs_proof = fewshot_proof(c, la.items_per_question(train_items), len(train_items), counted) \
        if c.get("fewshot") else None
    # driver-local absolute paths (data, base_dir, storage), the run's node id, the raw env dict and the frozen uid
    # lists (fewshot_rows) stay out of every checkpoint (the env is recorded only as KEY=VALUE strings in the run record)
    ray_cfg = {k: v for k, v in c.items()
               if k not in ("data", "base_dir", "storage", "driver_node_id", "expect_env", "fewshot_rows")}
    per_rank = len(train_items) // world                    # equal shards keep DDP collectives in step
    if per_rank == 0:                                       # same on every rank: never publish an untrained "fine-tune"
        raise SystemExit("no training items per rank (%d items, %d workers); use more messages or fewer workers"
                         % (len(train_items), world))
    shard = train_items[rank * per_rank:(rank + 1) * per_rank]
    if rank == 0:
        print("messages %d | neg %s: items train %d (counted %d, skipped %r, %d per rank x %d)"
              % (len(train_rows), c["neg"] or "all", len(train_items), counted, skipped, per_rank, world), flush=True)
        if fs_proof is not None:
            print("few-shot %s: held-out tags %s, their items %s" % (
                fs_proof["stage"], ",".join(fs_proof["tags"]),
                fs_proof["held_out_items"] if fs_proof["stage"] == "base" else fs_proof["new_tag_items"]), flush=True)

    if c["freeze_encoder"]:
        la.freeze_encoder(model)
    elif c["grad_checkpointing"] and not la.enable_gradient_checkpointing(model):
        raise SystemExit("encoder has no gradient checkpointing; pass --no-grad-checkpointing to train without it")
    if c["freeze_embeddings"]:                              # rung embeddings_frozen: after checkpointing, as the spike
        la.freeze_embeddings(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print("rung %s%s | trainable params %d of %d" % (c["rung"], " (head-only, weaker)" if runs.weaker(c["rung"])
                                                          else "", trainable_params, total_params), flush=True)
    # the model move (and a resume's weights) allocate on the device outside the training step (04-REVIEW WR-01); a
    # rerun with no epoch left only writes the final checkpoint, so its OOM says to rerun, never the next rung (a rung
    # flag would change the run_id and retrain every epoch)
    # a final seed stops after the val-chosen epoch K of the --epochs schedule (--stop-after-epoch): the cosine T_max and
    # the sigma anneal below keep c["epochs"], so epochs 1..K follow the selection run's own trajectory (RESEARCH A3)
    stop = c["stop_after_epoch"] or c["epochs"]
    move = runs.STAGE_FINALIZE_MOVE if training_done(state, stop) else "while moving the model to the device"
    with mps_oom_exit(None, None, c["rung"], move):
        if state is not None:                               # fp32 weights into the CPU model: one copy on the device
            t = time.perf_counter()
            model.load_state_dict(state["model"])
            resume_load_s += time.perf_counter() - t
        model = ray.train.torch.prepare_model(model, move_to_device=device,
                                              parallel_strategy_kwargs={"find_unused_parameters": True})
    amp = c["amp"]                                          # decoupled from checkpointing (02-REVIEW CR-01)
    gc.collect()                                            # drop the CPU copy before any memory sampling
    require_param_device(model, device.type)
    inner = model.module if hasattr(model, "module") else model
    inner.train()
    if c["freeze_encoder"]:
        inner.encoder.eval()

    groups = [{"params": [p for n, p in inner.named_parameters()
                          if not n.startswith("encoder.") and p.requires_grad], "lr": c["head_lr"]}]
    if not c["freeze_encoder"]:
        groups.insert(0, {"params": [p for n, p in inner.named_parameters()
                                     if n.startswith("encoder.") and p.requires_grad], "lr": c["encoder_lr"]})
    # foreach/fused only where resolve() pins them (mps: the single-tensor loop the spike measured)
    impl = {k: c["adamw_" + k] for k in ("foreach", "fused") if c["adamw_" + k] is not None}
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY, **impl)
    mb, accum = c["micro_batch"], c["grad_accum"]
    steps_per_epoch = -(-len(shard) // mb)
    updates = c["epochs"] * -(-steps_per_epoch // accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    params = [p for g in groups for p in g["params"]]
    rng = random.Random(c["seed"] + rank)
    shuffle = ("choice",) if c["shuffle_options"] else ()
    mem_reset, mem_sync, mem_sample, mem_current, peak = memory_sampler(device)

    start_epoch, rows_out, segments, resume_peak = 0, [], [], None
    if state is not None:
        # after the optimizer and scheduler exist (the scheduler's constructor sets the lr the saved state restores)
        t = time.perf_counter()
        restored = restore_optimizer(state, stop, opt, sched, scaler, c["rung"])
        set_rng_states(rng_state, device, rng)
        start_epoch, rows_out, segments = state["epoch"], list(state["history"]), list(state["segments"])
        rows_out[-1]["save_s"] = save_meta["save_s"]       # the saved epoch's own save time, written after its state
        del state, rng_state
        gc.collect()
        mem_sync()
        resume_load_s += time.perf_counter() - t
        mem_sample()
        resume_peak = peak["total"]
        if rank == 0:
            print("resuming %s after epoch %d (segment %d, load %.1fs)%s"
                  % (c["run_id"], start_epoch, len(segments), resume_load_s,
                     "" if restored else "; no epoch left: it only writes the final checkpoint, the optimizer state "
                                         "is not loaded"),
                  flush=True)
    segment = dict(zip(runs.SEGMENT_FIELDS, (len(segments), start_epoch, start_epoch > 0, c["code"]["git_sha"],
                                             c["code"]["git_dirty"], c["code"]["git_diff_sha256"], device.type,
                                             resume_load_s, resume_peak)))
    segments.append(segment)
    exports_dir = os.path.join(run_dir, "exports")
    record = {"device": device.type,
              "param_device": ",".join(sorted({p.device.type for p in inner.parameters()})),
              "torch_threads": torch.get_num_threads(),
              "env": runs.env_strings(env_seen) if device.type == "mps" else [],
              "grad_checkpointing": c["grad_checkpointing"], "amp": amp,
              "adamw_foreach": c["adamw_foreach"], "adamw_fused": c["adamw_fused"],
              "recommended_max_memory_bytes": torch.mps.recommended_max_memory() if device.type == "mps" else None,
              "rung": c["rung"], "fallback": c["rung"] != "none", "weaker": runs.weaker(c["rung"]),
              "spike_departures": runs.spike_departures(c, max_len),
              "trainable_params": trainable_params, "total_params": total_params,
              "neg": c["neg"], "neg_seed": c["neg_seed"], "n_train_messages": len(train_rows),
              "n_train_items": len(train_items), "n_train_items_counted": counted, "items_skipped": skipped,
              "load_s": load_s, "calib_s": None, "final_save_s": None, "run_id": c["run_id"],
              "canonical": c["canonical"], "segments": segments,
              "code_changed_on_resume": code_changed(segments),
              "exports": [], "train_state_bytes": state_bytes, "export_bytes": None, "save_s_includes_persist": True,
              # a final seed's binding to the val selection (LAYA-08): the record's sha256, chosen settings and T
              "postfit": ({k: c["postfit"][k] for k in ("sha256", "chosen", "temperature")} if c["postfit"]
                          else None),
              # a Phase 9 run's masking proof and binding (counts, tags and digests only, never a uid), else None
              "fewshot": fs_proof,
              "epochs": rows_out}

    def note_exports():
        names = sorted(n for n in os.listdir(exports_dir) if len(n) == 9 and n.startswith("epoch-")) \
            if os.path.isdir(exports_dir) else []
        record["exports"] = ["exports/" + n for n in names]   # relative names only, never the storage path
        if names:
            record["export_bytes"] = dir_bytes(os.path.join(exports_dir, names[-1]))

    if rank == 0:                                           # a dead segment's exports past the resumed epoch (WR-02)
        gone = prune_exports(exports_dir, start_epoch)
        if gone:
            print("dropped export(s) %s of a killed segment beyond epoch %d" % (", ".join(gone), start_epoch),
                  flush=True)
    note_exports()
    last = {"epoch": start_epoch, "loss": rows_out[-1]["loss"] if rows_out else None}
    for epoch in range(start_epoch, stop):
        t_epoch = time.time()
        order = list(shard)
        random.Random(c["seed"] + 1000 * epoch + rank).shuffle(order)
        sigma = la.sigma_at(epoch, c["epochs"], 0.4, 0.1)
        total, n, steps = 0.0, 0, 0
        mem_reset()
        mem_sync()
        swap0, t0 = spike.swap_used(), time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for s in range(0, len(order), mb):
            batch = la.collate_items([[la.encode_item(tok, it, max_len, head_max_len,
                                                      la.draw_option_order(it, rng, shuffle), parallel)
                                       for it in order[s:s + mb]]], tok.pad_token_id)
            step_now = (n + 1) % accum == 0 or s + mb >= len(order)
            # skip the gradient all-reduce on accumulation steps
            sync = nullcontext() if step_now or not hasattr(model, "no_sync") else model.no_sync()
            with mps_oom_exit(epoch + 1, n + 1, c["rung"]), sync:
                logits = la.forward(model, batch, device, amp, c["freeze_encoder"])
                mask, target = batch["marker_mask"].to(device), batch["target"].to(device)
                if c["loss"] == "rlcd":
                    loss = la.rlcd_loss(logits, target, mask, batch["qtype"].to(device), sigma)
                else:
                    loss = la.soft_ce_loss(logits, target, mask)
                (scaler.scale(loss / accum) if scaler else loss / accum).backward()
            mem_sample()
            n += 1
            if step_now:
                with mps_oom_exit(epoch + 1, n, c["rung"]):   # the step closes micro-batch n
                    if scaler:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, CLIP_GRAD_NORM)
                    (scaler.step(opt), scaler.update()) if scaler else opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                steps += 1
                mem_sample()
            total += loss.item()
        mem_sync()
        train_s, swap_delta = time.perf_counter() - t0, spike.swap_used() - swap0
        row = dict(zip(runs.EPOCH_FIELDS, (epoch + 1, total / max(1, n), train_s, 0.0, steps, len(shard),
                                           peak["total"], peak["driver"], peak["rss"], mem_current(),
                                           swap_delta, segment["segment"], segment["resumed"])))
        rows_out.append(row)
        metrics = {"epoch": epoch + 1, "loss": row["loss"], "epoch_seconds": round(time.time() - t_epoch, 1),
                   "train_s": train_s}
        if rank == 0:
            print("epoch %d/%d%s loss %.4f train %.1fs peak %.2f GiB swap %+.0f MiB"
                  % (epoch + 1, c["epochs"], " (stop after %d)" % stop if stop != c["epochs"] else "", row["loss"],
                     train_s, row["peak_total_bytes"] / 2**30, swap_delta / 2**20), flush=True)

        # Epoch checkpoint (LAYA-04): rank 0's fp32 train state + every rank's RNG; never the fp16 laya export
        if device.type == "mps":
            torch.mps.empty_cache()                         # the allocator cache before the state copies to CPU
        gc.collect()
        name = "epoch-%03d" % (epoch + 1)
        with tempfile.TemporaryDirectory() as d:
            t_save = time.perf_counter()
            if rank == 0:
                with mps_oom_exit(epoch + 1, None, c["rung"], "while saving the checkpoint"):
                    if c["export_epochs"]:
                        ecfg = epoch_config(cfg, max_len, head_max_len,
                                            dict(ray_cfg, record=export_record(record, name)), epoch + 1)
                        write_export(inner, tok, ecfg, run_dir, name)
                        note_exports()
                    path = os.path.join(d, "train_state.pt")
                    torch.save({"kind": "train_state", "run_id": c["run_id"], "canonical": c["canonical"],
                                "epoch": epoch + 1, "model": inner.state_dict(), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                                "history": rows_out, "segments": segments}, path)
                record["train_state_bytes"] = os.path.getsize(path)
            torch.save({"torch": torch.get_rng_state(), "device": device_rng_state(device), "option": rng.getstate()},
                       os.path.join(d, "rng-rank%d.pt" % rank))
            row["save_s"] = time.perf_counter() - t_save    # the state above holds 0.0; save.json carries the time
            if rank == 0:
                with open(os.path.join(d, SAVE_META), "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch + 1, "save_s": row["save_s"]}, f)
            ray.train.report(metrics, checkpoint=Checkpoint.from_directory(d),
                             checkpoint_dir_name="epoch-%03d" % (epoch + 1))
            # plus Ray's persist (the train-state copy into storage, the rotation): a per-epoch cost the compute
            # projection charges per epoch (04-REVIEW IN-02); the next epoch's state and the final record carry it,
            # save.json (the resumed epoch's row) only the time up to the report
            row["save_s"] = time.perf_counter() - t_save
        if c["fail_after_epoch"] == epoch + 1:              # test hook: every rank fails after the checkpoint
            raise RuntimeError("simulated failure after epoch %d (--fail-after-epoch)" % (epoch + 1))
        last = metrics

    # Publish the final checkpoint from rank 0: uncalibrated (T = 1; postfit.py fits T on val later, LAYA-05), or, for a
    # --postfit final seed, serving the record's val-fitted T bit for bit (a JSON float round trip, never rounded), which
    # laya.load then serves (LAYA-08). record["calib_s"] stays None: training never calibrates.
    del opt, sched, scaler, params, groups                 # free optimizer state before the final save
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()                             # hand the freed AdamW blocks back before the final save
    if rank == 0:
        inner.eval()
        out_cfg = dict(cfg, max_len=max_len, head_max_len=head_max_len, fine_tuned=True,
                       temperature=list(UNCALIBRATED_TEMPERATURE), temperature_by_options={})
        out_cfg["training"] = dict(out_cfg.get("training") or {}, ray_train=dict(ray_cfg, record=record),
                                   epoch=stop, calibrated=False)
        if c["postfit"]:                                    # [choice, score, noul]: only noul is val-fitted
            out_cfg["temperature"] = [1.0, 1.0, c["postfit"]["temperature"]]
            out_cfg["training"].update(calibrated=True, postfit=c["postfit"])
        with tempfile.TemporaryDirectory() as d:
            t_save = time.perf_counter()
            with mps_oom_exit(None, None, c["rung"], runs.STAGE_FINAL_SAVE):
                la.save_checkpoint(inner, tok, out_cfg, d)
            record["final_save_s"] = time.perf_counter() - t_save
            _rewrite_config(d, out_cfg)                      # the record now holds the final save time
            with open(os.path.join(d, FINAL_MARK), "w", encoding="utf-8") as f:
                json.dump({"kind": "final", "run_id": c["run_id"]}, f)   # a rerun returns this without training
            ray.train.report(last, checkpoint=Checkpoint.from_directory(d), checkpoint_dir_name="final")
    else:
        ray.train.report(last)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="multilingual")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None,
                    help="training device (default cpu); mps trains in one Ray worker on the Apple GPU")
    ap.add_argument("--gpu", action="store_true", help="alias for --device cuda")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                    help="CUDA mixed precision (default: on for cuda, off elsewhere)")
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                    help="gradient checkpointing on the encoder and the Laya head for full-encoder runs (default on)")
    ap.add_argument("--cpus-per-worker", type=int, default=None, help="CPUs and torch threads per worker "
                    "(default 4 on mps, the spike's timing basis; 1 otherwise)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--encoder-lr", type=float, default=2.5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--loss", choices=("rlcd", "soft-ce"), default="rlcd")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="fallback rung 3: train the head only (labelled head-only, weaker)")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="fallback rung 1: freeze only the token embeddings")
    ap.add_argument("--shuffle-options", action="store_true")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--head-max-len", type=int, default=None)
    ap.add_argument("--max-messages", type=int, default=None, help="subsample messages (smoke tests)")
    ap.add_argument("--neg", type=int, default=8,
                    help="negatives kept per training message; 0 = all 48 labels (data_full)")
    ap.add_argument("--neg-seed", type=int, default=None, help="negative-sampling seed; default --seed")
    ap.add_argument("--object-store-gb", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--storage", default=os.path.abspath("ray_results"),
                    help="Ray run storage (local dir; the script is single-node)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify --data against (default: the committed data/splits.json)")
    ap.add_argument("--print-run-id", action="store_true",
                    help="print the run id this command trains under (its Ray dir and resume key), then exit without "
                         "starting Ray")
    ap.add_argument("--export-epochs", action=argparse.BooleanOptionalAction, default=True,
                    help="write each epoch's uncalibrated laya export to storage/<run_id>/exports/epoch-NNN for Phase 5 "
                         "epoch selection (default on)")
    ap.add_argument("--stop-after-epoch", type=int, default=None,
                    help="stop after epoch K of the --epochs schedule (the cosine schedule and the sigma anneal keep "
                         "--epochs): the val-selected epoch of a final seed; 1..--epochs - 1, omit it for every epoch")
    ap.add_argument("--postfit", default=None,
                    help="the committed postfit record (results/postfit/laya.json): trains a final seed with the "
                         "val-chosen settings, refused unless every hyperparameter matches, and writes the val-fitted T "
                         "into the final config")
    ap.add_argument("--fewshot-base", action="store_true",
                    help="Phase 9: train a label-masked base of the frozen few-shot protocol (protocol.json); refused "
                         "unless every setting is the frozen one")
    ap.add_argument("--mask-tags", default=None,
                    help="with --fewshot-base: the held-out tags, comma-separated; refused unless they are the frozen "
                         "protocol's (the default)")
    ap.add_argument("--fewshot-draw", type=int, default=None,
                    help="Phase 9: continue the masked --base on the frozen support set of this draw plus its "
                         "replay set")
    ap.add_argument("--fewshot-k", type=int, default=None,
                    help="Phase 9: the frozen k of the --fewshot-draw support set")
    ap.add_argument("--fail-after-epoch", type=int, default=None, help=argparse.SUPPRESS)   # test hook, never an env var
    return ap


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    if a.epochs < 1:
        ap.error("--epochs must be at least 1")
    if a.neg < 0:
        ap.error("--neg must be 0 (all labels) or a positive count")
    # here, not in the worker: a bad value there fails after ray.init, the run dir and the fetch, with resume advice
    for flag, v in (("--workers", a.workers), ("--micro-batch", a.micro_batch), ("--grad-accum", a.grad_accum),
                    ("--cpus-per-worker", a.cpus_per_worker), ("--max-messages", a.max_messages),
                    # 0 would fall back to the config default while the run identity recorded 0 (IN-10)
                    ("--max-len", a.max_len), ("--head-max-len", a.head_max_len)):
        if v is not None and v < 1:
            ap.error("%s must be at least 1" % flag)
    if a.freeze_embeddings and a.freeze_encoder:
        ap.error("--freeze-encoder already freezes the embeddings; pass one of them")
    # K = --epochs has one spelling, the omitted flag (one run id per trajectory)
    if a.stop_after_epoch is not None and not 1 <= a.stop_after_epoch < a.epochs:
        ap.error("--stop-after-epoch must be at least 1 and below --epochs (%d); omit it to train every epoch"
                 % a.epochs)
    # Phase 9 (D-01): a few-shot mode is one frozen-protocol command; every other combination is an argument error
    continued = a.fewshot_draw is not None or a.fewshot_k is not None
    fs_flags = [f for f, on in (("--fewshot-base", a.fewshot_base), ("--fewshot-draw", a.fewshot_draw is not None),
                                ("--fewshot-k", a.fewshot_k is not None), ("--mask-tags", a.mask_tags is not None))
                if on]
    if (a.fewshot_draw is None) != (a.fewshot_k is None):
        ap.error("--fewshot-draw and --fewshot-k go together: a continued run names its frozen draw and k")
    if a.fewshot_base and continued:
        ap.error("--fewshot-base trains a masked base and --fewshot-draw continues one; pass one of them")
    if continued and (a.stop_after_epoch is not None or a.mask_tags is not None):
        ap.error("--fewshot-draw takes no --stop-after-epoch or --mask-tags: a continued run trains every frozen epoch "
                 "on the support and replay rows, the held-out tags labelled on support rows only")
    if a.mask_tags is not None and not a.fewshot_base:
        ap.error("--mask-tags masks a Phase 9 base only: pass it with --fewshot-base (the frozen protocol's tags)")
    if fs_flags and a.postfit:
        ap.error("%s takes no --postfit: a few-shot run is bound to the frozen protocol, not to a postfit record"
                 % fs_flags[0])
    if fs_flags and a.max_messages is not None:
        ap.error("%s takes no --max-messages: a few-shot run trains every message the frozen protocol selects"
                 % fs_flags[0])
    try:
        r = resolve(a)
    except ValueError as e:
        ap.error(str(e))
    if r["device"] == "mps" and a.workers != 1:
        ap.error("--device mps trains on one Ray worker (one GPU, no MPS collective backend); use --workers 1")
    if a.amp and r["device"] in ("mps", "cpu"):
        ap.error("--amp is CUDA-only (the MPS fine-tune is measured with AMP off); drop --amp")
    proto = pin = block = mode = None
    if a.fewshot_base or continued:                         # D-01: the frozen protocol first, before the split is read
        proto, pin = protocol.require_frozen()
        block = proto.get("fewshot")
        if not isinstance(block, dict):
            raise SystemExit("the frozen protocol holds no few-shot block; restore protocol.json with `git checkout -- "
                             "protocol.json`")
        mode = fewshot.read_mode(pin)                       # the D-08 decision record names primary or fallback
        if continued and a.base == "multilingual":
            raise SystemExit("--fewshot-draw continues a masked base: pass a ray_train.py --fewshot-base checkpoint as "
                             "--base, not multilingual")
    entry = verify_split_file(a.data, manifest_path=a.manifest, roles=("train",))   # before ray.init (D-03)
    rows = read_verified_rows(a.data, entry["sha256"])
    refuse_held_out(a.data, entry, rows)                    # keyed on the content (WR-02)
    mask = sel = None
    if block is not None:
        fewshot.check_source(block, entry, rows)            # --data is the frozen source; every draw regenerates (D-02)
        if a.fewshot_base:
            mask = sorted(block["tags"])
            given = None if a.mask_tags is None else [t.strip() for t in a.mask_tags.split(",")]
            if given is not None and (sorted(given) != mask or len(set(given)) != len(given)):
                raise SystemExit("--mask-tags %s is not the frozen protocol's held-out tags %s; pass them (or omit "
                                 "--mask-tags): Phase 9 runs the frozen protocol unchanged"
                                 % (a.mask_tags, ",".join(mask)))
        else:
            sel = fewshot.selection(block, a.fewshot_draw, a.fewshot_k)
            fewshot.laya_item_count(rows, sel, block["tags"], a.neg)   # a selected uid missing from the rows: refused
    # the postfit record (torch-free read: schema and sha256 of the parsed bytes) before any fetch or ray.init
    record, record_sha = postfit.read_record(a.postfit) if a.postfit else (None, None)
    if not a.print_run_id and os.path.exists(a.out) and (not os.path.isdir(a.out) or os.listdir(a.out)):
        # copying over old files could leave a mixed checkpoint that laya.load still accepts
        raise SystemExit("--out %s exists and is not empty; choose a new dir" % a.out)
    # the device must exist here before any pin check, fetch or ray.init (LAYA-02)
    if r["device"] == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps, but MPS is not available here; use --device cpu, or run on Apple silicon with "
                         "the locked torch build")
    if r["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda (or --gpu), but CUDA is not available here; use --device cpu or mps")
    la.check_version()                                      # the adapter's laya pin, before the fetch and ray.init
    if a.base == "multilingual":                            # download once on the driver, pinned + verified (ENV-02)
        base_dir = fetch(pins.LAYA)
    elif os.path.isfile(os.path.join(a.base, "rl_agent_config.json")):
        base_dir = os.path.abspath(a.base)
    else:
        raise SystemExit("--base must be 'multilingual' (pinned) or a local Laya checkpoint dir: %s" % a.base)
    base_name = "multilingual" if a.base == "multilingual" else base_dir
    base_ident = identity(base_name)
    base_sha256 = checkpoint_digests(base_name, base_ident)   # workers verify these bytes before loading
    check_tokenizer_config_rewritable(base_dir, "base checkpoint")   # before ray.init, not after a worker load (CR-01)
    fs_binding = base_info = None
    if a.fewshot_base:
        fs_binding = fewshot.binding(pin, "base")
    elif continued:                                         # the base names itself by its own config bytes
        base_info = fewshot_base_of(base_dir, base_ident)
        fs_binding = fewshot.binding(pin, "continued", a.fewshot_draw, a.fewshot_k,
                                     block["draws"][a.fewshot_draw].get("digest"), base_info["run_id"])
    canon = runs.canonical(identity_values(a, r, entry, base_name, base_ident, record_sha, mask_tags=mask,
                                           fewshot=fs_binding))
    pf_block = None
    if record is not None:                                  # bound by refusal, never by transcription (LAYA-08)
        postfit.check_final(record, canon)
        pf_block = postfit.final_block(record, record_sha, a.postfit)   # refuses a T outside laya's band here
    if a.fewshot_base:                                      # bound by refusal to the frozen protocol (D-01, D-03)
        fewshot.check_base(proto, pin, "laya", canon, fewshot.chosen_canonical(proto, "laya"), mode)
    elif continued:                                         # D-06, D-07: the frozen draw, k, epochs, LRs and base
        fewshot.check_continued(proto, pin, "laya", canon, base_info["canonical"], base_info["run_id"],
                                a.fewshot_draw, a.fewshot_k, mode)
    rid = runs.run_id(canon)                                # the Ray experiment name: storage/<rid> is the resume dir
    if a.print_run_id:
        print(rid)
        return
    run_dir = os.path.join(os.path.abspath(a.storage), rid)  # Ray's experiment dir: the resume key
    need = runs.disk_need_bytes(os.path.getsize(os.path.join(base_dir, "model.safetensors")),
                                a.stop_after_epoch or a.epochs, a.export_epochs)
    free = runs.free_bytes(a.storage)
    if free < need:                                         # before ray.init: a full disk mid-save corrupts the run
        raise SystemExit("%.1f GiB free under --storage; this run needs about %.1f GiB (three train states during "
                         "checkpoint rotation, per-epoch exports, the final checkpoint); free disk or pass another "
                         "--storage" % (free / 2**30, need / 2**30))

    device, cpus = r["device"], r["cpus_per_worker"]
    object_store = int(a.object_store_gb * 1024**3) if a.object_store_gb else None
    env = spike.spike_env("0", cpus) if device == "mps" else {}
    if device == "mps":
        # the spike's worker env (strict fallback, HIGH and LOW watermarks 1.0); one spare CPU for the Train controller
        ray.init(runtime_env={"env_vars": env}, num_cpus=cpus + 1, ignore_reinit_error=True, include_dashboard=False,
                 object_store_memory=object_store)
    else:
        ray.init(ignore_reinit_error=True, include_dashboard=False, object_store_memory=object_store)
    require_single_node(ray.nodes())
    cfg = dict({"data": os.path.abspath(a.data), "data_sha256": entry["sha256"], "base_dir": base_dir,
                "driver_node_id": ray.get_runtime_context().get_node_id(),
                "base_sha256": base_sha256, "epochs": a.epochs,
                "micro_batch": a.micro_batch, "grad_accum": a.grad_accum, "encoder_lr": a.encoder_lr,
                "head_lr": a.head_lr, "loss": a.loss, "freeze_encoder": a.freeze_encoder,
                "freeze_embeddings": a.freeze_embeddings,
                "shuffle_options": a.shuffle_options, "max_len": a.max_len, "head_max_len": a.head_max_len,
                "max_messages": a.max_messages, "neg": a.neg,
                "seed": a.seed, "expect_env": env, "run_id": rid, "canonical": canon, "code": code_state(),
                "storage": os.path.abspath(a.storage), "export_epochs": a.export_epochs,
                "fail_after_epoch": a.fail_after_epoch, "stop_after_epoch": a.stop_after_epoch,
                "postfit": pf_block,
                # Phase 9: the binding and mask (recorded), the frozen uid lists (never recorded: the worker leaves
                # fewshot_rows out of ray_cfg) and a continued run's base (run id and canonical config)
                "mask_tags": mask, "fewshot": fs_binding,
                "fewshot_rows": ({"support": sorted(sel["support"]), "replay": sorted(sel["replay"]),
                                  "tags": list(block["tags"])} if sel is not None else None),
                "fewshot_base": base_info,
                "provenance": provenance(entry, a.manifest, a.base, base_dir, base_ident)}, **r)
    cfg["rung"] = runs.rung_label(cfg)                      # derived from the flags, into the record and the config
    trainer = TorchTrainer(
        train_loop_per_worker, train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=a.workers, use_gpu=(device == "cuda"),
                                     resources_per_worker={"CPU": cpus}),
        # one train state kept (each is ~6x the fp16 weights); no in-process retry: a rerun of the command resumes
        run_config=RunConfig(name=rid, storage_path=os.path.abspath(a.storage),
                             checkpoint_config=CheckpointConfig(num_to_keep=1),
                             failure_config=FailureConfig(max_failures=0)))
    t0 = time.time()
    result = fit_or_exit(trainer, run_dir, cfg["rung"])
    print("metrics:", json.dumps(result.metrics, default=str))
    if result.checkpoint is None:
        raise SystemExit("the run reported no checkpoint; rerun the same command to resume from %s" % run_dir)
    if os.path.isdir(a.out) and not os.listdir(a.out):
        os.rmdir(a.out)                                     # an empty dir checked before training
    with result.checkpoint.as_directory() as d:
        shutil.copytree(d, a.out)                           # never merges: fails if --out was filled meanwhile
    print("checkpoint -> %s  (%.0fs)  load with laya.load(%r)" % (a.out, time.time() - t0, a.out))


if __name__ == "__main__":
    main()
