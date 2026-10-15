"""Torch-free BERTurk bookkeeping (Phase 6, BASE-01/02): the locked recipe, the run identity, the 48 labels in INTENTS
order, the checkpoint identity and dump run id, and the berturk_train.json record the scorer reads.

berturk_train.py trains dbmdz/bert-base-turkish-cased with the HF Trainer as a 48-label multi-label classifier;
score.py --model berturk scores its checkpoints into ordinary dumps. Both import this module before torch, so it never
imports torch, transformers, accelerate, Ray or laya (a test enforces it).

Recipe (confirmed by the user at plan 06-02, "Protocol decision: confirm"): A1 one LR for encoder and head, LR; A2 an
EPOCHS schedule whose every epoch is a selection candidate; A3 batch BATCH, a LR_SCHEDULER schedule with a WARMUP_RATIO
warm-up, AdamW (OPTIM, WEIGHT_DECAY), MAX_GRAD_NORM, MAX_LEN, fp32, no gradient checkpointing, the seed set right
before the head is created; A6 no resume (a run dir that exists is refused; a run is rerun from scratch); A7 a masked
BCE (LOSS): a masked (message, tag) label is MASK_LABEL and the loss averages over the unmasked entries only.

Labels: id2label/label2id and every label vector follow INTENTS order, the dump's column order. check_label_order
refuses a config.json that scores another order (plausible but wrong per-tag numbers otherwise); label_vector refuses
a row missing a tag (Phase 6 never masks by omission). check_cased refuses a lowercasing tokenizer: lowercasing merges
the Turkish dotted and dotless i.

Run identity: IDENTITY lists every value that changes what a run trains; canonical() keeps exactly those (a
DROPPED_WHEN_NONE key whose value is None is left out) and run_id() names the run berturk-<split>-s<seed>-<first 8 hex
of the sha256 of the sorted, compact canonical JSON>, so two seeds or configs never share <storage>/<run_id>/. Every
berturk_train.py flag is in exactly one of IDENTITY, NOT_IDENTITY or RESOLVED (a test enforces it).

Postfit (BASE-02, plan 06-04): PROTOCOL is BERTurk's selection grid for pilot.postfit.protocol("berturk"): LR_GRID
(explicit, kept equal to pilot.postfit.LR_GRID by a test: equal tuning effort, the 2026-10-15 user decision), epochs
1..EPOCHS, head_lr None (the head LR is the encoder LR) and pilot.postfit's selection and final seeds (A4). postfit.py
runs Laya's own functions on BERTurk val dumps under it. prune_export deletes one berturk_train.py epoch export only
after a verified full val dump of exactly that export exists (postfit.py --prune picks it for a dir holding RECORD).

Checkpoint (an export or, later, a final seed) = CHECKPOINT_FILES: model.safetensors, config.json (id2label in
INTENTS order), the base's vocab.txt and tokenizer_config.json copied byte for byte (never tokenizer.save_pretrained,
whose bytes depend on HF_HUB_OFFLINE) and RECORD (schema RECORD_SCHEMA). identity() hashes them without a load;
dump_run_id() is berturk-<model sha256[:12]>-<record sha256[:8]>-<digest of config.json and the tokenizer files[:8]>,
so checkpoints that differ in any file the scorer reads never share a dump dir.

usage: from pilot import berturk; rid = berturk.run_id(berturk.canonical(values)); berturk.label_vector(row)
       ident = berturk.identity(ckpt); berturk.dump_run_id(ident); seed, training = berturk.training_block(
       berturk.read_record(ckpt, ident)); berturk.prune_export(export_dir, dump_root, manifest_path)
"""
import hashlib
import json
import os
import shutil

from intents import INTENTS
from pilot import dumps, postfit
from pilot.data import DEFAULT_MANIFEST, sha256_bytes, sha256_file

# ---- the recipe (A1-A3, A7), confirmed at plan 06-02 ----
LR = 2e-05                                  # A1: one LR for encoder and head, equal to Laya's single LR
EPOCHS = 10                                 # A2: the selection schedule (approved berturk_epochs)
BATCH = 32                                  # A3: messages per optimizer step (approved berturk_batch)
MAX_LEN = 128                               # A3: Laya's max_len, in WordPiece tokens with [CLS] and [SEP]
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LR_SCHEDULER = "linear"
MAX_GRAD_NORM = 1.0
OPTIM = "adamw_torch"                       # explicit: the transformers 5.19 default is adamw_torch_fused
LOSS = "masked_bce"                         # A7
MASK_LABEL = -100.0                         # a masked (message, tag) label; the Trainer counts labels != -100

# ---- the postfit protocol (BASE-02, plan 06-04) ----
LR_GRID = (2e-05,)                          # A1: explicit; equal to pilot.postfit.LR_GRID (one LR for both models)
PROTOCOL = {"lr_grid": LR_GRID, "epochs": EPOCHS, "head_lr": None, "selection_seed": postfit.SELECTION_SEED,
            "final_seeds": postfit.FINAL_SEEDS}       # head_lr None: tied to encoder_lr; the seeds are Laya's (A4)

# ---- labels and checkpoint layout ----
TAGS = list(INTENTS)
PROBLEM_TYPE = "multi_label_classification"
TOKENIZER_FILES = ("tokenizer_config.json", "vocab.txt")
BASE_FILES = ("config.json", "model.safetensors") + TOKENIZER_FILES
RECORD = "berturk_train.json"
CHECKPOINT_FILES = BASE_FILES + (RECORD,)
RECORD_SCHEMA = "berturk-train/1"
DONE = "done.json"                          # <storage>/<run_id>/done.json: a finished run
# one row per completed epoch; peak_driver_bytes is None on cpu (never 0), where peak_total_bytes is the peak RSS
EPOCH_FIELDS = ("epoch", "loss", "train_s", "optimizer_steps", "n_messages", "peak_total_bytes", "peak_driver_bytes",
                "peak_rss_bytes", "swap_delta_bytes")
LOGIT_NOTE = "classifier logit per tag; P(tag) = sigmoid(logit / T)"

# ---- run identity ----
IDENTITY = ("split", "data_sha256", "data_role", "base_model_sha256", "base_config_sha256", "base_files_sha256",
            "transformers_version", "accelerate_version", "device", "epochs", "batch", "encoder_lr", "head_lr",
            "weight_decay", "warmup_ratio", "lr_scheduler", "max_grad_norm", "optim", "loss", "max_len",
            "max_messages", "mask_tags", "seed", "stop_after_epoch", "postfit_sha256")
# berturk_train.py flags that never change training, so never the run id:
#   data, base, manifest: paths, represented in IDENTITY by the verified sha256s (data_sha256, base_*_sha256)
#   storage: where the run dir goes; print_run_id: prints the run id and exits
#   export_epochs: whether per-epoch exports are written (outputs, not training); cpus: torch threads (a resource)
NOT_IDENTITY = ("data", "base", "storage", "manifest", "print_run_id", "export_epochs", "cpus")
# flags that enter IDENTITY through resolved keys: the HF Trainer has one LR for encoder and head (A1)
RESOLVED = {"lr": ("encoder_lr", "head_lr")}
# left out when None: base_files_sha256 (None for the pinned base, whose files the pin fixes), max_messages (a smoke
# subsample), mask_tags (Phase 6 trains unmasked), stop_after_epoch and postfit_sha256 (a final seed, plan 06-08), so
# every selection run id is fixed now
DROPPED_WHEN_NONE = ("base_files_sha256", "max_messages", "mask_tags", "stop_after_epoch", "postfit_sha256")


def canonical(values):
    """{key: values[key]} for exactly the IDENTITY keys, without a DROPPED_WHEN_NONE key whose value is None; ValueError
    naming any missing or unexpected key (a DROPPED_WHEN_NONE key must still be given, as None)."""
    missing = [k for k in IDENTITY if k not in values]
    extra = sorted(k for k in values if k not in IDENTITY)
    if missing or extra:
        raise ValueError("BERTurk run identity: missing %s; unexpected %s" % (", ".join(missing) or "none",
                                                                            ", ".join(extra) or "none"))
    return {k: values[k] for k in IDENTITY if not (k in DROPPED_WHEN_NONE and values[k] is None)}


def run_id(canon):
    """berturk-<split>-s<seed>-<first 8 hex of the sha256 of the sorted, compact canonical JSON>; independent of key
    order."""
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "berturk-%s-s%d-%s" % (canon["split"], canon["seed"], hashlib.sha256(blob).hexdigest()[:8])


# ---- labels ----

def id2label():
    """{index: tag} in INTENTS order (the classifier's output columns = the dump's columns)."""
    return dict(enumerate(TAGS))


def label2id():
    return {t: i for i, t in enumerate(TAGS)}


def check_mask_tags(tags):
    """Sorted tuple of the --mask-tags; SystemExit naming an unknown or a duplicated tag."""
    tags = [t for t in tags]
    unknown = sorted({t for t in tags if t not in INTENTS})
    if unknown:
        raise SystemExit("--mask-tags: unknown tag(s) %s; use INTENTS tag names (intents.py)" % ", ".join(unknown))
    dup = sorted({t for t in tags if tags.count(t) > 1})
    if dup:
        raise SystemExit("--mask-tags: tag(s) %s given more than once; list each tag once" % ", ".join(dup))
    return tuple(sorted(tags))


def label_vector(row, mask_tags=()):
    """48 floats in INTENTS order: MASK_LABEL for a masked tag, else 1.0 or 0.0 from row["expected"]. SystemExit naming
    the uid and the tag when the row does not label a tag (a missing label is never read as a negative)."""
    expected = row.get("expected") or {}
    out = []
    for t in TAGS:
        if t in mask_tags:
            out.append(MASK_LABEL)
        elif t not in expected:
            raise SystemExit("row %s does not label tag %s; every training row must label all 48 INTENTS tags "
                             "(regenerate the split with `make data`)" % (row.get("uid"), t))
        else:
            out.append(1.0 if expected[t] else 0.0)
    return out


def check_label_order(cfg, where):
    """SystemExit unless a config.json dict scores the 48 INTENTS tags in INTENTS order as a multi-label classifier."""
    want = id2label()
    try:
        got = {int(k): v for k, v in (cfg.get("id2label") or {}).items()}
    except (AttributeError, TypeError, ValueError):
        got = None
    if got != want or cfg.get("label2id") != label2id() or cfg.get("problem_type") != PROBLEM_TYPE:
        raise SystemExit("%s: config.json does not score the 48 INTENTS tags in INTENTS order (id2label, label2id, "
                         "problem_type); retrain it with berturk_train.py" % where)


def check_cased(tcfg, where):
    """SystemExit unless the tokenizer config keeps case (do_lower_case false): lowercasing merges İ/i and I/ı."""
    if not isinstance(tcfg, dict) or tcfg.get("do_lower_case") is not False:
        raise SystemExit("%s: tokenizer_config.json does not set do_lower_case false; BERTurk must use the pinned "
                         "cased tokenizer (lowercasing merges the Turkish dotted and dotless i)" % where)


# ---- checkpoint identity and dump run id ----

def _sha(path):
    return sha256_file(path) if os.path.isfile(path) else None


def identity(ckpt_dir):
    """The checkpoint identity of a BERTurk checkpoint dir: sha256 of every file the scorer reads (None for a missing
    one). Hashes only, never a load."""
    return {"local": True, "kind": "berturk",
            "model_sha256": _sha(os.path.join(ckpt_dir, "model.safetensors")),
            "config_sha256": _sha(os.path.join(ckpt_dir, "config.json")),
            "record_sha256": _sha(os.path.join(ckpt_dir, RECORD)),
            "files_sha256": {f: _sha(os.path.join(ckpt_dir, f)) for f in TOKENIZER_FILES}}


def require_files(ckpt_dir, what):
    """SystemExit naming every CHECKPOINT_FILES entry missing from ckpt_dir: without them transformers would read a
    config or tokenizer from the Hub, unpinned."""
    missing = [f for f in CHECKPOINT_FILES if not os.path.isfile(os.path.join(ckpt_dir, f))]
    if missing:
        raise SystemExit("%s %s lacks %s; a BERTurk checkpoint holds %s, as berturk_train.py writes it"
                         % (what, os.path.basename(os.path.normpath(ckpt_dir)), ", ".join(missing),
                            ", ".join(CHECKPOINT_FILES)))


def dump_run_id(ident):
    """berturk-<model sha256[:12]>-<record sha256[:8]>-<files_digest of config.json and the tokenizer files[:8]>."""
    files = ident.get("files_sha256") if isinstance(ident, dict) else None
    parts = [ident.get(k) if isinstance(ident, dict) else None for k in ("model_sha256", "record_sha256",
                                                                         "config_sha256")]
    if not all(parts) or not isinstance(files, dict) or any(not files.get(f) for f in TOKENIZER_FILES):
        raise SystemExit("BERTurk checkpoint identity lacks a digest (model.safetensors, %s, config.json or the "
                         "tokenizer files); recompute it with pilot.berturk.identity of a complete checkpoint" % RECORD)
    model, record, config = parts
    return "berturk-%s-%s-%s" % (model[:12], record[:8], dumps.files_digest(dict(files, **{"config.json": config}))[:8])


# ---- the record ----

def record_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def read_record(ckpt_dir, ident):
    """The parsed berturk_train.json of a checkpoint, from bytes whose sha256 is ident's record_sha256 (the bytes that
    name the dump run id); SystemExit on changed bytes or another schema."""
    path = os.path.join(ckpt_dir, RECORD)
    with open(path, "rb") as f:
        raw = f.read()
    if sha256_bytes(raw) != ident.get("record_sha256"):
        raise SystemExit("%s changed after it was hashed; rescore" % RECORD)
    try:
        rec = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise SystemExit("%s is not valid JSON (%s); retrain with berturk_train.py" % (RECORD, e))
    if not isinstance(rec, dict) or rec.get("schema") != RECORD_SCHEMA:
        raise SystemExit("%s schema %r, expected %s; retrain with this berturk_train.py"
                         % (RECORD, rec.get("schema") if isinstance(rec, dict) else None, RECORD_SCHEMA))
    canon = rec.get("canonical")
    if not rec.get("run_id") or not isinstance(canon, dict) or "seed" not in canon:
        raise SystemExit("%s has no run_id or canonical seed; retrain with this berturk_train.py" % RECORD)
    return rec


def training_block(rec):
    """(seed, training) of a BERTurk dump sidecar: exactly the keys of score.training_block (the ones pilot.postfit
    reads); neg_seed, rung, fallback and weaker are Laya-only and None."""
    canon = rec["canonical"]
    epoch = rec.get("epoch")
    rows = (rec.get("record") or {}).get("epochs") or []
    return canon["seed"], {
        "run_id": rec["run_id"], "split": canon.get("split"), "epoch": epoch, "epochs_schedule": canon.get("epochs"),
        "stop_after_epoch": canon.get("stop_after_epoch"), "encoder_lr": canon.get("encoder_lr"),
        "head_lr": canon.get("head_lr"), "neg_seed": None, "rung": None, "fallback": None, "weaker": None,
        "loss_by_epoch": [row.get("loss") for row in rows], "calibrated": rec.get("calibrated"),
        "postfit_sha256": (rec.get("postfit") or {}).get("sha256"), "canonical": canon,
        "config_sha256_sans_seed": dumps.training_digest(canon, epoch)}


def prune_export(export_dir, dump_root=dumps.DUMP_ROOT, manifest_path=DEFAULT_MANIFEST):
    """Delete one berturk_train.py epoch export once its verified full val dump exists; returns the dump's run id.

    Refused, and left in place: a symlink (the export or its exports/ dir), a path that is not a real
    <storage>/<run_id>/exports/epoch-NNN dir holding RECORD, and an export without a full val dump under dump_root
    whose sidecar checkpoint_identity equals the export's identity() (load_dump verifies the dump bytes). Only that one
    dir is ever deleted (mirrors pilot.postfit.prune_export, A8).
    """
    path = os.path.abspath(export_dir)
    if os.path.islink(path) or os.path.islink(os.path.dirname(path)):
        raise SystemExit("%s is a symlink (or sits in a symlinked exports dir); refusing to delete it" % export_dir)
    real = os.path.realpath(path)
    if not (os.path.isdir(real) and postfit.EXPORT_RE.match(os.path.basename(real))
            and os.path.basename(os.path.dirname(real)) == "exports" and os.path.isfile(os.path.join(real, RECORD))):
        raise SystemExit("%s is not a berturk_train.py export (<storage>/<run_id>/exports/epoch-NNN); refusing to "
                         "delete it" % export_dir)
    ident = identity(real)
    rid = dump_run_id(ident)
    try:
        _, _, side, _ = dumps.load_dump(os.path.join(dump_root, rid), "val", manifest_path)
    except SystemExit as e:
        raise SystemExit("no verified full val dump for this export under %s (%s); score it on val first, then prune"
                         % (dump_root, e))
    if side.get("n_subset") is not None or (side.get("system") or {}).get("checkpoint_identity") != ident:
        raise SystemExit("no verified full val dump for this export under %s (%s is a subset dump or another "
                         "checkpoint's); score it on val first, then prune" % (dump_root, rid))
    shutil.rmtree(real)
    return rid


def disk_need_bytes(model_bytes, epochs, export_epochs, final):
    """Bytes a run needs under --storage: one export per epoch, the final checkpoint and a 1 GiB margin (no Trainer
    checkpoint, optimizer state or RNG pickle is ever written, A6)."""
    return (epochs * model_bytes if export_epochs else 0) + (model_bytes if final else 0) + 2**30
