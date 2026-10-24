"""Torch-free BERTurk bookkeeping for the BERTurk baseline: the recipe, the run identity, the 48 labels in INTENTS
order, the checkpoint identity and dump run id, and the berturk_train.json record the scorer reads.

berturk_train.py trains dbmdz/bert-base-turkish-cased with the HF Trainer as a 48-label multi-label classifier;
score.py --model berturk scores its checkpoints into ordinary dumps. Both import this module before torch, so it never
imports torch, transformers, accelerate, Ray or laya (a test enforces it).

Recipe (confirmed by the user before the first BERTurk run): one encoder LR, LR (Laya's single encoder LR), and the
freshly initialised classifier head (the HEAD_PREFIX parameters) at HEAD_LR, Laya's head LR (the first reference-check
fix; SUPERSEDED holds the run that tied the head LR to LR); an EPOCHS schedule (30 since the second reference-check
fix) whose every epoch is a selection candidate; batch BATCH, a LR_SCHEDULER schedule with a WARMUP_RATIO warm-up,
AdamW (OPTIM, WEIGHT_DECAY), MAX_GRAD_NORM, MAX_LEN, fp32, no gradient checkpointing, the seed set right before the
head is created; no resume (a run dir that exists is refused; a run is rerun from scratch); a masked BCE (LOSS): a
masked (message, tag) label is MASK_LABEL and the loss averages over the unmasked entries only.

Labels: id2label/label2id and every label vector follow INTENTS order, the dump's column order. check_label_order
refuses a config.json that scores another order (plausible but wrong per-tag numbers otherwise); label_vector refuses
a row missing a tag (a tag is masked only by name, through mask_tags, never by omission). check_cased refuses a
lowercasing tokenizer: lowercasing merges the Turkish dotted and dotless i.

Run identity: IDENTITY lists every value that changes what a run trains; canonical() keeps exactly those (a
DROPPED_WHEN_NONE key whose value is None is left out) and run_id() names the run berturk-<split>-s<seed>-<first 8 hex
of the sha256 of the sorted, compact canonical JSON>, so two seeds or configs never share <storage>/<run_id>/. Every
berturk_train.py flag is in exactly one of IDENTITY, NOT_IDENTITY or RESOLVED (a test enforces it).

Postfit: PROTOCOL is BERTurk's selection grid for pilot.postfit.protocol("berturk"): LR_GRID (explicit, kept equal to
pilot.postfit.LR_GRID by a test: equal tuning effort, the user's single-LR decision), epochs 1..EPOCHS (30 since the
second reference-check fix; the 10-epoch protocols are kept in SUPERSEDED), head_lr HEAD_LR (Laya's, so both models
share both LRs; a test enforces it) and pilot.postfit's selection and final seeds. postfit.py runs Laya's own
functions on BERTurk val dumps under it. prune_export deletes one berturk_train.py epoch export only after a verified
full val dump of exactly that export exists (postfit.py --prune picks it for a dir holding RECORD).

Superseded runs: SUPERSEDED is an ordered chain, oldest first, of the BERTurk selections discarded after their val
outcome (each read "investigate" in the reference check, and the user chose a recipe fix). Each entry's delta holds the
canonical keys and values the run that replaced it changed, and nothing else changed (the replacing run is the next
entry, or the current run for the last entry; tests/test_berturk_results.py enforces both). Each run's dumps and
records moved byte for byte (git renames) from results/dumps and results/postfit to its own labelled dir under
results/superseded/, so results/dumps and results/postfit only ever hold the current BERTurk run. One pinned digest
anchors each run's chain: postfit_sha256 is the record's sha256, the reference check binds that sha256 and the record
binds every candidate's dump_sha256. The protocol each ran under is kept as literals, never as a reference to
PROTOCOL. A superseded run is never current: no tracked current dump or record may belong to it, and its reference
reads investigate without a resolution, so check_reference_for_final refuses it as the gate of any final seed
(tests/test_berturk_results.py enforces both).

The BERTurk reference check (fixed in code before any BERTurk val score): reference_check(record, ...) compares the
chosen candidate's val micro-F1 at its val-selected global θ after T (the selection objective) with THRESHOLD =
707/1000, more than 10 micro-F1 points below the lower published reference 80.7 (REFERENCE: Moghe et al., Multi3NLU++,
Table 12 Turkish BANKING, frozen-encoder MLPs at θ 0.3). The comparison is exact (Fraction of the summed counts, never
a rounded float): exactly 0.707 is "ok", anything below is "investigate". The result (REFERENCE_SCHEMA) quotes the
source, the setup DIFFERENCES verbatim and the CAVEATS (θ and T were chosen on the same val messages), and records the
user's resolution, which is allowed only on "investigate" (an investigate result may be written without one).

Final seeds: berturk_train.py --postfit RECORD --reference REF trains one final seed on final_train with the
val-chosen settings and stops after the chosen epoch of the selection schedule. check_reference_for_final refuses it
unless REF is the reference check of exactly that record (its sha256), equals reference_check recomputed from that
record and reads ok or carries a recorded resolution; pilot.postfit.check_final refuses any other setting. The final
checkpoint serves the record's T (calibrated true, postfit = pilot.postfit.final_block). final_command prints the argv
of each final seed from the record; final_manifest (pilot.postfit.FINAL_SCHEMA) lists the finished checkpoints only
when they are exactly the record's final seeds, each bound to its sha256, at the chosen epoch and T.

Checkpoint (an export or a final seed) = CHECKPOINT_FILES: model.safetensors, config.json (id2label in
INTENTS order), the base's vocab.txt and tokenizer_config.json copied byte for byte (never tokenizer.save_pretrained,
whose bytes depend on HF_HUB_OFFLINE) and RECORD (schema RECORD_SCHEMA). identity() hashes them without a load;
dump_run_id() is berturk-<model sha256[:12]>-<record sha256[:8]>-<digest of config.json and the tokenizer files[:8]>,
so checkpoints that differ in any file the scorer reads never share a dump dir.

Optimizer (the first reference-check fix): optimizer_groups splits the parameter names into four AdamW groups, the
encoder at the encoder LR and the head at the head LR, each with the HF Trainer's own weight-decay split; it refuses
unless the head (the HEAD_PREFIX names) is exactly the set of parameters from_pretrained newly initialised.
berturk_train.py hands the optimizer to the Trainer, which still builds the linear schedule and its warm-up over it
(both LRs scale alike). Its continued mode (continued=True, a continued run of the few-shot experiment) is the one
exception: a few-shot run continues from a masked base's own fine-tuned head, so the head is the HEAD_PREFIX names and
nothing may be newly initialised (a base of another architecture or label set is refused); the same four groups
otherwise (no fresh head and no weight imprinting, which this pilot does not try).

Few-shot experiment: fewshot is an IDENTITY key left out when None (the pilot.fewshot.binding of a masked base or a
continued run), so every committed BERTurk run id is unchanged; the --fewshot-base, --fewshot-draw and --fewshot-k
flags are NOT_IDENTITY (represented by fewshot and mask_tags).

README block: load_superseded() reads every SUPERSEDED entry's history record and reference check, refused unless both
are bound to its postfit_sha256; render_markdown(record, reference, final, superseded) renders the README's BERTurk
baseline block (postfit.py --render-berturk): the chosen settings, the LR and epoch-grid lines against Laya's protocol
and the approved compute budget, the grid-edge, θ and T bound positions, the reference check with every difference
and caveat, the 30 candidates, one "Protocol changes after a val outcome" row per superseded entry from its delta
(with the count of extra selection runs against Laya's) and the final seeds.

usage: from pilot import berturk; rid = berturk.run_id(berturk.canonical(values)); berturk.label_vector(row)
       ident = berturk.identity(ckpt); berturk.dump_run_id(ident); seed, training = berturk.training_block(
       berturk.read_record(ckpt, ident)); berturk.prune_export(export_dir, dump_root, manifest_path)
       berturk.reference_check(record, record_path, record_sha256, resolution=None)
       berturk.check_reference_for_final(reference_path, record_sha256, record)
       berturk.final_command(record, seed, record_path, reference_path, storage, out_prefix)
       berturk.final_manifest(record_path, ckpt_dirs)
       berturk.optimizer_groups(param_names, decay_names, new_names, encoder_lr, head_lr, weight_decay,
                                continued=False)
       for s in berturk.SUPERSEDED: s["record"], s["postfit_sha256"], s["protocol"], s["delta"]
       berturk.render_markdown(record, reference, final_manifest, berturk.load_superseded())
"""
import hashlib
import json
import math
import os
import shutil
from collections.abc import Collection, Iterable
from fractions import Fraction
from typing import Any

from intents import INTENTS
from pilot import dumps, pins, postfit
from pilot.data import DEFAULT_MANIFEST, REPO_ROOT, repo_relpath, sha256_bytes, sha256_file

# ---- the recipe, confirmed by the user before the first BERTurk run ----
LR = 2e-05                                  # the encoder LR, equal to Laya's single encoder LR
# the first reference-check fix (the user's reply "fix: head LR 1e-4 (Recommended)", 2026-10-21): the freshly
# initialised classifier head trains at Laya's head LR; the encoder keeps LR and every other recipe value is unchanged
HEAD_LR = postfit.HEAD_LR
HEAD_PREFIX = "classifier."                 # the head's parameter names: exactly the ones from_pretrained initialises
# the selection schedule. The second reference-check fix (the user's reply "fix: extend epochs (Recommended)",
# 2026-10-21): 30 epochs, the user's choice; SUPERSEDED[-1] holds the 10-epoch run. The linear schedule and its 10%
# warm-up span all 30 epochs. It deviates from the approved compute budget (berturk_epochs 10), which pilot/compute.py
# keeps as approved and the README discloses.
EPOCHS = 30
BATCH = 32                                  # messages per optimizer step (approved berturk_batch)
MAX_LEN = 128                               # Laya's max_len, in WordPiece tokens with [CLS] and [SEP]
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LR_SCHEDULER = "linear"
MAX_GRAD_NORM = 1.0
OPTIM = "adamw_torch"                       # explicit: the transformers 5.19 default is adamw_torch_fused
LOSS = "masked_bce"                         # BCE over the unmasked (message, tag) labels only
MASK_LABEL = -100.0                         # a masked (message, tag) label; the Trainer counts labels != -100

# ---- the postfit protocol ----
LR_GRID = (2e-05,)                          # explicit; equal to pilot.postfit.LR_GRID (one LR for both models)
PROTOCOL = {"lr_grid": LR_GRID, "epochs": EPOCHS, "head_lr": HEAD_LR, "selection_seed": postfit.SELECTION_SEED,
            "final_seeds": postfit.FINAL_SEEDS}       # head_lr: Laya's fixed head LR; seeds: Laya's

# ---- superseded selection runs (the two reference-check fixes) ----
# an ordered chain, oldest first, kept as disclosed history under results/superseded/<name>/, never current; protocol
# values and deltas are literals on purpose (delta: the canonical change of the run that replaced the entry);
# user_choice is the user's decision after the entry's val outcome, with the reply quoted verbatim
_TIED = "results/superseded/berturk-tied-head-lr"
_GRID10 = "results/superseded/berturk-10-epoch-grid"
SUPERSEDED = (
    {"name": "berturk-tied-head-lr", "dir": _TIED, "record": _TIED + "/postfit/berturk.json",
     "reference": _TIED + "/postfit/berturk-reference.json", "dumps": _TIED + "/dumps",
     "record_path_at_check": "results/postfit/berturk.json",
     "postfit_sha256": "8975cfa506deddaea77db6cdc883407a7a13b2cb7c6a95155e02e22e219db21a",
     "train_run_id": "berturk-selection_train-s0-c0c9505e",
     "protocol": {"lr_grid": (2e-05,), "epochs": 10, "head_lr": None, "selection_seed": 0, "final_seeds": (0, 1, 2)},
     "user_choice": {"date": "2026-10-21", "decision": "fix", "reply": "fix: head LR 1e-4 (Recommended)"},
     "change": ("the classifier head's LR stops being tied to the encoder LR 2e-05 and becomes 1e-4, Laya's head LR "
                "(pilot.postfit.HEAD_LR); the encoder LR and every other setting are unchanged"),
     "delta": {"head_lr": 0.0001}},
    {"name": "berturk-10-epoch-grid", "dir": _GRID10, "record": _GRID10 + "/postfit/berturk.json",
     "reference": _GRID10 + "/postfit/berturk-reference.json", "dumps": _GRID10 + "/dumps",
     "record_path_at_check": "results/postfit/berturk.json",
     "postfit_sha256": "7086c7f100007c1ab812e1f473a537cb460d0b911962b12de3940618ffd40d58",
     "train_run_id": "berturk-selection_train-s0-8c033154",
     "protocol": {"lr_grid": (2e-05,), "epochs": 10, "head_lr": 0.0001, "selection_seed": 0, "final_seeds": (0, 1, 2)},
     "user_choice": {"date": "2026-10-21", "decision": "fix", "reply": "fix: extend epochs (Recommended)"},
     "change": ("the epoch grid of the selection schedule grows from 10 to 30 epochs, the user's choice after this val "
                "outcome, while the encoder LR 2e-05, the head LR 1e-4 and every other setting stay; the linear "
                "schedule and its warm-up then span 30 epochs"),
     "delta": {"epochs": 30}},
)

# ---- the BERTurk reference check against the published numbers; the texts below land in berturk-reference/1 ----
REFERENCE_SCHEMA = "berturk-reference/1"
REFERENCE = {
    "paper": "Moghe et al., Multi3NLU++ (multilingual, multi-intent, multi-domain NLU dataset)",
    "venue": "ACL Findings 2023",
    "url": "https://arxiv.org/abs/2212.10455",
    "metric": "micro-F1",
    "table_12": {"title": "In-language in-domain results for intent detection with MLP-based setup (F1 x100)",
                 "language": "Turkish, in-language", "domain": "BANKING",
                 "setting": "Large: 10-fold cross-validation, train on 9 of 10 folds (about 1,860 messages)",
                 "values": [{"model": "MLP on frozen LaBSE", "encoder": "frozen", "theta": 0.3, "micro_f1": 82.8},
                            {"model": "MLP on frozen mpnet", "encoder": "frozen", "theta": 0.3, "micro_f1": 80.7}]},
    "table_18": {"setup": "QA-based intent detection", "model": "QA mDeBERTa", "micro_f1": 90.1},
    "lower_reference": 80.7,
}
THRESHOLD = Fraction(707, 1000)
THRESHOLD_RULE = ("below 0.707 (more than 10 micro-F1 points below the lower published reference 80.7) triggers an "
                  "investigation; exactly 0.707 passes")
DIFFERENCES = [
    "a fixed split (selection train folds 0–13 with about 1,450 messages; final train folds 0–15 with about 1,658)",
    "a different test portion (folds 16–19)",
    "a fine-tuned BERTurk rather than a frozen encoder",
    "a single split rather than cross-validation averages",
    "this check itself reads val folds 14-15",
]
CAVEATS = ["θ and T were chosen on the same val messages, so the value is in-sample and optimistic",
           "the paper uses a fixed θ 0.3 on frozen-encoder MLPs; neither of our numbers is like-for-like"]

# ---- labels and checkpoint layout ----
TAGS = list(INTENTS)
PROBLEM_TYPE = "multi_label_classification"
TOKENIZER_FILES = ("tokenizer_config.json", "vocab.txt")
BASE_FILES = ("config.json", "model.safetensors") + TOKENIZER_FILES
RECORD = "berturk_train.json"
CHECKPOINT_FILES = BASE_FILES + (RECORD,)
RECORD_SCHEMA = "berturk-train/1"
DONE = "done.json"                          # <storage>/<run_id>/done.json: a finished run
# one row per completed epoch; peak_driver_bytes is None on cpu (never 0), where peak_total_bytes is the peak RSS; loss
# is the unweighted mean of the epoch's per-step batch losses (diagnostic only)
EPOCH_FIELDS = ("epoch", "loss", "train_s", "optimizer_steps", "n_messages", "peak_total_bytes", "peak_driver_bytes",
                "peak_rss_bytes", "swap_delta_bytes")
LOGIT_NOTE = "classifier logit per tag; P(tag) = sigmoid(logit / T)"

# ---- run identity ----
IDENTITY = ("split", "data_sha256", "data_role", "base_model_sha256", "base_config_sha256", "base_files_sha256",
            "transformers_version", "accelerate_version", "device", "epochs", "batch", "encoder_lr", "head_lr",
            "weight_decay", "warmup_ratio", "lr_scheduler", "max_grad_norm", "optim", "loss", "max_len",
            "max_messages", "mask_tags", "seed", "stop_after_epoch", "postfit_sha256", "fewshot")
# berturk_train.py flags that never change training, so never the run id:
#   data, base, manifest: paths, represented in IDENTITY by the verified sha256s (data_sha256, base_*_sha256)
#   storage: where the run dir goes; print_run_id: prints the run id and exits
#   export_epochs: whether per-epoch exports are written (outputs, not training); cpus: torch threads (a resource)
#   out: where the final checkpoint goes; postfit: a path, represented by postfit_sha256 (the record's bytes);
#   reference: the reference check, a refusal only (check_reference_for_final binds it to the record's sha256)
#   fewshot_base, fewshot_draw, fewshot_k: the few-shot modes, represented by the fewshot binding
#   (pilot.fewshot.binding: stage, draw, k, the draw digest and the base's run id) and mask_tags
NOT_IDENTITY = ("data", "base", "storage", "manifest", "print_run_id", "export_epochs", "cpus", "out", "postfit",
                "reference", "fewshot_base", "fewshot_draw", "fewshot_k")
# flags that enter IDENTITY through resolved keys: --lr is the encoder LR; --head-lr is its own IDENTITY key
RESOLVED = {"lr": ("encoder_lr",)}
# left out when None: base_files_sha256 (None for the pinned base, whose files the pin fixes), max_messages (a smoke
# subsample), mask_tags (only a few-shot masked base masks), stop_after_epoch and postfit_sha256 (a final seed), and
# fewshot (a few-shot run), so every selection and final-seed run id stays as it was before those keys existed
DROPPED_WHEN_NONE = ("base_files_sha256", "max_messages", "mask_tags", "stop_after_epoch", "postfit_sha256", "fewshot")


def canonical(values: dict[str, Any]) -> dict[str, Any]:
    """{key: values[key]} for exactly the IDENTITY keys, without a DROPPED_WHEN_NONE key whose value is None; ValueError
    naming any missing or unexpected key (a DROPPED_WHEN_NONE key must still be given, as None)."""
    missing = [k for k in IDENTITY if k not in values]
    extra = sorted(k for k in values if k not in IDENTITY)
    if missing or extra:
        raise ValueError("BERTurk run identity: missing %s; unexpected %s" % (", ".join(missing) or "none",
                                                                            ", ".join(extra) or "none"))
    return {k: values[k] for k in IDENTITY if not (k in DROPPED_WHEN_NONE and values[k] is None)}


def run_id(canon: dict[str, Any]) -> str:
    """berturk-<split>-s<seed>-<first 8 hex of the sha256 of the sorted, compact canonical JSON>; independent of key
    order."""
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "berturk-%s-s%d-%s" % (canon["split"], canon["seed"], hashlib.sha256(blob).hexdigest()[:8])


# ---- the optimizer groups (the first reference-check fix) ----

def optimizer_groups(param_names: Iterable[str], decay_names: Iterable[str], new_names: Iterable[str],
                     encoder_lr: float, head_lr: float, weight_decay: float,
                     continued: bool = False) -> list[dict[str, Any]]:
    """Four AdamW param groups over the parameter names, in the fixed order encoder + decay, encoder + no decay,
    head + decay, head + no decay: {"group": "encoder" | "head", "decay": bool, "lr", "weight_decay" (weight_decay
    for a decay group, else 0.0), "params": [names in param_names order]}. The head is the names starting with
    HEAD_PREFIX, decay follows decay_names (the HF Trainer's own rule). SystemExit unless the head is non-empty and
    exactly new_names, the parameters from_pretrained newly initialised (a pretrained parameter at the head LR, or a
    fresh one left at the encoder LR, would be another recipe). Every name lands in exactly one group.

    continued=True (a continued few-shot run): the head is the loaded, fine-tuned head of a masked base,
    so new_names must be empty (SystemExit naming them otherwise: a base that newly initialised anything is of another
    architecture or label set, and its head would not be the base's); the same four groups."""
    names, decay, new = list(param_names), set(decay_names), set(new_names)
    head = {n for n in names if n.startswith(HEAD_PREFIX)}
    if continued:
        if not head or new:
            raise SystemExit("a continued few-shot run trains the masked base's own classifier head, loaded, never "
                             "newly initialised (head %s, newly initialised %s); pass a berturk_train.py "
                             "--fewshot-base checkpoint as --base" % (sorted(head) or "none", sorted(new) or "none"))
    elif not head or head != new:
        raise SystemExit("the classifier head must be exactly the newly initialised parameters (head %s, newly "
                         "initialised %s); check the base and the transformers version"
                         % (sorted(head) or "none", sorted(new) or "none"))
    groups = []
    for group, lr in (("encoder", encoder_lr), ("head", head_lr)):
        for d in (True, False):
            groups.append({"group": group, "decay": d, "lr": lr, "weight_decay": weight_decay if d else 0.0,
                           "params": [n for n in names if (n in head) == (group == "head") and (n in decay) == d]})
    return groups


# ---- labels ----

def id2label() -> dict[int, str]:
    """{index: tag} in INTENTS order (the classifier's output columns = the dump's columns)."""
    return dict(enumerate(TAGS))


def label2id() -> dict[str, int]:
    """{tag: index} in INTENTS order, the inverse of id2label."""
    return {t: i for i, t in enumerate(TAGS)}


def check_mask_tags(tags: Iterable[str]) -> tuple[str, ...]:
    """Sorted tuple of the --mask-tags; SystemExit naming an unknown or a duplicated tag."""
    tags = [t for t in tags]
    unknown = sorted({t for t in tags if t not in INTENTS})
    if unknown:
        raise SystemExit("--mask-tags: unknown tag(s) %s; use INTENTS tag names (intents.py)" % ", ".join(unknown))
    dup = sorted({t for t in tags if tags.count(t) > 1})
    if dup:
        raise SystemExit("--mask-tags: tag(s) %s given more than once; list each tag once" % ", ".join(dup))
    return tuple(sorted(tags))


def label_vector(row: dict[str, Any], mask_tags: Collection[str] = ()) -> list[float]:
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


def check_label_order(cfg: dict[str, Any], where: str) -> None:
    """SystemExit unless a config.json dict scores the 48 INTENTS tags in INTENTS order as a multi-label classifier."""
    want = id2label()
    try:
        got = {int(k): v for k, v in (cfg.get("id2label") or {}).items()}
    except (AttributeError, TypeError, ValueError):
        got = None
    if got != want or cfg.get("label2id") != label2id() or cfg.get("problem_type") != PROBLEM_TYPE:
        raise SystemExit("%s: config.json does not score the 48 INTENTS tags in INTENTS order (id2label, label2id, "
                         "problem_type); retrain it with berturk_train.py" % where)


def check_cased(tcfg: Any, where: str) -> None:
    """SystemExit unless the tokenizer config keeps case (do_lower_case false): lowercasing merges İ/i and I/ı."""
    if not isinstance(tcfg, dict) or tcfg.get("do_lower_case") is not False:
        raise SystemExit("%s: tokenizer_config.json does not set do_lower_case false; BERTurk must use the pinned "
                         "cased tokenizer (lowercasing merges the Turkish dotted and dotless i)" % where)


# ---- checkpoint identity and dump run id ----

def _sha(path: str) -> str | None:
    """The sha256 of a file, None when it is missing."""
    return sha256_file(path) if os.path.isfile(path) else None


def identity(ckpt_dir: str) -> dict[str, Any]:
    """The checkpoint identity of a BERTurk checkpoint dir: sha256 of every file the scorer reads (None for a missing
    one). Hashes only, never a load."""
    return {"local": True, "kind": "berturk",
            "model_sha256": _sha(os.path.join(ckpt_dir, "model.safetensors")),
            "config_sha256": _sha(os.path.join(ckpt_dir, "config.json")),
            "record_sha256": _sha(os.path.join(ckpt_dir, RECORD)),
            "files_sha256": {f: _sha(os.path.join(ckpt_dir, f)) for f in TOKENIZER_FILES}}


def require_files(ckpt_dir: str, what: str) -> None:
    """SystemExit naming every CHECKPOINT_FILES entry missing from ckpt_dir: without them transformers would read a
    config or tokenizer from the Hub, unpinned."""
    missing = [f for f in CHECKPOINT_FILES if not os.path.isfile(os.path.join(ckpt_dir, f))]
    if missing:
        raise SystemExit("%s %s lacks %s; a BERTurk checkpoint holds %s, as berturk_train.py writes it"
                         % (what, os.path.basename(os.path.normpath(ckpt_dir)), ", ".join(missing),
                            ", ".join(CHECKPOINT_FILES)))


def dump_run_id(ident: Any) -> str:
    """berturk-<model sha256[:12]>-<record sha256[:8]>-<files_digest of config.json and the tokenizer files[:8]>;
    SystemExit when ident is not an identity() dict with every digest."""
    files = ident.get("files_sha256") if isinstance(ident, dict) else None
    parts = [ident.get(k) if isinstance(ident, dict) else None for k in ("model_sha256", "record_sha256",
                                                                         "config_sha256")]
    if not all(parts) or not isinstance(files, dict) or any(not files.get(f) for f in TOKENIZER_FILES):
        raise SystemExit("BERTurk checkpoint identity lacks a digest (model.safetensors, %s, config.json or the "
                         "tokenizer files); recompute it with pilot.berturk.identity of a complete checkpoint" % RECORD)
    model, record, config = parts
    return "berturk-%s-%s-%s" % (model[:12], record[:8], dumps.files_digest(dict(files, **{"config.json": config}))[:8])


# ---- the record ----

def record_bytes(obj: dict[str, Any]) -> bytes:
    """The bytes of a berturk_train.json record (or done.json): indented JSON with a trailing newline."""
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def read_record(ckpt_dir: str, ident: dict[str, Any]) -> dict[str, Any]:
    """The parsed berturk_train.json of a checkpoint, from bytes whose sha256 is ident's record_sha256 (the bytes that
    name the dump run id); SystemExit on changed bytes, another schema, or a canonical max_len that is not a positive
    int or a temperature that is not a finite float > 0, the two values score.py scores with."""
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
    n, t = canon.get("max_len"), rec.get("temperature")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise SystemExit("%s: canonical max_len %r is not a positive int; retrain with this berturk_train.py"
                         % (RECORD, n))
    if not isinstance(t, float) or not math.isfinite(t) or t <= 0:
        raise SystemExit("%s: temperature %r is not a finite float > 0; retrain with this berturk_train.py"
                         % (RECORD, t))
    return rec


def training_block(rec: dict[str, Any]) -> tuple[int, dict[str, Any]]:
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


def prune_export(export_dir: str, dump_root: str = dumps.DUMP_ROOT, manifest_path: str = DEFAULT_MANIFEST) -> str:
    """Delete one berturk_train.py epoch export once its verified full val dump exists; returns the dump's run id.

    Refused, and left in place: a symlink (the export or its exports/ dir), a path that is not a real
    <storage>/<run_id>/exports/epoch-NNN dir holding RECORD, and an export without a full val dump under dump_root
    whose sidecar checkpoint_identity equals the export's identity() (load_dump verifies the dump bytes). Only that one
    dir is ever deleted (mirrors pilot.postfit.prune_export).
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


def reference_check(record: dict[str, Any], record_path: str, record_sha256: str | None,
                    resolution: str | None = None) -> dict[str, Any]:
    """The BERTurk reference check (REFERENCE_SCHEMA) of a BERTurk postfit record: the chosen candidate's val
    micro-F1 at its val-selected global θ after T, as the exact Fraction(2TP, 2TP + FP + FN) of its summed counts,
    against THRESHOLD. status "investigate" when below, else "ok". SystemExit on a record of another system kind, a
    chosen candidate missing from the candidates, or a resolution on an "ok" status (nothing to resolve). No timestamp,
    no local path: the same inputs give the same JSON bytes."""
    if record.get("system_kind") != "berturk":
        raise SystemExit("%s: system kind %r; the published reference check applies to BERTurk only"
                         % (record_path, record.get("system_kind")))
    chosen = record.get("chosen") or {}
    cand = next((c for c in record.get("candidates") or [] if c.get("dump_sha256") == chosen.get("dump_sha256")), None)
    if cand is None:
        raise SystemExit("%s: the chosen candidate (dump_sha256 %s) is not among the record's candidates; rebuild it "
                         "with postfit.py" % (record_path, chosen.get("dump_sha256")))
    o = cand["objective"]
    den = 2 * o["tp"] + o["fp"] + o["fn"]
    value = Fraction(2 * o["tp"], den) if den else Fraction(0)
    status = "investigate" if value < THRESHOLD else "ok"
    resolution = resolution.strip() if isinstance(resolution, str) and resolution.strip() else None
    if resolution is not None and status == "ok":
        raise SystemExit("the reference check is ok (val micro-F1 %d/%d >= 0.707); nothing to resolve, drop "
                         "--resolution" % (value.numerator, value.denominator))
    return {"schema": REFERENCE_SCHEMA, "postfit": {"path": repo_relpath(record_path), "sha256": record_sha256},
            "reference": REFERENCE, "threshold": float(THRESHOLD), "threshold_rule": THRESHOLD_RULE,
            "value": {"metric": postfit.OBJECTIVE,
                      "val_micro_f1_at_global_theta": cand.get("val_micro_f1_at_global_theta"),
                      "fraction": [value.numerator, value.denominator],
                      "val_micro_f1_at_0_5": cand.get("val_micro_f1_at_0_5"), "epoch": chosen.get("epoch"),
                      "encoder_lr": chosen.get("encoder_lr"), "temperature": chosen.get("temperature"),
                      "train_run_id": chosen.get("train_run_id"), "dump_run_id": chosen.get("dump_run_id")},
            "below_threshold": status == "investigate", "status": status, "differences": DIFFERENCES,
            "caveats": CAVEATS, "resolution": resolution}


def check_reference_for_final(reference_path: str, record_sha256: str, record: dict[str, Any]) -> dict[str, Any]:
    """The reference-check gate of a final seed: the parsed reference check at reference_path, refused (SystemExit) when
    it is missing, not valid JSON, of another schema, of another postfit record than record_sha256, its status
    disagrees with its own exact fraction, it reads "investigate" without a non-empty resolution, or it is not
    reference_check of `record` (the parsed record of that sha256) at its recorded path and resolution: the gate
    recomputes the value from the record and never trusts the file's own. No final seed is trained
    (or printed) before the shortfall is investigated and then fixed or documented."""
    if not os.path.isfile(reference_path):
        raise SystemExit("%s not found; build the BERTurk reference check with postfit.py --reference-check RECORD "
                         "--out %s before any final seed" % (reference_path, reference_path))
    with open(reference_path, "rb") as f:
        raw = f.read()
    try:
        ref = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise SystemExit("%s is not valid JSON (%s); rebuild it with postfit.py --reference-check" % (reference_path, e))
    if not isinstance(ref, dict) or ref.get("schema") != REFERENCE_SCHEMA:
        raise SystemExit("%s: schema %r, expected %s; rebuild it with postfit.py --reference-check"
                         % (reference_path, ref.get("schema") if isinstance(ref, dict) else None, REFERENCE_SCHEMA))
    if (ref.get("postfit") or {}).get("sha256") != record_sha256:
        raise SystemExit("the reference check %s is of another postfit record; rebuild it with postfit.py "
                         "--reference-check" % reference_path)
    value = ref.get("value") or {}
    frac = value.get("fraction")
    try:
        below = Fraction(frac[0], frac[1]) < THRESHOLD
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        raise SystemExit("%s holds no exact val micro-F1 fraction; rebuild it with postfit.py --reference-check"
                         % reference_path) from None
    status = ref.get("status")
    if status != ("investigate" if below else "ok"):
        raise SystemExit("%s: status %r disagrees with its val micro-F1 %d/%d against 0.707; rebuild it with postfit.py "
                         "--reference-check" % (reference_path, status, frac[0], frac[1]))
    res = ref.get("resolution")
    if status == "investigate" and not (isinstance(res, str) and res.strip()):
        raise SystemExit("reference check: BERTurk's val micro-F1 %s is below 0.707 and no resolution is recorded; "
                         "investigate, then fix (a new selection run) or document it (postfit.py --reference-check "
                         "--resolution) before any final seed" % value.get("val_micro_f1_at_global_theta"))
    got = reference_check(record, os.path.join(REPO_ROOT, ref["postfit"].get("path") or ""), record_sha256, res)
    if postfit._json(got) != postfit._json(ref):
        raise SystemExit("%s is not pilot.berturk.reference_check of the postfit record %s (its value, status or "
                         "another field was edited); rebuild it with postfit.py --reference-check"
                         % (reference_path, record_sha256))
    return ref


def final_command(record: dict[str, Any], seed: int, record_path: str, reference_path: str, storage: str,
                  out_prefix: str, base: str = "berturk", data: str = "data/splits/final_train.jsonl",
                  manifest: str | None = None) -> list[str]:
    """The berturk_train.py argv (starting with "berturk_train.py") that trains final seed `seed` of a BERTurk postfit
    record: every hyperparameter from the chosen canonical config, the chosen stop epoch, --postfit record_path,
    --reference reference_path and --no-export-epochs; --out is "<out_prefix>-s<seed>". berturk_train.py's
    pilot.postfit.check_final and check_reference_for_final are still the binding; this spares the transcription."""
    final, chosen = record["final"], record["chosen"]
    if record.get("system_kind") != "berturk":
        raise SystemExit("system kind %r; berturk_train.py trains the final seeds of a BERTurk record only"
                         % (record.get("system_kind"),))
    if seed not in (final.get("seeds") or []):
        raise SystemExit("seed %r is not one of the record's final seeds %s" % (seed, final.get("seeds")))
    c = chosen.get("canonical")
    if not isinstance(c, dict):
        raise SystemExit("the postfit record's chosen run has no canonical config; rebuild it with this postfit.py")
    if base == "berturk" and c.get("base_model_sha256") != pins.BERTURK["sha256"]["model.safetensors"]:
        raise SystemExit("the chosen run trained another base (base_model_sha256 %s, not the pinned BERTurk weights); "
                         "pass --base DIR" % c.get("base_model_sha256"))
    argv = ["berturk_train.py", "--data", data, "--base", base, "--device", c["device"], "--epochs", str(c["epochs"]),
            "--lr", repr(c["encoder_lr"]), "--head-lr", repr(c["head_lr"]), "--batch", str(c["batch"]),
            "--max-len", str(c["max_len"])]
    if c.get("max_messages") is not None:
        argv += ["--max-messages", str(c["max_messages"])]
    if c.get("mask_tags"):
        argv += ["--mask-tags", ",".join(c["mask_tags"])]
    if chosen.get("stop_after_epoch") is not None:
        argv += ["--stop-after-epoch", str(chosen["stop_after_epoch"])]
    argv += ["--seed", str(seed), "--postfit", record_path, "--reference", reference_path, "--no-export-epochs",
             "--storage", storage, "--out", "%s-s%d" % (out_prefix, seed)]
    if manifest is not None:
        argv += ["--manifest", manifest]
    return argv


def _final_entry(record: dict[str, Any], sha: str, ckpt_dir: str) -> dict[str, Any]:
    """One final_manifest entry of a finished berturk_train.py --postfit checkpoint, or SystemExit naming what does not
    bind it (mirrors pilot.postfit._final_entry)."""
    chosen, name = record["chosen"], os.path.basename(os.path.normpath(ckpt_dir))
    require_files(ckpt_dir, "final checkpoint")
    ident = identity(ckpt_dir)
    rec = read_record(ckpt_dir, ident)                      # the bytes ident hashed
    canon = rec["canonical"]
    prov = rec.get("provenance") or {}
    man, split = prov.get("manifest") or {}, prov.get("split") or {}
    # as for Laya's final seeds: a record built on the committed manifest binds final seeds trained on it, whose
    # provenance split bytes are the canonical data_sha256; other records keep it free, as check_final does
    if (record.get("manifest") or {}).get("is_committed") is True and \
            (man.get("is_committed") is not True or split.get("sha256") != canon.get("data_sha256")):
        raise SystemExit("%s: provenance manifest is_committed %r (split sha256 %s, canonical %s); a final seed of a "
                         "record on the committed manifest trains on the committed data/splits.json"
                         % (name, man.get("is_committed"), split.get("sha256"), canon.get("data_sha256")))
    if (rec.get("postfit") or {}).get("sha256") != sha or canon.get("postfit_sha256") != sha:
        raise SystemExit("%s is bound to postfit record %s, not %s; train it with berturk_train.py --postfit of this "
                         "record" % (name, (rec.get("postfit") or {}).get("sha256"), sha))
    if canon.get("split") != record["final"]["split"]:
        raise SystemExit("%s trained on %s; the final seeds train on %s" % (name, canon.get("split"),
                                                                            record["final"]["split"]))
    postfit.check_final(record, canon)
    if rec.get("epoch") != chosen["epoch"]:
        raise SystemExit("%s stopped at epoch %r; the chosen epoch is %r" % (name, rec.get("epoch"), chosen["epoch"]))
    if postfit._json(rec.get("temperature")) != postfit._json(chosen["temperature"]) or rec.get("calibrated") is not True:
        raise SystemExit("%s serves temperature %r (calibrated %r); a final seed serves the record's T %r"
                         % (name, rec.get("temperature"), rec.get("calibrated"), chosen["temperature"]))
    rows = (rec.get("record") or {}).get("epochs") or []
    return {"seed": canon["seed"], "train_run_id": rec["run_id"], "checkpoint_name": name,
            "checkpoint_identity": ident, "dump_run_id": dump_run_id(ident), "temperature": rec["temperature"],
            "loss_by_epoch": [row.get("loss") for row in rows],
            "epochs": [{k: row.get(k) for k in ("epoch", "loss", "train_s", "peak_total_bytes", "swap_delta_bytes")}
                       for row in rows],
            "config_sha256_sans_seed": dumps.training_digest(canon, rec.get("epoch"))}


def final_manifest(record_path: str, ckpt_dirs: Iterable[str]) -> dict[str, Any]:
    """The pilot.postfit.FINAL_SCHEMA manifest of a BERTurk record's final checkpoints: refused unless every dir is
    bound to the record's sha256 (berturk_train.json postfit and canonical), trained on its final split with the chosen
    recipe (check_final), stops at the chosen epoch, serves the chosen T exactly, the seeds are exactly the record's
    final seeds (each once) and every seed shares one config_sha256_sans_seed. Ids, digests and numbers only."""
    record, sha = postfit.read_record(record_path)
    if record.get("system_kind") != "berturk":
        raise SystemExit("%s: system kind %r; this is the BERTurk final manifest" % (record_path,
                                                                                     record.get("system_kind")))
    entries = [_final_entry(record, sha, d) for d in ckpt_dirs]
    seeds = sorted(e["seed"] for e in entries)
    want = sorted(record["final"].get("seeds") or [])
    if seeds != want:
        raise SystemExit("the checkpoints are seeds %s; the record's final seeds are %s (each exactly once)"
                         % (seeds, want))
    digests = sorted({e["config_sha256_sans_seed"] for e in entries})
    if len(digests) != 1:
        raise SystemExit("the final seeds differ in more than the seed (config_sha256_sans_seed %s); retrain them with "
                         "postfit.py --print-final-commands" % digests)
    chosen = record["chosen"]
    return {"schema": postfit.FINAL_SCHEMA, "system_kind": "berturk",
            "postfit": {"path": repo_relpath(record_path), "sha256": sha}, "final_split": record["final"]["split"],
            "chosen": {k: chosen.get(k) for k in ("train_run_id", "encoder_lr", "epoch", "epochs_schedule",
                                                  "stop_after_epoch", "temperature")},
            "config_sha256_sans_seed": digests[0],
            "seeds": [{k: v for k, v in e.items() if k != "config_sha256_sans_seed"}
                      for e in sorted(entries, key=lambda e: e["seed"])]}


# ---- the README block, with the disclosure of the reference-check fixes ----

def load_superseded(repo_root: str = REPO_ROOT) -> list[dict[str, Any]]:
    """The disclosed history of every SUPERSEDED entry, in chain order: [{"entry", "record", "reference"}], the parsed
    history record and its reference check. SystemExit naming the file unless it exists, is JSON, the record's sha256
    is the entry's postfit_sha256 and the reference check's postfit.sha256 is that sha256 too (the history is read
    from the committed bytes only, never retyped)."""
    out = []
    for s in SUPERSEDED:
        parsed = {}
        for key in ("record", "reference"):
            p = os.path.join(repo_root, s[key])
            try:
                with open(p, "rb") as f:
                    raw = f.read()
                parsed[key] = json.loads(raw.decode("utf-8"))
            except (OSError, ValueError) as e:
                raise SystemExit("%s cannot be read (%s); the superseded history under results/superseded/ is "
                                 "committed, restore it with git" % (s[key], e))
            if key == "record" and sha256_bytes(raw) != s["postfit_sha256"]:
                raise SystemExit("%s is not the superseded record pinned in pilot.berturk.SUPERSEDED (sha256 %s, "
                                 "pinned %s); restore it with git" % (s[key], sha256_bytes(raw), s["postfit_sha256"]))
        bound = (parsed["reference"].get("postfit") or {}).get("sha256") if isinstance(parsed["reference"], dict) \
            else None
        if bound != s["postfit_sha256"]:
            raise SystemExit("%s is bound to postfit record %s, not the pinned %s; restore it with git"
                             % (s["reference"], bound, s["postfit_sha256"]))
        out.append({"entry": s, "record": parsed["record"], "reference": parsed["reference"]})
    return out


def _ref_f1(ref: dict[str, Any]) -> str:
    """A reference check's value with its exact fraction in lowest terms, e.g. "0.8138 (389/478)"."""
    v = ref["value"]
    return "%.4f (%d/%d)" % (v["val_micro_f1_at_global_theta"], v["fraction"][0], v["fraction"][1])


def _old_value(key: str, canon: dict[str, Any]) -> str:
    """A delta key's value in the run it replaced; a head LR equal to the encoder LR is the tied recipe."""
    if key == "head_lr" and canon.get("head_lr") == canon.get("encoder_lr"):
        return "tied to the encoder LR (%g)" % canon.get("encoder_lr")
    v = canon.get(key)
    return "%g" % v if isinstance(v, float) else str(v)


def _gib(e: dict[str, Any]) -> tuple[str, str, str]:
    """(final-epoch loss, peak GiB, largest swap delta MiB) of a final-manifest seed; an unmeasured value is n/a."""
    ep = e.get("epochs") or []
    peaks = [r["peak_total_bytes"] for r in ep if r.get("peak_total_bytes") is not None]
    swaps = [r["swap_delta_bytes"] for r in ep if r.get("swap_delta_bytes") is not None]
    loss = ep[-1].get("loss") if ep else None
    return ("n/a" if loss is None else "%.4f" % loss, "n/a" if not peaks else "%.2f" % (max(peaks) / 2 ** 30),
            "n/a" if not swaps else "%.0f" % (max(swaps) / 2 ** 20))


def render_markdown(record: dict[str, Any], reference: dict[str, Any], final: dict[str, Any] | None = None,
                    superseded: Iterable[dict[str, Any]] = ()) -> str:
    """The README's BERTurk baseline block (markdown) of the BERTurk postfit record, its reference check, optionally its
    final manifest and the superseded history (load_superseded()): the chosen settings, the rule line, the LR and
    epoch-grid lines against Laya's protocol and the approved pilot.compute.MATRIX, the grid-edge, θ and T bound
    positions, the reference check (value against 0.707, status, the quoted source values, every difference, every
    caveat, the resolution), the candidates, one "Protocol changes after a val outcome" row per superseded entry (its
    delta's old and new value, its chosen epoch, its val micro-F1, status, the user's choice with its date and reply
    verbatim, its history dir)
    with the count of extra selection runs, and the final seeds. Display rounding only; every number comes from the
    records; no timestamp, no local path, no test number. SystemExit on a record of another system kind, a reference
    check that is not the recompute of this record's chosen candidate, or a manifest whose chosen block or final split
    is not the record's."""
    from pilot import compute                               # torch-free; lazy like pilot.postfit's render
    if record.get("system_kind") != "berturk":
        raise SystemExit("pilot.berturk.render_markdown renders BERTurk postfit records only (Laya: postfit.py "
                         "--render); this record's system kind is %r" % (record.get("system_kind"),))
    chosen, proto, rules = record["chosen"], record["protocol"], record.get("rules") or {}
    bound = (reference.get("postfit") or {}) if isinstance(reference, dict) else {}
    got = reference_check(record, os.path.join(REPO_ROOT, bound.get("path") or ""), bound.get("sha256"),
                          reference.get("resolution") if isinstance(reference, dict) else None)
    if json.dumps(got, sort_keys=True, ensure_ascii=False) != json.dumps(reference, sort_keys=True, ensure_ascii=False):
        raise SystemExit("the reference check is not pilot.berturk.reference_check of this record's chosen candidate; "
                         "rebuild it with postfit.py --reference-check")
    if final is not None:
        for key in postfit.FINAL_CHOSEN_KEYS:
            if postfit._json((final.get("chosen") or {}).get(key)) != postfit._json(chosen.get(key)):
                raise SystemExit("the final manifest's chosen %s is %r, the record's %r; it belongs to another record"
                                 % (key, (final.get("chosen") or {}).get(key), chosen.get(key)))
        if final.get("final_split") != record["final"]["split"]:
            raise SystemExit("the final manifest's final split is %r, the record's %r"
                             % (final.get("final_split"), record["final"]["split"]))
        if (final.get("postfit") or {}).get("sha256") != bound.get("sha256"):
            raise SystemExit("the final manifest is bound to postfit record %s, the reference check to %s"
                             % ((final.get("postfit") or {}).get("sha256"), bound.get("sha256")))
    cands = record["candidates"]
    by_id = {c["dump_run_id"]: c for c in cands}
    if chosen["dump_run_id"] not in by_id:
        raise SystemExit("the record's chosen dump %s is not among its candidates" % chosen["dump_run_id"])
    cc, th, canon = by_id[chosen["dump_run_id"]], chosen["theta"], chosen["canonical"]
    epoch, sched, stop, top = chosen["epoch"], chosen["epochs_schedule"], chosen["stop_after_epoch"], proto["epochs"]
    md = postfit._md
    out = ["**Chosen on val** (selection run on %s, seed %s, %d val messages x %d tags; val numbers, in-sample; no "
           "BERTurk test number exists before the final scoring)"
           % (proto["selection_split"], proto["selection_seed"], record["split"]["n_messages"], len(TAGS)), ""]
    out.append(md(("setting", "value"), [
        ("train run id", chosen["train_run_id"]),
        ("encoder LR / head LR", "%g / %g" % (canon["encoder_lr"], canon["head_lr"])),
        ("epoch", "%d of %d, %s" % (epoch, sched, "--stop-after-epoch omitted (last epoch)" if stop is None
                                    else "--stop-after-epoch %d" % stop)),
        ("val-fitted T", "%.4f (bounds [%s, %s])" % (chosen["temperature"], postfit.T_MIN, postfit.T_MAX)),
        ("val NLL at T / at T = 1", "%.5f / %.5f" % (cc["nll_fitted"], cc["nll_raw"])),
        ("global θ shipped / raw", "%.2f / %.2f" % (th["shipped"]["global"], th["raw"]["global"])),
        ("tags with their own θ (shipped / raw)", "%d / %d" % (len(th["shipped"]["per_tag"]), len(th["raw"]["per_tag"]))),
        ("val micro-F1 at the global θ", postfit._f1(cc)),
        ("val micro-F1 at θ 0.5", "%.4f" % cc["val_micro_f1_at_0_5"])]))
    out += ["", "- Rule: %s; tie order: %s (Laya's rule, run by pilot.postfit on BERTurk's val dumps)."
            % (rules.get("objective"), rules.get("tie_order"))]
    lr_grid, laya = proto["lr_grid"], postfit.protocol("laya")
    out.append("- LRs: encoder LR grid %s (%s; Laya's grid %s), head LR %g (Laya's %g)."
               % (", ".join("%g" % x for x in lr_grid), "one value, fixed, not tuned" if len(lr_grid) == 1
                  else "%d values, val-selected" % len(lr_grid), ", ".join("%g" % x for x in laya["lr_grid"]),
                  proto["head_lr"], laya["head_lr"]))
    out.append("- Epoch grid: 1..%d, every epoch of one run a candidate; Laya's is 1..%d (approved max_epochs %d), and "
               "the approved compute budget for BERTurk is berturk_epochs %d, so this grid is a disclosed deviation "
               "from both, and a selection over %d in-sample candidates is more optimistic than over %d."
               % (top, laya["epochs"], compute.MATRIX["max_epochs"], compute.MATRIX["berturk_epochs"], top,
                  laya["epochs"]))
    edges = {name: postfit.theta_edges(cands, name) for name in ("shipped", "raw")}
    clipped = edges["shipped"][0] + edges["shipped"][1]
    ranked = sorted(cands, key=lambda c: (postfit.score(c), -c["epoch"], -c["encoder_lr"]), reverse=True)
    runner = ranked[1] if len(ranked) > 1 else None
    edge = "epoch %d of the epoch grid 1..%d" % (epoch, top)
    if epoch == top:
        line = ("upper edge: %s; the objective may still rise beyond it, and extending the grid would be a protocol "
                "change after a val outcome, decided only by the user" % edge)
    elif epoch == 1:
        line = "lower edge: %s" % edge
    elif clipped:
        line = ("interior in the epoch grid: %s; θ clipped at the grid edge (see the θ-edge line), so the epoch is not "
                "a clean optimum of the objective" % edge)
    elif top - epoch == 1:
        line = "next to the upper edge: %s, one epoch below it" % edge
    elif epoch == 2:
        line = "next to the lower edge: %s, one epoch above it" % edge
    else:
        line = "interior optimum: %s, %d epochs below the upper edge" % (edge, top - epoch)
    if runner is not None:
        line += "; runner-up epoch %d at %s, %.4f below the chosen" % (
            runner["epoch"], postfit._f1(runner), cc["val_micro_f1_at_global_theta"] -
            runner["val_micro_f1_at_global_theta"])
    out.append("- Grid edge: %s." % line)
    pos = []
    for name in ("shipped", "raw"):
        g, grid = th[name]["global"], th[name]["grid"]
        pos.append("%s %.2f (%s)" % (name, g, "at the upper edge" if g == grid["stop"] else "at the lower edge"
                                     if g == grid["start"] else "inside"))
    temps = [c["temperature"] for c in cands]
    on_bound = sum(t in (postfit.T_MIN, postfit.T_MAX) for t in temps)
    t = chosen["temperature"]
    out.append("- Bounds: on the θ grid %s..%s the chosen global θ is %s; the candidates' fitted T span %.4f..%.4f in "
               "laya's band [%s, %s] (%d of %d on a bound), and the chosen T %.4f is %.4f above the lower bound %s and "
               "%.4f below the upper bound %s."
               % (th["shipped"]["grid"]["start"], th["shipped"]["grid"]["stop"], " and ".join(pos), min(temps),
                  max(temps), postfit.T_MIN, postfit.T_MAX, on_bound, len(temps), t, t - postfit.T_MIN, postfit.T_MIN,
                  postfit.T_MAX - t, postfit.T_MAX))
    (s_up, s_lo), (r_up, r_lo) = edges["shipped"], edges["raw"]
    grid = th["shipped"]["grid"]
    out.append("- θ grid edge: %d of %d candidates choose the shipped global θ on the upper edge %.2f and %d on the "
               "lower edge %.2f (raw: %d upper, %d lower)%s." % (
                   len(s_up), len(cands), grid["stop"], len(s_lo), grid["start"], len(r_up), len(r_lo),
                   "; the objective is clipped at a θ-grid edge for those candidates" if clipped
                   else "; no candidate's objective is clipped at a θ-grid edge"))
    out.append(postfit.per_tag_edge_line(cands, th))
    # the published reference check
    v, ref = reference["value"], reference["reference"]
    t12, t18 = ref["table_12"], ref["table_18"]
    out += ["", "**Published reference check** (%s, %s; %s)" % (ref["paper"], ref["venue"], ref["url"]), "",
            md(("check", "value"), [
                ("BERTurk val micro-F1 at the global θ (epoch %d, T %.4f)" % (v["epoch"], v["temperature"]),
                 _ref_f1(reference)),
                ("threshold", "%s: %s" % (reference["threshold"], reference["threshold_rule"])),
                ("status", "%s (below the threshold: %s)" % (reference["status"], reference["below_threshold"])),
                ("resolution", reference["resolution"] or "none recorded (nothing to resolve when the status is ok)"),
                ("checked record", "%s (sha256 %s)" % (bound["path"], bound["sha256"][:12]))]), ""]
    out.append("- Quoted source, Table 12 (%s; %s, %s; %s): %s." % (
        t12["title"], t12["language"], t12["domain"], t12["setting"],
        "; ".join("%s (%s encoder, θ %s) %s" % (x["model"], x["encoder"], x["theta"], x["micro_f1"])
                  for x in t12["values"])))
    out.append("- Quoted source, Table 18 (%s): %s %s; lower reference %s." % (t18["setup"], t18["model"],
                                                                             t18["micro_f1"], ref["lower_reference"]))
    out.append("- Not like-for-like. Setup differences: %s." % "; ".join(reference["differences"]))
    out.append("- Caveats: %s." % "; ".join(reference["caveats"]))
    # the candidates
    rows = []
    for c in cands:
        mark = ("chosen" if c["dump_run_id"] == chosen["dump_run_id"] else
                "runner-up" if runner is not None and c["dump_run_id"] == runner["dump_run_id"] else "")
        rows.append(("%d" % c["epoch"], "%.4f" % c["temperature"], "%.5f" % c["nll_fitted"], "%.5f" % c["nll_raw"],
                     "%.4f" % c["val_micro_f1_at_0_5"], postfit._f1(c), "%.2f" % c["theta"]["shipped"]["global"],
                     "%.2f" % c["theta"]["raw"]["global"], mark))
    out += ["", "**Candidates** (one full val dump per epoch export of %s, encoder LR %g; the objective is micro-F1 at "
                "the global θ, as 2TP/(2TP + FP + FN) unreduced; the reference check above reduces it)"
            % (chosen["train_run_id"], chosen["encoder_lr"]), "",
            md(("epoch", "T", "NLL at T", "NLL at T = 1", "micro-F1 @0.5", "micro-F1 @global θ", "θ shipped", "θ raw",
                "pick"), rows)]
    # the protocol changes after a val outcome (the reference-check fixes)
    items = list(superseded)
    if items:
        prows, notes = [], []
        for it in items:
            s, rec, sref = it["entry"], it["record"], it["reference"]
            sc = rec["chosen"]
            change = "; ".join("%s: %s -> %s" % (k, _old_value(k, sc["canonical"]),
                                                 "%g" % d if isinstance(d, float) else d)
                               for k, d in sorted(s["delta"].items()))
            g = s["user_choice"]
            prows.append((s["train_run_id"], change, "%d of %d" % (sc["epoch"], rec["protocol"]["epochs"]),
                          _ref_f1(sref), "%s (resolution: %s)" % (sref["status"], sref["resolution"] or "none"),
                          "%s: `%s`" % (g["date"], g["reply"]), "%s/" % s["dir"]))
            notes.append("- %s: %s." % (s["train_run_id"], s["change"]))
        n_laya, n_bert = len(laya["lr_grid"]), len(items) + 1   # Laya: one selection run per encoder LR
        out += ["", "**Protocol changes after a val outcome (reference-check fixes)** (each discarded run below scored "
                    "under 0.707 at val; the user chose the fix after that outcome; the run is kept as disclosed "
                    "history, never current)", "",
                md(("discarded train run", "changed by the fix (old -> new)", "chosen epoch", "val micro-F1 at the "
                    "global θ", "status", "user's choice: date and reply", "history"), prows), ""]
        out += notes
        more = n_bert - n_laya
        out.append("- BERTurk had %d more selection run%s than Laya: %d in all (the current %s and the %d discarded "
                   "above), against Laya's %d; every recipe change above was made after a val outcome, a disclosed "
                   "deviation from equal tuning effort."
                   % (more, "" if more == 1 else "s", n_bert, chosen["train_run_id"], len(items), n_laya))
    if final is not None:
        frows = []
        for e in final["seeds"]:
            frows.append(("%d" % e["seed"], e["train_run_id"], "%d" % len(e.get("epochs") or []), *_gib(e)))
        out += ["", "**Final seeds** (%s, --epochs %d stopped after epoch %d, serving T %.4f; not scored before the "
                    "final scoring)"
                % (final["final_split"], chosen["epochs_schedule"], epoch, chosen["temperature"]), "",
                md(("seed", "train run id", "epochs trained", "final-epoch loss", "peak GiB", "largest swap delta MiB"),
                   frows)]
    return "\n".join(out) + "\n"


def disk_need_bytes(model_bytes: int, epochs: int, export_epochs: bool, final: bool) -> int:
    """Bytes a run needs under --storage: one export per epoch, the final checkpoint and a 1 GiB margin (no Trainer
    checkpoint, optimizer state or RNG pickle is ever written: a run is never resumed)."""
    return (epochs * model_bytes if export_epochs else 0) + (model_bytes if final else 0) + 2**30
