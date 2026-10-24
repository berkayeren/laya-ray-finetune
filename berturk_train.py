"""Fine-tune BERTurk (dbmdz/bert-base-turkish-cased at its pinned revision) with the HF Trainer as a 48-label
multi-label intent classifier: the BERTurk baseline that Laya is compared against.

One process, no Ray (BERTurk is trained with the HF Trainer, not the Ray harness): --device cpu, mps or cuda. The
recipe is the one the user confirmed before the first BERTurk run (pilot.berturk holds the values):
- LRs: the encoder LR, --lr (default 2e-05, Laya's single encoder LR), and the classifier head's LR, --head-lr (default
  1e-4, pilot.berturk.HEAD_LR = Laya's head LR): the first fix after the BERTurk reference check (2026-10-21), which
  replaced one LR for encoder and head. The head is exactly the parameters from_pretrained newly initialised
  (classifier.weight, classifier.bias; refused otherwise, before the run dir). The canonical config records encoder_lr
  and head_lr, so pilot.postfit selects BERTurk candidates with the same code as Laya's.
- Schedule: --epochs (default pilot.berturk.EPOCHS = 30, the second reference-check fix; the linear schedule and its
  10% warm-up span all epochs); every epoch writes an export, every export is a val selection candidate.
- Optimizer and precision: batch 32 messages (--batch), a linear schedule with a 10% warm-up, AdamW (optim adamw_torch,
  weight decay 0.01) in four param groups (build_optimizer: encoder and head, each split by the Trainer's own
  weight-decay rule, with the Trainer's own AdamW class and kwargs), passed as Trainer(optimizers=(optimizer, None)) so
  the Trainer still builds the schedule and its warm-up over both LRs (one multiplier, no Trainer subclass), max grad
  norm 1.0, max_len 128 WordPiece tokens with [CLS] and [SEP] (--max-len; longer messages are truncated and counted,
  n_truncated), fp32 (no fp16, bf16 or autocast), no gradient checkpointing, default dropout,
  transformers.set_seed(--seed) right before the classifier head is created, seed and data_seed for the Trainer.
- No resume. A run takes minutes, so it is rerun from scratch: an existing <storage>/<run_id>/ is refused, naming
  the dir to delete. No Trainer checkpoint, training_args.bin, optimizer, scheduler or RNG pickle is written or read
  (save_strategy "no"); the exports are safetensors plus JSON. The Trainer's output_dir is removed after training, and
  a run is not marked done if the Trainer wrote anything into it.
- Loss: a masked BCE given to the Trainer as compute_loss_func: a masked (message, tag) label is -100, and the loss is
  the BCE summed over the unmasked entries divided by the Trainer's num_items_in_batch (its count of labels != -100
  over the accumulation window, clamped at 1). Unmasked, it equals the mean BCE over all 48 tags; a masked tag gets
  exactly zero gradient; an all-masked batch gives 0, never NaN. Selection runs and final seeds train unmasked; the
  few-shot modes below mask the held-out tags (--mask-tags is accepted only with --fewshot-base). A row that does not
  label a tag is refused, never read as a negative.

Labels: pilot.berturk.label_vector builds every row's 48 labels in INTENTS order and id2label/label2id come from the
same order, so config.json records the column order and score.py refuses any other (check_label_order). Text is
never lowercased or case-folded: the pinned cased WordPiece tokenizer (do_lower_case false, refused otherwise) is used
as is, loaded from local files.

Checkpoint layout: each epoch k writes <storage>/<run_id>/exports/epoch-NNN (a temp sibling, then one rename) holding
exactly model.safetensors and config.json (model.save_pretrained, fp32), vocab.txt and tokenizer_config.json copied
byte for byte from the base (never tokenizer.save_pretrained, whose tokenizer_config.json depends on HF_HUB_OFFLINE),
and berturk_train.json (schema berturk-train/1: run id, canonical config, epoch, epoch schedule, calibrated false,
temperature 1.0, postfit None, the run record and provenance). The run record holds the device, parameter device, dtype,
versions, optimizer and its groups (optimizer_groups: group, decay, lr, weight_decay, n_tensors; counts, never parameter
names or values), n_messages, n_truncated, max_len, bit_reproducible (true on cpu only: MPS is not bitwise
deterministic), the load time, one row per completed epoch with exactly pilot.berturk.EPOCH_FIELDS (loss: the unweighted
mean of the epoch's per-step batch losses, so the last, partial batch counts as much as a full one; diagnostic only, no
selection reads it, and kept so for comparability with the committed records; device-synchronised training seconds,
optimizer steps, peak memory, swap delta) and the code state (git sha, dirty flag, diff digest). A finished run writes
<storage>/<run_id>/done.json. Records hold ids, digests, numbers and repo-relative paths only, never a local path or
message text.

Final seeds: --stop-after-epoch K trains epochs 1..K of the --epochs schedule (the linear schedule and its warm-up stay
sized by --epochs, so epoch K is the selection trajectory's epoch K; K in 1..--epochs - 1, K = --epochs has one
spelling, the omitted flag). --out DIR writes the last trained epoch as the final checkpoint (a temp sibling, then one
rename; the same files as an export). --postfit RECORD --reference REF (each needs the other, and --out) binds a final
seed by refusal, never by transcription: before any fetch, load or run dir, also with --print-run-id,
pilot.berturk.check_reference_for_final refuses unless REF is the BERTurk reference check of exactly that record (its
sha256), equal to the check recomputed from the record, and reads ok or carries a recorded resolution, and
pilot.postfit.check_final refuses unless --data is the record's final split, --seed one of its final seeds,
--stop-after-epoch the chosen one and every other canonical value the chosen selection run's (naming the first differing
key). The final berturk_train.json serves the val-fitted T: calibrated true, temperature the record's chosen T bit for
bit (a JSON float round trip), postfit = pilot.postfit.final_block. stop_after_epoch and postfit_sha256 enter the run
identity and are left out when None, so every selection run id is unchanged. postfit.py --print-final-commands prints
the three commands.

Few-shot experiment: two modes run the frozen protocol.json few-shot block unchanged, each bound to it by refusal before
any fetch, load or run dir, also with --print-run-id. pilot.protocol.require_frozen() and the compute decision record
(pilot.fewshot.read_mode: primary or fallback mode) come first, before the split is read; the protocol's
rules.base_runs.berturk must name mask_label pilot.berturk.MASK_LABEL and dtype float32; pilot.fewshot.check_source then
proves --data is the frozen source (selection_train) and that every frozen draw regenerates from its verified bytes.
Neither mode takes --postfit, --reference or --max-messages.
- --fewshot-base trains a label-masked base: every selection_train message is kept and the protocol's held-out tags
  (--mask-tags is optional and refused unless it names exactly them) are MASK_LABEL in every label vector, so the
  masked BCE gives them zero gradient; the run refuses to train when any label of those columns is not MASK_LABEL.
  pilot.fewshot.check_base binds the command to rules.base_runs.berturk and to the chosen selection run's canonical
  config, read through the protocol-pinned postfit record.
- --fewshot-draw D --fewshot-k K --base <masked base> continues a masked base (a local berturk_train.py checkpoint,
  never the pinned base) on the frozen support set of (D, K) plus the draw's replay set, in selection_train file
  order (pilot.fewshot.berturk_rows): a support row keeps all 48 labels, a replay row masks the held-out tags. The
  base's run id and canonical config come from its berturk_train.json, read from bytes bound to its identity
  (pilot.berturk.read_record); pilot.fewshot.check_continued binds the epochs, LRs, seed (the draw seed), draw digest
  and base pairing (the recorded mode); --stop-after-epoch is refused. The model continues from the base's own
  fine-tuned classifier head: from_pretrained must report no missing, unexpected or mismatched key (nothing is newly
  initialised), and the head (the classifier.* names) gets its own AdamW groups at --head-lr
  (pilot.berturk.optimizer_groups, continued mode), the encoder at --lr, under the Trainer's linear schedule and its
  10% warm-up over the continued epochs; transformers.set_seed, the Trainer's seed and data_seed are the draw seed.
fewshot (pilot.fewshot.binding) is a run-identity key left out when None (mask_tags already was), so every committed
run id is unchanged. The run record's fewshot entry is the masking proof, counts, tags and digests only, never
a uid: a base {protocol_sha256, stage base, tags, unmasked_held_out_labels (all 0)}; a continued run {protocol_sha256,
stage continued, draw, k, digest, tags, support_rows, replay_rows, new_tag_labels (one label per support row),
replay_unmasked_held_out_labels (0), base {run_id, canonical}}; None for every other run.

Run identity: the run id is pilot.berturk.run_id of the canonical config (pilot.berturk.IDENTITY: split, data and base
digests, transformers and accelerate versions, device, every hyperparameter and the seed; never a path or a resource),
e.g. berturk-selection_train-s0-1a2b3c4d. --print-run-id prints it after the split and base checks and exits without
creating anything.

--data must be a split registered in the manifest with role train and a matching sha256, read once and parsed from the
verified bytes, and holding no committed held-out content under any manifest (pilot.data.refuse_held_out, shared with
ray_train.py). --base berturk (default) fetches the pinned, sha256-verified weights (pilot.weights.fetch); any other
value must be a local dir holding config.json, model.safetensors, tokenizer_config.json and vocab.txt (its tokenizer
files enter the run identity by digest). Every refusal comes before any model load. Module-level imports stay
torch-free: torch, transformers and the ray_train helpers are imported in main after the MPS env is set (torch reads
PYTORCH_ENABLE_MPS_FALLBACK once, at import).

usage:
  python berturk_train.py --data data/splits/selection_train.jsonl --device mps        # the selection run (30 epochs)
  python berturk_train.py --data data/splits/selection_train.jsonl --print-run-id      # its run id, nothing created
  python berturk_train.py --data ... --base DIR --device cpu --epochs 2 --batch 4 --max-len 32 --storage DIR  # smoke
  python berturk_train.py --data data/splits/final_train.jsonl --device mps --stop-after-epoch K --seed S \
      --postfit results/postfit/berturk.json --reference results/postfit/berturk-reference.json --no-export-epochs \
      --out ckpt/berturk-final-sS                   # a final seed (postfit.py --print-final-commands prints it)
  python berturk_train.py --data data/splits/selection_train.jsonl --device mps --epochs 30 --stop-after-epoch 29 \
      --seed 0 --fewshot-base --no-export-epochs --out ckpt-berturk-fewshot-base-s0     # a few-shot masked base
  python berturk_train.py --data data/splits/selection_train.jsonl --base ckpt-berturk-fewshot-base-s0 --device mps \
      --epochs 10 --seed 0 --fewshot-draw 0 --fewshot-k 10 --no-export-epochs --out ckpt-berturk-fewshot-d0-k10
                                                    # a few-shot continued run from it
"""
import argparse
import importlib.metadata
import json
import os
import shutil
import time
from typing import Any

from pilot import berturk, fewshot, pins, postfit, protocol, runs, spike
from pilot.data import (DEFAULT_MANIFEST, manifest_is_committed, parse_jsonl, refuse_held_out, repo_relpath,
                        sha256_file, write_atomic, verify_split_file)
from pilot import dumps


def masked_bce(outputs: Any, labels: Any, num_items_in_batch: Any = None) -> Any:
    """The Trainer's compute_loss_func: BCE with logits summed over the (message, tag) entries whose label is not
    pilot.berturk.MASK_LABEL, divided by num_items_in_batch (the Trainer's count of labels != -100 over the accumulation
    window, a tensor or an int) or, when None, by this batch's count; the divisor is clamped at 1, so an all-masked
    batch gives 0. outputs maps "logits" to a float32 [batch, 48] tensor; labels is a [batch, 48] tensor; returns the
    scalar loss tensor."""
    import torch
    logits = outputs["logits"]
    if logits.dtype != torch.float32:
        raise SystemExit("BERTurk logits are %s; training is fp32: no fp16, bf16 or autocast" % logits.dtype)
    labels = labels.to(logits.dtype)
    valid = labels != berturk.MASK_LABEL
    per = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.clamp(min=0.0), reduction="none")
    total = torch.where(valid, per, torch.zeros_like(per)).sum()
    n = valid.sum() if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(n):
        return total / n.to(total.device).clamp(min=1).to(total.dtype)
    return total / max(int(n), 1)


def build_optimizer(model: Any, args: Any, head_lr: float, new_names: list[str],
                    continued: bool = False) -> tuple[Any, list[dict[str, Any]]]:
    """(optimizer, groups): the HF Trainer's own AdamW (transformers.Trainer.get_optimizer_cls_and_kwargs of args,
    optim adamw_torch) over four param groups from pilot.berturk.optimizer_groups: the encoder at args.learning_rate,
    the head (exactly new_names, the parameters from_pretrained newly initialised; SystemExit otherwise) at head_lr,
    each split by the Trainer's own weight-decay rule (transformers.Trainer.get_decay_parameter_names, which reads no
    Trainer state in the pinned transformers) at args.weight_decay or 0.0. With head_lr equal to the encoder LR it
    trains exactly as the Trainer's own optimizer (tests/test_berturk_train.py). continued=True (a continued few-shot
    run): the head is the masked base's loaded classifier.* parameters and new_names must be empty. model is a
    transformers BertForSequenceClassification, args its TrainingArguments; returns (torch.optim.AdamW, the groups of
    pilot.berturk.optimizer_groups)."""
    import torch
    import transformers
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = transformers.Trainer.get_decay_parameter_names(None, model)
    cls, kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(args, model)
    if cls is not torch.optim.AdamW or set(kwargs) - {"lr", "betas", "eps"}:
        raise SystemExit("the Trainer's optimizer for optim %s is %s with %s, expected torch.optim.AdamW with lr, "
                         "betas and eps; check the transformers version" % (args.optim, cls, sorted(kwargs)))
    groups = berturk.optimizer_groups([n for n, _ in named], decay, new_names, args.learning_rate, head_lr,
                                      args.weight_decay, continued=continued)
    params = dict(named)
    optimizer = cls([{"params": [params[n] for n in g["params"]], "lr": g["lr"], "weight_decay": g["weight_decay"]}
                     for g in groups], **kwargs)
    return optimizer, groups


def parser() -> argparse.ArgumentParser:
    """The berturk_train.py argument parser."""
    ap = argparse.ArgumentParser(description="fine-tune BERTurk with the HF Trainer (48-label multi-label, masked BCE)")
    ap.add_argument("--data", required=True, help="a registered split with role train")
    ap.add_argument("--base", default="berturk",
                    help="berturk (the pinned, digest-verified weights; default) or a local base dir holding %s"
                         % ", ".join(berturk.BASE_FILES))
    ap.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    ap.add_argument("--epochs", type=int, default=berturk.EPOCHS)
    ap.add_argument("--lr", type=float, default=berturk.LR, help="the encoder LR (Laya's single encoder LR)")
    ap.add_argument("--head-lr", type=float, default=berturk.HEAD_LR,
                    help="the classifier head's LR (Laya's head LR, the first reference-check fix)")
    ap.add_argument("--batch", type=int, default=berturk.BATCH, help="messages per optimizer step")
    ap.add_argument("--max-len", type=int, default=berturk.MAX_LEN,
                    help="WordPiece tokens per message with [CLS] and [SEP]; longer ones are truncated and counted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-tags", default=None,
                    help="with --fewshot-base: the held-out tags, comma-separated; refused unless they are the frozen "
                         "protocol's (the default)")
    ap.add_argument("--max-messages", type=int, default=None,
                    help="train on the first N rows in split-file order (smoke tests)")
    ap.add_argument("--cpus", type=int, default=None, help="torch threads (default 4 on mps, else 1)")
    ap.add_argument("--storage", default=os.path.abspath("berturk_runs"),
                    help="run storage: <storage>/<run_id>/ holds the exports and done.json")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="splits.json to verify --data against (default: the committed data/splits.json)")
    ap.add_argument("--print-run-id", action="store_true",
                    help="print the run id (<storage>/<run_id>/) and exit without creating anything")
    ap.add_argument("--export-epochs", action=argparse.BooleanOptionalAction, default=True,
                    help="write each epoch's export to <storage>/<run_id>/exports/epoch-NNN (default on)")
    ap.add_argument("--stop-after-epoch", type=int, default=None,
                    help="stop after epoch K of the --epochs schedule (the schedule and warm-up keep --epochs): the "
                         "val-chosen epoch of a final seed; 1..--epochs - 1, omit it to train every epoch")
    ap.add_argument("--out", default=None,
                    help="write the last trained epoch here as the final checkpoint (required with --postfit)")
    ap.add_argument("--postfit", default=None,
                    help="the committed BERTurk postfit record: trains a final seed with the val-chosen settings, "
                         "refused unless every setting matches, and serves the val-fitted T (needs --reference, --out)")
    ap.add_argument("--reference", default=None,
                    help="the record's BERTurk reference check (postfit.py --reference-check --out REF); a final seed "
                         "is refused unless it is of that record and reads ok or carries a resolution")
    ap.add_argument("--fewshot-base", action="store_true",
                    help="few-shot: train a label-masked base of the frozen few-shot protocol (protocol.json); refused "
                         "unless every setting is the frozen one")
    ap.add_argument("--fewshot-draw", type=int, default=None,
                    help="few-shot: continue the masked --base from its own head on the frozen support set of this "
                         "draw plus its replay set")
    ap.add_argument("--fewshot-k", type=int, default=None,
                    help="few-shot: the frozen k of the --fewshot-draw support set")
    return ap


def _version(dist: str) -> str:
    """The installed version of a distribution; SystemExit naming `make env` when it is missing."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit("%s is not installed; run `make env` (the HF Trainer needs it)" % dist) from None


def _read_json(path: str, what: str) -> Any:
    """The parsed JSON at path; SystemExit naming `what` on invalid JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except ValueError as e:
        raise SystemExit("%s is not valid JSON (%s)" % (what, e))


def held_out_labels(labels: list[list[float]], tags: list[str]) -> dict[str, int]:
    """{tag: the number of label vectors (INTENTS order) whose tag column is not pilot.berturk.MASK_LABEL}, in `tags`
    order: the masking proof's count."""
    return {t: sum(1 for lab in labels if lab[berturk.TAGS.index(t)] != berturk.MASK_LABEL) for t in tags}


def fewshot_base_of(base_dir: str) -> dict[str, Any]:
    """{"run_id", "canonical"} of a continued run's base, from its berturk_train.json read from bytes bound to its
    identity (pilot.berturk.read_record); SystemExit naming the fix for a config.json that does not score the 48
    INTENTS tags in order, or a run id that is not the one its canonical config names."""
    berturk.check_label_order(_read_json(os.path.join(base_dir, "config.json"), "--base config.json"), "--base")
    rec = berturk.read_record(base_dir, berturk.identity(base_dir))
    rid, canon = rec["run_id"], rec["canonical"]
    if berturk.run_id(canon) != rid:
        raise SystemExit("--base %s names run %s, but its canonical config is run %s; a continued run starts from an "
                         "unedited berturk_train.py --fewshot-base checkpoint" % (repo_relpath(base_dir), rid,
                                                                                  berturk.run_id(canon)))
    return {"run_id": rid, "canonical": canon}


def check_float32(model: Any) -> None:
    """SystemExit unless every parameter of the torch model is float32 (the run is fp32 end to end)."""
    import torch
    bad = sorted({str(p.dtype) for p in model.parameters() if p.dtype != torch.float32})
    if bad:
        raise SystemExit("model parameters in %s; BERTurk trains in fp32" % ", ".join(bad))


def _write_checkpoint(model: Any, base_dir: str, final: str, rec: dict[str, Any]) -> str:
    """A checkpoint dir at `final`: model.save_pretrained (config.json + model.safetensors, fp32), the base's tokenizer
    files byte for byte, then the record; written to a temp sibling and renamed, so it is whole or absent. A temp
    sibling that already exists is refused, never deleted: this process did not make it. Returns `final`."""
    tmp = final + ".tmp"
    if os.path.lexists(tmp):
        raise SystemExit("%s exists (another run's, or one interrupted while writing); inspect it and remove it, then "
                         "rerun" % tmp)
    os.makedirs(tmp)
    model.save_pretrained(tmp)
    for f in berturk.TOKENIZER_FILES:
        shutil.copyfile(os.path.join(base_dir, f), os.path.join(tmp, f))
    with open(os.path.join(tmp, berturk.RECORD), "wb") as f:
        f.write(berturk.record_bytes(rec))
    have = sorted(os.listdir(tmp))
    if have != sorted(berturk.CHECKPOINT_FILES):
        raise SystemExit("checkpoint %s holds %s, expected exactly %s; check the transformers version"
                         % (os.path.basename(final), have, sorted(berturk.CHECKPOINT_FILES)))
    os.rename(tmp, final)
    return final


def write_export(model: Any, base_dir: str, run_dir: str, epoch: int, rec: dict[str, Any]) -> str:
    """<run_dir>/exports/epoch-NNN (_write_checkpoint), returned. An existing export is refused (a run dir is never
    reused)."""
    name = "epoch-%03d" % epoch
    final = os.path.join(run_dir, "exports", name)
    if os.path.lexists(final):
        raise SystemExit("exports/%s already exists; a BERTurk run dir is never reused: delete %s, then rerun"
                         % (name, run_dir))
    return _write_checkpoint(model, base_dir, final, rec)


def write_final(model: Any, base_dir: str, out: str, rec: dict[str, Any]) -> str:
    """The final checkpoint at --out (_write_checkpoint), its absolute path returned: refused when --out was filled
    meanwhile; an empty --out dir (checked before training) is replaced by the rename."""
    out = os.path.abspath(out)
    if os.path.lexists(out):
        if not os.path.isdir(out) or os.path.islink(out) or os.listdir(out):
            raise SystemExit("--out %s exists and is not empty; choose a new dir" % out)
        os.rmdir(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return _write_checkpoint(model, base_dir, out, rec)


def remove_trainer_dir(path: str) -> None:
    """Remove the Trainer's output_dir, which save_strategy "no" leaves empty; SystemExit naming what it holds
    otherwise (a run keeps no Trainer checkpoint, training_args.bin or RNG pickle), so such a run is never marked
    done."""
    if not os.path.isdir(path):
        return
    left = sorted(os.listdir(path))
    if left:
        raise SystemExit("the Trainer wrote %s under %s; a BERTurk run keeps no Trainer state: check the "
                         "transformers version and the TrainingArguments" % (", ".join(left), path))
    os.rmdir(path)


def main(argv: list[str] | None = None) -> str | None:
    """Train one BERTurk run (see the module docstring); returns its run dir, or None after --print-run-id."""
    ap = parser()
    a = ap.parse_args(argv)
    for flag, v in (("--epochs", a.epochs), ("--batch", a.batch), ("--max-len", a.max_len),
                    ("--max-messages", a.max_messages), ("--cpus", a.cpus)):
        if v is not None and v < 1:
            ap.error("%s must be at least 1" % flag)
    if not a.lr > 0:
        ap.error("--lr must be positive")
    if not a.head_lr > 0:
        ap.error("--head-lr must be positive")
    # K = --epochs has one spelling, the omitted flag (one run id per trajectory)
    if a.stop_after_epoch is not None and not 1 <= a.stop_after_epoch < a.epochs:
        ap.error("--stop-after-epoch must be at least 1 and below --epochs (%d); omit it to train every epoch"
                 % a.epochs)
    # a few-shot mode is one frozen-protocol command; every other combination is an argument error
    continued = a.fewshot_draw is not None or a.fewshot_k is not None
    fs_flags = [f for f, on in (("--fewshot-base", a.fewshot_base), ("--fewshot-draw", a.fewshot_draw is not None),
                                ("--fewshot-k", a.fewshot_k is not None), ("--mask-tags", a.mask_tags is not None))
                if on]
    if (a.fewshot_draw is None) != (a.fewshot_k is None):
        ap.error("--fewshot-draw and --fewshot-k go together: a continued run names its frozen draw and k")
    if a.fewshot_base and continued:
        ap.error("--fewshot-base trains a masked base and --fewshot-draw continues one; pass one of them")
    if continued and a.stop_after_epoch is not None:
        ap.error("--fewshot-draw takes no --stop-after-epoch: a continued run trains every frozen epoch on the support "
                 "and replay rows")
    if a.mask_tags is not None and not a.fewshot_base:
        ap.error("--mask-tags masks a few-shot base only: pass it with --fewshot-base (the frozen protocol's tags)")
    if fs_flags and (a.postfit is not None or a.reference is not None):
        ap.error("%s takes no --postfit or --reference: a few-shot run is bound to the frozen protocol, not to a "
                 "postfit record" % fs_flags[0])
    if fs_flags and a.max_messages is not None:
        ap.error("%s takes no --max-messages: a few-shot run trains every message the frozen protocol selects"
                 % fs_flags[0])
    if (a.postfit is None) != (a.reference is None):
        ap.error("--postfit and --reference go together: a final seed needs the record and its reference check")
    if a.postfit is not None and a.out is None:
        ap.error("--postfit needs --out DIR (the final checkpoint)")
    cpus = a.cpus or (4 if a.device == "mps" else 1)
    proto = pin = block = mode = None
    mask = ()
    if a.fewshot_base or continued:                         # the frozen protocol first, before the split is read
        proto, pin = protocol.require_frozen()
        block = proto.get("fewshot")
        if not isinstance(block, dict) or not isinstance(block.get("rules"), dict):
            raise SystemExit("the frozen protocol holds no few-shot block; restore protocol.json with `git checkout -- "
                             "protocol.json`")
        mode = fewshot.read_mode(pin)                       # the compute decision record names primary or fallback
        rule = (block["rules"].get("base_runs") or {}).get("berturk") or {}
        if rule.get("mask_label") != berturk.MASK_LABEL or rule.get("dtype") != "float32":
            raise SystemExit("the frozen rules.base_runs.berturk name mask_label %r and dtype %r; berturk_train.py "
                             "masks with %r in float32, so it cannot run this protocol unchanged"
                             % (rule.get("mask_label"), rule.get("dtype"), berturk.MASK_LABEL))
        if continued and a.base == "berturk":
            raise SystemExit("--fewshot-draw continues a masked base from its own head: pass a berturk_train.py "
                             "--fewshot-base checkpoint as --base, not berturk")
        if a.fewshot_base:
            mask = berturk.check_mask_tags(block.get("tags") or [])   # sorted; label_vector masks these columns
            given = None if a.mask_tags is None else [t.strip() for t in a.mask_tags.split(",")]
            if given is not None and (sorted(given) != list(mask) or len(set(given)) != len(given)):
                raise SystemExit("--mask-tags %s is not the frozen protocol's held-out tags %s; pass them (or omit "
                                 "--mask-tags): the few-shot experiment runs the frozen protocol unchanged"
                                 % (a.mask_tags, ",".join(mask)))
    if not os.path.isfile(a.data):
        raise SystemExit("%s not found; check the path, or run `make data`" % a.data)
    with open(a.data, "rb") as f:
        raw = f.read()                                      # hash and parse the same bytes (no reread window)
    entry = verify_split_file(a.data, manifest_path=a.manifest, roles=("train",), data=raw)
    rows = parse_jsonl(raw)
    refuse_held_out(a.data, entry, rows)                    # every row, keyed on the content
    if a.max_messages is not None:
        rows = rows[:a.max_messages]
    if not rows:
        raise SystemExit("%s: no messages to train on" % a.data)
    fs_proof = sel = None
    if block is None:
        labels = [berturk.label_vector(r, mask) for r in rows]
    else:
        fewshot.check_source(block, entry, rows)            # --data is the frozen source; every draw regenerates
        tags = sorted(block["tags"])
        if a.fewshot_base:                                  # every message kept, the held-out columns MASK_LABEL
            labels = [berturk.label_vector(r, mask) for r in rows]
            unmasked = held_out_labels(labels, tags)
            leaked = ["%s %d" % (t, n) for t, n in unmasked.items() if n]
            if leaked:
                raise SystemExit("the masked base holds unmasked labels of held-out tag(s) (%s); refusing to train: "
                                 "every label of a held-out tag is %r"
                                 % (", ".join(leaked), berturk.MASK_LABEL))
            fs_proof = {"protocol_sha256": pin, "stage": "base", "tags": tags, "unmasked_held_out_labels": unmasked}
        else:                                               # support rows: all 48 labels; replay rows: held-out masked
            sel = fewshot.selection(block, a.fewshot_draw, a.fewshot_k)
            pairs = fewshot.berturk_rows(rows, sel, block["tags"])   # refuses a selected uid missing from the rows
            rows = [r for r, _ in pairs]
            labels = [berturk.label_vector(r, m) for r, m in pairs]
            sup = [lab for r, lab in zip(rows, labels) if r["uid"] in sel["support"]]
            rest = [lab for r, lab in zip(rows, labels) if r["uid"] not in sel["support"]]
            new, replay = held_out_labels(sup, tags), sum(held_out_labels(rest, tags).values())
            off = ["%s %d" % (t, n) for t, n in new.items() if n != len(sup)]
            if off or replay:
                raise SystemExit("the continued run labels held-out tag(s) %s for %d support rows and %d held-out "
                                 "label(s) on replay rows; refusing to train: one label per support row, none on a "
                                 "replay row" % (", ".join(off) or "as expected", len(sup), replay))
            d = block["draws"][a.fewshot_draw]
            fs_proof = {"protocol_sha256": pin, "stage": "continued", "draw": a.fewshot_draw, "k": a.fewshot_k,
                        "digest": d.get("digest"), "tags": tags, "support_rows": len(sup), "replay_rows": len(rest),
                        "new_tag_labels": new, "replay_unmasked_held_out_labels": replay}
    # the postfit record and its reference-check gate (torch-free reads) before any fetch, load or run dir
    pf_record, record_sha = postfit.read_record(a.postfit) if a.postfit else (None, None)
    if pf_record is not None:
        if pf_record.get("system_kind") != "berturk":
            raise SystemExit("--postfit %s: system kind %r; berturk_train.py trains the final seeds of a BERTurk "
                             "record" % (a.postfit, pf_record.get("system_kind")))
        berturk.check_reference_for_final(a.reference, record_sha, pf_record)
    if (a.out is not None and not a.print_run_id and os.path.lexists(a.out)
            and (not os.path.isdir(a.out) or os.listdir(a.out))):
        raise SystemExit("--out %s exists and is not empty; choose a new dir" % a.out)
    if a.out is not None and not a.print_run_id and os.path.lexists(os.path.abspath(a.out) + ".tmp"):
        raise SystemExit("%s.tmp exists (another run's, or one interrupted while writing --out); inspect it and remove "
                         "it, then rerun" % os.path.abspath(a.out))  # before training, as the --out check

    if a.base == "berturk":                                 # the pin names the digests; fetched after the binding
        base_dir = None
        digests = dict(pins.BERTURK["sha256"])
    else:
        base_dir = os.path.abspath(a.base)
        if continued:                                       # a berturk_train.py checkpoint, its record included
            berturk.require_files(base_dir, "--base")
        missing = [f for f in berturk.BASE_FILES if not os.path.isfile(os.path.join(base_dir, f))]
        if missing:
            raise SystemExit("--base must be 'berturk' (pinned) or a local base dir holding %s; %s lacks %s"
                             % (", ".join(berturk.BASE_FILES), a.base, ", ".join(missing)))
        digests = {f: sha256_file(os.path.join(base_dir, f)) for f in berturk.BASE_FILES}
    fs_binding = base_info = None
    if a.fewshot_base:
        fs_binding = fewshot.binding(pin, "base")
    elif continued:                                         # the base names itself by its own record bytes
        base_info = fewshot_base_of(base_dir)
        fs_binding = fewshot.binding(pin, "continued", a.fewshot_draw, a.fewshot_k,
                                     block["draws"][a.fewshot_draw].get("digest"), base_info["run_id"])
        fs_proof["base"] = base_info
    canon = berturk.canonical({
        "split": os.path.splitext(os.path.basename(entry["path"]))[0], "data_sha256": entry["sha256"],
        "data_role": entry["role"], "base_model_sha256": digests["model.safetensors"],
        "base_config_sha256": digests["config.json"],
        "base_files_sha256": (None if a.base == "berturk"
                              else dumps.files_digest({f: digests[f] for f in berturk.TOKENIZER_FILES})),
        "transformers_version": _version("transformers"), "accelerate_version": _version("accelerate"),
        "device": a.device, "epochs": a.epochs, "batch": a.batch, "encoder_lr": a.lr, "head_lr": a.head_lr,
        "weight_decay": berturk.WEIGHT_DECAY, "warmup_ratio": berturk.WARMUP_RATIO,
        "lr_scheduler": berturk.LR_SCHEDULER, "max_grad_norm": berturk.MAX_GRAD_NORM, "optim": berturk.OPTIM,
        "loss": berturk.LOSS, "max_len": a.max_len, "max_messages": a.max_messages,
        "mask_tags": list(mask) or None, "seed": a.seed, "stop_after_epoch": a.stop_after_epoch,
        "postfit_sha256": record_sha, "fewshot": fs_binding})
    pf_block = None
    if pf_record is not None:                               # bound by refusal, never by transcription
        postfit.check_final(pf_record, canon)
        pf_block = postfit.final_block(pf_record, record_sha, a.postfit)   # refuses a T outside laya's band here
    if a.fewshot_base:                                      # bound by refusal to the frozen protocol
        fewshot.check_base(proto, pin, "berturk", canon, fewshot.chosen_canonical(proto, "berturk"), mode)
    elif continued:                                         # the frozen draw, k, epochs, LRs and base
        fewshot.check_continued(proto, pin, "berturk", canon, base_info["canonical"], base_info["run_id"],
                                a.fewshot_draw, a.fewshot_k, mode)
    if a.base == "berturk":                                 # pinned + verified before any load
        from pilot.weights import fetch
        base_dir = fetch(pins.BERTURK)
    berturk.check_cased(_read_json(os.path.join(base_dir, "tokenizer_config.json"), "base tokenizer_config.json"),
                        "base")
    rid = berturk.run_id(canon)
    if a.print_run_id:
        print(rid)
        return
    storage = os.path.abspath(a.storage)
    run_dir = os.path.join(storage, rid)
    if os.path.lexists(run_dir):
        raise SystemExit("%s exists; a BERTurk run is rerun from scratch, never resumed: delete %s, then rerun"
                         % (run_dir, run_dir))
    stop = a.stop_after_epoch or a.epochs                    # the last epoch this run trains
    need = berturk.disk_need_bytes(os.path.getsize(os.path.join(base_dir, "model.safetensors")), stop,
                                   a.export_epochs, a.out is not None)
    free = runs.free_bytes(storage)
    if free < need:
        raise SystemExit("%.1f GiB free under --storage; this run needs about %.1f GiB (one export per epoch); free "
                         "disk or pass another --storage" % (free / 2**30, need / 2**30))

    if a.device == "mps":
        os.environ.update(spike.spike_env("0", cpus))       # strict MPS before torch is imported
    import torch
    import transformers

    from ray_train import code_state, memory_sampler, require_param_device

    if a.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps, but MPS is not available here; use --device cpu")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda, but CUDA is not available here; use --device cpu or mps")
    torch.set_num_threads(cpus)
    from pilot.weights import verify_files
    verify_files(base_dir, digests, "BERTurk base")          # the bytes the identity names, right before the load

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    data, n_truncated = [], 0
    for r, lab in zip(rows, labels):
        n_truncated += len(tok(r["state"])["input_ids"]) > a.max_len
        enc = tok(r["state"], truncation=True, max_length=a.max_len)
        data.append({"input_ids": enc["input_ids"], "token_type_ids": enc["token_type_ids"],
                     "attention_mask": enc["attention_mask"], "labels": lab})
    transformers.set_seed(a.seed)                           # right before the head is created, so it is seeded
    model, info = transformers.BertForSequenceClassification.from_pretrained(
        base_dir, num_labels=len(berturk.TAGS), problem_type=berturk.PROBLEM_TYPE, id2label=berturk.id2label(),
        label2id=berturk.label2id(), dtype=torch.float32, local_files_only=True, output_loading_info=True)
    new_names = info["missing_keys"]                         # the newly initialised parameters: the fresh head
    if continued:                                           # the masked base's whole model, its head included
        off = {k: sorted(str(x) for x in info.get(k) or ()) for k in ("missing_keys", "unexpected_keys",
                                                                        "mismatched_keys")}
        if any(off.values()):
            raise SystemExit("--base %s: from_pretrained reports %s; a continued run loads the masked base's whole "
                             "model, its fine-tuned classifier head included (no fresh head, no weight imprinting): "
                             "pass an unedited berturk_train.py --fewshot-base checkpoint"
                             % (repo_relpath(base_dir), "; ".join("%s %s" % (k, v) for k, v in off.items() if v)))
    load_s = time.time() - t0
    args = transformers.TrainingArguments(
        output_dir=os.path.join(run_dir, "trainer"), per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=1, num_train_epochs=a.epochs, learning_rate=a.lr,
        weight_decay=berturk.WEIGHT_DECAY, warmup_steps=berturk.WARMUP_RATIO, lr_scheduler_type=berturk.LR_SCHEDULER,
        max_grad_norm=berturk.MAX_GRAD_NORM, optim=berturk.OPTIM, seed=a.seed, data_seed=a.seed,
        use_cpu=(a.device == "cpu"), fp16=False, bf16=False, save_strategy="no", eval_strategy="no",
        logging_strategy="epoch", report_to="none", disable_tqdm=True, dataloader_num_workers=0,
        dataloader_pin_memory=False)
    if args.device.type != a.device:
        raise SystemExit("the Trainer picked device %s, requested %s; check the visible devices"
                         % (args.device.type, a.device))
    device = args.device
    model = model.to(device)                                 # before the optimizer holds its parameters
    if continued:                                           # the loaded head of the masked base, nothing new
        optimizer, groups = build_optimizer(model, args, a.head_lr, new_names, continued=True)
    else:
        optimizer, groups = build_optimizer(model, args, a.head_lr, new_names)   # refuses a head that is not fresh
    reset, sync, sample, _, peak = memory_sampler(device)
    record = {"device": a.device, "param_device": None, "dtype": "torch.float32", "amp": False,
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "accelerate_version": canon["accelerate_version"], "optim": berturk.OPTIM,
              "optimizer_groups": [{"group": g["group"], "decay": g["decay"], "lr": g["lr"],
                                    "weight_decay": g["weight_decay"], "n_tensors": len(g["params"])} for g in groups],
              "n_messages": len(rows),
              "n_truncated": n_truncated, "max_len": a.max_len, "bit_reproducible": a.device == "cpu",
              "torch_threads": torch.get_num_threads(), "load_s": load_s, "epochs": [], "code": code_state(),
              "fewshot": fs_proof}
    provenance = {"split": {"name": canon["split"], "path": entry["path"], "sha256": entry["sha256"],
                            "role": entry["role"]},
                  "manifest": {"path": repo_relpath(a.manifest), "sha256": sha256_file(a.manifest),
                               "is_committed": manifest_is_committed(a.manifest)},
                  "base": {"name": "berturk" if a.base == "berturk" else repo_relpath(base_dir),
                           "revision": pins.BERTURK["revision"] if a.base == "berturk" else None,
                           "model_sha256": digests["model.safetensors"]}}

    def doc(epoch: int, final: bool = False) -> dict[str, Any]:
        """berturk_train.json: an export (and done.json) is uncalibrated (T = 1); a --postfit final checkpoint serves
        the record's val-fitted T, copied from final_block without arithmetic (a JSON float round trip)."""
        served = final and pf_block is not None
        return {"schema": berturk.RECORD_SCHEMA, "run_id": rid, "canonical": canon, "epoch": epoch,
                "epochs_schedule": a.epochs, "stop_after_epoch": a.stop_after_epoch, "calibrated": served,
                "temperature": pf_block["temperature"] if served else 1.0, "postfit": pf_block if served else None,
                "record": record, "provenance": provenance}

    ep = {"epoch": 0, "loss_sum": 0.0, "loss_n": 0, "steps": 0, "t0": 0.0, "swap0": 0}

    def loss_fn(outputs: Any, labels: Any, num_items_in_batch: Any = None) -> Any:
        """masked_bce, adding each step's loss to the epoch row's running mean."""
        # the epoch row's loss is the mean over steps, not over items (documented, unchanged, diagnostic only)
        loss = masked_bce(outputs, labels, num_items_in_batch)
        ep["loss_sum"] += float(loss.detach())
        ep["loss_n"] += 1
        return loss

    def check_model(model: Any) -> None:
        """The requested device and fp32 for every parameter."""
        require_param_device(model, a.device)
        check_float32(model)

    def on_epoch_begin(args: Any, state: Any, control: Any, **kw: Any) -> None:
        """Reset the epoch counters, the memory peaks and the clock."""
        ep.update(epoch=ep["epoch"] + 1, loss_sum=0.0, loss_n=0, steps=0, swap0=spike.swap_used())
        reset()
        sync()
        ep["t0"] = time.perf_counter()

    def on_step_end(args: Any, state: Any, control: Any, **kw: Any) -> None:
        """Count one optimizer step and sample memory."""
        ep["steps"] += 1                                    # called once per optimizer step
        sample()

    def on_epoch_end(args: Any, state: Any, control: Any, model: Any = None, **kw: Any) -> Any:
        """The epoch row, the export and the stop after the last epoch this run trains; returns control."""
        sync()
        train_s = time.perf_counter() - ep["t0"]
        sample()
        if round(state.epoch) != ep["epoch"]:
            raise SystemExit("the Trainer reports epoch %s, this callback counted %d" % (state.epoch, ep["epoch"]))
        check_model(model)
        record["epochs"].append({"epoch": ep["epoch"], "loss": ep["loss_sum"] / max(ep["loss_n"], 1),
                                 "train_s": train_s, "optimizer_steps": ep["steps"], "n_messages": len(rows),
                                 "peak_total_bytes": peak["total"], "peak_driver_bytes": peak["driver"],
                                 "peak_rss_bytes": peak["rss"], "swap_delta_bytes": spike.swap_used() - ep["swap0"]})
        if a.export_epochs:
            write_export(model, base_dir, run_dir, ep["epoch"], doc(ep["epoch"]))
        if ep["epoch"] >= stop:                             # the schedule stays sized by --epochs
            control.should_training_stop = True
        return control

    cb = transformers.TrainerCallback()                     # an instance with assigned hooks, no subclass
    cb.on_epoch_begin, cb.on_step_end, cb.on_epoch_end = on_epoch_begin, on_step_end, on_epoch_end
    os.makedirs(run_dir)
    trainer = transformers.Trainer(model=model, args=args, train_dataset=data,
                                   data_collator=transformers.DataCollatorWithPadding(tok),
                                   compute_loss_func=loss_fn, callbacks=[cb], optimizers=(optimizer, None))
    check_model(trainer.model)
    record["param_device"] = a.device
    trainer.train()
    if ep["epoch"] != stop:
        raise SystemExit("the run stopped after %d of %d epochs" % (ep["epoch"], stop))
    remove_trainer_dir(args.output_dir)
    if a.out is not None:
        write_final(trainer.model, base_dir, a.out, doc(stop, final=True))
    write_atomic(os.path.join(run_dir, berturk.DONE), berturk.record_bytes(doc(stop)))
    print("%s: %d of %d epochs -> %s" % (rid, stop, a.epochs, a.out if a.out is not None
                                         else os.path.join(run_dir, "exports")))
    return run_dir


if __name__ == "__main__":
    main()
