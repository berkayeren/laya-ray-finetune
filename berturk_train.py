"""Fine-tune BERTurk (dbmdz/bert-base-turkish-cased at its pinned revision) with the HF Trainer as a 48-label
multi-label intent classifier (Phase 6, BASE-01): the baseline Laya is compared against.

One process, no Ray (PROJECT.md: "BERTurk is trained with the HF Trainer, not the Ray harness"): --device cpu, mps or
cuda. The recipe is the one the user confirmed at plan 06-02 (pilot.berturk holds the values):
- A1: one LR for encoder and head, --lr (default 2e-05, Laya's LR); the canonical config records it as encoder_lr and
  head_lr, so pilot.postfit selects BERTurk candidates with the same code as Laya's.
- A2: --epochs (default 10); every epoch writes an export, every export is a val selection candidate.
- A3: batch 32 messages (--batch), a linear schedule with a 10% warm-up, AdamW (optim adamw_torch, weight decay 0.01),
  max grad norm 1.0, max_len 128 WordPiece tokens with [CLS] and [SEP] (--max-len; longer messages are truncated and
  counted, n_truncated), fp32 (no fp16, bf16 or autocast), no gradient checkpointing, default dropout,
  transformers.set_seed(--seed) right before the classifier head is created, seed and data_seed for the Trainer.
- A6: no resume. A run takes minutes, so it is rerun from scratch: an existing <storage>/<run_id>/ is refused, naming
  the dir to delete. No Trainer checkpoint, training_args.bin, optimizer, scheduler or RNG pickle is written or read
  (save_strategy "no"); the exports are safetensors plus JSON. The Trainer's output_dir is removed after training, and
  a run is not marked done if the Trainer wrote anything into it.
- A7: the loss is a masked BCE given to the Trainer as compute_loss_func: a masked (message, tag) label is -100, and
  the loss is the BCE summed over the unmasked entries divided by the Trainer's num_items_in_batch (its count of labels
  != -100 over the accumulation window, clamped at 1). Unmasked, it equals the mean BCE over all 48 tags; a masked tag
  gets exactly zero gradient; an all-masked batch gives 0, never NaN. --mask-tags masks tags for every message (the
  few-shot path); Phase 6 trains unmasked. A row that does not label a tag is refused, never read as a negative.

Labels: pilot.berturk.label_vector builds every row's 48 labels in INTENTS order and id2label/label2id come from the
same order, so config.json records the column order and score.py refuses any other (check_label_order). Text is
never lowercased or case-folded: the pinned cased WordPiece tokenizer (do_lower_case false, refused otherwise) is used
as is, loaded from local files.

Checkpoint layout: each epoch k writes <storage>/<run_id>/exports/epoch-NNN (a temp sibling, then one rename) holding
exactly model.safetensors and config.json (model.save_pretrained, fp32), vocab.txt and tokenizer_config.json copied
byte for byte from the base (never tokenizer.save_pretrained, whose tokenizer_config.json depends on HF_HUB_OFFLINE),
and berturk_train.json (schema berturk-train/1: run id, canonical config, epoch, epoch schedule, calibrated false,
temperature 1.0, postfit None, the run record and provenance). The run record holds the device, parameter device,
dtype, versions, optimizer, n_messages, n_truncated, max_len, bit_reproducible (true on cpu only: MPS is not
bitwise deterministic), the load time, one row per completed epoch with exactly pilot.berturk.EPOCH_FIELDS
(mean loss, device-synchronised training seconds, optimizer steps, peak memory, swap delta) and the code state
(git sha, dirty flag, diff digest). A finished run writes <storage>/<run_id>/done.json. Records hold ids, digests,
numbers and repo-relative paths only, never a local path or message text.

Run identity: the run id is pilot.berturk.run_id of the canonical config (pilot.berturk.IDENTITY: split, data and base
digests, transformers and accelerate versions, device, every hyperparameter and the seed; never a path or a resource),
e.g. berturk-selection_train-s0-1a2b3c4d. --print-run-id prints it after the split and base checks and exits without
creating anything.

--data must be a split registered in the manifest with role train and a matching sha256, read once and parsed from the
verified bytes. --base berturk (default) fetches the pinned, sha256-verified weights (pilot.weights.fetch); any other
value must be a local dir holding config.json, model.safetensors, tokenizer_config.json and vocab.txt (its tokenizer
files enter the run identity by digest). Every refusal comes before any model load. Module-level imports stay
torch-free: torch, transformers and the ray_train helpers are imported in main after the MPS env is set (torch reads
PYTORCH_ENABLE_MPS_FALLBACK once, at import).

usage:
  python berturk_train.py --data data/splits/selection_train.jsonl --device mps        # the selection run (10 epochs)
  python berturk_train.py --data data/splits/selection_train.jsonl --print-run-id      # its run id, nothing created
  python berturk_train.py --data ... --base DIR --device cpu --epochs 2 --batch 4 --max-len 32 --storage DIR  # smoke
"""
import argparse
import importlib.metadata
import json
import os
import shutil
import time

from pilot import berturk, pins, runs, spike
from pilot.data import (DEFAULT_MANIFEST, manifest_is_committed, parse_jsonl, repo_relpath, sha256_file, write_atomic,
                        verify_split_file)
from pilot import dumps


def masked_bce(outputs, labels, num_items_in_batch=None):
    """The Trainer's compute_loss_func (A7): BCE with logits summed over the (message, tag) entries whose label is not
    pilot.berturk.MASK_LABEL, divided by num_items_in_batch (the Trainer's count of labels != -100 over the accumulation
    window) or, when None, by this batch's count; the divisor is clamped at 1, so an all-masked batch gives 0."""
    import torch
    logits = outputs["logits"]
    if logits.dtype != torch.float32:
        raise SystemExit("BERTurk logits are %s; training is fp32 (A3): no fp16, bf16 or autocast" % logits.dtype)
    labels = labels.to(logits.dtype)
    valid = labels != berturk.MASK_LABEL
    per = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.clamp(min=0.0), reduction="none")
    total = torch.where(valid, per, torch.zeros_like(per)).sum()
    n = valid.sum() if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(n):
        return total / n.to(total.device).clamp(min=1).to(total.dtype)
    return total / max(int(n), 1)


def parser():
    ap = argparse.ArgumentParser(description="fine-tune BERTurk with the HF Trainer (48-label multi-label, masked BCE)")
    ap.add_argument("--data", required=True, help="a registered split with role train")
    ap.add_argument("--base", default="berturk",
                    help="berturk (the pinned, digest-verified weights; default) or a local base dir holding %s"
                         % ", ".join(berturk.BASE_FILES))
    ap.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    ap.add_argument("--epochs", type=int, default=berturk.EPOCHS)
    ap.add_argument("--lr", type=float, default=berturk.LR, help="one LR for encoder and head (A1)")
    ap.add_argument("--batch", type=int, default=berturk.BATCH, help="messages per optimizer step")
    ap.add_argument("--max-len", type=int, default=berturk.MAX_LEN,
                    help="WordPiece tokens per message with [CLS] and [SEP]; longer ones are truncated and counted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-tags", default=None,
                    help="comma-separated INTENTS tags whose labels are masked for every message (default none)")
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
    return ap


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit("%s is not installed; run `make env` (the HF Trainer needs it)" % dist) from None


def _read_json(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except ValueError as e:
        raise SystemExit("%s is not valid JSON (%s)" % (what, e))


def check_float32(model):
    """SystemExit unless every parameter is float32 (A3: the run is fp32 end to end)."""
    import torch
    bad = sorted({str(p.dtype) for p in model.parameters() if p.dtype != torch.float32})
    if bad:
        raise SystemExit("model parameters in %s; BERTurk trains in fp32 (A3)" % ", ".join(bad))


def write_export(model, base_dir, run_dir, epoch, rec):
    """<run_dir>/exports/epoch-NNN: model.save_pretrained (config.json + model.safetensors, fp32), the base's tokenizer
    files byte for byte, then the record; written to a temp sibling and renamed, so it is whole or absent. An existing
    export is refused (a run dir is never reused, A6)."""
    name = "epoch-%03d" % epoch
    final = os.path.join(run_dir, "exports", name)
    if os.path.lexists(final):
        raise SystemExit("exports/%s already exists; a BERTurk run dir is never reused (A6): delete %s, then rerun"
                         % (name, run_dir))
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    model.save_pretrained(tmp)
    for f in berturk.TOKENIZER_FILES:
        shutil.copyfile(os.path.join(base_dir, f), os.path.join(tmp, f))
    with open(os.path.join(tmp, berturk.RECORD), "wb") as f:
        f.write(berturk.record_bytes(rec))
    have = sorted(os.listdir(tmp))
    if have != sorted(berturk.CHECKPOINT_FILES):
        raise SystemExit("export %s holds %s, expected exactly %s; check the transformers version"
                         % (name, have, sorted(berturk.CHECKPOINT_FILES)))
    os.rename(tmp, final)
    return final


def remove_trainer_dir(path):
    """Remove the Trainer's output_dir, which save_strategy "no" leaves empty; SystemExit naming what it holds
    otherwise (A6: no Trainer checkpoint, training_args.bin or RNG pickle), so such a run is never marked done."""
    if not os.path.isdir(path):
        return
    left = sorted(os.listdir(path))
    if left:
        raise SystemExit("the Trainer wrote %s under %s; a BERTurk run keeps no Trainer state (A6): check the "
                         "transformers version and the TrainingArguments" % (", ".join(left), path))
    os.rmdir(path)


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    for flag, v in (("--epochs", a.epochs), ("--batch", a.batch), ("--max-len", a.max_len),
                    ("--max-messages", a.max_messages), ("--cpus", a.cpus)):
        if v is not None and v < 1:
            ap.error("%s must be at least 1" % flag)
    if not a.lr > 0:
        ap.error("--lr must be positive")
    cpus = a.cpus or (4 if a.device == "mps" else 1)
    if not os.path.isfile(a.data):
        raise SystemExit("%s not found; check the path, or run `make data`" % a.data)
    with open(a.data, "rb") as f:
        raw = f.read()                                      # hash and parse the same bytes (no reread window)
    entry = verify_split_file(a.data, manifest_path=a.manifest, roles=("train",), data=raw)
    rows = parse_jsonl(raw)
    if a.max_messages is not None:
        rows = rows[:a.max_messages]
    if not rows:
        raise SystemExit("%s: no messages to train on" % a.data)
    mask = berturk.check_mask_tags(a.mask_tags.split(",")) if a.mask_tags else ()
    labels = [berturk.label_vector(r, mask) for r in rows]

    if a.base == "berturk":                                 # pinned + verified before any load (ENV-02)
        from pilot.weights import fetch
        base_dir = fetch(pins.BERTURK)
        digests = dict(pins.BERTURK["sha256"])
    else:
        base_dir = os.path.abspath(a.base)
        missing = [f for f in berturk.BASE_FILES if not os.path.isfile(os.path.join(base_dir, f))]
        if missing:
            raise SystemExit("--base must be 'berturk' (pinned) or a local base dir holding %s; %s lacks %s"
                             % (", ".join(berturk.BASE_FILES), a.base, ", ".join(missing)))
        digests = {f: sha256_file(os.path.join(base_dir, f)) for f in berturk.BASE_FILES}
    berturk.check_cased(_read_json(os.path.join(base_dir, "tokenizer_config.json"), "base tokenizer_config.json"),
                        "base")
    canon = berturk.canonical({
        "split": os.path.splitext(os.path.basename(entry["path"]))[0], "data_sha256": entry["sha256"],
        "data_role": entry["role"], "base_model_sha256": digests["model.safetensors"],
        "base_config_sha256": digests["config.json"],
        "base_files_sha256": (None if a.base == "berturk"
                              else dumps.files_digest({f: digests[f] for f in berturk.TOKENIZER_FILES})),
        "transformers_version": _version("transformers"), "accelerate_version": _version("accelerate"),
        "device": a.device, "epochs": a.epochs, "batch": a.batch, "encoder_lr": a.lr, "head_lr": a.lr,
        "weight_decay": berturk.WEIGHT_DECAY, "warmup_ratio": berturk.WARMUP_RATIO,
        "lr_scheduler": berturk.LR_SCHEDULER, "max_grad_norm": berturk.MAX_GRAD_NORM, "optim": berturk.OPTIM,
        "loss": berturk.LOSS, "max_len": a.max_len, "max_messages": a.max_messages,
        "mask_tags": list(mask) or None, "seed": a.seed, "stop_after_epoch": None, "postfit_sha256": None})
    rid = berturk.run_id(canon)
    if a.print_run_id:
        print(rid)
        return
    storage = os.path.abspath(a.storage)
    run_dir = os.path.join(storage, rid)
    if os.path.lexists(run_dir):
        raise SystemExit("%s exists; a BERTurk run is rerun from scratch, never resumed (A6): delete %s, then rerun"
                         % (run_dir, run_dir))
    need = berturk.disk_need_bytes(os.path.getsize(os.path.join(base_dir, "model.safetensors")), a.epochs,
                                   a.export_epochs, False)
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
    transformers.set_seed(a.seed)                           # right before the head is created (Pitfall 6)
    model = transformers.BertForSequenceClassification.from_pretrained(
        base_dir, num_labels=len(berturk.TAGS), problem_type=berturk.PROBLEM_TYPE, id2label=berturk.id2label(),
        label2id=berturk.label2id(), dtype=torch.float32, local_files_only=True)
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
    reset, sync, sample, _, peak = memory_sampler(device)
    record = {"device": a.device, "param_device": None, "dtype": "torch.float32", "amp": False,
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "accelerate_version": canon["accelerate_version"], "optim": berturk.OPTIM, "n_messages": len(rows),
              "n_truncated": n_truncated, "max_len": a.max_len, "bit_reproducible": a.device == "cpu",
              "torch_threads": torch.get_num_threads(), "load_s": load_s, "epochs": [], "code": code_state()}
    provenance = {"split": {"name": canon["split"], "path": entry["path"], "sha256": entry["sha256"],
                            "role": entry["role"]},
                  "manifest": {"path": repo_relpath(a.manifest), "sha256": sha256_file(a.manifest),
                               "is_committed": manifest_is_committed(a.manifest)},
                  "base": {"name": "berturk" if a.base == "berturk" else repo_relpath(base_dir),
                           "revision": pins.BERTURK["revision"] if a.base == "berturk" else None,
                           "model_sha256": digests["model.safetensors"]}}

    def doc(epoch):
        return {"schema": berturk.RECORD_SCHEMA, "run_id": rid, "canonical": canon, "epoch": epoch,
                "epochs_schedule": a.epochs, "stop_after_epoch": None, "calibrated": False, "temperature": 1.0,
                "postfit": None, "record": record, "provenance": provenance}

    ep = {"epoch": 0, "loss_sum": 0.0, "loss_n": 0, "steps": 0, "t0": 0.0, "swap0": 0}

    def loss_fn(outputs, labels, num_items_in_batch=None):
        loss = masked_bce(outputs, labels, num_items_in_batch)
        ep["loss_sum"] += float(loss.detach())
        ep["loss_n"] += 1
        return loss

    def check_model(model):
        require_param_device(model, a.device)
        check_float32(model)

    def on_epoch_begin(args, state, control, **kw):
        ep.update(epoch=ep["epoch"] + 1, loss_sum=0.0, loss_n=0, steps=0, swap0=spike.swap_used())
        reset()
        sync()
        ep["t0"] = time.perf_counter()

    def on_step_end(args, state, control, **kw):
        ep["steps"] += 1                                    # called once per optimizer step
        sample()

    def on_epoch_end(args, state, control, model=None, **kw):
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

    cb = transformers.TrainerCallback()                     # an instance with assigned hooks, no subclass
    cb.on_epoch_begin, cb.on_step_end, cb.on_epoch_end = on_epoch_begin, on_step_end, on_epoch_end
    os.makedirs(run_dir)
    trainer = transformers.Trainer(model=model, args=args, train_dataset=data,
                                   data_collator=transformers.DataCollatorWithPadding(tok),
                                   compute_loss_func=loss_fn, callbacks=[cb])
    check_model(trainer.model)
    record["param_device"] = a.device
    trainer.train()
    if ep["epoch"] != a.epochs:
        raise SystemExit("the run stopped after %d of %d epochs" % (ep["epoch"], a.epochs))
    remove_trainer_dir(args.output_dir)
    write_atomic(os.path.join(run_dir, berturk.DONE), berturk.record_bytes(doc(a.epochs)))
    print("%s: %d epochs -> %s" % (rid, a.epochs, os.path.join(run_dir, "exports")))
    return run_dir


if __name__ == "__main__":
    main()
