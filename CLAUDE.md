# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fine-tunes and evaluates the Laya `laya-multilingual` model on the Turkish banking split of Multi3NLU++ (multi-label intent detection, 48 tags), using Ray Train for distributed (DDP) training and Ray Data for distributed inference. There is no package config, test suite, or linter. The scripts are run directly. Dependencies: `pip install laya "ray[train,data]"`.

## Commands

```bash
# Data: DATA_DIR holds turkish/banking fold0..19.json from uoe-nlp/multi3-nlu
python prepare_data.py DATA_DIR data_full              # all 48 tags labelled per message
python prepare_data.py DATA_DIR data_neg8 --neg 8      # positives + 8 random negatives (cheaper training)
laya-train --data data_neg8/train.jsonl --base multilingual --dry-run   # validate JSONL format

# Train (CPU smoke test; GPU: drop --freeze-encoder/--max-messages, add --gpu)
python ray_train.py --data data_neg8/train.jsonl --base multilingual --out ckpt \
    --workers 2 --freeze-encoder --max-messages 100 --epochs 2

# Evaluate (--workers 0 runs in-process without Ray, useful for debugging; --n N subsamples)
python evaluate.py data_full/test.jsonl --checkpoint ./ckpt --workers 0 --n 20
python evaluate.py data_full/test.jsonl --checkpoint multilingual --workers 4 --gpus-per-worker 1 --out results/x.json
```

## Architecture

- **Task framing:** each of the 48 intent tags in `intents.py` (`INTENTS`) becomes one Laya `noul` (yes/no) question, so one forward pass gives a probability per tag. `prepare_data.py` and `evaluate.py` both build `QUESTIONS` from `INTENTS`. If you change tag wording, regenerate the data and re-run the baselines, because the question text is the model input.
- **Splits:** train is folds 0–15 and test is folds 16–19 (hardcoded in `prepare_data.py`). The test JSONL always labels all 48 tags. `--neg` only affects train.
- **JSONL schema:** `{"state": text, "uid", "questions": {tag: {"type": "noul", "instructions": ...}}, "expected": {tag: bool}}`. This is Laya's training format. `evaluate.py` reads gold labels from `expected`.
- **`ray_train.py`:** `laya-train` is single-device, so this script re-implements the training loop inside `TorchTrainer.train_loop_per_worker` using Laya internals from `laya.train` (`items_from_rows`, `split_calibration`, `encode_item`, `_forward`, `rlcd_loss`/`soft_ce_loss`, `calibration_records`, `save_checkpoint`) and `laya.calibrate.fit_temperature_map`. Because it depends on private/internal Laya APIs, check them against the installed `laya` version when something breaks. Key invariants:
  - Every rank builds identical items and an identical calibration split from the same seed, then takes an **equal-sized** shard (the remainder is dropped) so DDP collectives stay in step.
  - Gradient accumulation uses `model.no_sync()` on non-step micro-batches.
  - Only rank 0 attaches checkpoints to `ray.train.report`. Intermediate epochs report an uncalibrated checkpoint. After training, rank 0 fits temperature calibration on the held-out slice and writes the final config (`temperature`, `temperature_by_options`, `training.ray_train`). The driver copies `result.checkpoint` to `--out`. The output loads with `laya.load(path)`.
  - The script does not fit abstention thresholds or produce `laya-train`'s before/after report. Held-out metrics come from `evaluate.py`.
- **`evaluate.py`:** a Ray Data `map_batches` with the `Predictor` actor class (one checkpoint load per actor). `--checkpoint` accepts the hub names `multilingual|english|typed-decisions` (loaded from `convaiinnovations/laya`) or a local dir. It reports top-1 hit vs. a majority baseline, micro-F1/exact match at thresholds 0.3/0.5/0.7/0.9, macro-F1 and best/worst per-tag F1 at 0.5. The test set is shuffled with a fixed seed of 0 before `--n` truncation, so subsets are comparable across runs.
- **`results/`:** JSON reports from `evaluate.py --out`. The README's results table summarizes them.

## Resource notes

Each Ray worker holds about 2.6 GB. Two CPU workers ran out of memory in a 7 GB sandbox, so plan for 16 GB or more per worker. On multi-node clusters, pass a shared `--storage` (NFS/S3) to `ray_train.py`. `--freeze-encoder` trains the head only. It fits small machines but gains much less.
