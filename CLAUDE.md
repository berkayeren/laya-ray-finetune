# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fine-tunes and evaluates the Laya `laya-multilingual` model on the Turkish banking split of Multi3NLU++ (multi-label intent detection, 48 tags), using Ray Train for distributed (DDP) training and Ray Data for distributed inference. Dependencies are pinned in `pyproject.toml` and hash-locked in `requirements.txt` (Python 3.12, binary wheels only); `make env` installs them into `.venv`. There is no linter. The scripts are run directly.

## Commands

```bash
# Environment and data (pinned, hash-verified)
make env && source .venv/bin/activate      # python3.12 venv + pip install --require-hashes --only-binary :all:
make data                                  # pinned Multi3NLU++ -> data/splits/*.jsonl, data/audit/, data/splits.json
make test                                  # pytest -m "not mps" in the ACTIVE interpreter (or: make test PYTHON=.venv/bin/python)
make weights                               # optional prefetch of the pinned Laya + BERTurk weights (~1.1 GB)
make lock / make lock-check                # maintainer: regenerate the lock / prove it resolves on all 3 platforms

# Train (CPU smoke test; GPU: drop --freeze-encoder/--max-messages, add --gpu)
python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --out ckpt \
    --workers 2 --freeze-encoder --max-messages 100 --epochs 2

# Evaluate (--workers 0 runs in-process without Ray, useful for debugging; --n N subsamples)
python evaluate.py data/splits/test.jsonl --checkpoint ./ckpt --workers 0 --n 20
python evaluate.py data/splits/test.jsonl --checkpoint multilingual --workers 4 --gpus-per-worker 1 --out results/x.json
```

## Architecture

- **Task framing:** each of the 48 intent tags in `intents.py` (`INTENTS`) becomes one Laya `noul` (yes/no) question, so one forward pass gives a probability per tag. `QUESTIONS` is defined only in `intents.py` and imported by `pilot/data.py` and `evaluate.py` (a test fails on a second definition). If you change tag wording, regenerate the data and re-run the baselines, because the question text is the model input; `data/splits.json` records `questions_sha256`, so a wording change without a refreeze fails `make test`.
- **Splits:** `make data` writes five outputs to `data/splits/`: `selection_train` (folds 0–13, 1,450 msgs), `val` (14–15, 208), `final_train` (0–15, 1,658), `test` (16–19, 413) and `hotels_probe` (HOTELS 0–19, 1,009, out-of-domain; only the 22 `ood_scoring_tags` are valid gold). Every row labels all 48 tags; no-intent messages are kept by default (`--no-include-no-intent` to another `--out`; consumers read that variant with `--manifest OTHER_DIR/splits.json`, which may not change any split's role, path or folds versus `pilot.data.SPLITS`; outputs record `manifest_is_committed`). The committed `data/splits.json` registers each file's path, role and sha256. `pilot.data.verify_split_file` is the only verifier and hard-fails on a tampered, unregistered or wrong-role file; there is no override (regenerate with `make data`, or deliberately refreeze with `prepare_data.py --rewrite-manifest`, which invalidates downstream results). Duplicate uids hard-fail at prepare time.
- **JSONL schema:** `{"state": text, "uid", "questions": {tag: {"type": "noul", "instructions": ...}}, "expected": {tag: bool}}`. This is Laya's training format. `evaluate.py` reads gold labels from `expected`.
- **`pilot/`:** `pins.py` (every pinned revision and sha256: dataset, 40 raw folds, Laya, BERTurk, OOD tag subset), `data.py` (split producer + `verify_split_file`), `audit.py` (report-only duplicate audit to `data/audit/`, hash-registered in the manifest, never changes splits), `weights.py` (`fetch(pin)` downloads only the pinned safetensors files at the full commit SHA and checks their sha256; `make weights`).
- **`tests/`:** pytest with strict markers `unit` (offline), `data` (pinned dataset download), `weights` (pinned weights), `ray`, `mps`. `make test` = `pytest -m "not mps"`. Guards include installed == lock, single `QUESTIONS` source, no unpinned load calls, split contracts, and repository hygiene (only `data/splits.json` tracked under `data/`, no message text, no local paths).
- **`ray_train.py`:** `laya-train` is single-device, so this script re-implements the training loop inside `TorchTrainer.train_loop_per_worker` using Laya internals from `laya.train` (`items_from_rows`, `split_calibration`, `encode_item`, `_forward`, `rlcd_loss`/`soft_ce_loss`, `calibration_records`, `save_checkpoint`) and `laya.calibrate.fit_temperature_map`. Because it depends on private/internal Laya APIs, check them against the installed `laya` version when something breaks. `--data` must be a registered split with role `train` (verified before `ray.init`); `--base multilingual` resolves the pinned weights via `pilot.weights.fetch`, any other value must be a local checkpoint dir. Key invariants:
  - Every rank builds identical items and an identical calibration split from the same seed (whole messages are held out for calibration, never single (message, question) items), then takes an **equal-sized** shard (the remainder is dropped) so DDP collectives stay in step.
  - Gradient accumulation uses `model.no_sync()` on non-step micro-batches.
  - Only rank 0 attaches checkpoints to `ray.train.report`. Intermediate epochs report an uncalibrated checkpoint. After training, rank 0 fits temperature calibration on the held-out messages and writes the final config (`temperature`, `temperature_by_options`, `training.ray_train`). The driver copies `result.checkpoint` to `--out`. The output loads with `laya.load(path)`.
  - The script does not fit abstention thresholds or produce `laya-train`'s before/after report. Held-out metrics come from `evaluate.py`.
- **`evaluate.py`:** a Ray Data `map_batches` with the `Predictor` actor class (one checkpoint load per actor). The input must be a registered split with role `test` or `val` and a matching sha256 (train splits and `hotels_probe` are refused). `--checkpoint multilingual|DIR`: `multilingual` loads the pinned, digest-verified hub weights; anything else must be a local Laya checkpoint dir. It reports top-1 hit vs. a majority baseline whose tag is the most frequent one in the matching training split, never the evaluated labels (`final_train` for `test`, `selection_train` for `val`; both over messages with at least one gold tag; `n_top1`/`n_no_intent` give the counts), micro-F1/exact match at thresholds 0.3/0.5/0.7/0.9, macro-F1 and best/worst per-tag F1 at 0.5. The test set is shuffled with a fixed seed of 0 before `--n` truncation, so subsets are comparable across runs.
- **`results/`:** JSON reports from `evaluate.py --out`. The two existing files use the legacy protocol (no no-intent rows, 100 test messages); the README's results table summarizes them.

## Resource notes

Each Ray worker holds about 2.6 GB. Two CPU workers ran out of memory in a 7 GB sandbox, so plan for 16 GB or more per worker. `ray_train.py` is single-node: workers read the driver-local base weights and split, so it refuses a multi-node Ray cluster. `--freeze-encoder` trains the head only. It fits small machines but gains much less. Disk: about 1.5 GB venv + 1.1 GB weights + a few MB data.
