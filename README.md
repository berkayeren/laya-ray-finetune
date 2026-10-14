# Laya × Multi3NLU++ (Turkish banking) with Ray

Fine-tune and evaluate [Laya](https://github.com/NandhaKishorM/laya) `laya-multilingual` on the
Turkish banking split of [Multi3NLU++](https://huggingface.co/datasets/uoe-nlp/multi3-nlu)
(CC-BY 4.0), using Ray Train for distributed training and Ray Data for distributed evaluation.

## Task framing

Multi3NLU++ is **multi-label**: each message carries 0–6 of 48 compositional intent tags
(`card`, `lost_stolen`, `request_info`, …). Each tag becomes one Laya `noul` (yes/no)
question, so one forward pass returns a probability per tag. Tag wordings and the single
`QUESTIONS` definition are in `intents.py`.

## Setup

Everything installs from one hash-locked `requirements.txt` (Python 3.12, binary wheels only):

```bash
make env                    # python3.12 venv in .venv + pip install --require-hashes --only-binary :all:
source .venv/bin/activate
make data                   # download the pinned dataset, verify it, write data/splits/ and data/audit/
make test                   # offline pytest in the ACTIVE interpreter (fails if it is not the locked env)
make test-all               # also the network tests: pinned dataset + ~1.1 GB pinned weights on first run
make weights                # optional: prefetch and verify the pinned Laya + BERTurk weights (~1.1 GB)
```

`make test` runs in whatever interpreter is active and prints `sys.executable` first. A test
fails, listing every mismatched package, if that interpreter does not match the lock. Use
`make test PYTHON=.venv/bin/python` to run without activating the venv. `make test` is offline
(`pytest -m "not mps and not data and not weights"`); `make test-all` (`pytest -m "not mps"`) adds
the `data` and `weights` tests, which download the pinned dataset and weights on first run.

Maintainer commands: `make lock` regenerates `requirements.txt` from `pyproject.toml` (universal,
hashed, resolved as of a frozen date), and `make lock-check` proves the lock still resolves
binary-only with hashes on Linux x86_64, Linux aarch64 and macOS arm64.

## Frozen splits

`make data` writes five Laya JSONL files under `data/splits/` (gitignored) and the manifest
`data/splits.json` (committed):

| output | folds | messages | no-intent | role |
|--------|-------|----------|-----------|------|
| `selection_train` | banking 0–13 | 1,450 | 66 | train (model selection) |
| `val` | banking 14–15 | 208 | 7 | val |
| `final_train` | banking 0–15 | 1,658 | 73 | train (final models) |
| `test` | banking 16–19 | 413 | 15 | test |
| `hotels_probe` | hotels 0–19 | 1,009 | 274 (without hotel intents) | out-of-domain probe |

**Hash-verified, no override.** `data/splits.json` records the sha256 of every split and audit
file. `score.py` (and its `evaluate.py` wrapper) and `ray_train.py` refuse any file whose sha256 differs from the manifest, or
that the manifest does not register, before loading a model or starting Ray. There is no flag to
skip this: if a split is missing or stale, regenerate it with `make data`. If a rebuilt manifest
would differ from the committed one, `prepare_data.py` writes nothing. `--rewrite-manifest`
deliberately refreezes `data/splits.json`. That invalidates every result produced from the old
splits, so treat it as a protocol change, not a fix.

**No-intent rows are kept.** Banking messages without any intent stay in as all-negative rows by
default. `python prepare_data.py --no-include-no-intent --out OTHER_DIR` builds the variant without
them (1,384 / 201 / 1,585 / 398 messages). Write it to a different `--out` dir, because prepare
refuses to overwrite the committed manifest with a differing one. Consume it with
`--manifest OTHER_DIR/splits.json` on `score.py`, `report.py`, `evaluate.py` and `ray_train.py`; the files are still
hash-verified against that manifest, and the manifest must match the current code and pins. It
cannot change which file is train, val, test or probe: every split's role, path and folds must equal
`pilot.data.SPLITS`. Reports and checkpoints record `manifest_is_committed`.

**Every row labels all 48 tags.** `final_train` is 1,658 × 48 = 79,584 Laya items. There is no
prepare-time negative sampling. Sampling negatives per run seed arrives with the Phase 4 training
harness.

**The HOTELS probe is out-of-domain only.** It is never trained on. All 48 banking tags are
`false` on it, but hotel messages genuinely carry the 26 generic tags banking shares (`affirm`,
`thank`, …). So only the 22 banking-only tags, recorded as `ood_scoring_tags` in the manifest,
are valid gold when scoring it. `prepare_data.py` re-derives that set from the data and fails if it
drifts from `pilot/pins.py`. Its manifest `n_no_intent` (274) counts messages with no *hotel* intent
in the raw data; every probe row has all 48 banking tags `false`, so it is not a banking no-intent
count like the other splits' `n_no_intent`.

**The duplicate audit is report-only.** `data/audit/audit.md` (with `audit.json` and
`dup_flags.json`) lists exact and Turkish-normalized duplicate messages, conflicting labels, and
test/val messages that duplicate training messages. It never changes split membership. Its files
are hash-registered too, so later phases can trust the flag list.

**Duplicate uids hard-fail.** A repeated uid anywhere in the 3,080 raw banking + HOTELS rows stops
`make data` before any file is written.

## Files

| file | what it does |
|------|--------------|
| `intents.py` | the 48 tags phrased as yes/no questions; the only `QUESTIONS` definition |
| `prepare_data.py` | CLI for `make data`: pinned folds → `data/splits/*.jsonl`, `data/audit/`, `data/splits.json` |
| `pilot/pins.py` | every pinned revision and sha256 (dataset, raw folds, weights, OOD tag subset) |
| `pilot/data.py` | split producer and `verify_split_file`, the one verifier every consumer calls |
| `pilot/audit.py` | duplicate audit (report-only) |
| `pilot/weights.py` | pinned, digest-verified weight fetch (`make weights`) |
| `pilot/laya_adapter.py` | the only module that uses Laya's private API, pinned to laya 0.4.6 and contract-tested |
| `pilot/dumps.py` | the logit dump format: writer, the one hard-failing reader, gold labels, run ids |
| `pilot/metrics.py` | torch-free metrics, the val θ rule and the paired bootstrap, the same for every system |
| `pilot/runs.py` | torch-free run-record schema of `ray_train.py`, the run id, the neg-K item count, the MPS fallback ladder |
| `pilot/compute.py` | torch-free compute-checkpoint logic: projection, what-if table, disk budget, report and markdown |
| `ray_train.py` | Ray Train `TorchTrainer` (DDP) fine-tune using Laya's own loss, calibration and checkpoint format; cpu, cuda or one MPS worker; resumes by rerunning |
| `score.py` | the only place a model runs: one system on one val/test split → a logit dump (in-process or Ray Data actors) |
| `report.py` | dumps → metrics (numpy only, no model); `--compare` paired bootstrap; `--table` results table |
| `evaluate.py` | one-step wrapper: `score.py --model laya`, then `report.py`'s report of that dump |
| `mps_spike.py` | Phase 2 MPS feasibility spike (`results/spike/`) |
| `compute_checkpoint.py` | Phase 4 compute checkpoint: a measured MPS fine-tune probe and the projected experiment-matrix runtime (`results/compute/`) |
| `results/` | committed dumps (`dumps/`), full-test reports (`reports/`), legacy-protocol reports (`legacy/`), spike (`spike/`), compute checkpoint (`compute/`) |
| `tests/` | pytest suite (markers `unit`, `data`, `weights`, `ray`, `mps`) |

## Run

```bash
# full-encoder fine-tune on the Apple GPU: one Ray worker, the Phase 2 spike's start config, neg8 items
python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --out ckpt \
    --device mps --workers 1 --micro-batch 8 --grad-accum 4 --max-len 128
python ray_train.py ... --print-run-id     # the run id: its Ray dir ray_results/<run_id>/ and resume key
# after a kill or crash, rerun the identical command: it resumes from the last completed epoch

# CPU smoke fine-tune (CUDA: drop --freeze-encoder/--max-messages, add --device cuda; --gpu is the same)
python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --out ckpt \
    --workers 2 --freeze-encoder --max-messages 100 --epochs 2

# score one system on one split into a logit dump (--workers 0 runs in-process without Ray)
python score.py data/splits/val.jsonl --model laya --checkpoint multilingual --workers 2 --cpus-per-worker 4
python score.py data/splits/val.jsonl --model laya --checkpoint ./ckpt --device mps --workers 0   # fp32 on MPS
python score.py data/splits/test.jsonl --model prior          # baselines: --model majority | prior
# an existing dump is never overwritten: --force rescores it (results/dumps is the committed audit record)

# dumps -> metrics, no model (add --out PATH to save the JSON)
python report.py results/dumps/laya-zeroshot-7b928d82 --split test

# paired bootstrap of A - B on test (B = 10,000 replicates, seed 0)
python report.py --compare results/dumps/laya-zeroshot-7b928d82 results/dumps/prior --split test

# one-step wrapper: score.py --model laya, then report.py (dumps go to a temp dir unless --out-root is given)
python evaluate.py data/splits/test.jsonl --checkpoint ./ckpt --workers 0 --n 20

# Phase 3 baselines: rescore majority, prior and zero-shot Laya on val + test, rebuild results/reports/
make baselines              # needs make data and the pinned weights; about 5 minutes on the M4
make baseline-reports       # rebuild the reports and print the results table from the committed dumps, no model
```

`--checkpoint` (score, evaluate) and `--base` (train) accept `multilingual`, which loads the pinned,
sha256-verified Laya weights, or a local Laya checkpoint dir. Nothing downloads an unpinned
revision. `ray_train.py --data` only accepts a registered split with role `train`; `score.py` only
a registered `val` or `test` split.

Useful flags: `--neg K` (negatives kept per training message, default 8; 0 = all 48 labels),
`--freeze-embeddings` / `--freeze-encoder` (the MPS out-of-memory fallback rungs; `--freeze-encoder`
trains the head only and gains much less), `--max-messages N` (subsample for smoke tests),
`--loss rlcd|soft-ce`. A training run records its device, memory, per-epoch wall clock and fallback rung
in the checkpoint config (`training.ray_train.record`). An MPS out-of-memory error during training names the
flags of the next rung. After the last epoch (calibration, the final save, or the model move of a rerun that only
recalibrates) every epoch is already saved, so that error says to rerun the same command instead: a rung flag would
change the run id and retrain from epoch 0.

## Scoring path

Every system (majority, label prior, zero-shot Laya, later the fine-tuned models) goes through the
same two steps. `score.py` writes `results/dumps/RUN_ID/SPLIT.npz` (float32 logits [n, 48] in
`INTENTS` order plus the message uids; no text) and a sidecar `SPLIT.json` (split, manifest,
question and tag-order hashes, system identity, the shipped temperature, `dump_sha256`).
`report.py` recomputes every number from verified dumps with numpy alone, so anyone with the repo
and `make data` can rebuild a report without the weights. A tampered, renamed or stale dump fails
to load; there is no override.

- **Temperatures:** each report has a `shipped` variant (the checkpoint's served temperature; the
  headline) and a `raw` variant (T = 1). P(tag) = sigmoid(logit / T).
- **Thresholds:** each variant has a row at θ 0.5 and a row at the val-selected θ. The global θ
  maximises val micro-F1 over 0.05..0.95 in 0.01 steps; a tag gets its own θ only with 10 or more
  val positives; ties go to the θ closest to 0.5. θ is chosen on the same run's full val dump and
  never on test, and the report discloses it.
- **Baselines** come from the training split matching the scored role (`final_train` for test,
  `selection_train` for val), never from the labels being scored.
- **Comparisons** (`--compare`) are a paired, message-level bootstrap: one resample of the test
  messages is shared by both systems, per-seed probabilities are averaged before thresholding, and
  the 95% percentile interval of A − B is reported. They are descriptive only.

## Pinned artifacts

- Dataset: `uoe-nlp/multi3-nlu` at `cde8e2537d7ac06212ef4b25df2cf3084f26ec0f` (CC-BY 4.0),
  Turkish banking and hotels folds, all 40 raw fold sha256s checked.
- Laya: `convaiinnovations/laya` subfolder `multilingual` at `7b928d828b7b0e022f929d9bd2e44165aa270148`.
- BERTurk baseline: `dbmdz/bert-base-turkish-cased` at `b6e1de16c983e0f2c70664591ea3f22810072608`.
- Weights load from `.safetensors` only, never pickle, TF or Flax files. Each pinned file's
  sha256 is checked after download, and a moving ref (branch or tag) is refused.

## Platform notes

macOS arm64 (Apple silicon, MPS) is the primary platform. On Linux the lock installs the default
PyPI `torch==2.14.1` wheel, which is built for CUDA 13. It needs an sm_75+ GPU (a T4 works, a P100
does not) and a 580-series or newer driver. Disk: about 1.5 GB for `.venv`, about 1.1 GB for the
weights in the Hugging Face cache, and a few MB for the data. A training run also needs room for its
train state, per-epoch exports and final checkpoint under `ray_results/`; `ray_train.py` refuses to start
when the free disk is too low.

Training (`ray_train.py --device mps --workers 1`) and scoring (`score.py --device mps --workers 0`, fp32)
run on the M4's GPU. Ray has no MPS device manager, so an MPS run is one Ray worker on an explicit
`torch.device("mps")`, and every run asserts that its parameters landed on the requested device.

The Phase 2 MPS feasibility spike (`results/spike/mps_spike.md`, raw numbers in
`results/spike/mps_spike.json`) found that a full-encoder Laya fine-tune **fits** on a 16 GB M4: verdict FITS at
micro-batch 8, max_len 128, grad-accum 4, no frozen layers and no CPU-fallback ops
(`PYTORCH_ENABLE_MPS_FALLBACK=0`), with a projected 14.3 min per `selection_train` neg8 epoch and a peak of about
8.8 GiB. Training starts from that config on MPS, and the Phase 4 compute checkpoint below measured it on the
real data, so the Kaggle GPU contingency stays unused. To rerun the spike on a quiet Mac (about 20-40 min):
`python mps_spike.py --out results/spike`.

## Compute checkpoint (Phase 4)

`results/compute/compute_checkpoint.md` (raw numbers in `results/compute/compute_checkpoint.json`) is one real
`compute_checkpoint.py` run on the 16 GB M4: a full-encoder Laya fine-tune of `selection_train` with neg8 items in
one Ray worker on MPS, from the spike's start config (micro-batch 8, max_len 128, grad-accum 4, rung none, no
fallback), with gradient checkpointing on, AMP off and `PYTORCH_ENABLE_MPS_FALLBACK=0`.

| epoch | train min | s / optimizer step | optimizer steps | items | peak GiB (driver + RSS) | swap delta MiB |
|---|---|---|---|---|---|---|
| 1 | 13.8 | 1.835 | 452 | 14,439 | 8.63 | 0 |
| 2 (resumed) | 13.8 | 1.827 | 452 | 14,439 | 9.09 | 423 |

- **Resume:** the run was killed 60 s into epoch 2 and the identical command was rerun; it continued from the
  epoch-1 checkpoint (resume load 3.6 s, peak 5.33 GiB while resuming).
- **Swap:** swap grew by 423 MiB during the resumed epoch 2 (none in epoch 1; the idle preflight saw no growth). The
  run was kept as the compute checkpoint by the user's decision, and the swap growth is stated here and in the
  approval; the epoch-2 rate is no slower than epoch 1.
- **Other rates:** BERTurk 0.396 s per optimizer step (batch 32, fp32, peak 5.01 GiB) and 0.0019 s per message for
  inference; Laya scoring on MPS in fp32 0.321 s per message.
- **Projection:** the whole experiment matrix (Laya selection, val scoring and final seeds, BERTurk selection and
  final seeds, few-shot base, incremental and scoring, final scoring) is projected at **14.81 h** at the
  provisional matrix (LR grid 3, epoch ceiling 4, 3 seeds); the what-if table ranges from 9.06 h (LR grid 1,
  2 epochs) to 14.81 h, and the few-shot compute fallback (1 base run per model) gives 12.79 h. One training run
  needs up to 13.79 GiB of disk; `ray_train.py` refuses to start below 14.79 GiB free.

No experiment-matrix run (Laya selection or final seeds, BERTurk training, few-shot) starts before the user
approves this budget. To rerun the checkpoint on a quiet Mac on AC power (about 45-55 min):
`caffeinate -i python compute_checkpoint.py --out results/compute`.

## Why a custom Ray loop

`laya.train.finetune` / `laya-train` are single-device. `ray_train.py` reuses Laya's building
blocks (`items_from_rows`, `encode_item`, `soft_ce_loss` / `rlcd_loss`, `calibration_records`,
`fit_temperature_map`, `save_checkpoint`) and adds: rank sharding, DDP via
`ray.train.torch.prepare_model`, `no_sync` on gradient-accumulation steps, per-epoch checkpoints
reported to Ray, and temperature calibration on whole messages held out before training (rank 0;
`--calib-max-messages`, default 40, capped at `--calib-frac` of the messages).
The output loads with `laya.load(path)`. It does not reproduce `laya-train`'s before/after report
or abstention-threshold fitting — use `score.py` + `report.py` for held-out metrics. The Laya
internals it uses are reached through `pilot/laya_adapter.py`, the only module allowed to touch
Laya's private API.

## Results (Phase 3: baselines and zero-shot, full test split)

Majority (the most frequent `final_train` tag), the label prior (each tag's `final_train`
frequency) and zero-shot `laya-multilingual` (pinned revision `7b928d82`) are scored on the full
test split: 413 messages, including the 15 no-intent rows. θ is selected on the full val split
(208 messages) of the same run and never on test; the "val-θ" columns use it, the "@0.5" columns
use θ 0.5. Every number below is generated by `python report.py --table results/reports` from the
committed reports, which `make baseline-reports` rebuilds from the committed dumps under
`results/dumps/` without loading a model (a test fails if this block differs). The second table is
the paired bootstrap (B = 10,000 resamples of the test messages): zero-shot Laya minus the label
prior, and zero-shot Laya minus itself, which must be exactly zero.

<!-- results-table:start -->
| system | run_id | temperature | θ global (val) | per-tag θ | micro-F1 @0.5 | micro-F1 @val-θ | macro-F1 @0.5 | macro-F1 @val-θ | exact match @0.5 | exact match @val-θ | top-1 | split, n (no-intent) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya (multilingual) | laya-zeroshot-7b928d82 | shipped, T = 1.00 | 0.32 | 13 | 0.197 | 0.205 | 0.201 | 0.216 | 0.007 | 0.007 | 0.214 | test, 413 (15) |
| laya (multilingual) | laya-zeroshot-7b928d82 | raw, T = 1.00 | 0.32 | 13 | 0.197 | 0.205 | 0.201 | 0.216 | 0.007 | 0.007 | 0.214 | test, 413 (15) |
| majority (train: final_train) | majority | shipped, T = 1.00 | 0.50 | 13 | 0.138 | 0.138 | 0.008 | 0.008 | 0.005 | 0.005 | 0.231 | test, 413 (15) |
| majority (train: final_train) | majority | raw, T = 1.00 | 0.50 | 13 | 0.138 | 0.138 | 0.008 | 0.008 | 0.005 | 0.005 | 0.231 | test, 413 (15) |
| prior (train: final_train) | prior | shipped, T = 1.00 | 0.12 | 13 | 0.000 | 0.184 | 0.000 | 0.048 | 0.036 | 0.000 | 0.231 | test, 413 (15) |
| prior (train: final_train) | prior | raw, T = 1.00 | 0.12 | 13 | 0.000 | 0.184 | 0.000 | 0.048 | 0.036 | 0.000 | 0.231 | test, 413 (15) |

| A | B | θ | metric | delta (A - B) | 95% CI | replicates |
|---|---|---|---|---|---|---|
| laya-zeroshot-7b928d82 | prior | 0.5 | micro_f1 | +0.197 | [0.180, 0.214] | 10000 |
| laya-zeroshot-7b928d82 | prior | 0.5 | macro_f1 | +0.201 | [0.172, 0.221] | 10000 |
| laya-zeroshot-7b928d82 | prior | 0.5 | exact_match | -0.029 | [-0.048, -0.012] | 10000 |
| laya-zeroshot-7b928d82 | prior | 0.5 | top1_hit | -0.018 | [-0.074, 0.042] | 10000 |
| laya-zeroshot-7b928d82 | prior | val-selected | micro_f1 | +0.021 | [0.005, 0.037] | 10000 |
| laya-zeroshot-7b928d82 | prior | val-selected | macro_f1 | +0.168 | [0.142, 0.186] | 10000 |
| laya-zeroshot-7b928d82 | prior | val-selected | exact_match | +0.007 | [0.000, 0.017] | 10000 |
| laya-zeroshot-7b928d82 | prior | val-selected | top1_hit | -0.018 | [-0.074, 0.042] | 10000 |
| laya-zeroshot-7b928d82 | laya-zeroshot-7b928d82 | 0.5 | micro_f1 | +0.000 | [0.000, 0.000] | 10000 |
| laya-zeroshot-7b928d82 | laya-zeroshot-7b928d82 | 0.5 | macro_f1 | +0.000 | [0.000, 0.000] | 10000 |
| laya-zeroshot-7b928d82 | laya-zeroshot-7b928d82 | 0.5 | exact_match | +0.000 | [0.000, 0.000] | 10000 |
| laya-zeroshot-7b928d82 | laya-zeroshot-7b928d82 | 0.5 | top1_hit | +0.000 | [0.000, 0.000] | 10000 |
| laya-zeroshot-7b928d82 | laya-zeroshot-7b928d82 | val-selected | micro_f1 | +0.000 | [0.000, 0.000] | 10000 |
| laya-zeroshot-7b928d82 | laya-zeroshot-7b928d82 | val-selected | macro_f1 | +0.000 | [0.000, 0.000] | 10000 |
| laya-zeroshot-7b928d82 | laya-zeroshot-7b928d82 | val-selected | exact_match | +0.000 | [0.000, 0.000] | 10000 |
| laya-zeroshot-7b928d82 | laya-zeroshot-7b928d82 | val-selected | top1_hit | +0.000 | [0.000, 0.000] | 10000 |

- laya-zeroshot-7b928d82: shipped temperature is 1.0, so the T = 1 rows equal the headline rows.
- majority: shipped temperature is 1.0, so the T = 1 rows equal the headline rows.
- prior: shipped temperature is 1.0, so the T = 1 rows equal the headline rows.
<!-- results-table:end -->

The label prior predicts no tag at θ 0.5 by construction: every `final_train` prior is below
0.26, so at 0.5 it scores micro-F1 0 and gets only the 15 no-intent messages exactly right
(15/413). These intervals are descriptive; the go/no-go decision rule and its thresholds are frozen
in Phase 8.

### Legacy protocol (not comparable)

These numbers predate the frozen splits and were run in a CPU sandbox (2 vCPU / 7 GB). They used
the old test split without no-intent rows and only 100 test messages, so they are not comparable to
the table above. The metric definitions changed too: the majority row used a test-derived tag and
the old top-1 denominator; current reports take the tag from the matching training split and
compute top-1 over messages with gold tags. The two reports are kept for provenance in
`results/legacy/`, whose [README](results/legacy/README.md) labels them as the legacy protocol.

| run (100 test messages) | micro-F1 @0.5 | macro-F1 @0.5 | top-1 hit | tags predicted / gold |
|---|---|---|---|---|
| always predict the most common tag (legacy: test-derived tag) | — | — | 0.26 | — |
| zero-shot `laya-multilingual` | 0.186 | 0.187 | 0.21 | 4.7 / 2.29 |
| Ray smoke fine-tune (120 msgs, head only, 2 epochs, CPU) | 0.194 | 0.168 | 0.24 | 3.6 / 2.29 |

The smoke run proves the pipeline (Ray Train → calibrated Laya checkpoint → Ray Data eval), not
accuracy: it saw 8% of the training data with the encoder frozen. A real result needs a full
fine-tune with the encoder unfrozen. Two Ray workers ran one DDP epoch here before the 7 GB sandbox
ran out of memory (each worker holds ~2.6 GB); use ≥16 GB per worker.
