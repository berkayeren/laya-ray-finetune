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
| `ray_train.py` | Ray Train `TorchTrainer` (DDP) fine-tune using Laya's own loss and checkpoint format, uncalibrated (T = 1) checkpoints; cpu, cuda or one MPS worker; resumes by rerunning |
| `score.py` | the only place a model runs: one system on one val/test split → a logit dump (in-process or Ray Data actors) |
| `report.py` | dumps → metrics (numpy only, no model); `--compare` paired bootstrap; `--table` results table |
| `evaluate.py` | one-step wrapper: `score.py --model laya`, then `report.py`'s report of that dump |
| `mps_spike.py` | Phase 2 MPS feasibility spike (`results/spike/`) |
| `compute_checkpoint.py` | Phase 4 compute checkpoint: a measured MPS fine-tune probe and the projected experiment-matrix runtime (`results/compute/`) |
| `pilot/postfit.py` | torch-free Phase 5 postfit: the val-fitted T, θ and the chosen epoch in one byte-reproducible record; the final-seed binding and manifest; the README Phase 5 block |
| `postfit.py` | Phase 5 and 6 CLI: val dumps → `results/postfit/laya.json` or `berturk.json` (the protocol by system kind), `--prune`, `--print-final-commands`, `--final-manifest`, `--render`, `--reference-check` and `--render-berturk` (BERTurk) |
| `berturk_train.py` | Phase 6 BERTurk trainer: the HF Trainer, masked BCE, encoder LR `--lr` and head LR `--head-lr` in their own param groups, `--epochs` (default 30), a per-epoch export, `--stop-after-epoch`, `--postfit` and `--reference` for the final seeds; one process, no Ray |
| `pilot/berturk.py` | torch-free BERTurk bookkeeping: the recipe and run id, labels in `INTENTS` order, the checkpoint identity, the postfit protocol, the BASE-03 reference check, the final commands and manifest, the superseded chain (`SUPERSEDED`, `load_superseded`) and the README Phase 6 block |
| `results/` | committed dumps (`dumps/`), full-test reports (`reports/`), legacy-protocol reports (`legacy/`), spike (`spike/`), compute checkpoint (`compute/`), postfit records (`postfit/`: `laya*.json`, and `berturk.json`, `berturk-reference.json`, `berturk-final.json`), the two discarded BERTurk selection runs kept as disclosed history (`superseded/berturk-tied-head-lr/`, `superseded/berturk-10-epoch-grid/`, never current) |
| `tests/` | pytest suite (markers `unit`, `data`, `weights`, `ray`, `mps`) |

## Run

```bash
# full-encoder fine-tune on the Apple GPU: one Ray worker, the Phase 2 spike's start config, neg8 items
python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --out ckpt \
    --device mps --workers 1 --micro-batch 8 --grad-accum 4 --max-len 128
python ray_train.py ... --print-run-id     # the run id: its Ray dir ray_results/<run_id>/ and resume key
# after a kill or crash, rerun the identical command: it resumes from the last completed epoch

# Phase 5 final seeds: print the commands from the val-selected record, then run each one (one at a time on MPS);
# each is ray_train.py on final_train with --stop-after-epoch at the chosen epoch and --postfit, which serves the val T
python postfit.py --print-final-commands results/postfit/laya.json --storage ray_results --out-prefix ckpt-final
python ray_train.py --data data/splits/final_train.jsonl ... --epochs 4 --stop-after-epoch 3 --seed 0 \
    --postfit results/postfit/laya.json --no-export-epochs --storage ray_results --out ckpt-final-s0

# Phase 6 BERTurk baseline: the 30-epoch selection run, val scoring of each export, postfit and the BASE-03 check
python berturk_train.py --data data/splits/selection_train.jsonl --device mps --epochs 30 --lr 2e-05 --head-lr 0.0001
python score.py data/splits/val.jsonl --model berturk --checkpoint berturk_runs/RUN_ID/exports/epoch-001 \
    --device mps --workers 0                     # BERTurk is refused on test until Phase 8
make berturk-postfit                             # results/postfit/berturk.json from the 30 committed val dumps
python postfit.py --reference-check results/postfit/berturk.json --out results/postfit/berturk-reference.json
python postfit.py --print-final-commands results/postfit/berturk.json \
    --reference results/postfit/berturk-reference.json --storage berturk_runs --out-prefix ckpt-berturk-final

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
flags of the next rung. After the last epoch (the final save, or the model move of a rerun that only writes the final
checkpoint) every epoch is already saved, so that error says to rerun the same command instead: a rung flag would
change the run id and retrain from epoch 0. There is no calibration stage: training writes uncalibrated checkpoints
(T = 1), and `postfit.py` fits T on val afterwards.

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
blocks (`items_from_rows`, `encode_item`, `soft_ce_loss` / `rlcd_loss`, `save_checkpoint`) and adds:
rank sharding, DDP via `ray.train.torch.prepare_model`, `no_sync` on gradient-accumulation steps and
per-epoch checkpoints reported to Ray. Training no longer calibrates: every message trains, and the
checkpoints ship T = 1. `postfit.py` fits the temperature on the val messages after training (LAYA-05);
laya's own temperature fit is only the oracle the tests check it against.
The output loads with `laya.load(path)`. It does not reproduce `laya-train`'s before/after report
or abstention-threshold fitting — use `score.py` + `report.py` for held-out metrics. The Laya
internals it uses are reached through `pilot/laya_adapter.py`, the only module allowed to touch
Laya's private API.

## Laya val selection and final seeds (Phase 5)

How Laya's epoch count, temperature and thresholds were chosen, on validation messages only, and how the three
final Laya seeds were trained with those settings fixed (LAYA-05, LAYA-08).

- **Temperature after training, on val only.** `ray_train.py` writes uncalibrated checkpoints (T = 1). After
  training, `postfit.py` fits one noul T on all 208 val messages x 48 tags: the minimiser of the mean binary
  cross-entropy over log T in [0.5, 5.0], laya's own clamp band (scipy's bounded minimiser). It equals laya's own
  temperature fit within 2e-3, which the tests check (LAYA-05).
- **One selection run, 4 epoch candidates.** One run on `selection_train` with seed 0, scored on val at every epoch
  of its 4-epoch schedule (the approved max_epochs 4), gives the 4 candidates.
- **The encoder LR is fixed at 2e-05, not tuned.** On 2026-10-15 the user decided to run one LR. That was before any
  fine-tuned val metric existed, so the LR 3e-05 and 5e-05 runs were never trained. The approved compute budget
  (lr_grid 3) is still recorded as approved in `pilot/compute.py` and `results/compute/`. A later LR search would be
  a separate decision, and its new postfit record would orphan the final seeds below.
- **θ and the selection rule.** θ uses the same val rule as the baselines (global θ over 0.05..0.95, a tag's own θ
  only with 10 or more val positives), for the shipped variant (after T) and the raw one (T = 1). The objective is
  val micro-F1 at the global θ after T, compared exactly as a fraction; ties go to fewer epochs, then the lower LR.
- **Bounds.** Every candidate's shipped global θ, not only the chosen one, sits on the top edge 0.95 of the
  0.05..0.95 θ grid (the chosen raw global θ, 0.33, does not). Every fitted T is close to the upper bound 5.0 (epoch
  1's is on it), because the raw margins are overconfident on val. So each candidate is scored at its own margin
  cutoff T x logit(0.95), set by its fitted T, and not at its best θ: the epoch ranking is a ranking at that clipped
  cutoff, not a clean optimum. Past the edge the ranking flips. Recomputed from the committed val dumps (these numbers
  are not in the record; the command is below the list), val micro-F1 at θ 0.95 / 0.96 / 0.97 is
  0.8107 / 0.8178 / 0.8260 for epoch 3 and 0.8033 / 0.8129 / 0.8294 for epoch 4, so at θ 0.97, two grid steps past
  the edge, epoch 4 would win. Both grids are part of the pre-declared rule. On 2026-10-21 the user decided to keep
  the rule and disclose this (05-REVIEW CR-01), so epoch 3 stands as the rule's result. The block below shows the
  θ-edge count and the margin cutoffs.
- **In-sample caveat.** The epoch, T and θ were all chosen on the same 208 val messages, so the val numbers below are
  optimistic. They are not test results.
- **Final seeds.** Seeds 0, 1 and 2 train on `final_train` (folds 0-15) with the same 4-epoch schedule, stopped after
  the chosen epoch (`--stop-after-epoch`), and serve the val-fitted T. Their checkpoints are kept outside git for
  Phase 8; `results/postfit/laya-final.json` binds them to the record's sha256.
- **Audit record.** MPS runs are not bit-reproducible, so the committed val dumps under `results/dumps/` are the audit
  record: `make laya-postfit` rebuilds `results/postfit/laya.json` from them byte for byte, without a model.
- **Nothing fine-tuned is scored on test before Phase 8.** `score.py` refuses a fine-tuned checkpoint on test, and
  `postfit.py` reads val dumps only.

The θ-edge numbers in the Bounds bullet, recomputed from the committed val dumps (needs `make data`, no model):

```bash
python - <<'EOF'
import json
from pilot import dumps, metrics
from pilot.data import DEFAULT_MANIFEST as M
for c in json.load(open("results/postfit/laya.json"))["candidates"]:
    logits, uids, _, entry = dumps.load_dump("results/dumps/" + c["dump_run_id"], "val", M)
    Y, P = dumps.read_gold(entry, M, uids), metrics.probabilities(logits, c["temperature"])
    print(c["epoch"], *("%.4f" % metrics.metrics(Y, P, t)["micro_f1"] for t in (0.95, 0.96, 0.97)))
EOF
```

```bash
# the selection run: one LR, 4 epochs, an uncalibrated export per epoch under ray_results/<run_id>/exports/
python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --device mps --workers 1 \
    --micro-batch 8 --grad-accum 4 --max-len 128 --epochs 4 --neg 8 --seed 0 --encoder-lr 2e-05 --head-lr 0.0001 \
    --storage ray_results --out ckpt-sel
# score each epoch export on val (fp32 on MPS), then delete the export once its verified val dump exists
python score.py data/splits/val.jsonl --model laya --checkpoint ray_results/RUN_ID/exports/epoch-001 \
    --device mps --workers 0
python postfit.py --prune ray_results/RUN_ID/exports/epoch-001
# fit T and θ and choose the epoch over the 4 val dumps (make laya-postfit rebuilds the committed record)
python postfit.py results/dumps/DUMP_1 results/dumps/DUMP_2 results/dumps/DUMP_3 results/dumps/DUMP_4 \
    --out results/postfit/laya.json
make laya-postfit
# the 3 final-seed commands, printed from the record; then the manifest of the finished checkpoints
python postfit.py --print-final-commands results/postfit/laya.json --storage ray_results --out-prefix ckpt-final
python postfit.py --final-manifest results/postfit/laya.json ckpt-final-s0 ckpt-final-s1 ckpt-final-s2 \
    --out results/postfit/laya-final.json
# the generated block below
python postfit.py --render results/postfit/laya.json results/postfit/laya-final.json
```

The block between the markers is generated by the last command from the committed record and final manifest; a test
fails if it is edited by hand or goes stale.

<!-- laya-postfit:start -->
**Chosen on val** (selection run on selection_train, seed 0, 208 val messages x 48 tags; val numbers, in-sample)

| setting | value |
|---|---|
| train run id | laya-selection_train-neg8-s0-d92447f4 |
| encoder LR | 2e-05 (head LR 0.0001) |
| epoch | 3 of 4, --stop-after-epoch 3 |
| val-fitted T | 4.7698 (bounds [0.5, 5.0]) |
| val NLL at T / at T = 1 | 0.09417 / 0.35502 |
| global θ shipped / raw | 0.95 / 0.33 |
| tags with their own θ (shipped / raw) | 13 / 13 |
| val micro-F1 at the global θ | 0.8107 (852/1051) |
| val micro-F1 at θ 0.5 | 0.8000 |

- Rule: val micro-F1 at the val-selected global θ on probabilities after the val-fitted T, compared exactly as 2TP / (2TP + FP + FN) of the summed counts; tie order: equal objectives: fewer epochs first, then the lower encoder LR.
- Encoder LR grid: 2e-05 (one value, fixed, not tuned; the approved budget allows lr_grid 3).
- Grid edge: interior in the epoch grid: epoch 3 of the epoch grid 1..4 (approved max_epochs 4); θ clipped at the grid edge (see the θ-edge line), so the epoch is not a clean optimum of the objective.
- Bounds: on the θ grid 0.05..0.95 the chosen global θ is shipped 0.95 (at the upper edge) and raw 0.33 (inside); the candidates' fitted T span 4.7698..5.0000 in laya's band [0.5, 5.0] (1 of 4 on a bound), and the chosen T 4.7698 is 0.2302 below the upper bound 5.0.
- θ grid edge: 4 of 4 candidates choose the shipped global θ on the upper edge 0.95 and 0 on the lower edge 0.05 (raw: 1 upper, 0 lower), so the epoch ranking is a ranking at a T-dependent margin cutoff (T x logit(θ): 14.04..14.72 over the clipped candidates), not at each candidate's best θ; the objective is clipped at the edge, and a θ outside the pre-declared grid may rank the epochs differently.

**Candidates** (one full val dump per epoch export; the objective is micro-F1 at the global θ)

| epoch | encoder LR | T | NLL at T | NLL at T = 1 | micro-F1 @0.5 | micro-F1 @global θ | θ shipped | θ raw | pick |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2e-05 | 5.0000 | 0.12536 | 0.49554 | 0.7386 | 0.7598 (810/1066) | 0.95 | 0.93 |  |
| 2 | 2e-05 | 4.9974 | 0.10344 | 0.40539 | 0.7705 | 0.7859 (822/1046) | 0.95 | 0.74 |  |
| 3 | 2e-05 | 4.7698 | 0.09417 | 0.35502 | 0.8000 | 0.8107 (852/1051) | 0.95 | 0.33 | chosen |
| 4 | 2e-05 | 4.8746 | 0.10398 | 0.39724 | 0.7747 | 0.8033 (866/1078) | 0.95 | 0.95 | runner-up |

**Final seeds** (final_train, --epochs 4 stopped after epoch 3, serving T 4.7698; not scored before Phase 8)

| seed | train run id | epochs trained | final-epoch loss | peak GiB | largest swap delta MiB | rung | code changed on resume |
|---|---|---|---|---|---|---|---|
| 0 | laya-final_train-neg8-s0-95cf6955 | 3 | 0.2826 | 8.75 | 0 | none | False |
| 1 | laya-final_train-neg8-s1-07a1ee59 | 3 | 0.2936 | 8.75 | 0 | none | False |
| 2 | laya-final_train-neg8-s2-bbb5230e | 3 | 0.2591 | 8.95 | 0 | none | False |
<!-- laya-postfit:end -->

## BERTurk baseline (Phase 6)

How the BERTurk baseline (`dbmdz/bert-base-turkish-cased`, pinned) was trained, selected on validation messages only,
checked against the published Multi3NLU++ numbers and kept as three final seeds (BASE-01, BASE-02, BASE-03). It is
the well-known baseline that fine-tuned Laya is compared against in Phase 8.

- **Recipe.** `berturk_train.py` fine-tunes a `BertForSequenceClassification` with the HF Trainer as a 48-label
  multi-label classifier, labels in `INTENTS` order (the dump's column order), with a masked BCE loss. It uses batch
  32, max_len 128 WordPiece tokens, fp32 and AdamW with weight decay 0.01. The encoder trains at 2e-05 and the freshly
  initialised classifier head at 1e-4, in their own param groups. A linear schedule with a 10% warm-up spans the 30
  epochs of the selection schedule. One process on MPS, no Ray; the HF Trainer needs `accelerate`, so
  `accelerate==1.15.0` was added to the lock.
- **Equal settings.** Both models use the same encoder LR, 2e-05. It is fixed, not tuned: the user decided on
  2026-10-15 to run one LR for both models. Both also use the same head LR 1e-4, the same split (`selection_train` for
  selection, `final_train` for the final seeds), the same seeds (0 for selection; 0, 1 and 2 for the final seeds),
  the same max_len and the same postfit: `postfit.py` runs Laya's own functions, objective, tie order, T rule and θ
  rule on BERTurk's val dumps.
- **Two protocol changes after a val outcome.** BERTurk's recipe was changed twice after its val numbers had been
  seen. First, the first selection run tied the head LR to the encoder LR (2e-05). It scored 0.272 at val, below the
  0.707 floor, and was discarded under the BASE-03 fix the user chose at the 06-10 gate: train the head at 1e-4.
  Second, that head-LR run scored 0.475 at the upper edge of its 10-epoch grid, still rising, and was discarded under
  the second fix the user chose at the 06-16 gate: raise the epoch grid to 30. So BERTurk had three selection runs
  while Laya had one. This is a disclosed deviation from equal tuning effort. Both discarded runs are kept byte for
  byte under `results/superseded/` (`berturk-tied-head-lr/`, `berturk-10-epoch-grid/`), each with its own README, and
  the generated table below lists them.
- **The epoch grid.** BERTurk has 30 epoch candidates; Laya has 4 (max_epochs 4). Picking the best of 30 in-sample
  candidates is more optimistic than picking the best of 4. 30 also exceeds the approved compute budget
  berturk_epochs 10 (approved 2026-10-14), which `pilot/compute.py` and `results/compute/` still record as approved.
  The linear schedule and its warm-up span all 30 epochs, so the 30-epoch run's epoch 10 is not the 10-epoch run's
  epoch 10. The warm-up now takes 138 of 1,380 optimizer steps (epochs 1-3, against 1 epoch before). At the end of
  epoch 10 the 30-epoch run still trains at about 0.74 of the peak LR, while the 10-epoch run had decayed to 0.
- **Selection on val only.** The candidates are epochs 1..30 of one run on `selection_train` with seed 0, each scored
  on val (fp32 on MPS). `postfit.py` fits one T and the θ on val and picks the epoch by Laya's rule. The chosen epoch
  29 sits next to the upper edge of the grid: epoch 30 is the runner-up, 0.0003 lower, so the objective has flattened
  at the top. The chosen T (0.7430) and both global θ (0.19 shipped, 0.26 raw) are inside their bounds and grids.
  The epoch, T and θ were all chosen on the same 208 val messages, so the val numbers below are optimistic. They
  are not test results.
- **The BASE-03 reference check.** The chosen candidate's val micro-F1 at its val-selected global θ is compared with
  0.707, more than 10 points below the lower published reference 80.7 (Moghe et al., Table 12, Turkish BANKING,
  frozen-encoder MLPs at θ 0.3). It reads 389/478 = 0.8138, status ok, so no resolution was needed. It is not a
  like-for-like comparison: the block lists every setup difference and the in-sample caveat next to the value.
- **Final seeds.** Seeds 0, 1 and 2 train on `final_train` with the 30-epoch schedule, stopped after the chosen epoch
  29 (`--stop-after-epoch 29`), and serve the val-fitted T. `berturk_train.py --postfit --reference` refuses to train
  unless the reference check is of that record and reads ok (or records a resolution). The checkpoints are kept
  outside git for Phase 8; `results/postfit/berturk-final.json` binds them to the record's sha256.
- **Audit record.** MPS runs are not bit-reproducible, so the committed val dumps under `results/dumps/` are the audit
  record: `make berturk-postfit` rebuilds `results/postfit/berturk.json` from them byte for byte, without a model.
- **Nothing fine-tuned is scored on test before Phase 8.** `score.py --model berturk` refuses any split with role
  test and, under any manifest, any split that holds committed test content (the committed test sha256 or uid list,
  or any of its (uid, message) pairs), as for Laya. `postfit.py` reads val dumps only. No BERTurk test number exists
  yet.

```bash
# the selection run: 30 epochs, encoder LR 2e-05, head LR 1e-4, an export per epoch under berturk_runs/<run_id>/exports/
python berturk_train.py --data data/splits/selection_train.jsonl --base berturk --device mps --epochs 30 \
    --lr 2e-05 --head-lr 0.0001 --seed 0 --storage berturk_runs
# score each epoch export on val (fp32 on MPS, in-process), then delete the export once its verified val dump exists
python score.py data/splits/val.jsonl --model berturk --checkpoint berturk_runs/RUN_ID/exports/epoch-001 \
    --device mps --workers 0
python postfit.py --prune berturk_runs/RUN_ID/exports/epoch-001
# fit T and θ and choose the epoch over the 30 val dumps (make berturk-postfit rebuilds the committed record)
python postfit.py results/dumps/DUMP_1 ... results/dumps/DUMP_30 --out results/postfit/berturk.json
make berturk-postfit
# the BASE-03 reference check of the record (--resolution TEXT only on an investigate status)
python postfit.py --reference-check results/postfit/berturk.json --out results/postfit/berturk-reference.json
# the 3 final-seed commands, printed from the record behind the BASE-03 gate; each one stops after the chosen epoch
python postfit.py --print-final-commands results/postfit/berturk.json \
    --reference results/postfit/berturk-reference.json --storage berturk_runs --out-prefix ckpt-berturk-final
python berturk_train.py --data data/splits/final_train.jsonl ... --epochs 30 --lr 2e-05 --head-lr 0.0001 \
    --stop-after-epoch 29 --seed 0 --postfit results/postfit/berturk.json \
    --reference results/postfit/berturk-reference.json --no-export-epochs --storage berturk_runs \
    --out ckpt-berturk-final-s0
# the manifest of the finished checkpoints, then the generated block below
python postfit.py --final-manifest results/postfit/berturk.json ckpt-berturk-final-s0 ckpt-berturk-final-s1 \
    ckpt-berturk-final-s2 --out results/postfit/berturk-final.json
python postfit.py --render-berturk results/postfit/berturk.json results/postfit/berturk-reference.json \
    results/postfit/berturk-final.json
```

The block between the markers is generated by the last command from the committed record, reference check and final
manifest, plus the superseded history under `results/superseded/` (read and sha256-checked by
`pilot.berturk.load_superseded`); a test fails if it is edited by hand or goes stale.

<!-- berturk-postfit:start -->
**Chosen on val** (selection run on selection_train, seed 0, 208 val messages x 48 tags; val numbers, in-sample; no BERTurk test number exists before Phase 8)

| setting | value |
|---|---|
| train run id | berturk-selection_train-s0-4f36ae05 |
| encoder LR / head LR | 2e-05 / 0.0001 |
| epoch | 29 of 30, --stop-after-epoch 29 |
| val-fitted T | 0.7430 (bounds [0.5, 5.0]) |
| val NLL at T / at T = 1 | 0.05821 / 0.06491 |
| global θ shipped / raw | 0.19 / 0.26 |
| tags with their own θ (shipped / raw) | 13 / 13 |
| val micro-F1 at the global θ | 0.8138 (778/956) |
| val micro-F1 at θ 0.5 | 0.7570 |

- Rule: val micro-F1 at the val-selected global θ on probabilities after the val-fitted T, compared exactly as 2TP / (2TP + FP + FN) of the summed counts; tie order: equal objectives: fewer epochs first, then the lower encoder LR (Laya's rule, run by pilot.postfit on BERTurk's val dumps).
- LRs: encoder LR grid 2e-05 (one value, fixed, not tuned; Laya's grid 2e-05), head LR 0.0001 (Laya's 0.0001).
- Epoch grid: 1..30, every epoch of one run a candidate; Laya's is 1..4 (approved max_epochs 4), and the approved compute budget for BERTurk is berturk_epochs 10, so this grid is a disclosed deviation from both, and a selection over 30 in-sample candidates is more optimistic than over 4.
- Grid edge: next to the upper edge: epoch 29 of the epoch grid 1..30, one epoch below it; runner-up epoch 30 at 0.8135 (772/949), 0.0003 below the chosen.
- Bounds: on the θ grid 0.05..0.95 the chosen global θ is shipped 0.19 (inside) and raw 0.26 (inside); the candidates' fitted T span 0.5000..0.9862 in laya's band [0.5, 5.0] (1 of 30 on a bound), and the chosen T 0.7430 is 0.2430 above the lower bound 0.5 and 4.2570 below the upper bound 5.0.
- θ grid edge: 0 of 30 candidates choose the shipped global θ on the upper edge 0.95 and 0 on the lower edge 0.05 (raw: 0 upper, 0 lower); no candidate's objective is clipped at a θ-grid edge.

**Published reference check (BASE-03)** (Moghe et al., Multi3NLU++ (multilingual, multi-intent, multi-domain NLU dataset), ACL Findings 2023; https://arxiv.org/abs/2212.10455)

| check | value |
|---|---|
| BERTurk val micro-F1 at the global θ (epoch 29, T 0.7430) | 0.8138 (389/478) |
| threshold | 0.707: below 0.707 (more than 10 micro-F1 points below the lower published reference 80.7) triggers an investigation; exactly 0.707 passes |
| status | ok (below the threshold: False) |
| resolution | none recorded (nothing to resolve when the status is ok) |
| checked record | results/postfit/berturk.json (sha256 83b34dfb524e) |

- Quoted source, Table 12 (In-language in-domain results for intent detection with MLP-based setup (F1 x100); Turkish, in-language, BANKING; Large: 10-fold cross-validation, train on 9 of 10 folds (about 1,860 messages)): MLP on frozen LaBSE (frozen encoder, θ 0.3) 82.8; MLP on frozen mpnet (frozen encoder, θ 0.3) 80.7.
- Quoted source, Table 18 (QA-based intent detection): QA mDeBERTa 90.1; lower reference 80.7.
- Not like-for-like. Setup differences: a fixed split (selection train folds 0–13 with about 1,450 messages; final train folds 0–15 with about 1,658); a different test portion (folds 16–19); a fine-tuned BERTurk rather than a frozen encoder; a single split rather than cross-validation averages; this check itself reads val folds 14-15.
- Caveats: θ and T were chosen on the same val messages, so the value is in-sample and optimistic; the paper uses a fixed θ 0.3 on frozen-encoder MLPs; neither of our numbers is like-for-like.

**Candidates** (one full val dump per epoch export of berturk-selection_train-s0-4f36ae05, encoder LR 2e-05; the objective is micro-F1 at the global θ, as 2TP/(2TP + FP + FN) unreduced; the reference check above reduces it)

| epoch | T | NLL at T | NLL at T = 1 | micro-F1 @0.5 | micro-F1 @global θ | θ shipped | θ raw | pick |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.5000 | 0.45520 | 0.54154 | 0.1252 | 0.1320 (352/2666) | 0.46 | 0.48 |  |
| 2 | 0.6665 | 0.18027 | 0.20944 | 0.0000 | 0.2393 (190/794) | 0.13 | 0.22 |  |
| 3 | 0.9353 | 0.17364 | 0.17444 | 0.0000 | 0.2654 (280/1055) | 0.10 | 0.11 |  |
| 4 | 0.9862 | 0.17101 | 0.17104 | 0.0000 | 0.2696 (272/1009) | 0.12 | 0.12 |  |
| 5 | 0.9263 | 0.16343 | 0.16441 | 0.0000 | 0.3197 (290/907) | 0.15 | 0.16 |  |
| 6 | 0.8890 | 0.15192 | 0.15414 | 0.1147 | 0.3979 (382/960) | 0.13 | 0.15 |  |
| 7 | 0.8226 | 0.13590 | 0.14153 | 0.2768 | 0.4736 (494/1043) | 0.11 | 0.16 |  |
| 8 | 0.8316 | 0.12473 | 0.12952 | 0.3609 | 0.5349 (506/946) | 0.13 | 0.17 |  |
| 9 | 0.8079 | 0.11338 | 0.11943 | 0.4071 | 0.6043 (510/844) | 0.16 | 0.18 |  |
| 10 | 0.8136 | 0.10631 | 0.11174 | 0.4606 | 0.6536 (632/967) | 0.12 | 0.17 |  |
| 11 | 0.7486 | 0.09490 | 0.10468 | 0.5433 | 0.6985 (672/962) | 0.13 | 0.19 |  |
| 12 | 0.7754 | 0.09054 | 0.09785 | 0.5581 | 0.7224 (700/969) | 0.12 | 0.18 |  |
| 13 | 0.7283 | 0.08254 | 0.09309 | 0.6240 | 0.7418 (698/941) | 0.16 | 0.23 |  |
| 14 | 0.7361 | 0.07963 | 0.08913 | 0.6451 | 0.7460 (696/933) | 0.17 | 0.24 |  |
| 15 | 0.7610 | 0.07786 | 0.08525 | 0.6433 | 0.7550 (718/951) | 0.16 | 0.22 |  |
| 16 | 0.7274 | 0.07239 | 0.08190 | 0.6789 | 0.7697 (722/938) | 0.18 | 0.25 |  |
| 17 | 0.7411 | 0.07097 | 0.07918 | 0.6824 | 0.7734 (710/918) | 0.21 | 0.27 |  |
| 18 | 0.7416 | 0.06879 | 0.07673 | 0.6918 | 0.7761 (714/920) | 0.21 | 0.28 |  |
| 19 | 0.7262 | 0.06581 | 0.07463 | 0.7141 | 0.7861 (702/893) | 0.27 | 0.33 |  |
| 20 | 0.7508 | 0.06553 | 0.07251 | 0.6933 | 0.7978 (734/920) | 0.21 | 0.27 |  |
| 21 | 0.7254 | 0.06274 | 0.07120 | 0.7241 | 0.7954 (766/963) | 0.17 | 0.24 |  |
| 22 | 0.7439 | 0.06276 | 0.06989 | 0.7236 | 0.7978 (742/930) | 0.21 | 0.27 |  |
| 23 | 0.7572 | 0.06258 | 0.06881 | 0.7250 | 0.8047 (750/932) | 0.20 | 0.26 |  |
| 24 | 0.7476 | 0.06099 | 0.06767 | 0.7336 | 0.8017 (744/928) | 0.22 | 0.28 |  |
| 25 | 0.7487 | 0.06025 | 0.06679 | 0.7398 | 0.8051 (764/949) | 0.20 | 0.26 |  |
| 26 | 0.7474 | 0.05951 | 0.06608 | 0.7383 | 0.8082 (784/970) | 0.17 | 0.24 |  |
| 27 | 0.7488 | 0.05923 | 0.06567 | 0.7469 | 0.8115 (762/939) | 0.21 | 0.24 |  |
| 28 | 0.7493 | 0.05888 | 0.06525 | 0.7488 | 0.8128 (764/940) | 0.21 | 0.27 |  |
| 29 | 0.7430 | 0.05821 | 0.06491 | 0.7570 | 0.8138 (778/956) | 0.19 | 0.26 | chosen |
| 30 | 0.7416 | 0.05808 | 0.06485 | 0.7600 | 0.8135 (772/949) | 0.20 | 0.26 | runner-up |

**Protocol changes after a val outcome (BASE-03 fixes)** (each discarded run below scored under 0.707 at val; the user chose the fix at its gate; the run is kept as disclosed history, never current)

| discarded train run | changed by the fix (old -> new) | chosen epoch | val micro-F1 at the global θ | status | gate plan, date and reply | history |
|---|---|---|---|---|---|---|
| berturk-selection_train-s0-c0c9505e | head_lr: tied to the encoder LR (2e-05) -> 0.0001 | 10 of 10 | 0.2722 (150/551) | investigate (resolution: none) | 06-10, 2026-10-21: `fix: head LR 1e-4 (Recommended)` | results/superseded/berturk-tied-head-lr/ |
| berturk-selection_train-s0-8c033154 | epochs: 10 -> 30 | 10 of 10 | 0.4748 (113/238) | investigate (resolution: none) | 06-16, 2026-10-21: `fix: extend epochs (Recommended)` | results/superseded/berturk-10-epoch-grid/ |

- berturk-selection_train-s0-c0c9505e: the classifier head's LR stops being tied to the encoder LR 2e-05 and becomes 1e-4, Laya's head LR (pilot.postfit.HEAD_LR); the encoder LR and every other setting are unchanged.
- berturk-selection_train-s0-8c033154: the epoch grid of the selection schedule grows from 10 to 30 epochs, the user's choice at the 06-16 gate, while the encoder LR 2e-05, the head LR 1e-4 and every other setting stay; the linear schedule and its warm-up then span 30 epochs.
- BERTurk had 2 more selection runs than Laya: 3 in all (the current berturk-selection_train-s0-4f36ae05 and the 2 discarded above), against Laya's 1; every recipe change above was made after a val outcome, a disclosed deviation from equal tuning effort.

**Final seeds** (final_train, --epochs 30 stopped after epoch 29, serving T 0.7430; not scored before Phase 8)

| seed | train run id | epochs trained | final-epoch loss | peak GiB | largest swap delta MiB |
|---|---|---|---|---|---|
| 0 | berturk-final_train-s0-8a167355 | 29 | 0.0416 | 6.26 | 0 |
| 1 | berturk-final_train-s1-3c44e7ff | 29 | 0.0391 | 6.07 | 0 |
| 2 | berturk-final_train-s2-44f81396 | 29 | 0.0401 | 5.26 | 0 |
<!-- berturk-postfit:end -->

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
