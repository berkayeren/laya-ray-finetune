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
make test                   # pytest -m "not mps" in the ACTIVE interpreter (fails if it is not the locked env)
make weights                # optional: prefetch and verify the pinned Laya + BERTurk weights (~1.1 GB)
```

`make test` runs in whatever interpreter is active and prints `sys.executable` first. A test
fails, listing every mismatched package, if that interpreter does not match the lock. Use
`make test PYTHON=.venv/bin/python` to run without activating the venv.

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
file. `evaluate.py` and `ray_train.py` refuse any file whose sha256 differs from the manifest, or
that the manifest does not register, before loading a model or starting Ray. There is no flag to
skip this: if a split is missing or stale, regenerate it with `make data`. If a rebuilt manifest
would differ from the committed one, `prepare_data.py` writes nothing. `--rewrite-manifest`
deliberately refreezes `data/splits.json`. That invalidates every result produced from the old
splits, so treat it as a protocol change, not a fix.

**No-intent rows are kept.** Banking messages without any intent stay in as all-negative rows by
default. `python prepare_data.py --no-include-no-intent --out OTHER_DIR` builds the variant without
them (1,384 / 201 / 1,585 / 398 messages). Write it to a different `--out` dir, because prepare
refuses to overwrite the committed manifest with a differing one. Consume it with
`--manifest OTHER_DIR/splits.json` on `evaluate.py` and `ray_train.py`; the files are still
hash-verified against that manifest, and the manifest must match the current code and pins.

**Every row labels all 48 tags.** `final_train` is 1,658 × 48 = 79,584 Laya items. There is no
prepare-time negative sampling. Sampling negatives per run seed arrives with the Phase 4 training
harness.

**The HOTELS probe is out-of-domain only.** It is never trained on. All 48 banking tags are
`false` on it, but hotel messages genuinely carry the 26 generic tags banking shares (`affirm`,
`thank`, …). So only the 22 banking-only tags, recorded as `ood_scoring_tags` in the manifest,
are valid gold when scoring it. `prepare_data.py` re-derives that set from the data and fails if it
drifts from `pilot/pins.py`.

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
| `ray_train.py` | Ray Train `TorchTrainer` (DDP) fine-tune using Laya's own loss, calibration and checkpoint format |
| `evaluate.py` | Ray Data batch inference + micro/macro F1, top-1, exact match, per-tag F1 |
| `tests/` | pytest suite (markers `unit`, `data`, `weights`, `ray`, `mps`) |

## Run

```bash
# CPU smoke fine-tune (GPU: drop --freeze-encoder/--max-messages, add --gpu)
python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --out ckpt \
    --workers 2 --freeze-encoder --max-messages 100 --epochs 2

# evaluate in-process on a 20-message subsample (--workers 0 runs without Ray)
python evaluate.py data/splits/test.jsonl --checkpoint multilingual --workers 0 --n 20

# evaluate with 4 GPU workers
python evaluate.py data/splits/test.jsonl --checkpoint ./ckpt --workers 4 --gpus-per-worker 1 --out results/x.json
```

`--checkpoint` (evaluate) and `--base` (train) accept `multilingual`, which loads the pinned,
sha256-verified Laya weights, or a local Laya checkpoint dir. Nothing downloads an unpinned
revision. `ray_train.py --data` only accepts a registered split with role `train`.

Useful flags: `--freeze-encoder` (head only, fits small CPUs/GPUs, gains much less),
`--max-messages N` (subsample for smoke tests), `--loss rlcd|soft-ce`.

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
weights in the Hugging Face cache, and a few MB for the data.

## Why a custom Ray loop

`laya.train.finetune` / `laya-train` are single-device. `ray_train.py` reuses Laya's building
blocks (`items_from_rows`, `encode_item`, `soft_ce_loss` / `rlcd_loss`, `calibration_records`,
`fit_temperature_map`, `save_checkpoint`) and adds: rank sharding, DDP via
`ray.train.torch.prepare_model`, `no_sync` on gradient-accumulation steps, per-epoch checkpoints
reported to Ray, and temperature calibration on whole messages held out before training (rank 0;
`--calib-max-messages`, default 40, capped at `--calib-frac` of the messages).
The output loads with `laya.load(path)`. It does not reproduce `laya-train`'s before/after report
or abstention-threshold fitting — use `evaluate.py` for held-out metrics.

## Results so far (legacy protocol, CPU sandbox, 2 vCPU / 7 GB)

**Legacy protocol:** these numbers predate the frozen splits. They used the old test split
without no-intent rows and only 100 test messages, so they are not comparable to results on
`data/splits/test.jsonl`. The reports are in `results/`.

| run (100 test messages) | micro-F1 @0.5 | macro-F1 @0.5 | top-1 hit | tags predicted / gold |
|---|---|---|---|---|
| always predict the most common tag | — | — | 0.26 | — |
| zero-shot `laya-multilingual` | 0.186 | 0.187 | 0.21 | 4.7 / 2.29 |
| Ray smoke fine-tune (120 msgs, head only, 2 epochs, CPU) | 0.194 | 0.168 | 0.24 | 3.6 / 2.29 |

The smoke run proves the pipeline (Ray Train → calibrated Laya checkpoint → Ray Data eval), not
accuracy: it saw 8% of the training data with the encoder frozen. A real result needs a full
fine-tune with the encoder unfrozen. Two Ray workers ran one DDP epoch here before the 7 GB sandbox
ran out of memory (each worker holds ~2.6 GB); use ≥16 GB per worker.
