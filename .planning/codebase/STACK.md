---
last_mapped_commit: 49868df263b5b89e8efe10909576f479c0330c44
last_mapped_at: 2026-10-13
---
# Technology Stack

**Analysis Date:** 2026-10-13

## Languages

**Primary:**
- Python (version not pinned; no `.python-version`/`pyproject.toml`) - all code: `prepare_data.py`, `ray_train.py`, `evaluate.py`, `intents.py`

**Secondary:**
- Not applicable

## Runtime

**Environment:**
- CPython, run directly as scripts (`python ray_train.py ...`); CLI via `argparse`
- Ray cluster (local or multi-node); CPU or GPU workers (CUDA via `--gpu`, `--gpus-per-worker`)

**Package Manager:**
- pip (`pip install laya "ray[train,data]"`)
- Lockfile: missing; no `requirements.txt`, no package config

## Frameworks

**Core:**
- `laya` (unpinned) - multilingual yes/no (`noul`) question model; imports `laya.train`, `laya.calibrate`, `laya.common`, and `laya.load` in `ray_train.py` and `evaluate.py`. Uses private/internal APIs, so verify against the installed version.
- `ray[train]` (unpinned) - `TorchTrainer`, `ScalingConfig`, `RunConfig`, `Checkpoint` for DDP training in `ray_train.py`
- `ray[data]` (unpinned) - `map_batches` with a `Predictor` actor class for distributed inference in `evaluate.py`
- `torch` - model, DDP, `no_sync()` gradient accumulation (`ray_train.py`)

**Testing:**
- None (no test suite)

**Build/Dev:**
- None (no linter, formatter, or build config). CLI entry `laya-train` comes from the `laya` package (used for `--dry-run` validation).

## Key Dependencies

**Critical:**
- `laya` - training internals (`items_from_rows`, `split_calibration`, `encode_item`, `rlcd_loss`, `soft_ce_loss`, `save_checkpoint`, `fit_temperature_map`)
- `ray` - distributed training and inference
- `torch` - installed transitively

**Infrastructure:**
- Stdlib only otherwise: `argparse`, `json`, `glob`, `random`, `shutil`, `tempfile`, `time`, `os`

## Configuration

**Environment:**
- Configured entirely through CLI flags (`ray_train.py`: `--data --base --out --workers --gpu --epochs --micro-batch --grad-accum --encoder-lr --head-lr --loss --freeze-encoder --storage --object-store-gb --seed` etc.; `evaluate.py`: `--checkpoint --n --workers --gpus-per-worker --out`; `prepare_data.py`: `--neg --seed`)
- No environment variables read by the code; no `.env` files

**Build:**
- None. Task definition lives in `intents.py` (`INTENTS`, 48 tags).

## Platform Requirements

**Development:**
- 16 GB or more RAM per Ray worker (about 2.6 GB each plus overhead)
- Multi3NLU++ Turkish banking folds downloaded manually into a `DATA_DIR`

**Production:**
- Not applicable; research/fine-tuning scripts. Multi-node runs need a shared `--storage` (NFS/S3). Default storage is `./ray_results`.

---

*Stack analysis: 2026-10-13*
