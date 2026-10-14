<!-- GSD:project-start source:PROJECT.md -->

## Project

**Laya Turkish Banking Intent Pilot**

A pilot that decides whether Laya (`laya-multilingual`) is good enough for Turkish banking intent classification at a bank. It uses the Turkish banking split of Multi3NLU++ (48 multi-label intent tags, CC-BY 4.0) as a public stand-in until the bank's own customer messages can be used. The deliverable is a defensible, reproducible go/no-go recommendation. It compares fine-tuned Laya against a well-known baseline (fine-tuned BERTurk) on accuracy, speed and confidence quality.

**Core Value:** A trustworthy, reproducible head-to-head comparison (fine-tuned Laya vs fine-tuned BERTurk vs zero-shot Laya vs majority) on the full test set. It has to be trustworthy enough to make a go/no-go call that humans can audit and override.

### Constraints

- **Hardware**: Apple M4 Mac with 16 GB unified memory, PyTorch MPS, no CUDA. Fall back to `--freeze-encoder` only if a full fine-tune OOMs, and label that run as weaker.
- **Data**: no real customer data, and no bank name anywhere in the repo.
- **Hosting**: everything runs self-hosted and locally. Hub downloads of public model weights are fine. Any exception needs the user's explicit confirmation. The only one under consideration is the Kaggle GPU training contingency (see Key Decisions, ⚠️ Revisit).
- **Reproducibility**: pinned dependencies, recorded seeds and configs, thresholds never tuned on test.
- **Laya internals**: the private API usage must be confined to one adapter module.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

The root `CLAUDE.md` is the authoritative description of commands and architecture; this is a summary.

## Languages and Runtime

- Python 3.12 (`requires-python = "==3.12.*"` in `pyproject.toml`), run directly as scripts with `argparse` CLIs: `prepare_data.py`, `ray_train.py`, `score.py`, `report.py`, `evaluate.py` (a thin wrapper over the two), `mps_spike.py` (Phase 2) and `compute_checkpoint.py` (Phase 4), plus `intents.py` and the `pilot/` package.
- Dependencies are exact-pinned in `pyproject.toml` and hash-locked in `requirements.txt` (universal lock, binary wheels only). `make env` builds `.venv` with `pip install --require-hashes --only-binary :all:`; `make lock` / `make lock-check` regenerate and check the lock.
- Ray on a single node: `ray_train.py` refuses a multi-node cluster (workers read driver-local paths); `score.py` scores with Ray Data actors, or in-process with `--workers 0` (`report.py` is numpy only and loads no model). Devices: `ray_train.py --device cpu|cuda|mps` (`--gpu` is an alias for `--device cuda`; mps trains in one Ray worker on the Apple GPU) and `score.py --device cpu|mps|cuda` (mps in-process with `--workers 0`, fp32 on every device). The Phase 2 spike `mps_spike.py` found the full fine-tune fits (`results/spike/mps_spike.md`), and the Phase 4 `compute_checkpoint.py` measured it on the M4 (2 epochs with a kill and resume, BERTurk and scoring probes, the projected experiment-matrix runtime; `results/compute/compute_checkpoint.md`).

## Frameworks and Key Dependencies

- `laya==0.4.6`: `score.py` loads every checkpoint through the public `laya.load(..., expected_sha256=...)`. Private internals (`laya.train`: `load_checkpoint`, `items_from_rows`, `split_calibration`, `encode_item`, `_forward`, `rlcd_loss`/`soft_ce_loss`, `calibration_records`, `save_checkpoint`; `laya.calibrate.fit_temperature_map` and the noul logit route; `laya.common`) are confined to `pilot/laya_adapter.py`, used by `ray_train.py`, `mps_spike.py` and `score.py`. `tests/test_laya_adapter.py` is the version and signature contract (exact laya version, an `inspect.signature` snapshot of every wrapped symbol, a tiny-model round trip) and an AST guard that fails on a private laya import in any other file.
- `ray[train,data]==2.59.0` (`TorchTrainer`, `ScalingConfig`, `RunConfig`, `Checkpoint`; Ray Data `map_batches`), `torch==2.14.1`, `transformers==5.19.0`, `huggingface-hub==2.2.0`, `safetensors==0.8.0`.
- Dev: `pytest==9.1.1`, `pytest-timeout`, `torchmetrics`. No linter or formatter.

## Configuration

- CLI flags only. `prepare_data.py`: `--data-dir --out --include-no-intent/--no-include-no-intent --rewrite-manifest`. `ray_train.py`: `--data --base --out --workers --device cpu|cuda|mps --gpu --amp --grad-checkpointing --epochs --micro-batch --grad-accum --max-len --neg --neg-seed --encoder-lr --head-lr --loss --freeze-embeddings --freeze-encoder --max-messages --calib-max-messages --calib-frac --storage --manifest --seed --print-run-id --export-epochs` etc. `score.py`: `SPLIT --model laya|majority|prior --checkpoint --device --workers --cpus-per-worker --gpus-per-worker --n --out-root --force --manifest`. `report.py`: `RUN_DIR --split --out --manifest`, `--compare A_DIRS B_DIRS --B --seed`, `--table PATH...`. `evaluate.py`: `--checkpoint --n --workers --cpus-per-worker --gpus-per-worker --device --out-root --force --out --manifest`. `compute_checkpoint.py`: `--out --device --data --manifest --base --epochs --seed --cpus --storage --min-available-gib --interrupt kill|fail|none --kill-delay-s --keep-run --spike-report --berturk/--no-berturk --score-probe/--no-score-probe`. Make: `make baselines` (rescore majority, prior and zero-shot Laya on val + test) and `make baseline-reports` (rebuild `results/reports/` and the table from the committed dumps, no model).
- Every pinned revision and sha256 (dataset, 40 raw folds, Laya, BERTurk, OOD tag subset) lives in `pilot/pins.py`. The code reads no environment variables; `HF_HUB_OFFLINE=1` reuses the hub cache after a first fetch.

## Platform Requirements

- About 2.6 GB RAM per Ray worker; plan for 16 GB or more per worker. Disk: about 1.5 GB venv + 1.1 GB weights.
- Data: `make data` downloads the pinned Multi3NLU++ revision and verifies all 40 raw fold hashes. Weights: `make weights` (optional prefetch). Ray run storage defaults to `./ray_results` (gitignored), one experiment dir per run.

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming and Style

- snake_case modules and functions, short verbs (`to_case`, `metrics`, `fetch`, `verify_split_file`); UPPER_CASE module constants (`INTENTS`, `QUESTIONS`, `SPLITS`, `TRAINED_ON`); Ray callback `train_loop_per_worker`.
- Terse local names are normal: `a` (parsed args), `c` (train config), `cfg` (model config), `tok`, `mb`, `tp/fp/fn`, `g/p`.
- No type hints. The only class is `Predictor` in `score.py` (a Ray Data actor).
- PEP 8-like, 4-space indent, double quotes, lines up to ~120 characters, hand-formatted. Compact style: tuple-unpack counters, conditional expressions.
- `%`-style formatting for messages (`"%s: %d messages" % (...)`).

## Error Handling

- Refusals are `raise SystemExit("<what is wrong>; <how to fix>")` with %-formatted messages, raised before any model load or `ray.init` where possible (tampered, unregistered, missing or wrong-role splits; stale manifests; unpinned or mismatched weights). Argument errors use `ap.error(...)`.
- Zero-division is guarded with conditional expressions (`_prf` in `pilot/metrics.py`). try/except is rare (e.g. `write_atomic` cleans up its temp file and re-raises).

## Logging and Comments

- Progress goes through `print` (`flush=True` inside Ray workers; only rank 0 prints in `ray_train.py`). Reports print as `json.dumps(report, ensure_ascii=False, indent=2)`.
- Every script and `pilot/` module opens with a docstring (purpose, design notes, `usage:`). Inline comments explain the "why" of invariants. Helpers that encode a contract carry a short docstring.

## Data and I/O

- Open files with `with` and `encoding="utf-8"`; write JSON with `ensure_ascii=False`. Outputs are written atomically (`pilot.data.write_atomic`).
- JSONL in Laya's training schema, one case per line; parse split bytes with `pilot.data.parse_jsonl` (splits on `\n` only).
- Seeds are explicit (`random.Random(seed)`); `score.py` shuffles with a fixed seed of 0 before `--n`; the bootstrap seed defaults to 0.
- Reports and checkpoints carry repo-relative paths and sha256s, never local paths (`pilot.data.repo_relpath`).

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Intent schema | 48 tags -> yes/no question text (the model input); the only `QUESTIONS` definition | `intents.py` |
| Pins | Every pinned revision and sha256 | `pilot/pins.py` |
| Splits | Producer (`prepare`, `SPLITS`, `TRAINED_ON`) and the only verifier (`verify_split_file`) | `pilot/data.py` |
| Audit | Report-only duplicate audit, hash-registered in the manifest | `pilot/audit.py` |
| Weights | Pinned, digest-verified fetch (`fetch`), provenance hashes (`identity`) | `pilot/weights.py` |
| Data prep CLI | Pinned Multi3NLU++ -> `data/splits/*.jsonl`, `data/audit/`, `data/splits.json` | `prepare_data.py` |
| Trainer | Single-node DDP fine-tuning on Ray Train (cpu, cuda, or one MPS worker), neg-K items, message-level calibration holdout, fallback ladder, run id, per-epoch resume, run record | `ray_train.py` |
| Run record | Torch-free run-record schema, run identity (`run_id`), neg-K item count, LAYA-03 fallback ladder | `pilot/runs.py` |
| Laya adapter | The only module using Laya's private API, pinned to laya 0.4.6 and contract-tested | `pilot/laya_adapter.py` |
| Dumps | Logit dump format: `write_dump`, `load_dump` (the only, hard-failing reader), `read_gold`, `run_id` | `pilot/dumps.py` |
| Metrics | Torch-free metrics, the val θ rule (`select_theta`), `seed_average`, `paired_bootstrap` | `pilot/metrics.py` |
| Scorer | The only place a model runs: one system (laya, majority, prior) on one val/test split -> logit dump | `score.py` |
| Reporter | Dumps -> metrics at θ 0.5 and val-selected θ (shipped and T = 1), `--compare` bootstrap, `--table` | `report.py` |
| Evaluator | Thin wrapper: `score.py --model laya` then `report.run_report` | `evaluate.py` |
| Compute checkpoint | 2-epoch MPS probe with a kill and resume, BERTurk and scoring probes, matrix projection -> `results/compute/` | `compute_checkpoint.py` |
| Compute logic | Torch-free projection, what-if, disk budget, assumptions and markdown of the compute checkpoint | `pilot/compute.py` |
| Results | Committed dumps (`dumps/`), full-test reports (`reports/`), legacy-protocol reports (`legacy/`), spike (`spike/`), compute checkpoint (`compute/`) | `results/` |
| Tests | Offline contracts and guards (`make test`); network tests via `make test-all` | `tests/` |

## Pattern Overview

- Each of the 48 tags is one Laya `noul` question; one forward pass yields a probability per tag.
- Splits: `selection_train` (folds 0-13), `val` (14-15), `final_train` (0-15), `test` (16-19), `hotels_probe` (HOTELS 0-19, out-of-domain). Fold boundaries live only in `pilot.data.SPLITS`.
- The committed `data/splits.json` registers each split's path, role, folds and sha256. Consumers verify the exact bytes they parse; a `--manifest` may differ only in what `prepare()` varies, never in roles, paths or folds.
- `score.py` scores only `val`/`test`; `ray_train.py` trains only on role `train`. θ is selected on val only (`report.py`), never tuned on test.

## Architectural Constraints

- **DDP invariants:** same seed and items on every rank, equal-sized shards (remainder dropped), `model.no_sync()` on non-step micro-batches, every rank attaches its `rng-rank<r>.pt` to each epoch checkpoint (rank 0 adds `train_state.pt` and `save.json`; resume reads every rank's file), and only the final checkpoint (rank 0, `final.json`) is a calibrated laya checkpoint; uncalibrated epoch exports go to `exports/epoch-NNN`.
- **Single node:** `ray_train.py` refuses a multi-node Ray cluster.
- **Laya internals:** private API use is confined to `pilot/laya_adapter.py` (an AST guard in `tests/test_laya_adapter.py` fails on any other importer).
- **Global state:** module-level constants only (`INTENTS`, `QUESTIONS`, `SPLITS`, `TRAINED_ON`, pins).

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-fast` for a trivial task inline, with no subagents and no PLAN.md
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
