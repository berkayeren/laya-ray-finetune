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

## Languages

- Python (version not pinned; no `.python-version`/`pyproject.toml`) - all code: `prepare_data.py`, `ray_train.py`, `evaluate.py`, `intents.py`
- Not applicable

## Runtime

- CPython, run directly as scripts (`python ray_train.py ...`); CLI via `argparse`
- Ray cluster (local or multi-node); CPU or GPU workers (CUDA via `--gpu`, `--gpus-per-worker`)
- pip (`pip install laya "ray[train,data]"`)
- Lockfile: missing; no `requirements.txt`, no package config

## Frameworks

- `laya` (unpinned) - multilingual yes/no (`noul`) question model; imports `laya.train`, `laya.calibrate`, `laya.common`, and `laya.load` in `ray_train.py` and `evaluate.py`. Uses private/internal APIs, so verify against the installed version.
- `ray[train]` (unpinned) - `TorchTrainer`, `ScalingConfig`, `RunConfig`, `Checkpoint` for DDP training in `ray_train.py`
- `ray[data]` (unpinned) - `map_batches` with a `Predictor` actor class for distributed inference in `evaluate.py`
- `torch` - model, DDP, `no_sync()` gradient accumulation (`ray_train.py`)
- None (no test suite)
- None (no linter, formatter, or build config). CLI entry `laya-train` comes from the `laya` package (used for `--dry-run` validation).

## Key Dependencies

- `laya` - training internals (`items_from_rows`, `split_calibration`, `encode_item`, `rlcd_loss`, `soft_ce_loss`, `save_checkpoint`, `fit_temperature_map`)
- `ray` - distributed training and inference
- `torch` - installed transitively
- Stdlib only otherwise: `argparse`, `json`, `glob`, `random`, `shutil`, `tempfile`, `time`, `os`

## Configuration

- Configured entirely through CLI flags (`ray_train.py`: `--data --base --out --workers --gpu --epochs --micro-batch --grad-accum --encoder-lr --head-lr --loss --freeze-encoder --storage --object-store-gb --seed` etc.; `evaluate.py`: `--checkpoint --n --workers --gpus-per-worker --out`; `prepare_data.py`: `--neg --seed`)
- No environment variables read by the code; no `.env` files
- None. Task definition lives in `intents.py` (`INTENTS`, 48 tags).

## Platform Requirements

- 16 GB or more RAM per Ray worker (about 2.6 GB each plus overhead)
- Multi3NLU++ Turkish banking folds downloaded manually into a `DATA_DIR`
- Not applicable; research/fine-tuning scripts. Multi-node runs need a shared `--storage` (NFS/S3). Default storage is `./ray_results`.

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- lowercase snake_case, one script per job, at the repo root: `prepare_data.py`, `ray_train.py`.
- snake_case, short verbs: `load`, `to_case`, `metrics`, `load_agent`, `f1`, `main`.
- Ray entry callback is `train_loop_per_worker` (Ray naming).
- Terse names are normal: `a` (parsed args), `c` (train config dict), `cfg` (model config), `tok`, `mb`, `accum`, `tp/fp/fn`, `g/p` (gold/probs).
- Module constants are UPPER_CASE: `INTENTS` in `intents.py`, `QUESTIONS` in `prepare_data.py` and `evaluate.py`.
- Intent tags are snake_case string keys, such as `transfer_payment_deposit`.
- No type hints, no classes except `Predictor` in `evaluate.py` (a Ray Data actor).

## Code Style

- PEP 8-like, 4-space indent, double-quoted strings, lines up to ~120 characters. Hand-formatted, no tool.
- Compact style: tuple-unpack counters (`tp, fp, fn = tp + ..., ...`), conditional expressions, and one-line `import gc; gc.collect()`.
- `%`-style string formatting is used for messages (`"%s: %d messages" % (...)`). Use it for new log lines.
- None configured.

## Import Organization

## Error Handling

- Almost none. There are no try/except blocks. Failures surface as raw exceptions.
- Inputs are filtered with guard clauses, such as `if r.get("intents")` in `prepare_data.py`.
- Zero-division is guarded with conditional expressions (`f1` in `evaluate.py`: `tp / (tp + fp) if tp + fp else 0.0`).
- Argparse enforces required and choice arguments (`--loss` choices, `--data` required).

## Logging

- Progress and summary lines go through `print`. `flush=True` is used inside Ray workers.
- Reports print as `json.dumps(report, ensure_ascii=False, indent=2)`.
- Only rank 0 prints in `ray_train.py`.

## Comments

- Every script opens with a module docstring: purpose, design notes, and a `usage:` block.
- Inline comments explain the "why" of invariants, such as "equal shards keep DDP collectives in step".
- Docstrings on functions are rare, with one-line exceptions (`Predictor`).

## Function Design

## Module Design

## Data and I/O Conventions

- Open text files with `encoding="utf-8"` and write JSON with `ensure_ascii=False` (Turkish text).
- JSONL, one case per line, in Laya's training schema (see `CLAUDE.md`).
- Seeds are explicit: `random.Random(seed)`, never the global RNG. `evaluate.py` uses a fixed seed of 0.
- File handles are often opened inline without `with` (`json.load(open(...))`). New code should prefer `with`.

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Intent schema | 48 tag -> yes/no question text (the model input) | `intents.py` |
| Data prep | Multi3NLU++ folds -> Laya JSONL; train folds 0-15, test 16-19 | `prepare_data.py` |
| Trainer | DDP fine-tuning loop on Ray Train, calibration, checkpoint | `ray_train.py` |
| Evaluator | Distributed inference + multi-label metrics | `evaluate.py` |
| Reports | Saved evaluation JSON | `results/` |

## Pattern Overview

- Each of 48 tags is one Laya `noul` question; one forward pass yields a probability per tag.
- No package, no tests, no shared library beyond `intents.py`.
- Training depends on private Laya internals (`laya.train`, `laya.common`, `laya.calibrate`).

## Layers

## Data Flow

### Primary Path

### Evaluation Path

## Key Abstractions

## Entry Points

- `prepare_data.py`, `ray_train.py`, `evaluate.py`: each has `main()` guarded by `if __name__ == "__main__"` and argparse CLI.

## Architectural Constraints

- **Threading:** Ray worker processes; each worker about 2.6 GB RAM.
- **Global state:** Module-level `INTENTS` and `QUESTIONS` constants only.
- **DDP invariants:** same seed on all ranks, equal-sized shards (remainder dropped), `model.no_sync()` on non-step micro-batches, only rank 0 attaches checkpoints.
- **Duplication:** `QUESTIONS` is built separately in `prepare_data.py` and `evaluate.py`; keep in sync via `intents.py`.
- **Splits hardcoded:** folds 0-15 / 16-19 in `prepare_data.py:main`.

## Anti-Patterns

### Divergent question text

## Error Handling

## Cross-Cutting Concerns

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
