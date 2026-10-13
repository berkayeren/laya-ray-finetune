---
last_mapped_commit: 49868df263b5b89e8efe10909576f479c0330c44
last_mapped_at: 2026-10-13
---
<!-- refreshed: 2026-10-13 -->

# Architecture

**Analysis Date:** 2026-10-13

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                  CLI scripts (run directly)                  │
├──────────────────┬──────────────────┬───────────────────────┤
│ prepare_data.py  │  ray_train.py    │   evaluate.py         │
│ raw folds->JSONL │  Ray Train DDP   │   Ray Data inference  │
└────────┬─────────┴────────┬─────────┴──────────┬────────────┘
         │                  │                     │
         ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Shared label schema: `intents.py` (INTENTS -> noul QUESTIONS)│
└─────────────────────────────────────────────────────────────┘
         │                  │                     │
         ▼                  ▼                     ▼
  data_*/train|test.jsonl   ckpt/ (laya.load)    results/*.json
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

**Overall:** Flat set of independent pipeline scripts (prepare -> train -> evaluate) connected only through files on disk and the shared `intents.py` module.

**Key Characteristics:**
- Each of 48 tags is one Laya `noul` question; one forward pass yields a probability per tag.
- No package, no tests, no shared library beyond `intents.py`.
- Training depends on private Laya internals (`laya.train`, `laya.common`, `laya.calibrate`).

## Layers

**Schema:** `intents.py` exports `INTENTS`; `prepare_data.py` and `evaluate.py` each build `QUESTIONS = {k: {"type": "noul", "instructions": v}}`.
**Data:** `prepare_data.py` (`load`, `to_case`, `main`). Depends on `intents.py`.
**Training:** `ray_train.py` (`train_loop_per_worker`, `main`). Depends on Laya, Ray Train, torch.
**Evaluation:** `evaluate.py` (`load_agent`, `Predictor`, `f1`, `metrics`, `main`). Depends on `intents.py`, Laya, Ray Data (imported lazily).

## Data Flow

### Primary Path

1. `python prepare_data.py DATA_DIR OUT [--neg K]` reads `foldN.json`, writes `train.jsonl` (optionally positives + K negatives per message) and `test.jsonl` (all 48 tags) (`prepare_data.py:main`).
2. `python ray_train.py --data train.jsonl --out ckpt` resolves the base checkpoint on the driver, launches `TorchTrainer`; each worker builds identical items, takes an equal shard, trains with grad accumulation (`ray_train.py:train_loop_per_worker`).
3. Rank 0 fits temperature calibration on a held-out slice, reports the final checkpoint; driver copies it to `--out` (`ray_train.py:main`).
4. `python evaluate.py test.jsonl --checkpoint ckpt` scores messages via `Predictor` actors and prints/saves a metrics report (`evaluate.py:main`).

### Evaluation Path

1. Load cases, shuffle with `random.Random(0)`, truncate to `--n`.
2. `--workers 0`: call `Predictor` in-process in chunks of 16. Otherwise `ray.data.from_items(...).map_batches(Predictor, concurrency=workers)`.
3. Probabilities are JSON-encoded per row, decoded by uid, passed to `metrics()`.

**State Management:** None persistent; all state is files (JSONL, checkpoint dir, Ray run storage `ray_results`).

## Key Abstractions

**Predictor:** Ray Data actor class loading one checkpoint per worker. `evaluate.py`.
**Train loop config dict `c`:** plain dict passed as `train_loop_config`, also stored into checkpoint config under `training.ray_train`. `ray_train.py`.
**Case record:** `{"state","uid","questions","expected"}` JSONL line. `prepare_data.py:to_case`.

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

**What happens:** Changing `INTENTS` wording without regenerating data.
**Why it's wrong:** Question text is model input; train/eval mismatch.
**Do this instead:** Edit only `intents.py`, then rerun `prepare_data.py` and baselines.

## Error Handling

**Strategy:** Minimal; failures propagate as exceptions. No retries or custom exception types.

## Cross-Cutting Concerns

**Logging:** `print` only (rank 0 for training summary).
**Validation:** `laya-train --dry-run` for JSONL format; `prepare_data.py` drops rows without intents.
**Authentication:** None (hub checkpoints loaded from `convaiinnovations/laya`).

---

*Architecture analysis: 2026-10-13*
