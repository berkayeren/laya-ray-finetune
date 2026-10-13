---
last_mapped_commit: 49868df263b5b89e8efe10909576f479c0330c44
last_mapped_at: 2026-10-13
---
# Codebase Structure

**Analysis Date:** 2026-10-13

## Directory Layout

```
laya-ray-finetune/
├── intents.py          # INTENTS: 48 tag -> yes/no question
├── prepare_data.py     # Multi3NLU++ folds -> train/test JSONL
├── ray_train.py        # Ray Train DDP fine-tuning
├── evaluate.py         # Ray Data distributed evaluation
├── results/            # JSON reports from evaluate.py --out
├── README.md           # Usage and results table
├── CLAUDE.md           # Agent guidance
├── LICENSE
├── .gitignore
└── .planning/codebase/ # Generated codebase maps
```

Generated at runtime (not in repo): `data_full/`, `data_neg8/` (JSONL), `ckpt/` (checkpoint), `ray_results/` (Ray run storage).

## Directory Purposes

**`results/`:**
- Purpose: Evaluation reports
- Contains: `finetuned_smoke.json`, `zeroshot_multilingual.json`

## Key File Locations

**Entry Points:** `prepare_data.py`, `ray_train.py`, `evaluate.py`
**Configuration:** CLI flags via argparse; no config files. Label wording in `intents.py`.
**Core Logic:** `ray_train.py:train_loop_per_worker`, `evaluate.py:Predictor` and `metrics`
**Testing:** Not applicable (no test suite; use `--workers 0 --n 20` smoke runs)

## Naming Conventions

**Files:** lowercase snake_case flat modules at repo root (`prepare_data.py`).
**Directories:** lowercase; data outputs `data_<variant>` (`data_neg8`); results named `<model>_<setup>.json`.
**Code:** snake_case functions, PascalCase classes (`Predictor`), UPPER_CASE constants (`INTENTS`, `QUESTIONS`); CLI flags kebab-case (`--max-messages`).

## Where to Add New Code

**New intent/tag:** `intents.py`, then regenerate data.
**New data variant/split logic:** `prepare_data.py`.
**New training option:** add `ap.add_argument` and a key in the `cfg` dict in `ray_train.py:main`, consume in `train_loop_per_worker`.
**New metric:** `evaluate.py:metrics`.
**New script:** root-level `<name>.py` with argparse `main()` and `__main__` guard; import shared constants from `intents.py`.
**Saved reports:** `results/<name>.json` via `evaluate.py --out`.

## Special Directories

**`results/`:** evaluation output; generated; committed.
**`ray_results/`, `ckpt/`, `data_*/`:** generated at runtime; not part of the source tree.

---

*Structure analysis: 2026-10-13*
