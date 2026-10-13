---
last_mapped_commit: 49868df263b5b89e8efe10909576f479c0330c44
last_mapped_at: 2026-10-13
---
# Coding Conventions

**Analysis Date:** 2026-10-13

Flat Python scripts (`prepare_data.py`, `evaluate.py`, `ray_train.py`, `intents.py`). No linter, formatter, or package config is present.

## Naming Patterns

**Files:**
- lowercase snake_case, one script per job, at the repo root: `prepare_data.py`, `ray_train.py`.

**Functions:**
- snake_case, short verbs: `load`, `to_case`, `metrics`, `load_agent`, `f1`, `main`.
- Ray entry callback is `train_loop_per_worker` (Ray naming).

**Variables:**
- Terse names are normal: `a` (parsed args), `c` (train config dict), `cfg` (model config), `tok`, `mb`, `accum`, `tp/fp/fn`, `g/p` (gold/probs).
- Module constants are UPPER_CASE: `INTENTS` in `intents.py`, `QUESTIONS` in `prepare_data.py` and `evaluate.py`.
- Intent tags are snake_case string keys, such as `transfer_payment_deposit`.

**Types:**
- No type hints, no classes except `Predictor` in `evaluate.py` (a Ray Data actor).

## Code Style

**Formatting:**
- PEP 8-like, 4-space indent, double-quoted strings, lines up to ~120 characters. Hand-formatted, no tool.
- Compact style: tuple-unpack counters (`tp, fp, fn = tp + ..., ...`), conditional expressions, and one-line `import gc; gc.collect()`.
- `%`-style string formatting is used for messages (`"%s: %d messages" % (...)`). Use it for new log lines.

**Linting:**
- None configured.

## Import Organization

**Order:**
1. Standard library (`argparse`, `json`, `os`, `random`, ...)
2. Third-party (`ray`, `torch`, `laya`)
3. Local (`from intents import INTENTS`)

**Heavy imports:** `evaluate.py` imports `laya` and `ray` lazily inside functions (`load_agent`, `main`) so `--workers 0` runs without Ray. Keep that pattern.

**Path Aliases:** None. Scripts import siblings directly, so run them from the repo root.

## Error Handling

**Patterns:**
- Almost none. There are no try/except blocks. Failures surface as raw exceptions.
- Inputs are filtered with guard clauses, such as `if r.get("intents")` in `prepare_data.py`.
- Zero-division is guarded with conditional expressions (`f1` in `evaluate.py`: `tp / (tp + fp) if tp + fp else 0.0`).
- Argparse enforces required and choice arguments (`--loss` choices, `--data` required).

## Logging

**Framework:** `print`.
- Progress and summary lines go through `print`. `flush=True` is used inside Ray workers.
- Reports print as `json.dumps(report, ensure_ascii=False, indent=2)`.
- Only rank 0 prints in `ray_train.py`.

## Comments

**When to Comment:**
- Every script opens with a module docstring: purpose, design notes, and a `usage:` block.
- Inline comments explain the "why" of invariants, such as "equal shards keep DDP collectives in step".
- Docstrings on functions are rare, with one-line exceptions (`Predictor`).

## Function Design

**Size:** Small helpers plus a long `main()`. `train_loop_per_worker` is a ~115-line single function.

**Parameters:** Training config travels as a single plain dict `c` (built in `main()`, keys mirror CLI flags). Add new options as an argparse flag plus a `cfg` key plus a `c["..."]` read.

**Return Values:** Plain dicts and lists. JSON-serializable, with numbers rounded via `round(x, 3)`.

## Module Design

**Exports:** No `__all__`. Each script ends with `if __name__ == "__main__": main()`.

**Shared code:** `intents.py` holds the single source of truth for tags. `QUESTIONS` is rebuilt from it in each script with the same dict comprehension.

**Barrel Files:** None.

## Data and I/O Conventions

- Open text files with `encoding="utf-8"` and write JSON with `ensure_ascii=False` (Turkish text).
- JSONL, one case per line, in Laya's training schema (see `CLAUDE.md`).
- Seeds are explicit: `random.Random(seed)`, never the global RNG. `evaluate.py` uses a fixed seed of 0.
- File handles are often opened inline without `with` (`json.load(open(...))`). New code should prefer `with`.

---

*Convention analysis: 2026-10-13*
