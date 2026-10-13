---
last_mapped_commit: 49868df263b5b89e8efe10909576f479c0330c44
last_mapped_at: 2026-10-13
---
# Codebase Concerns

**Analysis Date:** 2026-10-13

## Tech Debt

**Dependence on private Laya internals:**
- Issue: `ray_train.py` imports `laya.train` private helpers (`lt._forward`, `lt.encode_item`, `lt.items_from_rows`, `lt.split_calibration`, `lt.sigma_at`, `lt.draw_option_order`, `lt.calibration_records`, `lt.save_checkpoint`, `lt.load_checkpoint`, `lt.resolve_checkpoint_dir`), plus `laya.common.collate_items`/`uses_parallel_layout` and `laya.calibrate.fit_temperature_map`. Magic constants are passed positionally, e.g. `lt.rlcd_loss(logits, target, mask, qtype, sigma, 4, 0.75, 1.0)` and `lt.sigma_at(epoch, epochs, 0.4, 0.1)` (`ray_train.py` lines 95, 109).
- Files: `ray_train.py`
- Impact: Any `laya` upgrade can silently change signatures or semantics. No version pin exists, and there is no package config or requirements file.
- Fix approach: Add `requirements.txt` or `pyproject.toml` pinning `laya` and `ray`. Add a startup check of the `laya` version. Name the loss hyperparameters as constants or CLI flags.

**No dependency manifest, tests, or linter:**
- Issue: Dependencies are only documented in `CLAUDE.md` and `README.md`. Metrics, sharding and calibration code have no tests.
- Files: repo root, `evaluate.py`, `ray_train.py`, `prepare_data.py`
- Impact: Regressions go unnoticed. The pure functions `metrics`, `f1`, `to_case` and the shard arithmetic are easy to unit test.
- Fix approach: Add pytest for `metrics()` and `to_case()`, and a CPU smoke test for `ray_train.py`.

**Duplicated `QUESTIONS` construction:**
- Issue: `QUESTIONS` is built identically in `prepare_data.py` (line 21) and `evaluate.py` (line 16).
- Impact: Drift risk. Tag wording is the model input, so a mismatch invalidates results.
- Fix approach: Define `QUESTIONS` once in `intents.py` and import it.

**Resource and style rough edges:**
- Issue: `open(...)` without a context manager in `evaluate.py` (lines 88, 120) and `prepare_data.py` (line 27). Inline `import gc; gc.collect()` at `ray_train.py` line 134. The `ray.init` call in `evaluate.py` hides worker logs with `log_to_driver=False`, which obscures actor failures.
- Fix approach: Use `with open(...)`. Move imports to the top of the file.

## Known Bugs

**Fixed positional-index assumption in `evaluate.py` metrics:**
- Symptoms: `by_uid[u]` raises `KeyError` if two test rows share a `uid`, or if Ray drops a row. `p[k]` raises `KeyError` if the model omits a tag.
- Files: `evaluate.py` (lines 64-66, 113-117)
- Trigger: Duplicate `uid` across folds in `data_full/test.jsonl`. Duplicates also collapse `gold` and `by_uid` silently while `cases` keeps both.
- Workaround: Verify `uid` uniqueness in the source folds.

**Empty-intent rows dropped from both splits:**
- Symptoms: `load()` in `prepare_data.py` (line 28) filters rows with no `intents`. The model is never trained or tested on "no intent" messages, so false-positive behavior on out-of-scope text is unmeasured. `top1_hit` in `evaluate.py` (line 47) also assumes gold is non-empty.
- Files: `prepare_data.py`, `evaluate.py`
- Workaround: None. Reported scores are optimistic for real traffic.

## Security Considerations

**Remote model loading and unsafe deserialization surface:**
- Risk: `laya.load("convaiinnovations/laya", ...)` downloads weights from the Hugging Face hub with no revision pin or hash check (`evaluate.py` line 23; `lt.resolve_checkpoint_dir` in `ray_train.py` line 185).
- Files: `evaluate.py`, `ray_train.py`
- Current mitigation: None.
- Recommendations: Pin a revision. Treat checkpoints from untrusted sources as code.

**Training config persisted into the checkpoint:**
- Risk: `ray_train.py` (line 146) writes the whole training config, including the absolute local `data` path, into the checkpoint config. Sharing checkpoints leaks local paths.
- Recommendations: Strip `data` before saving, as is already done for `base_dir`.

## Performance Bottlenecks

**Per-rank memory and tokenization:**
- Problem: Every rank tokenizes the full dataset (`lt.items_from_rows` over all rows, lines 51-56) before taking its shard, and each worker holds about 2.6 GB. Two CPU workers ran out of memory in a 7 GB sandbox.
- Files: `ray_train.py`
- Improvement path: Shard rows before tokenizing, but keep the calibration split identical across ranks.

**Single-rank calibration stalls the others:**
- Problem: After the final epoch rank 0 runs `calibration_records` and `fit_temperature_map` (lines 135-150) while other ranks wait in `ray.train.report(last)`. On GPU this may exceed collective or report timeouts with large calibration sets.
- Files: `ray_train.py`
- Improvement path: Keep `--calib-max` small, or run calibration on the driver after training.

**Evaluation batching is fixed:**
- Problem: `batch_size=8` inside `Predictor.__call__` and `batch_size=16` in `map_batches` are hardcoded (`evaluate.py` lines 34, 107). Each message costs 48 question forward passes.
- Improvement path: Expose batch sizes as CLI flags.

## Fragile Areas

**DDP step-synchronization logic:**
- Files: `ray_train.py` (lines 57-58, 98-120)
- Why fragile: Correctness relies on equal shard sizes (remainder dropped, so up to `world-1` items are never used). `no_sync` is skipped on `step_now`, which depends on `n` and the shard length. `find_unused_parameters=True` is needed for the frozen encoder or unused head paths. If shards differ in length or a rank skips a collective, training hangs.
- Safe modification: Keep `per_rank = len // world`. If `per_rank` is 0, which happens with a tiny `--max-messages` and many workers, the shard is empty and the loop silently trains nothing, so guard against it.
- Test coverage: None. Only `results/finetuned_smoke.json` records a smoke run.

**Scheduler step count:**
- Files: `ray_train.py` (lines 83-85)
- Why fragile: `updates` is computed from `steps_per_epoch` and `accum`. The loop resets `n` each epoch and steps on the last partial batch, so the count must match exactly or the cosine LR ends early or late.

**Intermediate versus final checkpoints:**
- Files: `ray_train.py` (lines 124-129, 150, 200)
- Why fragile: Intermediate checkpoints are uncalibrated. Only the final report carries calibration, and the driver copies `result.checkpoint`. No `FailureConfig` or resume logic exists, so a mid-run failure leaves only an uncalibrated checkpoint in `--storage`.

**Reproducibility of evaluation subsets:**
- Files: `evaluate.py` (line 89)
- Why fragile: Shuffle seed 0 is hardcoded and not recorded in the report.

## Scaling Limits

**Training cost:**
- Current capacity: Full 48-tag training is about 77k items (`prepare_data.py` docstring). `--neg 8` reduces this.
- Limit: Full-tag training on CPU is impractical. `--freeze-encoder` fits small machines but gains little.
- Scaling path: Use `--gpu`, and a shared `--storage` on multi-node clusters.

## Dependencies at Risk

**`laya` (internal APIs) and `ray`:**
- Risk: Unpinned. `ray.train.torch.get_device`, `prepare_model(parallel_strategy_kwargs=...)` and `TorchTrainer` APIs change between Ray releases.
- Impact: `ray_train.py` breaks at import or runtime.
- Migration plan: Pin versions and record them in the README.

## Missing Critical Features

**Abstention thresholds and before/after report:**
- Problem: `ray_train.py` does not fit abstention thresholds or produce `laya-train`'s report. Thresholds are fixed at 0.3-0.9 in `evaluate.py`.
- Blocks: Checkpoints from `ray_train.py` are not equivalent to `laya-train` output for abstention behavior.

**No train/validation split by fold:**
- Problem: The calibration slice is random from train, and there is no validation-based early stopping or model selection.

## Test Coverage Gaps

**Everything:**
- What's not tested: `metrics()` correctness, `to_case()` negative sampling, DDP sharding, calibration output, JSONL schema.
- Files: `evaluate.py`, `prepare_data.py`, `ray_train.py`
- Risk: Silent metric or data errors affect the README results table, built from `results/*.json`.
- Priority: High for `metrics()` and `to_case()`, Medium for `ray_train.py`.

---

*Concerns audit: 2026-10-13*
