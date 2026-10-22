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

- Python 3.12 (`requires-python = "==3.12.*"` in `pyproject.toml`), run directly as scripts with `argparse` CLIs: `prepare_data.py`, `ray_train.py`, `score.py`, `report.py`, `evaluate.py` (a thin wrapper over the two), `mps_spike.py` (Phase 2), `compute_checkpoint.py` (Phase 4), `postfit.py` (Phase 5), `berturk_train.py` (Phase 6, the BERTurk baseline: the HF Trainer in one process, no Ray) and `latency.py` (Phase 7, per-message latency of the final seeds, in-process, reported, not gated), plus `intents.py` and the `pilot/` package.
- Dependencies are exact-pinned in `pyproject.toml` and hash-locked in `requirements.txt` (universal lock, binary wheels only). `make env` builds `.venv` with `pip install --require-hashes --only-binary :all:`; `make lock` / `make lock-check` regenerate and check the lock.
- Ray on a single node: `ray_train.py` refuses a multi-node cluster (workers read driver-local paths); `score.py` scores with Ray Data actors, or in-process with `--workers 0` (`report.py` is numpy only and loads no model). Devices: `ray_train.py --device cpu|cuda|mps` (`--gpu` is an alias for `--device cuda`; mps trains in one Ray worker on the Apple GPU) and `score.py --device cpu|mps|cuda` (mps in-process with `--workers 0`, fp32 on every device). The Phase 2 spike `mps_spike.py` found the full fine-tune fits (`results/spike/mps_spike.md`), and the Phase 4 `compute_checkpoint.py` measured it on the M4 (2 epochs with a kill and resume, BERTurk and scoring probes, the projected experiment-matrix runtime; `results/compute/compute_checkpoint.md`).

## Frameworks and Key Dependencies

- `laya==0.4.6`: `score.py` loads every checkpoint through the public `laya.load(..., expected_sha256=...)`. Private internals (`laya.train`: `load_checkpoint`, `items_from_rows`, `encode_item`, `_forward`, `rlcd_loss`/`soft_ce_loss`, `save_checkpoint`; the noul logit route; `laya.common`) are confined to `pilot/laya_adapter.py`, used by `ray_train.py`, `mps_spike.py` and `score.py`. Training no longer calibrates (LAYA-05), so `laya.calibrate.fit_temperature_map` stays in the adapter only as the test oracle of `pilot/postfit.py`'s T fit; the production T fit is scipy's bounded scalar minimiser (`scipy==1.18.1`) in `pilot/postfit.py`. `tests/test_laya_adapter.py` is the version and signature contract (exact laya version, an `inspect.signature` snapshot of every wrapped symbol, a tiny-model round trip) and an AST guard that fails on a private laya import in any other file.
- `ray[train,data]==2.59.0` (`TorchTrainer`, `ScalingConfig`, `RunConfig`, `Checkpoint`; Ray Data `map_batches`), `torch==2.14.1`, `transformers==5.19.0`, `huggingface-hub==2.2.0`, `safetensors==0.8.0`.
- `accelerate==1.15.0` (Phase 6): the HF Trainer that `berturk_train.py` uses needs it, so it was added to the lock. The BERTurk base is `dbmdz/bert-base-turkish-cased` at its pinned revision (`pilot.pins.BERTURK`, safetensors only, sha256-checked); `pilot/berturk.py` is torch-free and never imports torch, transformers, accelerate, Ray or laya.
- Dev: `pytest==9.1.1`, `pytest-timeout`, `torchmetrics`. No linter or formatter.

## Configuration

- CLI flags only. `prepare_data.py`: `--data-dir --out --include-no-intent/--no-include-no-intent --rewrite-manifest`. `ray_train.py`: `--data --base --out --workers --device cpu|cuda|mps --gpu --amp --grad-checkpointing --epochs --micro-batch --grad-accum --max-len --neg --neg-seed --encoder-lr --head-lr --loss --freeze-embeddings --freeze-encoder --max-messages --storage --manifest --seed --print-run-id --export-epochs/--no-export-epochs --stop-after-epoch --postfit` etc. `postfit.py` (Laya modes): `RUN_DIR... --out --manifest`, `--prune EXPORT_DIR... --dump-root`, `--print-final-commands RECORD --storage --out-prefix --base`, `--final-manifest RECORD CKPT... --out`, `--render RECORD [FINAL_MANIFEST]`; BERTurk modes: the same RUN_DIR, `--prune` and `--final-manifest` (dispatched by system kind), `--print-final-commands RECORD --reference REF --storage --out-prefix --base`, `--reference-check RECORD --resolution --out`, `--render-berturk RECORD REFERENCE [FINAL_MANIFEST]`. `berturk_train.py`: `--data --base --device cpu|mps|cuda --epochs (default 30) --lr (encoder, default 2e-05) --head-lr (default 1e-4) --batch --max-len --seed --mask-tags --max-messages --cpus --storage (default berturk_runs) --manifest --print-run-id --export-epochs/--no-export-epochs --stop-after-epoch --out --postfit --reference`. `score.py`: `SPLIT --model laya|berturk|majority|prior --checkpoint --device --workers --cpus-per-worker --gpus-per-worker --n --out-root --force --manifest`. `report.py`: `RUN_DIR --split --out --manifest`, `--compare A_DIRS B_DIRS --B --seed`, `--table PATH...`, `--confidence RUN_DIR --split val|test|hotels_probe --postfit RECORD --out X.json` (Phase 7: the confidence/1 record, plus X.svg), `--confidence-table PATH...`. `latency.py` (Phase 7): `--system laya|berturk --device cpu|mps|cuda --final-manifest --checkpoint-root --split --manifest --threads --warmup --min-available-gib --out`, `--render DIR --out`. `evaluate.py`: `--checkpoint --n --workers --cpus-per-worker --gpus-per-worker --device --out-root --force --out --manifest`. `compute_checkpoint.py`: `--out --device --data --manifest --base --epochs --seed --cpus --storage --min-available-gib --interrupt kill|fail|none --kill-delay-s --probe-timeout-s --keep-run --spike-report --berturk/--no-berturk --score-probe/--no-score-probe`. Make: `make baselines` (rescore majority, prior and zero-shot Laya on val + test) and `make baseline-reports` (rebuild `results/reports/` and the table from the committed dumps, no model), `make laya-postfit` (rebuild `results/postfit/laya.json` from the committed Laya selection val dumps, no model), `make berturk-postfit` (rebuild `results/postfit/berturk.json` from the 30 committed BERTurk selection val dumps, no model), `make confidence-reports` (rebuild the three Phase 7 val confidence records and their reliability SVGs under `results/confidence/` from the committed dumps and postfit records, no model).
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
| Weights | Pinned, digest-verified fetch (`fetch`), provenance hashes (`identity`: every file `laya.load` reads) | `pilot/weights.py` |
| Data prep CLI | Pinned Multi3NLU++ -> `data/splits/*.jsonl`, `data/audit/`, `data/splits.json` | `prepare_data.py` |
| Trainer | Single-node DDP fine-tuning on Ray Train (cpu, cuda, or one MPS worker), neg-K items, uncalibrated (T = 1) checkpoints, `--stop-after-epoch`, the `--postfit` binding of a final seed, fallback ladder, run id, per-epoch resume, run record | `ray_train.py` |
| BERTurk trainer | HF Trainer fine-tune of the pinned BERTurk (one process, no Ray; cpu, mps or cuda): masked BCE, encoder LR and head LR in their own AdamW param groups, 30-epoch schedule, per-epoch exports, run id, no resume, `--stop-after-epoch`, the `--postfit`/`--reference` binding of a final seed | `berturk_train.py` |
| BERTurk bookkeeping | Torch-free recipe, run and checkpoint identity, labels in `INTENTS` order, postfit protocol, BASE-03 reference check, final commands and manifest, the superseded chain (`SUPERSEDED`, `load_superseded`), the README Phase 6 block | `pilot/berturk.py` |
| Run record | Torch-free run-record schema, run identity (`run_id`), neg-K item count, LAYA-03 fallback ladder | `pilot/runs.py` |
| Laya adapter | The only module using Laya's private API, pinned to laya 0.4.6 and contract-tested | `pilot/laya_adapter.py` |
| Dumps | Logit dump format: `write_dump`, `load_dump` (the only, hard-failing reader), `read_gold`, `run_id` | `pilot/dumps.py` |
| Metrics | Torch-free metrics, the val θ rule (`select_theta`), `seed_average`, `paired_bootstrap` | `pilot/metrics.py` |
| Scorer | The only place a model runs: one system (laya, berturk, majority, prior) on one val/test split -> logit dump; BERTurk refused on test until Phase 8 | `score.py` |
| Reporter | Dumps -> metrics at θ 0.5 and val-selected θ (shipped and T = 1), `--compare` bootstrap, `--table` | `report.py` |
| Confidence metrics | Calibration (Brier, NLL, classwise ECE, reliability), abstention (one message-confidence rule, coverage, AURC) and false positives (no-intent and out-of-domain, Wilson 95%) as torch-free functions; the `confidence/1` record (`--confidence`, three T/θ bindings), its reliability SVG and the README block (`--confidence-table`) | `pilot/metrics.py`, `report.py` |
| Latency harness | Per-message latency of every final seed, batch 1, in-process, fp32, through the scoring load path (`score.load_bert` for BERTurk); torch-free timing contract, nearest-rank summaries, `latency/1` report and markdown; reported, not gated | `latency.py`, `pilot/latency.py` |
| Evaluator | Thin wrapper: `score.py --model laya` then `report.run_report` | `evaluate.py` |
| Compute checkpoint | 2-epoch MPS probe with a kill and resume, BERTurk and scoring probes, matrix projection -> `results/compute/` | `compute_checkpoint.py` |
| Compute logic | Torch-free projection, what-if, disk budget, assumptions and markdown of the compute checkpoint | `pilot/compute.py` |
| Postfit | Val-only T fit, θ and epoch selection into a byte-reproducible record, export pruning, final-seed commands and manifest, the README Phase 5 block; BERTurk through the same functions (protocol by system kind), its reference check and the README Phase 6 block (`--render-berturk`) | `pilot/postfit.py`, `postfit.py` |
| Results | Committed dumps (`dumps/`), full-test reports (`reports/`), legacy-protocol reports (`legacy/`), spike (`spike/`), compute checkpoint (`compute/`), postfit records (`postfit/`: Laya and BERTurk), the superseded history (`superseded/`: the two discarded BERTurk selection runs, never current), Phase 7 val confidence records and reliability SVGs (`confidence/`) and latency reports (`latency/`) | `results/` |
| Tests | Offline contracts and guards (`make test`); network tests via `make test-all` | `tests/` |

## Pattern Overview

- Each of the 48 tags is one Laya `noul` question; one forward pass yields a probability per tag.
- Splits: `selection_train` (folds 0-13), `val` (14-15), `final_train` (0-15), `test` (16-19), `hotels_probe` (HOTELS 0-19, out-of-domain). Fold boundaries live only in `pilot.data.SPLITS`.
- The committed `data/splits.json` registers each split's path, role, folds and sha256. Consumers verify the exact bytes they parse; a `--manifest` may differ only in what `prepare()` varies, never in roles, paths or folds.
- `score.py` scores only `val`/`test`; `ray_train.py` trains only on role `train`. θ is selected on val only (`report.py`), never tuned on test.

## Architectural Constraints

- **DDP invariants:** same seed and items on every rank, equal-sized shards (remainder dropped), `model.no_sync()` on non-step micro-batches, every rank attaches its `rng-rank<r>.pt` to each epoch checkpoint (rank 0 adds `train_state.pt` and `save.json`; resume reads every rank's file), and the final checkpoint (rank 0, `final.json`) is uncalibrated (T = 1) unless `--postfit` stamps the val-fitted T; uncalibrated epoch exports go to `exports/epoch-NNN`.
- **Val-only selection:** Laya's epoch, temperature and θ are chosen on val only (`postfit.py`, which refuses test input), at one encoder LR (2e-05, fixed, not tuned); `score.py` refuses a fine-tuned checkpoint on test until Phase 8.
- **BERTurk (Phase 6):** BERTurk's epoch, T and θ are val-only too, through `pilot.postfit` (`protocol("berturk")`); both models share the encoder LR 2e-05 (fixed, not tuned) and the head LR 1e-4. BERTurk's epoch grid is 30 against Laya's 4 (and the approved `berturk_epochs` 10), and BERTurk had three selection runs against Laya's one (two recipe changes after a val outcome, the BASE-03 fixes): a disclosed deviation, rendered in the README from the records. A superseded run (`pilot.berturk.SUPERSEDED`, `results/superseded/`) is never current and can never gate a final seed. No fine-tuned checkpoint of either model is scored on test before Phase 8.
- **Confidence and latency (Phase 7):** the calibration, abstention and false-positive numbers are val-only and in-sample (T, θ and the epoch were chosen on the same val messages); every record carries the caveat and both temperature variants. T and θ come only from val or the postfit record (`report.confidence_binding`), never from test or HOTELS; a val split holding committed test, HOTELS or selection_train (training) content is refused as their source, keyed on the content whatever the manifest says (`report.refuse_held_out_theta`), and a final seed is never reported on val or on content it trained on, in any report mode: `report.py RUN_DIR` (so `evaluate.py`), `--compare` and `--confidence` (`report.refuse_final_seed_training_content`); an `ood_probe` dump never reaches `select_theta`. There is one message-confidence rule (`pilot.metrics.CONFIDENCE_RULE`, the joint decision probability), fixed on principle and its val sensitivity disclosed (`report.CONFIDENCE_DISCLOSURE`); Phase 8 freezes it. Latency is reported, not gated. No fine-tuned model is scored on test or HOTELS before Phase 8: `score.py` still refuses `hotels_probe`, and the hygiene tests keep any HOTELS dump and any fine-tuned test dump out of git.
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
