# Laya Turkish Banking Intent Pilot

## What This Is

A pilot that decides whether Laya (`laya-multilingual`) is good enough for Turkish banking intent classification at a bank. It uses the Turkish banking split of Multi3NLU++ (48 multi-label intent tags, CC-BY 4.0) as a public stand-in until the bank's own customer messages can be used. The deliverable is a defensible, reproducible go/no-go recommendation. It compares fine-tuned Laya against a well-known baseline (fine-tuned BERTurk) on accuracy, speed and confidence quality.

## Core Value

A trustworthy, reproducible head-to-head comparison (fine-tuned Laya vs fine-tuned BERTurk vs zero-shot Laya vs majority) on the full test set. It has to be trustworthy enough to make a go/no-go call that humans can audit and override.

## Requirements

### Validated

<!-- Inferred from the existing codebase (.planning/codebase/). -->

- ✓ Convert Multi3NLU++ Turkish banking folds to Laya JSONL. Train is folds 0–15 and test is folds 16–19. `--neg K` negative sampling. (`prepare_data.py`). Existing.
- ✓ 48 intent tags phrased as Laya `noul` yes/no questions, in one shared schema (`intents.py`). Existing.
- ✓ Ray Train `TorchTrainer` fine-tune loop using Laya's own loss (`rlcd`/`soft-ce`), temperature calibration on a held-out slice, and a checkpoint loadable with `laya.load` (`ray_train.py`). Existing. CPU smoke run proven.
- ✓ Ray Data distributed evaluation: top-1 vs majority baseline, micro-F1 and exact match at 0.3/0.5/0.7/0.9, macro-F1, best/worst per-tag F1, JSON reports (`evaluate.py`). Existing.
- ✓ Zero-shot `laya-multilingual` baseline on 100 test messages (micro-F1 0.186) (`results/zeroshot_multilingual.json`). Existing.
- ✓ Pinned, hash-locked dependencies (`pyproject.toml` + `requirements.txt`, Python 3.12, binary wheels only, `make env`). Phase 1.
- ✓ Frozen, hash-registered splits (`data/splits.json`): selection_train/val/final_train/test plus the HOTELS OOD probe; no-intent rows kept by default; tampered, unregistered or wrong-role files hard-fail. Phase 1.
- ✓ Pinned, digest-verified Laya and BERTurk weights (safetensors only) via `pilot.weights.fetch`. Phase 1.
- ✓ Local CPU test suite (`make test`, no hosted CI) with env-lock, single-`QUESTIONS`, split-contract and repo-hygiene guards. Phase 1.

### Active

**1. Trustworthy pipeline (first, small)**
- [ ] Include the no-intent rows in the main metrics and also report the false-positive rate on no-intent messages separately (rows are kept in the splits since Phase 1)
- [ ] Isolate private `laya.train` internals behind one adapter module. Add a test that fails loudly if their signatures change.
- [ ] Tests: data conversion (counts, splits, labels), metric functions on toy inputs, and a tiny CPU smoke run of ray_train + evaluate (few messages, 1 epoch, frozen encoder)
- [ ] One command reproduces data prep → training → evaluation from `pip install -r requirements.txt`

**2. Real fine-tune of Laya on Apple Silicon**
- [ ] `--device` flag supporting `mps` (Apple GPU) with CPU fallback. Ray Train runs with 1 worker. The existing CUDA `--gpu`/DDP path must keep working.
- [ ] Full encoder fine-tune of `laya-multilingual` on `data_neg8` within 16 GB unified memory: micro-batch 4–8, gradient accumulation, gradient checkpointing, max_len 128–256. Use `data_full` only if time allows.
- [ ] Checkpoint every epoch, and long runs can resume
- [ ] Validation split carved from train (never test). Per-tag or global decision thresholds are chosen on it.
- [ ] Evaluate on all 398 test messages (plus the no-intent rows)
- [ ] Report micro/macro F1, top-1, exact match, per-tag F1, ECE/calibration, precision at fixed coverage (abstention), latency per message (p50/p95, MPS and CPU), peak memory, and wall-clock time per epoch
- [ ] 3 seeds per fine-tune, with run configs and seeds recorded in the results

**3. Baseline comparison and decision**
- [ ] Fine-tune BERTurk (`dbmdz/bert-base-turkish-cased`) as a standard multi-label classifier on the same split, hardware, settings and seeds
- [ ] Few-shot new-intent experiment: hold out some tags and re-add them with k=5 or k=10 examples. Compare how fast Laya and BERTurk learn them.
- [ ] Paired bootstrap confidence intervals on the test set for the Laya vs BERTurk differences
- [ ] One results table: majority, zero-shot Laya, fine-tuned Laya and fine-tuned BERTurk (mean ± std over seeds)
- [ ] A README section with a go/no-go recommendation applying the GO criteria below, plus caveats (translated data, small dataset, laptop hardware as a proxy)

### GO criteria (decision rule)

GO if fine-tuned Laya meets all of the following:
- (a) It is within 2 points of BERTurk on micro-F1 and within 3 points on macro-F1 (paired bootstrap on the test set).
- (b) It is stable across 3 seeds with no collapsed run.
- (c) It meets the latency budget on the target hardware. **In v1 latency is reported, not gated**, because the budget is deferred.
- (d) It is clearly better on at least one of: calibration or precision at fixed coverage (abstention), or few-shot new-intent learning.

It is also a GO if Laya beats BERTurk by more than 3 points with a confidence interval that excludes zero. Otherwise it is a NO-GO, because BERTurk is simpler. All numbers are reported either way so humans can override the call.

### Out of Scope

- Other languages and domains. The pilot is Turkish banking only.
- Serving and deployment. This is a decision pilot, not production.
- Bank or customer data. None is available yet, and none may be committed.
- Multi-GPU and multi-node DDP work. The hardware is a single M4 Mac, but the existing path must not be broken.
- A data-efficiency curve (10/25/50/100% of train). It was considered, and the user chose calibration and few-shot as the edge tests instead.
- Hosted CI. A local test script is enough for now.

## Context

- The existing code is a flat set of scripts (`prepare_data.py`, `ray_train.py`, `evaluate.py`) sharing `intents.py`. See `.planning/codebase/`.
- Known issues from the codebase map: `ray_train.py` depends on private Laya APIs with magic positional constants. Nothing is pinned. There are no tests. `QUESTIONS` is duplicated in two files. `evaluate.py` raises KeyError on duplicate uids. The shuffle seed is not recorded. Each rank tokenizes the full dataset. The absolute data path leaks into the checkpoint config. There is no validation split or threshold fitting.
- Dataset: 1,585 train and 398 test messages that carry intents, plus 88 no-intent messages that are currently dropped. Laya trains one item per labelled question: 76,080 items for all tags, 16,417 with `--neg 8`.
- Prior results (100-message subset, CPU sandbox): majority top-1 0.26. Zero-shot Laya micro-F1 0.186. A head-only smoke fine-tune reached micro-F1 0.194, which proves the pipeline only.
- Each message costs 48 question forward passes in Laya, which matters for latency compared with a single-pass BERTurk classifier.

## Constraints

- **Hardware**: Apple M4 Mac with 16 GB unified memory, PyTorch MPS, no CUDA. Fall back to `--freeze-encoder` only if a full fine-tune OOMs, and label that run as weaker.
- **Data**: no real customer data, and no bank name anywhere in the repo.
- **Hosting**: everything runs self-hosted and locally. Hub downloads of public model weights are fine. Any exception needs the user's explicit confirmation. The only one under consideration is the Kaggle GPU training contingency (see Key Decisions, ⚠️ Revisit).
- **Reproducibility**: pinned dependencies, recorded seeds and configs, thresholds never tuned on test.
- **Laya internals**: the private API usage must be confined to one adapter module.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| BERTurk (`dbmdz/bert-base-turkish-cased`) is the reference baseline | A well-known Turkish encoder with a standard multi-label head is the simplest credible alternative | — Pending |
| Keep no-intent messages as all-negative rows and report their false-positive rate separately | The closest proxy for out-of-scope traffic. Dropping them gives optimistic scores. | — Pending |
| Train on an M4 MPS with a 1-worker Ray Train, keeping the CUDA path intact | The only available hardware. Avoids DDP work while keeping the code portable. | — Pending |
| 3 seeds per fine-tune and paired bootstrap CIs | GO criteria (a) and (b) need stability and significance | — Pending |
| Calibration/abstention and few-shot new intent are the "Laya edge" tests | GO criterion (d). The data-efficiency curve was deferred. | — Pending |
| Latency is reported only, budget TBD | Target serving hardware is unknown | ⚠️ Revisit |
| Use `data_neg8` first and `data_full` only if time allows | 4.6× cheaper on laptop hardware | — Pending |
| Local test script instead of hosted CI | Self-hosted constraint, single developer | — Pending |
| Val = folds 14–15 for selecting epochs, thresholds and temperature. Final models are retrained on folds 0–15 with those settings fixed, and test is scored once. | Avoids tuning on test while still using all training data for the final models. Changes the Validated split, which was folds 0–15 train. | — Pending |
| Per-tag thresholds only where val has ≥10 positives, otherwise one global threshold | Per-tag thresholds on small val support are noise | — Pending |
| GO (a) read as non-inferiority plus an INCONCLUSIVE outcome (counts as NO-GO) | With ~413 test messages the CIs can't resolve a 3-point macro-F1 margin | — Pending |
| Bootstrap uses seed-averaged probabilities per message, with paired resampling over messages | One frozen procedure. Avoids picking among seeds. | — Pending |
| BERTurk is trained with the HF Trainer, not the Ray harness | A standard recipe makes the baseline credible. Fairness comes from the shared split, scorer, val-only postfit and metrics. | — Pending |
| HOTELS out-of-domain probe added to v1 | Only 15 no-intent test messages, too few to measure false positives | — Pending |
| Compute checkpoint after the first MPS epoch, waiting for user approval | Runtime on the laptop is unknown | — Pending |
| Few-shot uses label masking, k-shot as short continued training, max 3 draws | Fair to both models and keeps compute bounded | — Pending |
| Kaggle GPU training is a contingency only, used if the Phase 2 MPS feasibility spike shows a full Laya fine-tune does not fit in 16 GB. It would use the preserved CUDA `--gpu` path. | Kaggle is a hosted service, so this conflicts with the self-hosted constraint. It needs the user's explicit confirmation before any use. The exception is acceptable only because the work uses public Multi3NLU++ data and no bank data. | ⚠️ Revisit |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-10-13 after Phase 1*
