# Compute checkpoint (Phase 4, LAYA-07)

**Swap grew during measured epoch(s) 2** (largest 423 MiB): the run was under memory pressure (the spike's D-01 gate); confirm the rate before approval.

- Generated 2026-10-14T16:55:33Z
- Split: `splits/selection_train.jsonl` sha256 `a3b5fad486b3ce472178bdbaea4f0016916a4164fd861e0b7a43a8b259623331` (1450 messages)
- Manifest: `data/splits.json` sha256 `7376c5199685318fe459d603d5d257759303a48fc467ccdd29406e84de7e4103` (committed: True)
- Model: `convaiinnovations/laya/multilingual@7b928d828b7b0e022f929d9bd2e44165aa270148`
- Versions: laya 0.4.6, psutil 7.2.2, python 3.12.7, ray 2.59.0, torch 2.14.1, transformers 5.19.0
- macOS 26.6.2, arm64, 16.0 GiB memory
- Protocol: device mps, 2 epoch(s), seed 0, 4 CPUs, neg 8; spike start micro-batch 8, max_len 128, grad-accum 4, rung none, PYTORCH_ENABLE_MPS_FALLBACK=0; protocol complete: True
- Preflight: 8.75 GiB available (threshold 7.0), swap grew while idle: False

## Attempts

Every probe attempt runs in a fresh process; an MPS out-of-memory attempt is followed by the next rung of the LAYA-03 ladder. Chain status: **ok**.

| # | rung | status | run id | detail |
|---|---|---|---|---|
| 0 | none | ok | laya-selection_train-neg8-s0-db11040d | - |

## Interruption and resume

- Mode kill after epoch 1, 60 s into the next epoch; missed: False; leftover processes after the kill: 0; available memory before the rerun: 10.75 GiB
- Segments 2; resume load 3.6 s; peak memory while resuming 5.33 GiB
- Resume demonstrated: the rerun of the same command continued from the epoch-1 checkpoint.

## Measured run

- Device mps (parameters on mps), rung none, neg 8, micro-batch 8, max_len 128, grad-accum 4, gradient checkpointing on, AMP off
- Worker env: `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=4`, `PYTORCH_ENABLE_MPS_FALLBACK=0`, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`, `PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0`, `TOKENIZERS_PARALLELISM=false`
- Load 3.9 s, calibration 12.6 s, final save 0.4 s
- Projection rate: 1.8355 s per optimizer step, the largest train_s / optimizer_steps over 2 measured epoch(s) (epoch 1: 1.8355 s; RESEARCH Pitfall 6)

| epoch | train min | s/step | optimizer steps | items | peak GiB (driver + RSS) | driver GiB | RSS GiB | swap delta MiB | save s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.8 | 1.8355 | 452 | 14439 | 8.63 | 7.80 | 1.65 | 0.0 | 5.4 |
| 2 | 13.8 | 1.8274 | 452 | 14439 | 9.09 | 7.94 | 1.90 | 422.8 | 6.0 |

## BERTurk timing probe

- plain PyTorch loop, fp32, AdamW, batch 32, max_len 128; Phase 6 trains with the HF Trainer
- Device mps (parameters on mps), torch.float32, 4 torch threads, 110654256 parameters
- 0.3955 s per optimizer step (median of 20 measured steps after 3 warm-up; p90 0.4246 s); 0.0019 s per message for inference (64 messages, one batch)
- Peak memory 5.01 GiB (driver 4.19 GiB, RSS 0.82 GiB)

## Scoring probe

- --n 32 val subset of the --base weights (the pinned zero-shot model in the real run; same architecture as a fine-tuned checkpoint), in-process, fp32
- Device mps, torch.float32, autocast off: 0.3206 s per message over 32 messages (inference time, the sidecar's inference_seconds / n)

## Projection

Basis is the weakest input of a line: estimate (a labelled estimate, not measured), provisional (few-shot, protocol frozen only in Phase 8) or measured. Every matrix value is provisional; see Assumptions to confirm.

| line | phase | what | runs | epochs | units / epoch | unit | s / unit | basis | hours |
|---|---|---|---|---|---|---|---|---|---|
| laya_selection | 5 | Laya selection runs (LR grid x val-selected epoch ceiling) | 3 | 4 | 465 | optimizer step | 1.8355 | measured | 2.88 |
| laya_val_scoring | 5 | Laya val scoring of every selection epoch export | 12 | - | 208 | message | 0.3206 | measured | 0.22 |
| laya_final | 5 | Laya final seeds on final_train (projected at the epoch ceiling) | 3 | 4 | 532 | optimizer step | 1.8355 | measured | 3.29 |
| berturk_selection | 6 | BERTurk selection runs (LR grid) | 3 | 10 | 46 | optimizer step | 0.3955 | measured | 0.15 |
| berturk_final | 6 | BERTurk final seeds on final_train | 3 | 10 | 52 | optimizer step | 0.3955 | measured | 0.17 |
| fewshot_base | 9 | Few-shot base runs, held-out labels masked (both models) | 6 | - | - | optimizer step | - | provisional | 3.03 |
| fewshot_incremental | 9 | Few-shot incremental runs on k-shot support sets plus replay (both models) | 12 | 10 | - | optimizer step | - | provisional | 4.41 |
| fewshot_scoring | 9 | Few-shot scoring of every incremental model on test (both models) | 12 | - | 413 | message | - | provisional | 0.22 |
| final_scoring | 8 | Final scoring: every seed of both models on val, test and HOTELS | 6 | - | 1630 | message | - | measured | 0.44 |

**Total: 14.81 h** (53326 s, the sum of the lines above)

Per-model parts:

| line | model | k | messages | runs | epochs | units / epoch | unit | s / unit | basis | hours |
|---|---|---|---|---|---|---|---|---|---|---|
| fewshot_base | laya | - | - | 3 | 4 | 465 | optimizer step | 1.8355 | measured | 2.88 |
| fewshot_base | berturk | - | - | 3 | 10 | 46 | optimizer step | 0.3955 | measured | 0.15 |
| fewshot_incremental | laya | 5 | 350 | 3 | 10 | 113 | optimizer step | 1.8355 | measured | 1.79 |
| fewshot_incremental | laya | 10 | 500 | 3 | 10 | 161 | optimizer step | 1.8355 | measured | 2.53 |
| fewshot_incremental | berturk | 5 | 350 | 3 | 10 | 11 | optimizer step | 0.3955 | measured | 0.04 |
| fewshot_incremental | berturk | 10 | 500 | 3 | 10 | 16 | optimizer step | 0.3955 | measured | 0.05 |
| fewshot_scoring | laya | - | - | 6 | - | 413 | message | 0.3206 | measured | 0.22 |
| fewshot_scoring | berturk | - | - | 6 | - | 413 | message | 0.0019 | measured | 0.00 |
| final_scoring | laya | - | - | 3 | - | 1630 | message | 0.3206 | measured | 0.44 |
| final_scoring | berturk | - | - | 3 | - | 1630 | message | 0.0019 | measured | 0.00 |

Formulas and sources:

- `laya_selection` (measured): lr_grid (3) x [max_epochs (4) x (465 optimizer steps (14854 selection_train neg8 items, micro-batch 8, grad-accum 4) x 1.8355 s + 6.0 s save) + 3.9 s load + 12.6 s calibration + 0.4 s final save] = 10365 s. Source: probe run on mps: s/step = the largest train_s / optimizer_steps over 2 measured epoch(s); save, load, calibration and final-save seconds from the same record
- `laya_val_scoring` (measured): lr_grid (3) x max_epochs (4) epoch exports x 208 val messages x 0.3206 s = 800 s. Source: scoring probe on mps: --n 32 val subset of the --base weights (the pinned zero-shot model in the real run; same architecture as a fine-tuned checkpoint), in-process, fp32, 32 messages
- `laya_final` (measured): final_seeds (3) x [max_epochs (4) x (532 optimizer steps (17001 final_train neg8 items, micro-batch 8, grad-accum 4) x 1.8355 s + 6.0 s save) + 3.9 s load + 12.6 s calibration + 0.4 s final save] = 11840 s. Source: probe run on mps: s/step = the largest train_s / optimizer_steps over 2 measured epoch(s); save, load, calibration and final-save seconds from the same record
- `berturk_selection` (measured): lr_grid (3) x berturk_epochs (10) x 46 optimizer steps (ceil(1450 selection_train messages / batch 32)) x 0.3955 s = 546 s. Source: BERTurk timing probe on mps: plain PyTorch loop, fp32, AdamW, batch 32, max_len 128; Phase 6 trains with the HF Trainer
- `berturk_final` (measured): final_seeds (3) x berturk_epochs (10) x 52 optimizer steps (ceil(1658 final_train messages / batch 32)) x 0.3955 s = 617 s. Source: BERTurk timing probe on mps: plain PyTorch loop, fp32, AdamW, batch 32, max_len 128; Phase 6 trains with the HF Trainer
- `fewshot_base` (provisional): per model fewshot_base_runs (3) on selection_train: Laya at max_epochs (4), 465 optimizer steps per epoch; BERTurk at berturk_epochs (10), 46 optimizer steps per epoch = 10911 s. Source: Laya: probe run on mps: s/step = the largest train_s / optimizer_steps over 2 measured epoch(s); save, load, calibration and final-save seconds from the same record; BERTurk: BERTurk timing probe on mps: plain PyTorch loop, fp32, AdamW, batch 32, max_len 128; Phase 6 trains with the HF Trainer
- `fewshot_incremental` (provisional): per model and k: fewshot_draws (3) runs x fewshot_epochs (10) on fewshot_tags (6) x k x (1 + fewshot_neg_per_pos 4) + fewshot_replay (200) messages (k 5: 350, k 10: 500); Laya items = ceil(messages x 10.244 selection_train neg8 items per message) = 15870 s. Source: Laya: probe run on mps: s/step = the largest train_s / optimizer_steps over 2 measured epoch(s); save, load, calibration and final-save seconds from the same record; BERTurk: BERTurk timing probe on mps: plain PyTorch loop, fp32, AdamW, batch 32, max_len 128; Phase 6 trains with the HF Trainer
- `fewshot_scoring` (provisional): per model len(fewshot_k) (2) x fewshot_draws (3) models x 413 test messages x s/msg (Laya 0.3206, BERTurk 0.0019) = 799 s. Source: Laya: scoring probe on mps: --n 32 val subset of the --base weights (the pinned zero-shot model in the real run; same architecture as a fine-tuned checkpoint), in-process, fp32, 32 messages; BERTurk: BERTurk timing probe on mps: plain PyTorch loop, fp32, AdamW, batch 32, max_len 128; Phase 6 trains with the HF Trainer
- `final_scoring` (measured): per model eval_seeds (3) x (208 val + 413 test + 1009 hotels_probe = 1630 messages) x s/msg (Laya 0.3206, BERTurk 0.0019) = 1577 s. Source: Laya: scoring probe on mps: --n 32 val subset of the --base weights (the pinned zero-shot model in the real run; same architecture as a fine-tuned checkpoint), in-process, fp32, 32 messages; BERTurk: BERTurk timing probe on mps: plain PyTorch loop, fp32, AdamW, batch 32, max_len 128; Phase 6 trains with the HF Trainer

## What-if

Total projected hours when the matrix is trimmed (everything else unchanged):

| option | lr_grid | max_epochs | few-shot base runs | total h |
|---|---|---|---|---|
| lr_grid 1 x max_epochs 2 | 1 | 2 | 3 | 9.06 |
| lr_grid 1 x max_epochs 3 | 1 | 3 | 3 | 10.85 |
| lr_grid 1 x max_epochs 4 | 1 | 4 | 3 | 12.64 |
| lr_grid 2 x max_epochs 2 | 2 | 2 | 3 | 9.63 |
| lr_grid 2 x max_epochs 3 | 2 | 3 | 3 | 11.68 |
| lr_grid 2 x max_epochs 4 | 2 | 4 | 3 | 13.73 |
| lr_grid 3 x max_epochs 2 | 3 | 2 | 3 | 10.20 |
| lr_grid 3 x max_epochs 3 | 3 | 3 | 3 | 12.51 |
| lr_grid 3 x max_epochs 4 | 3 | 4 | 3 | 14.81 |
| few-shot compute fallback (1 base run per model) | 3 | 4 | 1 | 12.79 |

## Disk

- Base model (fp16 safetensors): 0.60 GiB; train state: 3.60 GiB (measured)
- One run's peak: 13.79 GiB (three train states during checkpoint rotation, its per-epoch exports, its final checkpoint); ray_train.py refuses below 14.79 GiB free
- Selection epoch exports kept until Phase 5 scores them on val: 7.20 GiB; Laya final checkpoints kept: 3.60 GiB

## Assumptions to confirm

- Matrix value lr_grid = 3 is provisional (RESEARCH A2): FEATURES TS-12 example shared LR grid {2e-5, 3e-5, 5e-5}, used by Laya (Phase 5) and BERTurk (Phase 6)
- Matrix value max_epochs = 4 is provisional (RESEARCH A2): ray_train.py default --epochs 4, the ceiling of the val-selected epoch (final seeds at it)
- Matrix value final_seeds = 3 is provisional (RESEARCH A2): LAYA-08 / BASE-01: three final seeds per model on final_train
- Matrix value neg = 8 is provisional (RESEARCH A2): the spike's data_neg8 item mix (ray_train.py --neg 8): every positive plus 8 negatives per message
- Matrix value effective_batch = 32 is provisional (RESEARCH A2): the spike's micro-batch 8 x grad-accum 4; the projection uses the measured run's own values
- Matrix value berturk_epochs = 10 is provisional (RESEARCH A2): STACK: BERTurk 5-10 epochs; the upper bound is projected
- Matrix value berturk_batch = 32 is provisional (RESEARCH A2): BERTurk batch 32 messages (one sequence per message, all 48 labels)
- Matrix value fewshot_tags = 6 is provisional (RESEARCH A2): FEATURES TS-14: 6 held-out tags (frozen only in the Phase 8 protocol.json)
- Matrix value fewshot_k = 5, 10 is provisional (RESEARCH A2): FEATURES TS-14 / ROADMAP Phase 8: k = 5 and 10 support messages per new tag
- Matrix value fewshot_draws = 3 is provisional (RESEARCH A2): ROADMAP Phase 8: at most 3 support-set draws
- Matrix value fewshot_neg_per_pos = 4 is provisional (RESEARCH A2): FEATURES TS-14 default: 4 negative support messages per positive
- Matrix value fewshot_replay = 200 is provisional (RESEARCH A2): FEATURES TS-14 default: 200 replay messages of the old tags
- Matrix value fewshot_epochs = 10 is provisional (RESEARCH A2): FEATURES TS-14 default: about 10 continued-training epochs
- Matrix value fewshot_base_runs = 3 is provisional (RESEARCH A2): base runs per model with the held-out labels masked (one per final seed); the pre-declared compute fallback is 1
- Matrix value eval_seeds = 3 is provisional (RESEARCH A2): ROADMAP Phase 8: 3 seeds x 2 models scored once on val, test and HOTELS
- Negatives are drawn per seed (RESEARCH A4): ray_train.py --neg-seed defaults to --seed, so the seed variance includes the neg8 sampling; pin --neg-seed 0 to exclude it
- The git sha is recorded per run segment and kept out of the run_id (RESEARCH A3): a resume across a code change is flagged code_changed_on_resume, not refused
- BERTurk is timed with a plain PyTorch loop (fp32, AdamW, batch 32, max_len 128), not the HF Trainer Phase 6 trains with; the Trainer's own overhead (data collation, logging, per-epoch evaluation hooks) is assumed small
- Scoring time is measured on the pinned zero-shot weights (a --n 32 val subset); a fine-tuned checkpoint has the same architecture and forward cost
- The Laya lines leave out each run's fixed overhead (process and Ray start, the driver's split and weight checks, item building, the --out copy): this probe predates its measurement (the probe child's wall clock minus the record's terms), so they are slightly optimistic, by that overhead once per Laya training run
- The per-epoch save seconds stop before Ray persists the epoch checkpoint (ray.train.report copies the 3.60 GiB train state into storage and drops the previous one): this probe predates timing it, so the Laya lines charge it at most once per run (inside the run overhead, when the probe measured one), not once per epoch; slightly optimistic, by that copy for each of the matrix's 96 Laya training epochs
- The Laya scoring lines leave out each score.py run's own overhead (process start, the weights' sha256 check, laya.load): this probe predates its measurement (the scoring probe's wall clock minus its inference seconds), so they are slightly optimistic, by that overhead once per scoring run (27 Laya score.py runs in the matrix); the BERTurk scoring lines have no such term either
- Swap grew during measured epoch(s) 2 (largest 423 MiB): the run was under memory pressure (the spike's D-01 gate), so the measured rate may be slower than on a quiet machine; confirm it before approval
- Scoring runs in fp32 on MPS (la.force_fp32 turns laya's fp16 autocast off); the scoring rate assumes fp32
- Per-epoch laya exports stay on (--export-epochs) so Phase 5 picks the epoch on val without retraining; Phase 5 prunes them after val scoring (see Disk)
- The measurement probe trains 2 epoch(s), interrupted after epoch 1 and resumed by rerunning the same command (see Interruption and resume), so the resumed process's warm-up is in the rate; the projection uses the slowest measured epoch
- Every split is counted over all its messages, without the 40-message calibration holdout, because Phase 5 calibrates on val (LAYA-05); slightly conservative
- BERTurk's per-epoch val evaluation for its epoch choice is not a separate line (lr_grid x berturk_epochs val passes at 0.002 s per message, about 12 s)
