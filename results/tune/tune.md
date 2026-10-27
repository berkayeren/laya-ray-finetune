# Exploratory hyperparameter search (Ray Tune)

This search is exploratory and not a decision input: it does not change the frozen verdict, protocol.json or any committed record. Every trial trains on tuning-train, selection_train folds 0-11, and is scored on tuning-dev, selection_train folds 12-13, never val, test or HOTELS. It is a seeded random search (Ray Tune's BasicVariantGenerator, one trial at a time; each system's seed and trial count are given below) with ASHAScheduler early stopping (time attribute epoch, grace period 1, reduction factor 3, max_t the longest epoch schedule). Objective: dev_micro_f1: the micro-F1 over the tuning-dev rows (selection_train folds 12-13) at the global θ pilot.metrics.select_theta picks on those same rows, with probabilities at T = 1 from the in-memory fp32 model scored in-process after every epoch; a trial's score is its best epoch.

## Laya

- Search: 20 trials, search seed 0, BasicVariantGenerator, rungs at epochs 1, 3, max_t 8; device mps, 4 CPUs per trial
- Data: tuning-train 1244 messages (folds 0-11), tuning-dev 206 messages (folds 12-13); 4 tuning-dev rows share their exact text with a tuning-train row; committed manifest: yes
- Base: multilingual (model sha256 9d628fd971b7)
- Versions: ray 2.59.0, torch 2.14.1, laya 0.4.6
- Driver starts: 1 (1: git 7480c7a9, clean; preflight 8.4 GiB available)
- Restarted trials (attempts above 1): 0 of 20
- ASHA-stopped trials: 13 of 20

Highest tuning-dev score: trial 14, epoch 6, dev micro-F1 0.8661 (TP 430, FP 95, FN 38); encoder_lr 3.0e-05, head_lr 9.8e-05, neg 16, epochs 6

| trial | encoder_lr | head_lr | neg | epochs | attempts | epochs run | best epoch | best dev micro-F1 | θ at best | ASHA stopped |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3.4e-05 | 7.7e-05 | 4 | 4 | 1 | 4 of 4 | 4 | 0.7197 | 0.95 | no |
| 1 | 5.3e-06 | 5.2e-04 | 8 | 8 | 1 | 8 of 8 | 3 | 0.7586 | 0.84 | no |
| 2 | 3.1e-05 | 3.9e-04 | 8 | 6 | 1 | 6 of 6 | 4 | 0.8173 | 0.92 | no |
| 3 | 8.2e-05 | 5.2e-04 | 16 | 4 | 1 | 1 of 4 | 1 | 0.0000 | 0.50 | yes |
| 4 | 6.5e-05 | 3.4e-05 | 16 | 8 | 1 | 1 of 8 | 1 | 0.4624 | 0.64 | yes |
| 5 | 8.5e-06 | 6.2e-04 | 4 | 6 | 1 | 1 of 6 | 1 | 0.6286 | 0.95 | yes |
| 6 | 1.2e-05 | 1.3e-04 | 8 | 4 | 1 | 4 of 4 | 3 | 0.7782 | 0.95 | no |
| 7 | 7.3e-06 | 3.2e-04 | 8 | 6 | 1 | 6 of 6 | 6 | 0.7917 | 0.91 | no |
| 8 | 3.2e-05 | 1.2e-04 | 8 | 8 | 1 | 1 of 8 | 1 | 0.5879 | 0.95 | yes |
| 9 | 9.4e-05 | 3.3e-04 | 16 | 6 | 1 | 1 of 6 | 1 | 0.4595 | 0.33 | yes |
| 10 | 3.9e-05 | 1.2e-04 | 16 | 4 | 1 | 1 of 4 | 1 | 0.6929 | 0.92 | yes |
| 11 | 4.3e-05 | 1.9e-04 | 8 | 4 | 1 | 1 of 4 | 1 | 0.6833 | 0.94 | yes |
| 12 | 2.1e-05 | 6.8e-04 | 4 | 8 | 1 | 1 of 8 | 1 | 0.5299 | 0.95 | yes |
| 13 | 1.5e-05 | 2.2e-04 | 4 | 4 | 1 | 1 of 4 | 1 | 0.5767 | 0.95 | yes |
| 14 | 3.0e-05 | 9.8e-05 | 16 | 6 | 1 | 6 of 6 | 6 | 0.8661 | 0.13 | no |
| 15 | 7.2e-05 | 6.7e-05 | 16 | 6 | 1 | 1 of 6 | 1 | 0.5341 | 0.53 | yes |
| 16 | 6.4e-06 | 5.6e-04 | 8 | 8 | 1 | 3 of 8 | 2 | 0.7725 | 0.94 | yes |
| 17 | 1.0e-05 | 6.5e-04 | 4 | 4 | 1 | 1 of 4 | 1 | 0.5822 | 0.95 | yes |
| 18 | 1.4e-05 | 5.1e-05 | 16 | 6 | 1 | 6 of 6 | 6 | 0.8646 | 0.54 | no |
| 19 | 5.4e-05 | 6.7e-05 | 16 | 4 | 1 | 1 of 4 | 1 | 0.5940 | 0.94 | yes |

## BERTurk

- Search: 20 trials, search seed 0, BasicVariantGenerator, rungs at epochs 1, 3, 9, 27, max_t 30; device mps, 4 CPUs per trial
- Data: tuning-train 1244 messages (folds 0-11), tuning-dev 206 messages (folds 12-13); 4 tuning-dev rows share their exact text with a tuning-train row; committed manifest: yes
- Base: berturk (model sha256 18e29c6c61a0)
- Versions: ray 2.59.0, torch 2.14.1, transformers 5.19.0, accelerate 1.15.0
- Driver starts: 1 (1: git 0ae2617a, clean; preflight 8.2 GiB available)
- Restarted trials (attempts above 1): 0 of 20
- ASHA-stopped trials: 17 of 20

Highest tuning-dev score: trial 4, epoch 20, dev micro-F1 0.8852 (TP 405, FP 42, FN 63); encoder_lr 6.6e-05, head_lr 2.0e-04, batch 16, warmup_ratio 0.06, weight_decay 0.0, epochs 20

| trial | encoder_lr | head_lr | batch | warmup_ratio | weight_decay | epochs | attempts | epochs run | best epoch | best dev micro-F1 | θ at best | ASHA stopped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3.4e-05 | 7.7e-05 | 16 | 0.06 | 0.0 | 10 | 1 | 10 of 10 | 10 | 0.6856 | 0.23 | no |
| 1 | 5.7e-05 | 7.4e-04 | 32 | 0.1 | 0.01 | 30 | 1 | 1 of 30 | 1 | 0.2009 | 0.15 | yes |
| 2 | 2.5e-05 | 8.0e-04 | 16 | 0.1 | 0.01 | 10 | 1 | 10 of 10 | 10 | 0.8027 | 0.38 | no |
| 3 | 6.5e-05 | 3.4e-05 | 32 | 0.1 | 0.01 | 10 | 1 | 1 of 10 | 1 | 0.2049 | 0.29 | yes |
| 4 | 6.6e-05 | 2.0e-04 | 16 | 0.06 | 0.0 | 20 | 1 | 20 of 20 | 20 | 0.8852 | 0.36 | no |
| 5 | 5.4e-06 | 4.6e-05 | 16 | 0.1 | 0.01 | 20 | 1 | 1 of 20 | 1 | 0.1129 | 0.46 | yes |
| 6 | 3.2e-05 | 1.2e-04 | 16 | 0.1 | 0.01 | 30 | 1 | 1 of 30 | 1 | 0.1381 | 0.28 | yes |
| 7 | 3.9e-05 | 2.9e-04 | 32 | 0.1 | 0.01 | 20 | 1 | 1 of 20 | 1 | 0.1406 | 0.22 | yes |
| 8 | 7.5e-06 | 3.8e-04 | 32 | 0.1 | 0.0 | 10 | 1 | 1 of 10 | 1 | 0.2067 | 0.16 | yes |
| 9 | 2.1e-05 | 6.8e-04 | 16 | 0.1 | 0.01 | 20 | 1 | 3 of 20 | 3 | 0.4723 | 0.20 | yes |
| 10 | 2.8e-05 | 9.3e-05 | 32 | 0.1 | 0.01 | 20 | 1 | 1 of 20 | 1 | 0.1366 | 0.47 | yes |
| 11 | 1.6e-05 | 6.8e-04 | 16 | 0.06 | 0.01 | 20 | 1 | 3 of 20 | 3 | 0.4896 | 0.22 | yes |
| 12 | 6.4e-06 | 5.6e-04 | 16 | 0.1 | 0.0 | 10 | 1 | 3 of 10 | 3 | 0.3084 | 0.14 | yes |
| 13 | 6.9e-05 | 3.7e-05 | 32 | 0.06 | 0.01 | 10 | 1 | 1 of 10 | 1 | 0.2007 | 0.23 | yes |
| 14 | 1.9e-05 | 4.9e-04 | 32 | 0.06 | 0.01 | 10 | 1 | 1 of 10 | 1 | 0.2189 | 0.05 | yes |
| 15 | 1.7e-05 | 6.0e-05 | 32 | 0.06 | 0.01 | 20 | 1 | 1 of 20 | 1 | 0.1355 | 0.47 | yes |
| 16 | 1.2e-05 | 3.2e-04 | 32 | 0.06 | 0.01 | 30 | 1 | 1 of 30 | 1 | 0.1245 | 0.29 | yes |
| 17 | 1.5e-05 | 4.3e-05 | 32 | 0.1 | 0.01 | 30 | 1 | 1 of 30 | 1 | 0.1185 | 0.50 | yes |
| 18 | 1.9e-05 | 8.5e-04 | 16 | 0.06 | 0.0 | 20 | 1 | 9 of 20 | 9 | 0.8031 | 0.41 | yes |
| 19 | 3.2e-05 | 9.8e-04 | 32 | 0.1 | 0.0 | 20 | 1 | 1 of 20 | 1 | 0.2169 | 0.05 | yes |

## Notes

- This search is exploratory and not a decision input: protocol.json, the frozen verdict, every committed record and every existing run id are unchanged by it.
- No trial model is scored on val, test or HOTELS, and no trial weights are kept.
- Tuning-train is selection_train folds 0-11 and tuning-dev is selection_train folds 12-13, split in memory by the fold in each uid; no trial reads val (folds 14-15), test or HOTELS.
- θ is chosen on the same tuning-dev rows it scores (in-sample) at T = 1, so dev micro-F1 is optimistic and not comparable to the val or test numbers.
- Dev scoring uses the in-memory fp32 weights (Laya: the training-side noul route, equal to laya.load's; BERTurk: the classifier logits), not fp16 exports.
- ASHA compares trials at the same epoch across schedule lengths (Laya 4, 6 or 8 epochs; BERTurk 10, 20 or 30), which favours short schedules at the rungs.
- A trial with attempts above 1 was restarted after a kill and may have passed a rung it had been judged at, and Ray Tune snapshots the experiment at most every 10 s, so a trial that ended seconds before a kill can rerun.
- Duplicate message texts across the two partitions are kept (the duplicate audit is report-only); dev_rows_with_train_text counts the tuning-dev rows whose exact text is also a tuning-train text.
- peak_total_bytes and swap_delta_bytes cover the epoch's training steps only.
- The pipeline's single encoder LR 2e-05 stays fixed, not tuned; this search reopens it for exploration only.
