# MPS feasibility spike (Phase 2, LAYA-09)

- Split: `splits/selection_train.jsonl` sha256 `a3b5fad486b3ce472178bdbaea4f0016916a4164fd861e0b7a43a8b259623331` (1450 messages)
- Model: `convaiinnovations/laya/multilingual@7b928d828b7b0e022f929d9bd2e44165aa270148`
- Versions: laya 0.4.6, psutil 7.2.2, python 3.12.7, ray 2.59.0, torch 2.14.1, transformers 5.19.0
- macOS 26.6.2, arm64, 16.0 GiB memory
- Protocol: grid 4x128,4x256,8x128,8x256, 5 warm-up + 20 measured optimizer steps per regime, effective batch 32, seed 0, 4 CPUs
- Run policy: min_available_gib 7.0, swap_rerun True, tie_tolerance 0.05

| micro-batch | max_len | grad-accum | rung | fallback | purpose | status | padded median s | padded p90 s | real median s | real p90 s | peak GiB | swap delta MiB | swapouts delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 8 | none | 0 | initial | ok | 3.127 | 3.154 | 2.038 | 2.094 | 8.80 | 0.0 | 0 |
| 4 | 256 | 8 | none | 0 | initial | ok | 5.760 | 5.785 | 2.034 | 2.096 | 8.84 | 0.0 | 0 |
| 8 | 128 | 4 | none | 0 | initial | ok | 3.043 | 3.104 | 1.849 | 1.883 | 8.76 | 0.0 | 0 |
| 8 | 256 | 4 | none | 0 | initial | ok | 5.603 | 5.635 | 1.842 | 1.879 | 8.68 | 0.0 | 0 |

## Preflight

- Available memory: 7.77 GiB (threshold 7.0 GiB): passed
- System swap used while idle (5 s apart): 807.0 -> 807.0 MiB; grew: no
- Swapouts counter: 3168236 -> 3168236; grew: no

## Token lengths (D-09b)

Laya's own tokenizer, untruncated (message, question) sequences over all 1450 messages and 69600 items:
- 0.00% of messages over 128 tokens; 0 items over 128
- 0.00% of messages over 256 tokens; 0 items over 256
- sequence length: min 32, median 47, p90 57, p99 67, max 89
- state length: min 1, median 11, p90 22, p99 30, max 45
- projection basis: 14854 neg8 items (3254 positives + up to 8 negatives for each of 1450 messages, seed 0)

## Fallback-op inventory

strict PYTORCH_ENABLE_MPS_FALLBACK=0 pass raised nowhere: no op falls back from MPS

## Ray vs in-process (D-09a)

- Basis: the verdict's chosen config, 8x128, rung none, PYTORCH_ENABLE_MPS_FALLBACK=0
- Ray (1-worker TorchTrainer): padded median 3.043 s, real median 1.849 s, 4 torch threads
- In-process, no Ray (status ok): padded median 2.987 s, real median 1.832 s, 4 torch threads
- Ray overhead: padded +1.9%, real +0.9% (positive = Ray slower); never changes the verdict

## Verdict

| micro-batch | max_len | rung | memory ok | projected epoch min | time ok | device ok | FITS |
|---|---|---|---|---|---|---|---|
| 4 | 128 | none | yes (completed 20 measured padded steps without OOM or swap growth) | 15.8 | yes | yes | yes |
| 4 | 256 | none | yes (completed 20 measured padded steps without OOM or swap growth) | 15.8 | yes | yes | yes |
| 8 | 128 | none | yes (completed 20 measured padded steps without OOM or swap growth) | 14.3 | yes | yes | yes |
| 8 | 256 | none | yes (completed 20 measured padded steps without OOM or swap growth) | 14.3 | yes | yes | yes |

- Tie-break: FITS configs within 5% of the fastest projected epoch resolve to the smaller max_len, then the larger micro-batch
- Truncation rule (D-03): applied: 0.0% of messages over 128 tokens (<= 1%, max_len 128 is allowed)
- D-01: memory: every measured padded step completes on MPS with no OOM and no system swap growth
- D-02: time: real-batch median optimizer step x optimizer steps per neg8 epoch <= 7200 s
- D-03: choice: fastest projected epoch among FITS configs with max_len >= 128; configs within the tie tolerance resolve to the smaller max_len, then the larger micro-batch; > 1% of messages over 128 tokens prefers a passing max_len 256
- D-04: rung: only a full-encoder or embeddings-frozen run on MPS counts; a frozen encoder never FITS
- D-08: preference: rung none configs win over embeddings_frozen ones (embedding freeze is a fallback rung only)

**Verdict: FITS** — Phase 4 starts from micro-batch 8, max_len 128, grad-accum 4, rung none, PYTORCH_ENABLE_MPS_FALLBACK=0 (projected neg8 epoch 14.3 min)
