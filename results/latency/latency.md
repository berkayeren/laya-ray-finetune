### Latency per message (Phase 7, EVAL-06): reported, not gated

Latency is reported, not gated: it is not a decision criterion (a latency budget is DEC-01, v2).

Rules:

1. batch 1, in-process, no Ray; one process per (system, device), every final seed of the final manifest
2. a message is one val row's text, passed unchanged to the checkpoint's tokenizer; its latency is the wall-clock seconds from that text to 48 probabilities on the host
3. warm-up (full_pass_per_seed): after each seed's load, one untimed forward of every message the seed then times, in split-file order, then one device sync; only then is every message timed; so every seed follows the same sequence and every input it times has already run once on its model in this process; the seeds run in manifest order, so a residual seed-order effect is not separated from a seed effect; the report records the policy and the count
4. the device is synchronised right before t0 and right after the host copy (torch.mps.synchronize on mps, torch.cuda.synchronize on cuda, nothing on cpu), under torch.no_grad
5. fp32 with autocast off, checked after the load and after the timed loop
6. val messages only; only times are kept (no outputs, text or uids)
7. p50 and p95 are nearest-rank with integer ranks s[(q * n + 99) // 100 - 1], never interpolated

#### berturk on cpu

Timed path: BertForSequenceClassification fp32 -> tokenizer of one message (truncated at the record max_len) -> one forward -> logits to host -> sigmoid(logit / T).

- Device cpu, torch.float32, autocast off, torch threads 4, batch 1; warm-up full_pass_per_seed: one untimed pass over all 208 message(s) per seed before its timed loop
- Recorded rules: this report's `rules` field differs from the rules above at rule(s) 3; it keeps the text in force when the report was timed, and the rules above are the current wording
- Versions: numpy 2.5.4, torch 2.14.1, transformers 5.19.0; PYTORCH_ENABLE_MPS_FALLBACK unset
- Code: 3beea81ca966
- Messages: 208 val messages of splits/val.jsonl (sha256 95f5cfe5ae59); final manifest results/postfit/berturk-final.json (sha256 0a611f198fcf, committed: True)
- Preflight: 7.98 GiB available (threshold 7.0), swap grew while idle: False
- Swap in use: 0 MiB before, 0 MiB after the timed loops

| Seed | Checkpoint | n | p50 ms | p95 ms | mean ms | min ms | max ms |
|---|---|---|---|---|---|---|---|
| 0 | ckpt-berturk-final-s0 | 208 | 24.6 | 26.8 | 24.8 | 19.1 | 28.7 |
| 1 | ckpt-berturk-final-s1 | 208 | 24.5 | 26.9 | 24.9 | 20.8 | 30.3 |
| 2 | ckpt-berturk-final-s2 | 208 | 24.5 | 26.8 | 24.9 | 19.2 | 28.1 |
| pooled | 3 seed(s) | 624 | 24.5 | 26.8 | 24.9 | 19.1 | 30.3 |

#### berturk on mps

Timed path: BertForSequenceClassification fp32 -> tokenizer of one message (truncated at the record max_len) -> one forward -> logits to host -> sigmoid(logit / T).

- Device mps, torch.float32, autocast off, torch threads 4, batch 1; warm-up full_pass_per_seed: one untimed pass over all 208 message(s) per seed before its timed loop
- Recorded rules: this report's `rules` field differs from the rules above at rule(s) 3; it keeps the text in force when the report was timed, and the rules above are the current wording
- Versions: numpy 2.5.4, torch 2.14.1, transformers 5.19.0; PYTORCH_ENABLE_MPS_FALLBACK unset
- Code: e8e326f4e1a3
- Messages: 208 val messages of splits/val.jsonl (sha256 95f5cfe5ae59); final manifest results/postfit/berturk-final.json (sha256 0a611f198fcf, committed: True)
- Preflight: 7.68 GiB available (threshold 7.0), swap grew while idle: False
- Swap in use: 0 MiB before, 0 MiB after the timed loops

| Seed | Checkpoint | n | p50 ms | p95 ms | mean ms | min ms | max ms |
|---|---|---|---|---|---|---|---|
| 0 | ckpt-berturk-final-s0 | 208 | 9.4 | 11.8 | 9.6 | 7.8 | 13.0 |
| 1 | ckpt-berturk-final-s1 | 208 | 9.4 | 11.7 | 9.6 | 7.9 | 13.1 |
| 2 | ckpt-berturk-final-s2 | 208 | 9.5 | 11.7 | 9.6 | 7.9 | 12.7 |
| pooled | 3 seed(s) | 624 | 9.4 | 11.7 | 9.6 | 7.8 | 13.1 |

#### laya on cpu

Timed path: laya.load (digests) -> la.force_fp32 -> la.noul_logits of one message (48 noul questions, one forward) -> sigmoid(margin / T) on the host.

- Device cpu, torch.float32, autocast off, torch threads 4, batch 1; warm-up full_pass_per_seed: one untimed pass over all 208 message(s) per seed before its timed loop
- Recorded rules: this report's `rules` field differs from the rules above at rule(s) 3; it keeps the text in force when the report was timed, and the rules above are the current wording
- Versions: laya 0.4.6, numpy 2.5.4, torch 2.14.1, transformers 5.19.0; PYTORCH_ENABLE_MPS_FALLBACK unset
- Code: bbf1a8543ed9, uncommitted changes
- Messages: 208 val messages of splits/val.jsonl (sha256 95f5cfe5ae59); final manifest results/postfit/laya-final.json (sha256 3d0b1a91e00e, committed: True)
- Preflight: 8.45 GiB available (threshold 7.0), swap grew while idle: False
- Swap in use: 0 MiB before, 0 MiB after the timed loops

| Seed | Checkpoint | n | p50 ms | p95 ms | mean ms | min ms | max ms |
|---|---|---|---|---|---|---|---|
| 0 | ckpt-final-s0 | 208 | 684.2 | 831.8 | 696.0 | 567.3 | 1014.5 |
| 1 | ckpt-final-s1 | 208 | 681.2 | 812.7 | 689.7 | 571.5 | 1016.0 |
| 2 | ckpt-final-s2 | 208 | 682.2 | 813.3 | 690.4 | 572.1 | 1020.6 |
| pooled | 3 seed(s) | 624 | 682.3 | 814.5 | 692.0 | 567.3 | 1020.6 |

#### laya on mps

Timed path: laya.load (digests) -> la.force_fp32 -> la.noul_logits of one message (48 noul questions, one forward) -> sigmoid(margin / T) on the host.

- Device mps, torch.float32, autocast off, torch threads 4, batch 1; warm-up full_pass_per_seed: one untimed pass over all 208 message(s) per seed before its timed loop
- Recorded rules: this report's `rules` field differs from the rules above at rule(s) 3; it keeps the text in force when the report was timed, and the rules above are the current wording
- Versions: laya 0.4.6, numpy 2.5.4, torch 2.14.1, transformers 5.19.0; PYTORCH_ENABLE_MPS_FALLBACK unset
- Code: bbf1a8543ed9
- Messages: 208 val messages of splits/val.jsonl (sha256 95f5cfe5ae59); final manifest results/postfit/laya-final.json (sha256 3d0b1a91e00e, committed: True)
- Preflight: 7.93 GiB available (threshold 7.0), swap grew while idle: False
- Swap in use: 0 MiB before, 0 MiB after the timed loops

| Seed | Checkpoint | n | p50 ms | p95 ms | mean ms | min ms | max ms |
|---|---|---|---|---|---|---|---|
| 0 | ckpt-final-s0 | 208 | 332.1 | 406.3 | 339.9 | 277.9 | 517.0 |
| 1 | ckpt-final-s1 | 208 | 330.6 | 406.5 | 339.4 | 278.0 | 514.6 |
| 2 | ckpt-final-s2 | 208 | 331.9 | 405.2 | 339.5 | 277.9 | 516.8 |
| pooled | 3 seed(s) | 624 | 331.9 | 406.3 | 339.6 | 277.9 | 517.0 |

Laya's own fp16 autocast default on MPS (laya_default_amp) is not what is timed: every run is fp32 with autocast off, the path that scored the dumps.
