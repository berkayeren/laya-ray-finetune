### Latency per message (Phase 7, EVAL-06): reported, not gated

Latency is reported, not gated: it is not a decision criterion (a latency budget is DEC-01, v2).

Rules:

- batch 1, in-process, no Ray; one process per (system, device), every final seed of the final manifest
- a message is one val row's text, passed unchanged to the checkpoint's tokenizer; its latency is the wall-clock seconds from that text to 48 probabilities on the host
- the first --warmup messages (default 10) of each seed run first and are excluded; the report records the count
- the device is synchronised right before t0 and right after the host copy (torch.mps.synchronize on mps, torch.cuda.synchronize on cuda, nothing on cpu), under torch.no_grad
- fp32 with autocast off, checked after the load and after the timed loop
- val messages only; only times are kept (no outputs, text or uids)
- p50 and p95 are nearest-rank with integer ranks s[(q * n + 99) // 100 - 1], never interpolated

#### berturk on cpu

Timed path: BertForSequenceClassification fp32 -> tokenizer of one message (truncated at the record max_len) -> one forward -> logits to host -> sigmoid(logit / T).

- Device cpu, torch.float32, autocast off, torch threads 4, warm-up 10 message(s) excluded, batch 1
- Versions: numpy 2.5.4, torch 2.14.1, transformers 5.19.0; PYTORCH_ENABLE_MPS_FALLBACK unset
- Code: ff65d2f1c675, uncommitted changes
- Messages: 208 val messages of splits/val.jsonl (sha256 95f5cfe5ae59); final manifest results/postfit/berturk-final.json (sha256 0a611f198fcf, committed: True)
- Preflight: 8.01 GiB available (threshold 7.0), swap grew while idle: False
- Swap in use: 0 MiB before, 0 MiB after the timed loops

| Seed | Checkpoint | n | p50 ms | p95 ms | mean ms | min ms | max ms |
|---|---|---|---|---|---|---|---|
| 0 | ckpt-berturk-final-s0 | 208 | 25.4 | 27.0 | 25.4 | 20.4 | 50.2 |
| 1 | ckpt-berturk-final-s1 | 208 | 25.9 | 27.5 | 25.9 | 21.1 | 29.1 |
| 2 | ckpt-berturk-final-s2 | 208 | 25.9 | 27.2 | 25.9 | 20.5 | 29.8 |
| pooled | 3 seed(s) | 624 | 25.8 | 27.2 | 25.7 | 20.4 | 50.2 |

#### berturk on mps

Timed path: BertForSequenceClassification fp32 -> tokenizer of one message (truncated at the record max_len) -> one forward -> logits to host -> sigmoid(logit / T).

- Device mps, torch.float32, autocast off, torch threads 4, warm-up 10 message(s) excluded, batch 1
- Versions: numpy 2.5.4, torch 2.14.1, transformers 5.19.0; PYTORCH_ENABLE_MPS_FALLBACK unset
- Code: a5d790e560b3
- Messages: 208 val messages of splits/val.jsonl (sha256 95f5cfe5ae59); final manifest results/postfit/berturk-final.json (sha256 0a611f198fcf, committed: True)
- Preflight: 7.58 GiB available (threshold 7.0), swap grew while idle: False
- Swap in use: 0 MiB before, 0 MiB after the timed loops

| Seed | Checkpoint | n | p50 ms | p95 ms | mean ms | min ms | max ms |
|---|---|---|---|---|---|---|---|
| 0 | ckpt-berturk-final-s0 | 208 | 9.7 | 14.2 | 13.5 | 7.9 | 177.0 |
| 1 | ckpt-berturk-final-s1 | 208 | 9.4 | 11.7 | 9.6 | 7.9 | 12.6 |
| 2 | ckpt-berturk-final-s2 | 208 | 9.5 | 11.7 | 9.6 | 8.0 | 12.9 |
| pooled | 3 seed(s) | 624 | 9.5 | 11.8 | 10.9 | 7.9 | 177.0 |

#### laya on cpu

Timed path: laya.load (digests) -> la.force_fp32 -> la.noul_logits of one message (48 noul questions, one forward) -> sigmoid(margin / T) on the host.

- Device cpu, torch.float32, autocast off, torch threads 4, warm-up 10 message(s) excluded, batch 1
- Versions: laya 0.4.6, numpy 2.5.4, torch 2.14.1, transformers 5.19.0; PYTORCH_ENABLE_MPS_FALLBACK unset
- Code: ff65d2f1c675, uncommitted changes
- Messages: 208 val messages of splits/val.jsonl (sha256 95f5cfe5ae59); final manifest results/postfit/laya-final.json (sha256 3d0b1a91e00e, committed: True)
- Preflight: 8.46 GiB available (threshold 7.0), swap grew while idle: False
- Swap in use: 0 MiB before, 0 MiB after the timed loops

| Seed | Checkpoint | n | p50 ms | p95 ms | mean ms | min ms | max ms |
|---|---|---|---|---|---|---|---|
| 0 | ckpt-final-s0 | 208 | 682.2 | 829.8 | 693.2 | 568.2 | 1021.2 |
| 1 | ckpt-final-s1 | 208 | 693.0 | 817.1 | 698.3 | 578.8 | 1033.2 |
| 2 | ckpt-final-s2 | 208 | 692.2 | 822.1 | 700.9 | 573.2 | 1041.1 |
| pooled | 3 seed(s) | 624 | 691.0 | 821.8 | 697.5 | 568.2 | 1041.1 |

#### laya on mps

Timed path: laya.load (digests) -> la.force_fp32 -> la.noul_logits of one message (48 noul questions, one forward) -> sigmoid(margin / T) on the host.

- Device mps, torch.float32, autocast off, torch threads 4, warm-up 10 message(s) excluded, batch 1
- Versions: laya 0.4.6, numpy 2.5.4, torch 2.14.1, transformers 5.19.0; PYTORCH_ENABLE_MPS_FALLBACK unset
- Code: ff65d2f1c675
- Messages: 208 val messages of splits/val.jsonl (sha256 95f5cfe5ae59); final manifest results/postfit/laya-final.json (sha256 3d0b1a91e00e, committed: True)
- Preflight: 7.85 GiB available (threshold 7.0), swap grew while idle: False
- Swap in use: 0 MiB before, 0 MiB after the timed loops

| Seed | Checkpoint | n | p50 ms | p95 ms | mean ms | min ms | max ms |
|---|---|---|---|---|---|---|---|
| 0 | ckpt-final-s0 | 208 | 320.9 | 390.1 | 325.8 | 264.5 | 492.6 |
| 1 | ckpt-final-s1 | 208 | 330.7 | 406.3 | 339.0 | 277.4 | 518.0 |
| 2 | ckpt-final-s2 | 208 | 331.5 | 406.9 | 339.6 | 280.7 | 518.0 |
| pooled | 3 seed(s) | 624 | 329.4 | 404.8 | 334.8 | 264.5 | 518.0 |

Laya's own fp16 autocast default on MPS (laya_default_amp) is not what is timed: every run is fp32 with autocast off, the path that scored the dumps.
