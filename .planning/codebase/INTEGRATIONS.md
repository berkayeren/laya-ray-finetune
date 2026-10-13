---
last_mapped_commit: 49868df263b5b89e8efe10909576f479c0330c44
last_mapped_at: 2026-10-13
---
# External Integrations

**Analysis Date:** 2026-10-13

## APIs & External Services

**Model hub:**
- Hugging Face Hub, repo `convaiinnovations/laya` - base checkpoints `multilingual`, `english`, `typed-decisions`
  - SDK/Client: `laya.load("convaiinnovations/laya", subfolder=..., device=...)` in `evaluate.py`; `--base` in `ray_train.py`/`laya-train`
  - Auth: none required (public)

**Dataset:**
- Multi3NLU++ (`uoe-nlp/multi3-nlu`, CC-BY 4.0) - Turkish banking folds `fold0..19.json`, downloaded manually and consumed by `prepare_data.py`. No programmatic API call.

## Data Storage

**Databases:**
- None

**File Storage:**
- Local filesystem: JSONL in `data_full/`, `data_neg8/`; checkpoints in `--out`; Ray run output in `./ray_results` (`--storage`, may be NFS/S3 for multi-node); eval reports in `results/*.json`

**Caching:**
- None (Hugging Face cache handled implicitly by the hub client)

## Authentication & Identity

**Auth Provider:**
- None

## Monitoring & Observability

**Error Tracking:**
- None

**Logs:**
- `print` to stdout; Ray's own logging and dashboard

## CI/CD & Deployment

**Hosting:**
- None (local or Ray cluster)

**CI Pipeline:**
- None

## Environment Configuration

**Required env vars:**
- None

**Secrets location:**
- Not applicable

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-10-13*
