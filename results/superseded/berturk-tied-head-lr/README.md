# Superseded: BERTurk selection with the head LR tied to the encoder LR (not comparable)

This directory holds the first BERTurk selection run of the BERTurk baseline. It is **superseded**, kept for audit only,
and **not comparable** to the current BERTurk selection or to any result under `results/dumps/`, `results/postfit/` or
`results/reports/`.

## What it is

- Training run `berturk-selection_train-s0-c0c9505e`: `berturk_train.py` on `selection_train` (folds 0–13), seed 0,
  one LR of 2e-05 for both the encoder and the freshly initialized 48-way classifier head (head LR tied to the encoder
  LR), 10 epochs, batch 32, max_len 128, fp32 on MPS. Every epoch was a selection candidate.
- The 10 val dumps of its epoch exports, plus the postfit record and the BERTurk reference check (the
  published-reference check of `pilot.berturk.reference_check`) built from them.

## Its reference check result

The chosen candidate (epoch 10, T 0.8659, global θ 0.10) reached a val micro-F1 of 150/551 = 0.2722 at the val-selected
global θ after T. That is below the reference check's floor of 0.707, so the status is `investigate`, with no resolution
recorded. In the diagnostics run after that outcome the epoch-10 train loss sat at the label-prior BCE floor (0.1737
against 0.1719), and only 6 of 48 tags scored above F1 0. The model had collapsed to the label priors.

## The decision

The user's reply after that outcome, verbatim (2026-10-21): `fix: head LR 1e-4 (Recommended)`. Decision: **fix**.

## What replaced it

A new selection run in which the classifier head trains at 1e-4 (Laya's head LR, `pilot.postfit.HEAD_LR`), while the
encoder stays at 2e-05 and every other setting is unchanged. This recipe change was made after a val outcome had been
seen. It is a disclosed protocol change, recorded in code as `pilot.berturk.SUPERSEDED`. That head-LR run
(`berturk-selection_train-s0-8c033154`) was itself superseded after its own reference check; see
`results/superseded/berturk-10-epoch-grid/`.

## Why it is kept

The files moved here byte for byte (git renames from `results/dumps/` and `results/postfit/`; nothing was rewritten or
regenerated). The reference check's `postfit.path` still names the original location, `results/postfit/berturk.json`.
Its `postfit.sha256` binds the record in this directory, and that record binds every dump's `dump_sha256`. The tests
in `tests/test_berturk_results.py` check that this chain still verifies and that the record rebuilds byte for byte
from these dumps.

This run can never drive a final seed: `pilot.berturk.check_reference_for_final` refuses an `investigate` check that
has no resolution. No test dump of it exists.

| file | what it is |
|---|---|
| `dumps/<dump run id>/val.npz`, `val.json` | the 10 val logit dumps (one per epoch export, epochs 1–10), from commit 4856642 |
| `postfit/berturk.json` | the postfit record (`postfit/1`, system kind berturk, protocol: LR 2e-05, head LR tied, 10 epochs), from commit aa92148 |
| `postfit/berturk-reference.json` | the BERTurk reference check (`berturk-reference/1`, status investigate, 150/551), from commit aa92148 |
