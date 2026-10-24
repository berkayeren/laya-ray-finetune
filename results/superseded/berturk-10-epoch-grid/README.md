# Superseded: BERTurk selection with a 10-epoch grid (not comparable)

This directory holds the second BERTurk selection run of the BERTurk baseline. It is **superseded**, kept for audit
only, and **not comparable** to the current BERTurk selection or to any result under `results/dumps/`,
`results/postfit/` or `results/reports/`.

## What it is

- Training run `berturk-selection_train-s0-8c033154`: `berturk_train.py` on `selection_train` (folds 0–13), seed 0,
  encoder LR 2e-05, the freshly initialized 48-way classifier head at 1e-4 (the first reference-check fix), 10 epochs,
  batch 32, max_len 128, fp32 on MPS. Every epoch was a selection candidate.
- The 10 val dumps of its epoch exports, plus the postfit record and the BERTurk reference check (the
  published-reference check of `pilot.berturk.reference_check`) built from them.

## Its reference check result

The chosen candidate (epoch 10, T 0.8560, global θ 0.13) reached a val micro-F1 of 113/238 = 0.4748 at the val-selected
global θ after T. That is below the reference check's floor of 0.707, so the status is `investigate`, with no resolution
recorded. The chosen epoch was the upper edge of the 1..10 grid, and the objective rose at every epoch. In the
diagnostics run after that outcome the train loss went below the 0.1719 label-prior floor from epoch 4 and was still
falling at 0.1409 in epoch 10, and 14 of 48 tags scored above F1 0 at global θ.

## The decision

The user's reply after that outcome, verbatim (2026-10-21): `fix: extend epochs (Recommended)`. Decision: **fix**. This
is the second reference-check fix.

## What replaced it

A new selection run whose epoch grid is 30, the user's choice. The encoder LR 2e-05, the head LR 1e-4 and every other
setting are unchanged; the linear schedule and its 10% warm-up then span 30 epochs.

This was the second recipe change made after a val outcome had been seen. The first (the head LR untied from the encoder
LR, after the first run's reference check) is under `results/superseded/berturk-tied-head-lr/`. BERTurk therefore has
three selection runs while Laya has one. Both changes are disclosed protocol changes, recorded in code as the ordered
chain `pilot.berturk.SUPERSEDED`, whose entries each carry the one canonical change (`delta`) of the run that replaced
them.

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
| `dumps/<dump run id>/val.npz`, `val.json` | the 10 val logit dumps (one per epoch export, epochs 1–10), from commit 4cdadf2 |
| `postfit/berturk.json` | the postfit record (`postfit/1`, system kind berturk, protocol: LR 2e-05, head LR 1e-4, 10 epochs), from commit 0df2036 |
| `postfit/berturk-reference.json` | the BERTurk reference check (`berturk-reference/1`, status investigate, 113/238), from commit 0df2036 |
