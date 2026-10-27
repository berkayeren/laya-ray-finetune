"""The defaults of new training runs: the best configuration of each committed exploratory search record.

ray_train.py (Laya) and berturk_train.py (BERTurk) read their recipe defaults from here: a plain command, one that
passes none of these flags and has no context to resolve them from, trains the best trial of the committed Ray Tune
search of its system. An explicit flag always wins.

Source (the `best` block of each record, its `config` copied exactly):
- LAYA: results/tune/laya.json (schema tune/1, sha256 b111b6e21db971fa765650c46a9534d530317fce620cea7f9e9f0c61f26feef5),
  best trial 14, epoch 6: encoder LR, head LR, neg 16, epochs 6.
- BERTURK: results/tune/berturk.json (schema tune/1, sha256
  acb143c13bc3b64dfbab5c819a435790e64671c39422654cc2342f2f10d3e70d), best trial 4, epoch 20: encoder LR, head LR,
  batch 16, warm-up ratio 0.06, weight decay 0.0, epochs 20.

Design notes:
- No import at all and no file read at import: a training CLI never depends on a results file, and the module stays
  inside the stdlib-only budget of the torch-free modules (tests/test_defaults.py runs it in a fresh interpreter).
- The values are the records' own numbers, unrounded and typed as the records hold them: neg, epochs and batch are
  int; the LRs, warmup_ratio and weight_decay are float (weight_decay is written 0.0, never 0, because the canonical
  run config is hashed as JSON text, where 0 and 0.0 differ). Each dict keeps its record's key order.
- tests/test_defaults.py binds every constant to pilot.tune.read_record of the committed records at their pinned
  sha256s, by JSON text (types included), so a drift by value or type fails a test.
- These are defaults of new runs only. Every committed run, record and the verdict used the frozen recipe of
  pilot.postfit and pilot.berturk, which stays unchanged; every command that reproduces a past run passes its recipe
  itself or resolves it from its context (a postfit record, the frozen few-shot protocol, a masked base).
- SOURCES names each record: its repo-relative path, its sha256, the best trial's index and its epoch.

usage:
    from pilot import defaults
    defaults.LAYA["encoder_lr"]; defaults.BERTURK["batch"]; defaults.SOURCES["laya"]["sha256"]
"""

# results/tune/laya.json best.config (trial 14, epoch 6)
LAYA: dict[str, float | int] = {"encoder_lr": 2.9660069889947058e-05, "head_lr": 9.811131074677546e-05, "neg": 16,
                                "epochs": 6}
# results/tune/berturk.json best.config (trial 4, epoch 20)
BERTURK: dict[str, float | int] = {"encoder_lr": 6.637303455962178e-05, "head_lr": 0.00020030926548084864,
                                   "batch": 16, "warmup_ratio": 0.06, "weight_decay": 0.0, "epochs": 20}
SOURCES: dict[str, dict[str, str | int]] = {
    "laya": {"record": "results/tune/laya.json",
             "sha256": "b111b6e21db971fa765650c46a9534d530317fce620cea7f9e9f0c61f26feef5", "trial": 14, "epoch": 6},
    "berturk": {"record": "results/tune/berturk.json",
                "sha256": "acb143c13bc3b64dfbab5c819a435790e64671c39422654cc2342f2f10d3e70d", "trial": 4, "epoch": 20},
}
