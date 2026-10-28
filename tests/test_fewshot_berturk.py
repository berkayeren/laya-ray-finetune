"""The BERTurk few-shot tracer: a label-masked base, then a continued run from its own
fine-tuned head, end to end through berturk_train.py and the HF Trainer on CPU, under a synthetic frozen protocol.

tests/fewshot_synthetic.synthetic_protocol installs a protocol whose fewshot block holds the tag card, k 1 and 2, one
draw (seed 0) and tiny recipes (base: --epochs 2 --stop-after-epoch 1, batch 4, max_len 32, cpu, fp32, mask label
-100; continued: 1 epoch at the base LRs). The chosen selection run is stood in for by a 2-epoch unmasked tiny run on
the same synthetic selection_train, trained at the frozen warm-up 0.1 and weight decay 0.01 (both differ from the
tuned defaults of new runs, as its batch 4 does): its berturk_train.json canonical config is the synthetic postfit
record's chosen.canonical, so check_base binds the masked base to it, and the few-shot commands, which omit the batch,
warm-up and weight decay, prove they resolve them from that run (the base) and from the masked base (the continued
run), never from pilot.defaults.BERTURK. berturk_train.main runs in this process (--manifest
passed explicitly), like a user's command:

- `--fewshot-base` keeps every message, sets the card column of every label vector to pilot.berturk.MASK_LABEL and
  records the proof (fewshot.stage base, unmasked_held_out_labels {card: 0});
- `--fewshot-draw 0 --fewshot-k 2 --base <masked base>` loads the masked base's own classifier head (no missing,
  unexpected or mismatched key: the head is loaded, never re-initialised), trains the frozen support rows (all 48
  labels) and replay rows (card masked) with the head in its own AdamW groups at the head LR, and records the proof
  (new_tag_labels {card: support rows}, replay_unmasked_held_out_labels 0, the base's run id and canonical config);
- `--print-run-id` of each command prints the run id the run then uses and creates nothing.
In `make test` (unit: no Ray, no weights, no data download; seconds).
"""
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from conftest import SyntheticRaw, prepared
from fewshot_synthetic import postfit_systems, synthetic_protocol
from pilot import berturk
from pilot import fewshot as fs
from pilot.data import sha256_file
from tiny_berturk import tiny_base

pytestmark = pytest.mark.unit

TIMEOUT = 600
# the frozen rules of the synthetic protocol: the BERTurk base recipe and the continued training, at tiny sizes
BERTURK_RULE = {"trainer": "berturk_train.py", "epochs": 2, "stop_after_epoch": 1, "encoder_lr": 2e-05,
                "head_lr": 0.0001, "batch": 4, "max_len": 32, "device": "cpu", "dtype": "float32",
                "mask_label": -100.0}
CONTINUED = {"epochs": 1, "encoder_lr": 2e-05, "head_lr": 0.0001, "rule": "the synthetic continued training"}
COMMON = ["--device", "cpu", "--batch", "4", "--max-len", "32", "--lr", "2e-05", "--head-lr", "0.0001",
          "--seed", "0"]
# the stand-in chosen run's warm-up and weight decay: the frozen recipe's, which differ from the tuned defaults of new
# runs, so a few-shot command that resolved from the plain context (pilot.defaults.BERTURK) would be told apart
FROZEN_TAIL = ["--warmup-ratio", "0.1", "--weight-decay", "0.01"]
GROUPS = [("encoder", True, 2e-05), ("encoder", False, 2e-05), ("head", True, 0.0001), ("head", False, 0.0001)]


def rules() -> dict[str, Any]:
    base_runs = json.loads(json.dumps(fs.RULES["base_runs"]))
    base_runs.update(seeds=[0], berturk=dict(BERTURK_RULE))
    return {"base_runs": base_runs, "continued_training": dict(CONTINUED)}


@pytest.fixture
def keep_threads() -> Iterator[None]:
    import torch
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def train(argv: list[Any]) -> Any:
    import berturk_train
    return berturk_train.main([str(x) for x in argv])


def print_run_id(argv: list[Any], capsys: pytest.CaptureFixture[str]) -> str:
    capsys.readouterr()
    assert train(argv + ["--print-run-id"]) is None
    return capsys.readouterr().out.strip().splitlines()[-1]


def read(ckpt: str | Path) -> tuple[str, dict[str, Any]]:
    with open(os.path.join(str(ckpt), berturk.RECORD), encoding="utf-8") as f:
        raw = f.read()
    return raw, json.loads(raw)


def without(argv: list[Any], flag: str) -> list[Any]:
    """argv with `flag` and its value removed (the flag must be there once)."""
    assert argv.count(flag) == 1, flag
    i = argv.index(flag)
    return argv[:i] + argv[i + 2:]


RESOLVED_KEYS = ("batch", "warmup_ratio", "weight_decay")   # passed by no committed few-shot command: resolved


def groups(rec: dict[str, Any]) -> list[tuple[str, bool, float]]:
    return [(g["group"], g["decay"], g["lr"]) for g in rec["record"]["optimizer_groups"]]


@pytest.mark.timeout(TIMEOUT)
def test_berturk_masked_base_then_continued_run(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                                keep_threads: None) -> None:
    import torch
    import transformers
    from safetensors.torch import load_file

    tiny = tiny_base(tmp_path / "tiny")
    # ---- the chosen selection run's stand-in: unmasked, the base recipe's full schedule ----
    pre_dir, pre_manifest = prepared(tmp_path, synthetic_raw, name="pre")
    chosen_out = tmp_path / "ckpt-chosen"
    train(["--data", os.path.join(pre_dir, "splits", "selection_train.jsonl"), "--manifest", pre_manifest,
           "--base", tiny, "--out", chosen_out, "--storage", tmp_path / "runs-chosen", "--epochs", "2"] + COMMON
          + FROZEN_TAIL)
    chosen = read(chosen_out)[1]["canonical"]
    from pilot import defaults
    # every resolved key of the stand-in differs from the tuned default, so each context check below tells the
    # chosen run's (or the masked base's) value from the plain context's
    assert all(chosen[k] != defaults.BERTURK[k] for k in RESOLVED_KEYS), chosen
    assert all(f not in COMMON for f in ("--warmup-ratio", "--weight-decay"))   # the few-shot commands omit them
    assert "fewshot" not in chosen and "mask_tags" not in chosen       # dropped when None
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, rules=rules(),
                            systems=postfit_systems(tmp_path, {"berturk": chosen}))
    assert fs.chosen_canonical(ns.protocol, "berturk") == chosen
    sel = os.path.join(ns.out_dir, "splits", "selection_train.jsonl")
    tags = ns.block["tags"]
    assert tags == ["card"]

    # ---- the label-masked base ----
    storage, base_out = tmp_path / "runs-base", tmp_path / "ckpt-base"
    base_argv = ["--data", sel, "--manifest", ns.manifest, "--base", tiny, "--out", base_out, "--storage", storage,
                 "--epochs", "2", "--stop-after-epoch", "1", "--fewshot-base"] + COMMON
    base_rid = print_run_id(base_argv, capsys)
    assert not storage.exists() and not base_out.exists()           # --print-run-id creates nothing
    # --batch omitted: resolved from the frozen selection run (the tuned 16 would be refused by check_base)
    assert print_run_id(without(base_argv, "--batch"), capsys) == base_rid
    # the tuned default given explicitly departs from the frozen selection run and is refused by check_base
    for flag, key in (("--warmup-ratio", "warmup_ratio"), ("--weight-decay", "weight_decay")):
        with pytest.raises(SystemExit, match=key):
            train(base_argv + [flag, repr(defaults.BERTURK[key]), "--print-run-id"])
    assert not storage.exists() and not base_out.exists()
    train(base_argv)
    raw, rec = read(base_out)
    assert rec["run_id"] == base_rid == berturk.run_id(rec["canonical"]) and rec["epoch"] == 1
    canon = rec["canonical"]
    assert canon["mask_tags"] == ["card"] and canon["fewshot"] == {"protocol_sha256": ns.pin, "stage": "base"}
    assert {k for k in set(canon) | set(chosen) if canon.get(k) != chosen.get(k)} <= set(fs.BASE_FREE)
    assert rec["record"]["n_messages"] == len(ns.rows)              # every message is kept
    assert rec["record"]["fewshot"] == {"protocol_sha256": ns.pin, "stage": "base", "tags": ["card"],
                                        "unmasked_held_out_labels": {"card": 0}}
    assert groups(rec) == GROUPS                                     # the fresh head at the head LR
    assert "bank_" not in raw                                        # counts and digests only, never a uid
    assert {k: canon[k] for k in RESOLVED_KEYS} == {k: chosen[k] for k in RESOLVED_KEYS}

    # ---- the continued run from that base's own head, on support plus replay ----
    seen = {}
    load = transformers.BertForSequenceClassification.from_pretrained

    def spy(*args: Any, **kwargs: Any) -> tuple[Any, Any]:  # the head as loaded, before any training step
        model, info = load(*args, **kwargs)
        seen.update(info=info, weight=model.classifier.weight.detach().clone(),
                    bias=model.classifier.bias.detach().clone())
        return model, info
    monkeypatch.setattr(transformers.BertForSequenceClassification, "from_pretrained", spy)
    storage, cont_out = tmp_path / "runs-cont", tmp_path / "ckpt-cont"
    cont_argv = ["--data", sel, "--manifest", ns.manifest, "--base", base_out, "--out", cont_out, "--storage", storage,
                 "--epochs", "1", "--fewshot-draw", "0", "--fewshot-k", "2"] + COMMON
    cont_rid = print_run_id(cont_argv, capsys)
    assert not storage.exists() and not seen                        # nothing created, nothing loaded
    # the omitted recipe flags resolve from the masked base: --batch omitted gives the same run; --epochs omitted
    # resolves the base's 2 and is refused naming the frozen continued 1; an explicit departure is refused
    assert print_run_id(without(cont_argv, "--batch"), capsys) == cont_rid
    with pytest.raises(SystemExit, match="epochs is 2 here but 1"):
        train(without(cont_argv, "--epochs") + ["--print-run-id"])
    with pytest.raises(SystemExit, match="weight_decay"):    # the tuned 0.0, not the base's resolved 0.01
        train(cont_argv + ["--weight-decay", "0.0", "--print-run-id"])
    assert not storage.exists() and not seen
    train(cont_argv)
    head = load_file(os.path.join(str(base_out), "model.safetensors"))
    assert all(not seen["info"][k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys"))
    assert torch.equal(seen["weight"], head["classifier.weight"]) and torch.equal(seen["bias"], head["classifier.bias"])
    raw, rec = read(cont_out)
    assert rec["run_id"] == cont_rid == berturk.run_id(rec["canonical"]) != base_rid
    canon = rec["canonical"]
    d = ns.block["draws"][0]
    assert canon["fewshot"] == {"protocol_sha256": ns.pin, "stage": "continued", "draw": 0, "k": 2,
                                "digest": d["digest"], "base_run_id": base_rid}
    assert "mask_tags" not in canon and canon["epochs"] == 1 and canon["seed"] == d["seed"] == 0
    assert canon["base_model_sha256"] == sha256_file(os.path.join(str(base_out), "model.safetensors"))
    assert canon["base_config_sha256"] == sha256_file(os.path.join(str(base_out), "config.json"))
    sel_k = fs.selection(ns.block, 0, 2)
    n_sup, n_rep = len(sel_k["support"]), len(sel_k["replay"])
    assert rec["record"]["n_messages"] == n_sup + n_rep
    assert rec["record"]["fewshot"] == {
        "protocol_sha256": ns.pin, "stage": "continued", "draw": 0, "k": 2, "digest": d["digest"], "tags": ["card"],
        "support_rows": n_sup, "replay_rows": n_rep, "new_tag_labels": {"card": n_sup},
        "replay_unmasked_held_out_labels": 0, "base": {"run_id": base_rid, "canonical": read(base_out)[1]["canonical"]}}
    assert groups(rec) == GROUPS                                     # the loaded head, in its own groups
    base_canon = read(base_out)[1]["canonical"]
    assert {k: canon[k] for k in RESOLVED_KEYS} == {k: base_canon[k] for k in RESOLVED_KEYS}
    trained = load_file(os.path.join(str(cont_out), "model.safetensors"))
    assert not torch.equal(trained["classifier.weight"], head["classifier.weight"])   # the head trained on
    assert "bank_" not in raw
