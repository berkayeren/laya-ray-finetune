"""Pure contracts of the BERTurk baseline: the masked BCE, labels and masks in INTENTS order, the label-order and
casing refusals, the run identity, the optimizer groups, the checkpoint identity, and the torch-free pilot.berturk.

Offline (unit): the loss tests use torch on CPU tensors only; only the Trainer-contract test loads the tiny offline
BERT base (tests/tiny_berturk.py).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import berturk_train
import score
from intents import INTENTS
from pilot import berturk, defaults, dumps
from pilot.data import REPO_ROOT, sha256_file

pytestmark = pytest.mark.unit

TAGS = list(INTENTS)


# ---- masked BCE ---------------------------------------------------------------------------------------------------

def _batch(n: int = 4, seed: int = 0, p: float = 0.2) -> tuple[Any, Any]:
    """(logits, labels): seeded float32 CPU tensors [n, 48], each label 1.0 with probability p."""
    import torch
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, len(TAGS), generator=g, dtype=torch.float32)
    labels = (torch.rand(n, len(TAGS), generator=g) < p).float()
    return logits, labels


def _loss(logits: Any, labels: Any, n: Any = None) -> Any:
    """berturk_train.masked_bce of logits and labels, num_items_in_batch n."""
    return berturk_train.masked_bce({"logits": logits}, labels, n)


def test_masked_bce_equals_mean_bce_when_unmasked() -> None:
    import torch
    logits, labels = _batch()
    want = torch.nn.BCEWithLogitsLoss()(logits, labels)
    assert abs(float(_loss(logits, labels)) - float(want)) < 1e-6
    n = torch.tensor(labels.numel())                         # what the Trainer counts: labels != -100
    assert abs(float(_loss(logits, labels, n)) - float(want)) < 1e-6


def test_masked_tag_gets_zero_gradient() -> None:
    logits, labels = _batch()
    labels[:, 3] = berturk.MASK_LABEL
    logits.requires_grad_(True)
    _loss(logits, labels).backward()
    assert (logits.grad[:, 3] == 0).all()                    # exactly zero, not merely small
    assert (logits.grad[:, [0, 1, 2, 4]] != 0).all()


def test_all_masked_batch_is_finite_zero() -> None:
    import torch
    logits, labels = _batch()
    labels[:] = berturk.MASK_LABEL
    logits.requires_grad_(True)
    for n in (None, torch.tensor(0), 0):                     # the Trainer's count of an all-masked window is 0
        logits.grad = None
        loss = _loss(logits, labels, n)
        assert torch.isfinite(loss) and float(loss.detach()) == 0.0
        loss.backward()
        assert (logits.grad == 0).all()


def test_loss_divides_by_num_items_across_micro_batches() -> None:
    """The Trainer's accumulation contract: sum / num_items over the window, so the micro-batch losses add up to the
    merged batch's loss (a per-batch mean would not)."""
    import torch
    la, ya = _batch(3, seed=1)
    lb, yb = _batch(5, seed=2)
    ya[0, :10] = berturk.MASK_LABEL                           # unequal valid counts per micro-batch
    yb[2, 5:] = berturk.MASK_LABEL
    n = (ya != berturk.MASK_LABEL).sum() + (yb != berturk.MASK_LABEL).sum()
    merged = _loss(torch.cat([la, lb]), torch.cat([ya, yb]))
    assert abs(float(_loss(la, ya, n) + _loss(lb, yb, n)) - float(merged)) < 1e-6


def test_loss_int_and_tensor_num_items_agree() -> None:
    import torch
    logits, labels = _batch()
    assert float(_loss(logits, labels, 100)) == float(_loss(logits, labels, torch.tensor(100)))


def test_loss_ignores_masked_logits_and_label_values() -> None:
    """A masked entry contributes nothing: changing its logit leaves the loss value unchanged."""
    logits, labels = _batch()
    labels[1, 7] = berturk.MASK_LABEL
    before = float(_loss(logits, labels))
    logits[1, 7] = 50.0
    assert float(_loss(logits, labels)) == before


def test_loss_gets_the_trainers_unmasked_label_count(tmp_path: Path) -> None:
    """The transformers contract the masked BCE rests on: with compute_loss_func set, the Trainer pops the labels and
    passes num_items_in_batch = the count of labels != -100 over the whole accumulation window (2 micro-batches here).
    """
    import torch
    import transformers
    from tiny_berturk import tiny_base
    base = tiny_base(tmp_path / "base")
    tok = transformers.AutoTokenizer.from_pretrained(base, local_files_only=True)
    rows = []
    for i, text in enumerate(("Kartım kayboldu 1", "Evet .", "İptal etmek istiyorum", "cari hesap")):
        labels = [float(j % 7 == i) for j in range(len(TAGS))]
        labels[:3 * i] = [berturk.MASK_LABEL] * (3 * i)        # 0, 3, 6 and 9 masked entries
        rows.append(dict(tok(text), labels=labels))
    want = sum(sum(v != berturk.MASK_LABEL for v in r["labels"]) for r in rows)
    seen = []

    def loss_fn(outputs: Any, labels: Any, num_items_in_batch: Any = None) -> Any:
        """masked_bce, recording the Trainer's num_items_in_batch."""
        seen.append(int(num_items_in_batch))
        return berturk_train.masked_bce(outputs, labels, num_items_in_batch)

    threads = torch.get_num_threads()
    try:
        transformers.set_seed(0)
        model = transformers.BertForSequenceClassification.from_pretrained(
            base, num_labels=len(TAGS), problem_type=berturk.PROBLEM_TYPE, dtype=torch.float32, local_files_only=True)
        args = transformers.TrainingArguments(
            output_dir=str(tmp_path / "t"), per_device_train_batch_size=2, gradient_accumulation_steps=2, max_steps=1,
            use_cpu=True, save_strategy="no", report_to="none", disable_tqdm=True, dataloader_num_workers=0,
            optim=berturk.OPTIM, seed=0)
        transformers.Trainer(model=model, args=args, train_dataset=rows, compute_loss_func=loss_fn,
                             data_collator=transformers.DataCollatorWithPadding(tok)).train()
    finally:
        torch.set_num_threads(threads)
    assert seen == [want, want]                               # one count for the window, both micro-batches


def test_masked_bce_refuses_non_fp32_logits() -> None:
    import torch
    logits, labels = _batch()
    for dtype in (torch.float16, torch.bfloat16):
        with pytest.raises(SystemExit, match="fp32"):
            _loss(logits.to(dtype), labels)


# ---- labels and masks ---------------------------------------------------------------------------------------------

def _row(pos: tuple[str, ...] = ("card", "lost_stolen"), uid: str = "bank_0_0") -> dict[str, Any]:
    """A training row labelling all 48 tags, positive exactly on `pos`."""
    return {"uid": uid, "expected": {t: t in pos for t in TAGS}}


def test_label_vector_follows_intents_order() -> None:
    vec = berturk.label_vector(_row())
    assert len(vec) == len(TAGS)
    assert [t for t, v in zip(TAGS, vec) if v == 1.0] == [t for t in TAGS if t in ("card", "lost_stolen")]
    assert set(vec) == {0.0, 1.0}
    masked = berturk.label_vector(_row(), ("card", "affirm"))
    assert masked[TAGS.index("card")] == masked[TAGS.index("affirm")] == berturk.MASK_LABEL
    assert masked[TAGS.index("lost_stolen")] == 1.0
    assert berturk.id2label() == dict(enumerate(TAGS))
    assert berturk.label2id() == {t: i for i, t in enumerate(TAGS)}


def test_label_vector_refuses_a_missing_tag() -> None:
    row = _row(uid="bank_7_3")
    del row["expected"]["pin"]
    with pytest.raises(SystemExit, match="bank_7_3.*pin"):
        berturk.label_vector(row)
    assert berturk.label_vector(row, ("pin",))[TAGS.index("pin")] == berturk.MASK_LABEL   # masked: never read


def test_mask_tags_are_validated() -> None:
    assert berturk.check_mask_tags(["pin", "card"]) == ("card", "pin")
    assert berturk.check_mask_tags([]) == ()
    with pytest.raises(SystemExit, match="unknown.*not_a_tag"):
        berturk.check_mask_tags(["card", "not_a_tag"])
    with pytest.raises(SystemExit, match="card given more than once"):
        berturk.check_mask_tags(["card", "pin", "card"])


def _cfg() -> dict[str, Any]:
    """The label part of the config.json berturk_train.py writes."""
    return {"id2label": {str(i): t for i, t in enumerate(TAGS)}, "label2id": {t: i for i, t in enumerate(TAGS)},
            "problem_type": berturk.PROBLEM_TYPE}


def test_label_order_refuses_a_permutation() -> None:
    berturk.check_label_order(_cfg(), "ok")                  # the config berturk_train.py writes
    swapped = _cfg()
    swapped["id2label"]["0"], swapped["id2label"]["1"] = TAGS[1], TAGS[0]
    wrong_inverse = _cfg()
    wrong_inverse["label2id"][TAGS[0]] = 5
    for bad in (swapped, wrong_inverse, dict(_cfg(), problem_type="single_label_classification"),
                dict(_cfg(), id2label={str(i): "LABEL_%d" % i for i in range(len(TAGS))}), {}):
        with pytest.raises(SystemExit, match="INTENTS order"):
            berturk.check_label_order(bad, "ckpt")


def test_check_cased_refuses_a_lowercasing_tokenizer() -> None:
    berturk.check_cased({"do_lower_case": False, "max_len": 512, "init_inputs": []}, "pinned")
    for bad in ({"do_lower_case": True}, {}, {"do_lower_case": 0}, []):
        with pytest.raises(SystemExit, match="do_lower_case"):
            berturk.check_cased(bad, "base")


# ---- run identity -------------------------------------------------------------------------------------------------

def _values(**over: Any) -> dict[str, Any]:
    """Every pilot.berturk.IDENTITY value of a 10-epoch selection run (digests are placeholders), `over` applied."""
    v = {"split": "selection_train", "data_sha256": "a" * 64, "data_role": "train", "base_model_sha256": "b" * 64,
         "base_config_sha256": "c" * 64, "base_files_sha256": None, "transformers_version": "5.19.0",
         "accelerate_version": "1.15.0", "device": "mps", "epochs": 10, "batch": 32, "encoder_lr": 2e-05,
         "head_lr": 0.0001, "weight_decay": 0.01, "warmup_ratio": 0.1, "lr_scheduler": "linear", "max_grad_norm": 1.0,
         "optim": "adamw_torch", "loss": "masked_bce", "max_len": 128, "max_messages": None, "mask_tags": None,
         "seed": 0, "stop_after_epoch": None, "postfit_sha256": None, "fewshot": None, "tune": None}
    v.update(over)
    return v


def test_identity_classifies_every_flag() -> None:
    dests = [act.dest for act in berturk_train.parser()._actions if act.dest != "help"]
    for d in dests:
        homes = [d in berturk.IDENTITY, d in berturk.NOT_IDENTITY, d in berturk.RESOLVED]
        assert sum(homes) == 1, "%s is in %d of IDENTITY, NOT_IDENTITY, RESOLVED" % (d, sum(homes))
    assert all(k in berturk.IDENTITY for keys in berturk.RESOLVED.values() for k in keys)
    assert set(berturk.DROPPED_WHEN_NONE) <= set(berturk.IDENTITY)
    assert not set(berturk.NOT_IDENTITY) & set(berturk.IDENTITY)


def test_identity_resolves_lr_to_the_encoder_only() -> None:
    """The first reference-check fix: --lr is the encoder LR only; the head LR is its own flag and IDENTITY key, so two
    runs that differ only in it never share a run dir."""
    assert berturk.RESOLVED == {"lr": ("encoder_lr",)}
    assert dict(berturk_train.RECIPE_FLAGS)["lr"] == "encoder_lr"
    dests = {act.dest: act for act in berturk_train.parser()._actions}
    assert "head_lr" in dests and "head_lr" in berturk.IDENTITY and dests["head_lr"].default is None
    assert dests["lr"].default is None                    # resolved from the run's context
    v = _identity(_parsed())                              # a plain command: the tuned defaults
    assert (v["encoder_lr"], v["head_lr"]) == (defaults.BERTURK["encoder_lr"], defaults.BERTURK["head_lr"])
    ids = {berturk.run_id(berturk.canonical(_values(head_lr=h))) for h in (berturk.HEAD_LR, 2e-05)}
    assert len(ids) == 2


def _superseded_canonical(entry: dict[str, Any]) -> dict[str, Any]:
    """The chosen canonical config of a SUPERSEDED entry's committed postfit record."""
    with open(os.path.join(REPO_ROOT, entry["record"]), encoding="utf-8") as f:
        return json.load(f)["chosen"]["canonical"]


RECIPE_KEYS = ("encoder_lr", "batch", "max_len", "weight_decay", "warmup_ratio", "lr_scheduler", "max_grad_norm",
               "optim", "loss", "seed")


def _recipe() -> dict[str, Any]:
    """The current recipe constants other than the epoch grid and the head LR, keyed like the canonical config."""
    return {"encoder_lr": berturk.LR, "batch": berturk.BATCH, "max_len": berturk.MAX_LEN,
            "weight_decay": berturk.WEIGHT_DECAY, "warmup_ratio": berturk.WARMUP_RATIO,
            "lr_scheduler": berturk.LR_SCHEDULER, "max_grad_norm": berturk.MAX_GRAD_NORM, "optim": berturk.OPTIM,
            "loss": berturk.LOSS, "seed": 0}


def test_head_lr_is_the_only_change_from_the_superseded_run() -> None:
    """The head-LR run's canonical config is the tied run's with its recorded delta (head_lr
    HEAD_LR), nothing else; the tied run's recipe is compared with its own protocol literals (epochs 10), never with
    the current EPOCHS."""
    tied, head = berturk.SUPERSEDED[0], berturk.SUPERSEDED[1]
    old = _superseded_canonical(tied)
    assert berturk.run_id(old) == tied["train_run_id"] == "berturk-selection_train-s0-c0c9505e"
    assert old["head_lr"] == old["encoder_lr"]                         # the superseded run tied the head LR
    assert old["epochs"] == tied["protocol"]["epochs"] == 10            # its own grid, a literal
    assert tied["delta"] == {"head_lr": berturk.HEAD_LR}
    new = dict(old, **tied["delta"])
    assert berturk.canonical(dict({k: None for k in berturk.DROPPED_WHEN_NONE}, **new)) == new
    changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    assert changed == sorted(tied["delta"]) == ["head_lr"]
    assert berturk.run_id(new) == head["train_run_id"] == "berturk-selection_train-s0-8c033154"
    assert {k: old[k] for k in RECIPE_KEYS} == _recipe()                # every other recipe value is unchanged
    assert berturk.HEAD_LR == 0.0001 and berturk.THRESHOLD == __import__("fractions").Fraction(707, 1000)
    assert berturk.PROTOCOL["final_seeds"] == (0, 1, 2) and berturk.PROTOCOL["lr_grid"] == (berturk.LR,)


def test_epochs_is_the_only_change_from_the_head_lr_run() -> None:
    """The second reference-check fix: the 30-epoch selection command's canonical config is the head-LR run's
    (SUPERSEDED[-1]) with its recorded delta, epochs EPOCHS = 30, and nothing else, so its run id is known offline (the
    committed selection run carries it); every other recipe constant equals the head-LR run's canonical value."""
    last = berturk.SUPERSEDED[-1]
    old = _superseded_canonical(last)
    assert berturk.run_id(old) == last["train_run_id"] == "berturk-selection_train-s0-8c033154"
    assert old["epochs"] == last["protocol"]["epochs"] == 10 and old["head_lr"] == berturk.HEAD_LR
    new = dict(old, **last["delta"])
    changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    assert changed == sorted(last["delta"]) == ["epochs"]               # the delta names exactly what changed
    assert new["epochs"] == berturk.EPOCHS == 30
    assert berturk.canonical(dict({k: None for k in berturk.DROPPED_WHEN_NONE}, **new)) == new
    assert berturk.run_id(new) == "berturk-selection_train-s0-4f36ae05"
    assert {k: old[k] for k in RECIPE_KEYS} == _recipe()                # the LRs and the recipe stay
    assert old["head_lr"] == berturk.HEAD_LR == 0.0001
    dests = {act.dest: act for act in berturk_train.parser()._actions}
    assert dests["epochs"].default is None                # resolved: a plain command gets the tuned epochs
    assert _identity(_parsed())["epochs"] == defaults.BERTURK["epochs"] == 20


def test_identity_canonical_refuses_missing_and_unexpected_keys() -> None:
    v = _values()
    del v["seed"]
    with pytest.raises(ValueError, match="missing seed"):
        berturk.canonical(v)
    with pytest.raises(ValueError, match="unexpected neg"):
        berturk.canonical(_values(neg=8))
    canon = berturk.canonical(_values())
    assert not set(berturk.DROPPED_WHEN_NONE) & set(canon)    # left out when None
    assert berturk.canonical(_values(mask_tags=["card"]))["mask_tags"] == ["card"]
    assert canon["epochs"] == 10 and canon["seed"] == 0       # never dropped


def test_identity_run_id_is_order_independent() -> None:
    canon = berturk.canonical(_values())
    rid = berturk.run_id(canon)
    assert rid == berturk.run_id(dict(reversed(list(canon.items()))))
    assert rid.startswith("berturk-selection_train-s0-") and len(rid.rsplit("-", 1)[1]) == 8
    assert berturk.run_id(berturk.canonical(_values(epochs=4))) != rid
    assert berturk.run_id(berturk.canonical(_values(mask_tags=["card"]))) != rid


def test_identity_seeds_never_share_a_run_dir() -> None:
    ids = [berturk.run_id(berturk.canonical(_values(seed=s))) for s in (0, 1, 2)]
    assert len(set(ids)) == 3
    assert [i.split("-")[2] for i in ids] == ["s0", "s1", "s2"]


def test_identity_fewshot_key_is_dropped_when_none() -> None:
    """fewshot is an IDENTITY key left out when None, and the three few-shot flags are
    NOT_IDENTITY (represented by fewshot and mask_tags); a bound base and a continued binding each get their own id."""
    assert "fewshot" in berturk.IDENTITY and "fewshot" in berturk.DROPPED_WHEN_NONE
    assert {"fewshot_base", "fewshot_draw", "fewshot_k"} <= set(berturk.NOT_IDENTITY)
    plain = berturk.canonical(_values())
    assert "fewshot" not in plain
    base = berturk.canonical(_values(mask_tags=["card"], fewshot={"protocol_sha256": "p" * 64, "stage": "base"}))
    cont = berturk.canonical(_values(fewshot={"protocol_sha256": "p" * 64, "stage": "continued", "draw": 0, "k": 10,
                                              "digest": "d" * 64, "base_run_id": berturk.run_id(base)}))
    assert len({berturk.run_id(c) for c in (plain, base, cont)}) == 3


def test_identity_committed_final_seed_run_ids_are_unchanged() -> None:
    """The three committed BERTurk final seeds' train run ids recompute from the committed postfit record and
    data/splits.json with every DROPPED_WHEN_NONE key (fewshot included) given as None; the selection run's id is
    test_epochs_is_the_only_change_from_the_head_lr_run's (berturk-selection_train-s0-4f36ae05)."""
    with open(os.path.join(REPO_ROOT, "results", "postfit", "berturk-final.json"), encoding="utf-8") as f:
        final = json.load(f)
    with open(os.path.join(REPO_ROOT, "results", "postfit", "berturk.json"), encoding="utf-8") as f:
        chosen = json.load(f)["chosen"]["canonical"]
    with open(os.path.join(REPO_ROOT, "data", "splits.json"), encoding="utf-8") as f:
        entry = json.load(f)["splits"]["final_train"]
    pf_sha = sha256_file(os.path.join(REPO_ROOT, final["postfit"]["path"]))
    assert pf_sha == final["postfit"]["sha256"]
    assert berturk.run_id(berturk.canonical(dict({k: None for k in berturk.DROPPED_WHEN_NONE}, **chosen))) \
        == final["chosen"]["train_run_id"] == "berturk-selection_train-s0-4f36ae05"
    for s in final["seeds"]:
        values = dict({k: None for k in berturk.DROPPED_WHEN_NONE}, **chosen)
        values.update(split="final_train", data_sha256=entry["sha256"], data_role=entry["role"], seed=s["seed"],
                      stop_after_epoch=final["chosen"]["stop_after_epoch"], postfit_sha256=pf_sha)
        assert berturk.run_id(berturk.canonical(values)) == s["train_run_id"]
    assert [s["train_run_id"] for s in final["seeds"]] == ["berturk-final_train-s0-8a167355",
                                                           "berturk-final_train-s1-3c44e7ff",
                                                           "berturk-final_train-s2-44f81396"]


def test_identity_tune_key_dropped_when_none_and_named_when_set() -> None:
    """tune (the tuning partition of a tune.py search trial) is an IDENTITY key left out when None: a plain command's
    canonical config and run id are as before, and a trial with a partition block gets its own run id."""
    assert "tune" in berturk.IDENTITY and "tune" in berturk.DROPPED_WHEN_NONE
    plain = berturk.canonical(_values())
    assert "tune" not in plain
    before = {k: v for k, v in _values().items() if k != "tune" and not (k in berturk.DROPPED_WHEN_NONE and v is None)}
    assert plain == before and berturk.run_id(plain) == berturk.run_id(before)
    block = {"source": "selection_train", "train_folds": list(range(12)), "dev_folds": [12, 13],
             "train_uids_sha256": "e" * 64, "dev_uids_sha256": "f" * 64}
    tuned = berturk.canonical(_values(tune=block))
    assert tuned["tune"] == block and berturk.run_id(tuned) != berturk.run_id(plain)


def _committed_berturk_canonicals() -> list[tuple[str, dict[str, Any], str]]:
    """(tracked path, canonical config, recorded train run id) of every BERTurk canonical config under results/: each
    dump sidecar's training block and each postfit record's chosen run."""
    tracked = subprocess.run(["git", "ls-files", "--", "results"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout.split()
    out = []
    for path in tracked:
        if not path.endswith(".json"):
            continue
        with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            continue
        tr = doc.get("training")
        if (isinstance(doc.get("system"), dict) and doc["system"].get("kind") == "berturk" and isinstance(tr, dict)
                and isinstance(tr.get("canonical"), dict)):
            out.append((path, tr["canonical"], tr["run_id"]))
        chosen = doc.get("chosen")
        if doc.get("system_kind") == "berturk" and isinstance(chosen, dict) and isinstance(chosen.get("canonical"),
                                                                                             dict):
            out.append((path, chosen["canonical"], chosen["train_run_id"]))
    return out


def test_committed_berturk_run_ids_are_unchanged_by_the_tune_key() -> None:
    """Every committed BERTurk canonical config whose keys are all current IDENTITY keys rebuilds unchanged through
    pilot.berturk.canonical with every DROPPED_WHEN_NONE key (tune included) given as None, so its recorded run id
    recomputes: the selection runs, the three final seeds and the few-shot runs among them."""
    found = _committed_berturk_canonicals()
    checked = {}
    for path, canon, rid in found:
        if not set(canon) <= set(berturk.IDENTITY):
            continue
        assert "tune" not in canon, path
        assert berturk.canonical(dict({k: None for k in berturk.DROPPED_WHEN_NONE}, **canon)) == canon, path
        assert berturk.run_id(canon) == rid, path
        checked[rid] = canon
    selection = [r for r, c in checked.items() if c["split"] == "selection_train"
                 and not {"fewshot", "stop_after_epoch", "postfit_sha256"} & set(c)]
    finals = sorted(r for r, c in checked.items() if "postfit_sha256" in c)
    few = [r for r, c in checked.items() if "fewshot" in c]
    assert "berturk-selection_train-s0-4f36ae05" in selection
    assert finals == ["berturk-final_train-s0-8a167355", "berturk-final_train-s1-3c44e7ff",
                      "berturk-final_train-s2-44f81396"]
    assert len(few) == 9                                    # three masked bases and six continued runs


def _parsed(*extra: str) -> Any:
    """berturk_train.py's parsed plain selection command on the pinned base, `extra` appended, its omitted recipe
    flags resolved exactly as main resolves a command with no --postfit record, few-shot mode or base: from the tuned
    defaults (pilot.defaults.BERTURK)."""
    a = berturk_train.parser().parse_args(["--data", "data/splits/selection_train.jsonl", *extra])
    berturk_train.resolve_recipe(a, *berturk_train.recipe_context(a, None, None, None))
    return a


def _identity(a: Any) -> dict[str, Any]:
    """identity_values of a parsed command with placeholder entry and digests, no mask, postfit or few-shot."""
    entry = {"path": "data/splits/selection_train.jsonl", "sha256": "a" * 64, "role": "train"}
    digests = {f: "%x" % i * 64 for i, f in enumerate(berturk.BASE_FILES, start=1)}
    return berturk_train.identity_values(a, entry, digests, (), None, None)


FROZEN_RECIPE = ("--epochs", "30", "--lr", "2e-05", "--head-lr", "0.0001", "--batch", "32", "--warmup-ratio", "0.1",
                 "--weight-decay", "0.01")


def test_recipe_flags_resolve_to_the_tuned_defaults_without_context(capsys: pytest.CaptureFixture[str]) -> None:
    """The six recipe flags default to None; a plain command (no --postfit, few-shot mode or masked base) resolves
    them to the tuned defaults (pilot.defaults.BERTURK, by JSON text: weight_decay is the float 0.0), while the frozen
    recipe spelled out gives exactly the identity values the former default command gave (the recipe constants), so
    its run id is the committed one's; --warmup-ratio 0.06 or --weight-decay 0.0 alone each name another run than the
    frozen one; a ratio outside [0, 1) or a negative weight decay is an argument error before anything is read."""
    dests = {act.dest: act for act in berturk_train.parser()._actions}
    assert all(dests[d].default is None for d, _ in berturk_train.RECIPE_FLAGS)
    assert dests["max_len"].default == berturk.MAX_LEN
    plain = _identity(_parsed())
    got = {key: plain[key] for _, key in berturk_train.RECIPE_FLAGS}
    assert json.dumps(got, sort_keys=True) == json.dumps(defaults.BERTURK, sort_keys=True)
    assert repr(plain["weight_decay"]) == "0.0"
    a = _parsed(*FROZEN_RECIPE)
    v = _identity(a)
    assert list(v) == list(berturk.IDENTITY)
    digests = {f: "%x" % i * 64 for i, f in enumerate(berturk.BASE_FILES, start=1)}
    old = {"split": "selection_train", "data_sha256": "a" * 64, "data_role": "train",
           "base_model_sha256": digests["model.safetensors"], "base_config_sha256": digests["config.json"],
           "base_files_sha256": None, "transformers_version": v["transformers_version"],
           "accelerate_version": v["accelerate_version"], "device": "cpu", "epochs": berturk.EPOCHS,
           "batch": berturk.BATCH, "encoder_lr": berturk.LR, "head_lr": berturk.HEAD_LR,
           "weight_decay": berturk.WEIGHT_DECAY, "warmup_ratio": berturk.WARMUP_RATIO,
           "lr_scheduler": berturk.LR_SCHEDULER, "max_grad_norm": berturk.MAX_GRAD_NORM, "optim": berturk.OPTIM,
           "loss": berturk.LOSS, "max_len": berturk.MAX_LEN, "max_messages": None, "mask_tags": None, "seed": 0,
           "stop_after_epoch": None, "postfit_sha256": None, "fewshot": None, "tune": None}
    assert v == old
    assert json.dumps(v, sort_keys=True) == json.dumps(old, sort_keys=True)
    rid = berturk.run_id(berturk.canonical(v))
    assert berturk.run_id(berturk.canonical(plain)) != rid
    for extra in (("--warmup-ratio", "0.06"), ("--weight-decay", "0.0")):
        frozen = [x for i, x in enumerate(FROZEN_RECIPE) if FROZEN_RECIPE[i - (i % 2)] != extra[0]]
        assert berturk.run_id(berturk.canonical(_identity(_parsed(*frozen, *extra)))) != rid, extra
    for extra, flag in ((("--warmup-ratio", "1.0"), "--warmup-ratio"), (("--warmup-ratio", "-0.1"), "--warmup-ratio"),
                        (("--weight-decay", "-0.1"), "--weight-decay")):
        with pytest.raises(SystemExit) as e:
            berturk_train.main(["--data", "data/splits/selection_train.jsonl", *extra])
        assert e.value.code == 2 and flag in capsys.readouterr().err, extra


FAKE_CHOSEN = {"epochs": 30, "encoder_lr": 2e-05, "head_lr": 0.0001, "batch": 32, "warmup_ratio": 0.1,
               "weight_decay": 0.01, "seed": 0}
EXPLICIT = {"epochs": ("--epochs", "7", 7), "lr": ("--lr", "3e-05", 3e-05), "head_lr": ("--head-lr", "0.0003", 0.0003),
            "batch": ("--batch", "8", 8), "warmup_ratio": ("--warmup-ratio", "0.2", 0.2),
            "weight_decay": ("--weight-decay", "0.05", 0.05)}


def test_an_explicit_recipe_flag_wins_over_the_context() -> None:
    """Each recipe flag given on the command line survives the resolution, against the tuned defaults and against a
    chosen canonical config; every other flag takes the context's value."""
    for context in (defaults.BERTURK, FAKE_CHOSEN):
        for dest, (flag, text, want) in EXPLICIT.items():
            a = berturk_train.parser().parse_args(["--data", "d", flag, text])
            berturk_train.resolve_recipe(a, context, "the test context")
            assert getattr(a, dest) == want and type(getattr(a, dest)) is type(want), (dest, context)
            for other, key in berturk_train.RECIPE_FLAGS:
                if other != dest:
                    assert getattr(a, other) == context[key], (other, context)


def test_recipe_context_follows_the_command() -> None:
    """recipe_context picks the --postfit record's chosen canonical, then (--fewshot-base) the frozen selection run's,
    then a masked base's canonical, else the tuned defaults; a context without a key refuses naming the key and the
    flag."""
    def parsed(*extra: str) -> Any:
        return berturk_train.parser().parse_args(["--data", "d", *extra])

    record = {"chosen": {"canonical": dict(FAKE_CHOSEN, epochs=29)}}
    base_info = {"run_id": "berturk-selection_train-s0-00000000", "canonical": dict(FAKE_CHOSEN, epochs=28)}
    chosen = dict(FAKE_CHOSEN, epochs=27)
    ctx, what = berturk_train.recipe_context(parsed(), record, None, None)
    assert ctx is record["chosen"]["canonical"] and "--postfit" in what
    ctx, what = berturk_train.recipe_context(parsed("--fewshot-base"), None, chosen, None)
    assert ctx is chosen and "frozen selection run" in what
    ctx, what = berturk_train.recipe_context(parsed("--fewshot-draw", "0", "--fewshot-k", "5"), None, None, base_info)
    assert ctx is base_info["canonical"] and "masked base" in what
    ctx, what = berturk_train.recipe_context(parsed(), None, None, None)
    assert ctx is defaults.BERTURK and "pilot/defaults.py" in what
    with pytest.raises(ValueError, match="--fewshot-base"):
        berturk_train.recipe_context(parsed("--fewshot-base"), None, None, None)
    for dest, key in berturk_train.RECIPE_FLAGS:
        lacking = {k: v for k, v in FAKE_CHOSEN.items() if k != key}
        flag = "--" + dest.replace("_", "-")
        with pytest.raises(SystemExit, match="the test context has no %s to resolve %s from; pass %s"
                                             % (key, flag, flag)):
            berturk_train.resolve_recipe(parsed(), lacking, "the test context")


def test_identity_values_refuses_an_unresolved_recipe() -> None:
    """identity_values and fit refuse a recipe flag still None, naming it: no direct caller hashes a null."""
    a = berturk_train.parser().parse_args(["--data", "data/splits/selection_train.jsonl"])
    with pytest.raises(ValueError, match="--epochs, --lr, --head-lr, --batch, --warmup-ratio, --weight-decay"):
        _identity(a)
    a = _parsed()
    a.warmup_ratio = None
    with pytest.raises(ValueError, match="unresolved recipe flag\\(s\\) --warmup-ratio: call resolve_recipe"):
        _identity(a)
    with pytest.raises(ValueError, match="--warmup-ratio"):
        berturk_train.fit(a, [], [], "", {}, {}, {}, "rid", "run_dir", cpus=1)


OG_NAMES = ["bert.embeddings.word_embeddings.weight", "bert.embeddings.LayerNorm.weight",
            "bert.encoder.layer.0.attention.self.query.weight", "bert.encoder.layer.0.attention.self.query.bias",
            "classifier.weight", "classifier.bias"]
OG_DECAY = [n for n in OG_NAMES if n.endswith(".weight") and "LayerNorm" not in n]


def test_optimizer_groups_continued_mode_trains_the_loaded_head() -> None:
    """continued=True (a continued few-shot run) takes the classifier.* names as the head of a masked base, loaded, so
    nothing may be newly initialised; the same four groups, the head at the head LR."""
    groups = berturk.optimizer_groups(OG_NAMES, OG_DECAY, [], 2e-05, 1e-4, 0.01, continued=True)
    assert [(g["group"], g["decay"], g["lr"], g["weight_decay"]) for g in groups] == [
        ("encoder", True, 2e-05, 0.01), ("encoder", False, 2e-05, 0.0), ("head", True, 1e-4, 0.01),
        ("head", False, 1e-4, 0.0)]
    assert groups[2]["params"] == ["classifier.weight"] and groups[3]["params"] == ["classifier.bias"]
    assert sorted(n for g in groups for n in g["params"]) == sorted(OG_NAMES)
    for new in (["classifier.weight", "classifier.bias"], ["classifier.bias"], ["bert.embeddings.LayerNorm.weight"]):
        with pytest.raises(SystemExit, match="loaded, never newly initialised"):
            berturk.optimizer_groups(OG_NAMES, OG_DECAY, new, 2e-05, 1e-4, 0.01, continued=True)
    with pytest.raises(SystemExit, match="head none"):
        berturk.optimizer_groups(OG_NAMES[:4], OG_DECAY, [], 2e-05, 1e-4, 0.01, continued=True)


def test_optimizer_groups_default_mode_still_needs_a_fresh_head() -> None:
    """continued=False (every non-few-shot run) keeps the first reference-check fix's rule: the head is exactly the
    newly initialised names, so a loaded head (nothing newly initialised) is refused."""
    fresh = ["classifier.bias", "classifier.weight"]
    assert berturk.optimizer_groups(OG_NAMES, OG_DECAY, fresh, 2e-05, 1e-4, 0.01) \
        == berturk.optimizer_groups(OG_NAMES, OG_DECAY, [], 2e-05, 1e-4, 0.01, continued=True)
    for new in ([], ["classifier.weight"]):
        with pytest.raises(SystemExit, match="the classifier head must be exactly the newly initialised parameters"):
            berturk.optimizer_groups(OG_NAMES, OG_DECAY, new, 2e-05, 1e-4, 0.01)


def _checkpoint(d: str) -> str:
    """A fake BERTurk checkpoint dir holding every CHECKPOINT_FILES entry with placeholder bytes; returns d."""
    os.makedirs(d, exist_ok=True)
    for name in berturk.CHECKPOINT_FILES:
        with open(os.path.join(d, name), "wb") as f:
            f.write(("%s bytes\n" % name).encode("utf-8"))
    return d


def test_identity_dump_run_id_binds_every_file(tmp_path: Path) -> None:
    d = _checkpoint(str(tmp_path / "ckpt"))
    berturk.require_files(d, "ckpt")
    ident = berturk.identity(d)
    rid = berturk.dump_run_id(ident)
    assert rid.startswith("berturk-%s-%s-" % (ident["model_sha256"][:12], ident["record_sha256"][:8]))
    seen = {rid}
    for name in berturk.CHECKPOINT_FILES:                    # any one changed file -> another dump dir
        e = _checkpoint(str(tmp_path / ("c-" + name)))
        with open(os.path.join(e, name), "ab") as f:
            f.write(b"x")
        seen.add(berturk.dump_run_id(berturk.identity(e)))
    assert len(seen) == len(berturk.CHECKPOINT_FILES) + 1
    for key in ("model_sha256", "record_sha256", "config_sha256"):
        with pytest.raises(SystemExit, match="lacks a digest"):
            berturk.dump_run_id(dict(ident, **{key: None}))
    with pytest.raises(SystemExit, match="lacks a digest"):
        berturk.dump_run_id(dict(ident, files_sha256={"vocab.txt": "f" * 64}))
    os.remove(os.path.join(d, "vocab.txt"))
    with pytest.raises(SystemExit, match="lacks vocab.txt"):
        berturk.require_files(d, "ckpt")


def _record(**over: Any) -> dict[str, Any]:
    """A berturk_train.json record of a 2-epoch run at epoch 1, `over` applied."""
    canon = berturk.canonical(_values(epochs=2))
    rec = {"schema": berturk.RECORD_SCHEMA, "run_id": berturk.run_id(canon), "canonical": canon, "epoch": 1,
           "epochs_schedule": 2, "stop_after_epoch": None, "calibrated": False, "temperature": 1.0, "postfit": None,
           "record": {"epochs": [dict.fromkeys(berturk.EPOCH_FIELDS, 0) | {"epoch": 1, "loss": 0.5}]}}
    rec.update(over)
    return rec


def test_identity_record_changed_after_hash_is_refused(tmp_path: Path) -> None:
    d = _checkpoint(str(tmp_path / "ckpt"))
    path = os.path.join(d, berturk.RECORD)
    with open(path, "wb") as f:
        f.write(berturk.record_bytes(_record()))
    ident = berturk.identity(d)
    assert berturk.read_record(d, ident)["epoch"] == 1
    with open(path, "wb") as f:
        f.write(berturk.record_bytes(_record(temperature=1.5)))
    with pytest.raises(SystemExit, match="changed after it was hashed"):
        berturk.read_record(d, ident)
    with open(path, "wb") as f:
        f.write(berturk.record_bytes(_record(schema="berturk-train/0")))
    with pytest.raises(SystemExit, match="schema"):
        berturk.read_record(d, berturk.identity(d))


@pytest.mark.parametrize("over, match", [
    ({"temperature": None}, "temperature None"),
    ({"temperature": 0.0}, "temperature 0.0"),
    ({"temperature": -1.5}, "temperature -1.5"),
    ({"temperature": 1}, "temperature 1 "),                  # an int is not the JSON float the record writes
    ({"temperature": True}, "temperature True"),
    ({"canonical": {"seed": 0, "max_len": None}}, "max_len None"),
    ({"canonical": {"seed": 0}}, "max_len None"),
    ({"canonical": {"seed": 0, "max_len": 0}}, "max_len 0"),
    ({"canonical": {"seed": 0, "max_len": 128.0}}, "max_len 128.0"),
])
def test_read_record_refuses_values_the_scorer_cannot_use(tmp_path: Path, over: dict[str, Any], match: str) -> None:
    """A refusal, not a KeyError or TypeError, before the scorer reads max_len and the temperature."""
    d = _checkpoint(str(tmp_path / "ckpt"))
    with open(os.path.join(d, berturk.RECORD), "wb") as f:
        f.write(berturk.record_bytes(_record(**over)))
    with pytest.raises(SystemExit, match=re.escape(match)):
        berturk.read_record(d, berturk.identity(d))


def test_identity_training_block_has_the_laya_keys(tmp_path: Path) -> None:
    """The sidecar training block of a BERTurk dump has exactly the keys score.training_block writes for Laya, the
    ones pilot.postfit.candidate reads."""
    laya = tmp_path / "laya"
    laya.mkdir()
    cfg = {"training": {"epoch": 1, "calibrated": False,
                        "ray_train": {"run_id": "laya-x", "canonical": {"seed": 0, "split": "selection_train"},
                                      "record": {"epochs": []}}}}
    (laya / "rl_agent_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    _, want = score.training_block(str(laya), {"config_sha256": sha256_file(str(laya / "rl_agent_config.json"))})
    seed, tr = berturk.training_block(_record())
    assert set(tr) == set(want)
    assert seed == 0 and tr["epoch"] == 1 and tr["epochs_schedule"] == 2 and tr["loss_by_epoch"] == [0.5]
    assert tr["encoder_lr"] == 2e-05 and tr["head_lr"] == 0.0001 and tr["calibrated"] is False
    assert tr["config_sha256_sans_seed"] == dumps.training_digest(tr["canonical"], 1)


def test_berturk_module_is_torch_free() -> None:
    code = ("import sys, pilot.berturk, berturk_train; "
            "print(sorted(m for m in ('torch', 'transformers', 'accelerate', 'ray', 'laya') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
