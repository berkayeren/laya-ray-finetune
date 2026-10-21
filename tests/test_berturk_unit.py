"""Pure contracts of the BERTurk slice (Phase 6, BASE-01/02): the masked BCE (A7), labels and masks in INTENTS order,
the label-order and casing refusals, the run identity and the checkpoint identity, and the torch-free pilot.berturk.

Offline (unit): the loss tests use torch on CPU tensors only; nothing loads a model.
"""
import json
import os
import subprocess
import sys

import pytest

import berturk_train
import score
from intents import INTENTS
from pilot import berturk, dumps
from pilot.data import REPO_ROOT, sha256_file

pytestmark = pytest.mark.unit

TAGS = list(INTENTS)


# ---- masked BCE (A7) ----------------------------------------------------------------------------------------------

def _batch(n=4, seed=0, p=0.2):
    import torch
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, len(TAGS), generator=g, dtype=torch.float32)
    labels = (torch.rand(n, len(TAGS), generator=g) < p).float()
    return logits, labels


def _loss(logits, labels, n=None):
    return berturk_train.masked_bce({"logits": logits}, labels, n)


def test_masked_bce_equals_mean_bce_when_unmasked():
    import torch
    logits, labels = _batch()
    want = torch.nn.BCEWithLogitsLoss()(logits, labels)
    assert abs(float(_loss(logits, labels)) - float(want)) < 1e-6
    n = torch.tensor(labels.numel())                         # what the Trainer counts: labels != -100
    assert abs(float(_loss(logits, labels, n)) - float(want)) < 1e-6


def test_masked_tag_gets_zero_gradient():
    logits, labels = _batch()
    labels[:, 3] = berturk.MASK_LABEL
    logits.requires_grad_(True)
    _loss(logits, labels).backward()
    assert (logits.grad[:, 3] == 0).all()                    # exactly zero, not merely small
    assert (logits.grad[:, [0, 1, 2, 4]] != 0).all()


def test_all_masked_batch_is_finite_zero():
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


def test_loss_divides_by_num_items_across_micro_batches():
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


def test_loss_int_and_tensor_num_items_agree():
    import torch
    logits, labels = _batch()
    assert float(_loss(logits, labels, 100)) == float(_loss(logits, labels, torch.tensor(100)))


def test_loss_ignores_masked_logits_and_label_values():
    """A masked entry contributes nothing: changing its logit leaves the loss value unchanged."""
    logits, labels = _batch()
    labels[1, 7] = berturk.MASK_LABEL
    before = float(_loss(logits, labels))
    logits[1, 7] = 50.0
    assert float(_loss(logits, labels)) == before


def test_loss_gets_the_trainers_unmasked_label_count(tmp_path):
    """The transformers contract A7 rests on: with compute_loss_func set, the Trainer pops the labels and passes
    num_items_in_batch = the count of labels != -100 over the whole accumulation window (2 micro-batches here)."""
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

    def loss_fn(outputs, labels, num_items_in_batch=None):
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


def test_masked_bce_refuses_non_fp32_logits():
    import torch
    logits, labels = _batch()
    for dtype in (torch.float16, torch.bfloat16):
        with pytest.raises(SystemExit, match="fp32"):
            _loss(logits.to(dtype), labels)


# ---- labels and masks ---------------------------------------------------------------------------------------------

def _row(pos=("card", "lost_stolen"), uid="bank_0_0"):
    return {"uid": uid, "expected": {t: t in pos for t in TAGS}}


def test_label_vector_follows_intents_order():
    vec = berturk.label_vector(_row())
    assert len(vec) == len(TAGS)
    assert [t for t, v in zip(TAGS, vec) if v == 1.0] == [t for t in TAGS if t in ("card", "lost_stolen")]
    assert set(vec) == {0.0, 1.0}
    masked = berturk.label_vector(_row(), ("card", "affirm"))
    assert masked[TAGS.index("card")] == masked[TAGS.index("affirm")] == berturk.MASK_LABEL
    assert masked[TAGS.index("lost_stolen")] == 1.0
    assert berturk.id2label() == dict(enumerate(TAGS))
    assert berturk.label2id() == {t: i for i, t in enumerate(TAGS)}


def test_label_vector_refuses_a_missing_tag():
    row = _row(uid="bank_7_3")
    del row["expected"]["pin"]
    with pytest.raises(SystemExit, match="bank_7_3.*pin"):
        berturk.label_vector(row)
    assert berturk.label_vector(row, ("pin",))[TAGS.index("pin")] == berturk.MASK_LABEL   # masked: never read


def test_mask_tags_are_validated():
    assert berturk.check_mask_tags(["pin", "card"]) == ("card", "pin")
    assert berturk.check_mask_tags([]) == ()
    with pytest.raises(SystemExit, match="unknown.*not_a_tag"):
        berturk.check_mask_tags(["card", "not_a_tag"])
    with pytest.raises(SystemExit, match="card given more than once"):
        berturk.check_mask_tags(["card", "pin", "card"])


def _cfg():
    return {"id2label": {str(i): t for i, t in enumerate(TAGS)}, "label2id": {t: i for i, t in enumerate(TAGS)},
            "problem_type": berturk.PROBLEM_TYPE}


def test_label_order_refuses_a_permutation():
    berturk.check_label_order(_cfg(), "ok")                  # the config berturk_train.py writes
    swapped = _cfg()
    swapped["id2label"]["0"], swapped["id2label"]["1"] = TAGS[1], TAGS[0]
    wrong_inverse = _cfg()
    wrong_inverse["label2id"][TAGS[0]] = 5
    for bad in (swapped, wrong_inverse, dict(_cfg(), problem_type="single_label_classification"),
                dict(_cfg(), id2label={str(i): "LABEL_%d" % i for i in range(len(TAGS))}), {}):
        with pytest.raises(SystemExit, match="INTENTS order"):
            berturk.check_label_order(bad, "ckpt")


def test_check_cased_refuses_a_lowercasing_tokenizer():
    berturk.check_cased({"do_lower_case": False, "max_len": 512, "init_inputs": []}, "pinned")
    for bad in ({"do_lower_case": True}, {}, {"do_lower_case": 0}, []):
        with pytest.raises(SystemExit, match="do_lower_case"):
            berturk.check_cased(bad, "base")


# ---- run identity -------------------------------------------------------------------------------------------------

def _values(**over):
    v = {"split": "selection_train", "data_sha256": "a" * 64, "data_role": "train", "base_model_sha256": "b" * 64,
         "base_config_sha256": "c" * 64, "base_files_sha256": None, "transformers_version": "5.19.0",
         "accelerate_version": "1.15.0", "device": "mps", "epochs": 10, "batch": 32, "encoder_lr": 2e-05,
         "head_lr": 0.0001, "weight_decay": 0.01, "warmup_ratio": 0.1, "lr_scheduler": "linear", "max_grad_norm": 1.0,
         "optim": "adamw_torch", "loss": "masked_bce", "max_len": 128, "max_messages": None, "mask_tags": None,
         "seed": 0, "stop_after_epoch": None, "postfit_sha256": None}
    v.update(over)
    return v


def test_identity_classifies_every_flag():
    dests = [act.dest for act in berturk_train.parser()._actions if act.dest != "help"]
    for d in dests:
        homes = [d in berturk.IDENTITY, d in berturk.NOT_IDENTITY, d in berturk.RESOLVED]
        assert sum(homes) == 1, "%s is in %d of IDENTITY, NOT_IDENTITY, RESOLVED" % (d, sum(homes))
    assert all(k in berturk.IDENTITY for keys in berturk.RESOLVED.values() for k in keys)
    assert set(berturk.DROPPED_WHEN_NONE) <= set(berturk.IDENTITY)
    assert not set(berturk.NOT_IDENTITY) & set(berturk.IDENTITY)


def test_identity_resolves_lr_to_the_encoder_only():
    """The BASE-03 fix: --lr is the encoder LR only; the head LR is its own flag and IDENTITY key, so two runs that
    differ only in it never share a run dir."""
    assert berturk.RESOLVED == {"lr": ("encoder_lr",)}
    dests = {act.dest: act for act in berturk_train.parser()._actions}
    assert "head_lr" in dests and "head_lr" in berturk.IDENTITY and dests["head_lr"].default == berturk.HEAD_LR
    assert dests["lr"].default == berturk.LR
    ids = {berturk.run_id(berturk.canonical(_values(head_lr=h))) for h in (berturk.HEAD_LR, 2e-05)}
    assert len(ids) == 2


def _superseded_canonical(entry):
    """The chosen canonical config of a SUPERSEDED entry's committed postfit record."""
    with open(os.path.join(REPO_ROOT, entry["record"]), encoding="utf-8") as f:
        return json.load(f)["chosen"]["canonical"]


RECIPE_KEYS = ("encoder_lr", "batch", "max_len", "weight_decay", "warmup_ratio", "lr_scheduler", "max_grad_norm",
               "optim", "loss", "seed")


def _recipe():
    """The current recipe constants other than the epoch grid and the head LR, keyed like the canonical config."""
    return {"encoder_lr": berturk.LR, "batch": berturk.BATCH, "max_len": berturk.MAX_LEN,
            "weight_decay": berturk.WEIGHT_DECAY, "warmup_ratio": berturk.WARMUP_RATIO,
            "lr_scheduler": berturk.LR_SCHEDULER, "max_grad_norm": berturk.MAX_GRAD_NORM, "optim": berturk.OPTIM,
            "loss": berturk.LOSS, "seed": 0}


def test_head_lr_is_the_only_change_from_the_superseded_run():
    """RP-3 (plan 06-14): the head-LR run's canonical config is the tied run's with its recorded delta (head_lr
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


def test_epochs_is_the_only_change_from_the_head_lr_run():
    """R2-3 (plan 06-18, the second BASE-03 fix): the 30-epoch selection command's canonical config is the head-LR
    run's (SUPERSEDED[-1]) with its recorded delta, epochs EPOCHS = 30, and nothing else, so its run id is known offline
    (checked again when plan 06-19 trains it); every other recipe constant equals the head-LR run's canonical value."""
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
    assert dests["epochs"].default == berturk.EPOCHS


def test_identity_canonical_refuses_missing_and_unexpected_keys():
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


def test_identity_run_id_is_order_independent():
    canon = berturk.canonical(_values())
    rid = berturk.run_id(canon)
    assert rid == berturk.run_id(dict(reversed(list(canon.items()))))
    assert rid.startswith("berturk-selection_train-s0-") and len(rid.rsplit("-", 1)[1]) == 8
    assert berturk.run_id(berturk.canonical(_values(epochs=4))) != rid
    assert berturk.run_id(berturk.canonical(_values(mask_tags=["card"]))) != rid


def test_identity_seeds_never_share_a_run_dir():
    ids = [berturk.run_id(berturk.canonical(_values(seed=s))) for s in (0, 1, 2)]
    assert len(set(ids)) == 3
    assert [i.split("-")[2] for i in ids] == ["s0", "s1", "s2"]


def _checkpoint(d):
    os.makedirs(d, exist_ok=True)
    for name in berturk.CHECKPOINT_FILES:
        with open(os.path.join(d, name), "wb") as f:
            f.write(("%s bytes\n" % name).encode("utf-8"))
    return d


def test_identity_dump_run_id_binds_every_file(tmp_path):
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


def _record(**over):
    canon = berturk.canonical(_values(epochs=2))
    rec = {"schema": berturk.RECORD_SCHEMA, "run_id": berturk.run_id(canon), "canonical": canon, "epoch": 1,
           "epochs_schedule": 2, "stop_after_epoch": None, "calibrated": False, "temperature": 1.0, "postfit": None,
           "record": {"epochs": [dict.fromkeys(berturk.EPOCH_FIELDS, 0) | {"epoch": 1, "loss": 0.5}]}}
    rec.update(over)
    return rec


def test_identity_record_changed_after_hash_is_refused(tmp_path):
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


def test_identity_training_block_has_the_laya_keys(tmp_path):
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


def test_berturk_module_is_torch_free():
    code = ("import sys, pilot.berturk, berturk_train; "
            "print(sorted(m for m in ('torch', 'transformers', 'accelerate', 'ray', 'laya') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
