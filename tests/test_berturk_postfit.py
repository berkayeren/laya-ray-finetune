"""BERTurk through Laya's own postfit (BASE-02): pilot.postfit.protocol("berturk") and fit() dispatch by system kind
over BERTurk val dumps (1 LR x epochs 1..10, the head LR tied to the encoder LR, Laya's seeds, objective, tie order,
T and θ rules), the val-only refusals for BERTurk input, and the Laya record unchanged by the dispatch.

No model, torch or Ray: candidate val dumps are written with pilot.dumps.write_dump on the synthetic val split
(conftest), with a sidecar shaped like score.py --model berturk's (score.sidecar_base + pilot.berturk.training_block).
"""
import json
import os
import random

import numpy as np
import pytest

import pilot.data
import postfit as postfit_cli
import score
from conftest import prepared
from pilot import berturk, dumps, postfit
from pilot.data import sha256_bytes
from test_postfit import GRID as LAYA_GRID
from test_postfit import write_candidate as write_laya_candidate

pytestmark = pytest.mark.unit

LR = 2e-05
EPOCHS = berturk.EPOCHS
SCORER = {"device": "mps", "dtype": "torch.float32", "amp_enabled": False}


@pytest.fixture
def snap(tmp_path, synthetic_raw):
    """(out_dir, manifest_path, val entry, val uids) of the synthetic snapshot prepared under tmp_path."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    path = os.path.join(out_dir, "splits", "val.jsonl")
    with open(path, "rb") as f:
        raw = f.read()
    entry = pilot.data.verify_split_file(path, manifest_path=manifest, roles=("val",), data=raw)
    return out_dir, manifest, entry, [r["uid"] for r in pilot.data.parse_jsonl(raw)]


def _values(**over):
    """Every pilot.berturk.IDENTITY value of a selection run on the pinned base (digests are placeholders)."""
    v = {k: None for k in berturk.IDENTITY}
    v.update(split="selection_train", data_sha256="s" * 64, data_role="train", base_model_sha256="b" * 64,
             base_config_sha256="c" * 64, transformers_version="5.19.0", accelerate_version="1.15.0", device="mps",
             epochs=EPOCHS, batch=berturk.BATCH, encoder_lr=LR, head_lr=LR, weight_decay=berturk.WEIGHT_DECAY,
             warmup_ratio=berturk.WARMUP_RATIO, lr_scheduler=berturk.LR_SCHEDULER, max_grad_norm=berturk.MAX_GRAD_NORM,
             optim=berturk.OPTIM, loss=berturk.LOSS, max_len=berturk.MAX_LEN, seed=0)
    v.update(over)
    return v


def write_berturk_candidate(root, snap, epoch, lr=LR, seed=0, split="selection_train", calibrated=False, shipped=1.0,
                            n=None, head_lr=None, kind="berturk", logits=None):
    """A BERTurk candidate val dump under root/<dump run id>: random margins (or `logits`), a fake local BERTurk
    identity and the training block pilot.berturk.training_block builds from a berturk_train.json record (canonical
    with split, epochs 10, encoder_lr = head_lr = lr unless head_lr is given, seed). Returns the run dir."""
    _, manifest, entry, uids = snap
    uids = uids if n is None else uids[:n]
    if logits is None:
        rng = np.random.default_rng(int(lr * 1e6) * 1000 + epoch * 10 + seed)
        logits = rng.normal(0.0, 3.0, size=(len(uids), 48))
    tag = "%s-%d-%s-%s-%s-%s-%s-%s" % (lr, epoch, seed, split, calibrated, shipped, head_lr, kind)
    ident = {"local": True, "kind": "berturk", "model_sha256": sha256_bytes(("model " + tag).encode()),
             "config_sha256": sha256_bytes(("config " + tag).encode()),
             "record_sha256": sha256_bytes(("record " + tag).encode()),
             "files_sha256": {f: sha256_bytes(f.encode()) for f in berturk.TOKENIZER_FILES}}
    canon = berturk.canonical(_values(split=split, encoder_lr=lr, head_lr=lr if head_lr is None else head_lr,
                                      seed=seed))
    rec = {"run_id": berturk.run_id(canon), "canonical": canon, "epoch": epoch, "calibrated": calibrated,
           "postfit": None, "record": {"epochs": [{"epoch": k, "loss": 0.5 / k} for k in range(1, epoch + 1)]}}
    side_seed, training = berturk.training_block(rec)
    rid = berturk.dump_run_id(ident)
    fields = {"run_id": rid, "seed": side_seed, "training": training, "logit": berturk.LOGIT_NOTE,
              "system": {"kind": kind, "checkpoint": "berturk_runs/%s/exports/epoch-%03d" % (rec["run_id"], epoch),
                         "checkpoint_identity": ident, "baseline_source": None, "clip_eps": None},
              "temperature": {"shipped": shipped, "source": "berturk_train.json temperature"},
              "scorer": dict(SCORER, workers=0), "timing": {"seconds": 0.0}}
    sidecar = dict(score.sidecar_base(entry, "val", manifest, None if n is None else len(uids)), **fields)
    run_dir = os.path.join(str(root), rid)
    dumps.write_dump(run_dir, "val", np.asarray(logits, dtype=np.float32), uids, sidecar)
    return run_dir


def _ten(tmp_path, snap, root="dumps"):
    return [write_berturk_candidate(tmp_path / root, snap, e) for e in range(1, EPOCHS + 1)]


def _grid_kw(kind):
    return {k: v for k, v in postfit.protocol(kind).items() if k != "final_seeds"}


# --- the BERTurk record ------------------------------------------------------------------------------------------

def test_berturk_record_from_ten_dumps(tmp_path, snap, capsys):
    run_dirs = _ten(tmp_path, snap)
    out = tmp_path / "postfit" / "berturk.json"
    postfit_cli.main(run_dirs + ["--manifest", snap[1], "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    rec = json.loads(text)
    assert json.loads(capsys.readouterr().out) == rec
    assert rec["schema"] == "postfit/1" and rec["system_kind"] == "berturk"
    assert rec["protocol"] == {"selection_split": "selection_train", "final_split": "final_train",
                               "lr_grid": [2e-05], "head_lr": None, "epochs": 10, "selection_seed": 0,
                               "final_seeds": [0, 1, 2]}
    assert "--epochs 10" in rec["rules"]["epoch_transfer"] and "--stop-after-epoch" in rec["rules"]["epoch_transfer"]
    assert rec["rules"]["epoch_transfer"] == postfit.EPOCH_TRANSFER_TEMPLATE % 10
    assert rec["rules"]["objective"] == postfit.OBJECTIVE and rec["rules"]["tie_order"] == postfit.TIE_ORDER
    assert rec["rules"]["temperature"] == postfit.TEMPERATURE_RULE and rec["rules"]["temperature_bounds"] == [0.5, 5.0]
    cands = rec["candidates"]
    assert [(c["encoder_lr"], c["epoch"]) for c in cands] == [(LR, e) for e in range(1, 11)]
    assert all(c["system_kind"] == "berturk" and c["canonical"]["head_lr"] == c["encoder_lr"] for c in cands)
    chosen = rec["chosen"]
    assert chosen["dump_sha256"] in {c["dump_sha256"] for c in cands}
    best = max(postfit.score(c) for c in cands)
    assert postfit.score(next(c for c in cands if c["dump_sha256"] == chosen["dump_sha256"])) == best
    for c in cands + [chosen]:
        assert 0.5 <= c["temperature"] <= 5.0
    assert chosen["stop_after_epoch"] == (None if chosen["epoch"] == 10 else chosen["epoch"])
    assert rec["final"] == {"split": "final_train", "seeds": [0, 1, 2], "epochs": 10,
                            "stop_after_epoch": chosen["stop_after_epoch"]}
    assert os.path.expanduser("~") not in text and str(tmp_path) not in text
    first = postfit.record_bytes(rec)
    assert out.read_bytes() == first
    for seed in range(3):
        shuffled = list(run_dirs)
        random.Random(seed).shuffle(shuffled)
        assert postfit.record_bytes(postfit.fit(shuffled, snap[1])) == first


def test_berturk_grid_equals_laya_grid():
    assert berturk.LR_GRID == postfit.LR_GRID == (2e-05,)            # one LR for both models (equal tuning effort)
    assert berturk.LR in berturk.LR_GRID
    p = postfit.protocol("berturk")
    assert p == {"lr_grid": (2e-05,), "epochs": 10, "head_lr": None, "selection_seed": postfit.SELECTION_SEED,
                 "final_seeds": postfit.FINAL_SEEDS}
    assert p["selection_seed"] == 0 and p["final_seeds"] == (0, 1, 2)
    assert p is not berturk.PROTOCOL                                  # a copy: a caller cannot edit the protocol
    assert postfit.protocol("laya") == {"lr_grid": postfit.LR_GRID, "epochs": postfit.EPOCHS,
                                        "head_lr": postfit.HEAD_LR, "selection_seed": 0, "final_seeds": (0, 1, 2)}


def test_berturk_grid_must_be_complete(tmp_path, snap):
    run_dirs = _ten(tmp_path, snap)
    with pytest.raises(SystemExit, match="missing \\[\\(2e-05, 10\\)\\], unexpected \\[\\]"):
        postfit.fit(run_dirs[:-1], snap[1])                           # 9 epochs
    eleven = write_berturk_candidate(tmp_path / "more", snap, 11)
    with pytest.raises(SystemExit, match="missing \\[\\], unexpected \\[\\(2e-05, 11\\)\\]"):
        postfit.fit(run_dirs + [eleven], snap[1])                     # 11 epochs
    other_lr = write_berturk_candidate(tmp_path / "more", snap, 1, lr=3e-05)
    with pytest.raises(SystemExit, match="unexpected \\[\\(3e-05, 1\\)\\]"):
        postfit.fit(run_dirs + [other_lr], snap[1])                   # a second LR
    with pytest.raises(SystemExit, match="selection grid \\[2e-05\\] x epochs 1..10"):
        postfit_cli.main(run_dirs[1:] + ["--manifest", snap[1]])      # the CLI refuses it the same way


def test_berturk_ties_go_to_fewer_epochs(tmp_path, snap):
    cands = [postfit.candidate(d, snap[1]) for d in _ten(tmp_path, snap)]
    for c in cands:
        c["objective"] = {"tp": 1, "fp": 3, "fn": 3}                  # 1/4 each
    cands[6]["objective"] = {"tp": 4, "fp": 2, "fn": 2}               # epoch 7: 2/3
    cands[2]["objective"] = {"tp": 2, "fp": 1, "fn": 1}               # epoch 3: exactly 2/3 too
    kw = _grid_kw("berturk")
    for seed in range(5):
        shuffled = list(cands)
        random.Random(seed).shuffle(shuffled)
        assert postfit.select(shuffled, **kw)["epoch"] == 3
    rec = postfit.build_record(cands, postfit.select(cands, **kw), snap[1], **postfit.protocol("berturk"))
    assert rec["chosen"]["epoch"] == 3 and rec["chosen"]["stop_after_epoch"] == 3
    assert rec["final"]["stop_after_epoch"] == 3 and rec["final"]["epochs"] == 10
    cands[9]["objective"] = {"tp": 9, "fp": 1, "fn": 1}               # epoch 10 alone best: no stop epoch
    rec = postfit.build_record(cands, postfit.select(cands, **kw), snap[1], **postfit.protocol("berturk"))
    assert rec["chosen"]["epoch"] == 10 and rec["chosen"]["stop_after_epoch"] is None


def test_berturk_temperature_band_bounds(tmp_path, snap):
    """The T rule keeps its band for BERTurk: an out-of-band optimum returns exactly 5.0 or 0.5."""
    _, manifest, entry, uids = snap
    Y = dumps.read_gold(entry, manifest, uids).astype(np.float64)
    noise = np.random.default_rng(1).normal(0.0, 50.0, size=Y.shape)          # loud and uninformative: T -> high
    hi = write_berturk_candidate(tmp_path / "hi", snap, 1, logits=noise)
    lo = write_berturk_candidate(tmp_path / "lo", snap, 1, logits=(2.0 * Y - 1.0) * 0.01)   # faint, exact: T -> low
    assert postfit.candidate(hi, manifest)["temperature"] == postfit.T_MAX == 5.0
    assert postfit.candidate(lo, manifest)["temperature"] == postfit.T_MIN == 0.5


def test_berturk_head_lr_must_be_tied(tmp_path, snap):
    run_dirs = _ten(tmp_path, snap)
    untied = write_berturk_candidate(tmp_path / "untied", snap, 5, head_lr=0.0001)
    dirs = [d for k, d in enumerate(run_dirs, 1) if k != 5] + [untied]
    with pytest.raises(SystemExit, match="canonical head_lr 0.0001 is not its encoder_lr 2e-05"):
        postfit.fit(dirs, snap[1])


def test_fit_refuses_mixed_kinds(tmp_path, snap):
    run_dirs = _ten(tmp_path, snap)
    laya = write_laya_candidate(tmp_path / "laya", snap, LR, 1)
    with pytest.raises(SystemExit, match="mix system kinds \\['berturk', 'laya'\\]"):
        postfit.fit(run_dirs[1:] + [laya], snap[1])
    with pytest.raises(SystemExit, match="mix system kinds"):
        postfit_cli.main(run_dirs + [laya, "--manifest", snap[1]])


def test_fit_refuses_unknown_kind(tmp_path, snap):
    with pytest.raises(SystemExit, match="no postfit protocol for system kind 'fasttext'; expected laya or berturk"):
        postfit.protocol("fasttext")
    run_dirs = [write_berturk_candidate(tmp_path / "dumps", snap, e, kind="fasttext") for e in range(1, EPOCHS + 1)]
    with pytest.raises(SystemExit, match="no postfit protocol for system kind 'fasttext'"):
        postfit.fit(run_dirs, snap[1])


@pytest.mark.parametrize("kw, match", [
    ({"split": "final_train"}, "trained on final_train"),
    ({"calibrated": True}, "calibrated True"),
    ({"shipped": 1.3}, "shipped temperature 1.3"),
    ({"n": 2}, "subset val dump"),
])
def test_berturk_postfit_refuses_test_input(tmp_path, snap, kw, match):
    good = write_berturk_candidate(tmp_path / "dumps", snap, 1)
    assert postfit.candidate(good, snap[1])["system_kind"] == "berturk"    # a full selection val dump is accepted
    bad = write_berturk_candidate(tmp_path / "bad", snap, 2, **kw)
    with pytest.raises(SystemExit, match=match):
        postfit.candidate(bad, snap[1])
    with pytest.raises(SystemExit, match=match):
        postfit.fit(_ten(tmp_path, snap, "ten")[:-1] + [bad], snap[1])


def test_berturk_postfit_refuses_a_test_dump(tmp_path, snap):
    run_dirs = _ten(tmp_path, snap)
    with open(os.path.join(run_dirs[3], "test.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    with pytest.raises(SystemExit, match="holds a test dump"):
        postfit.fit(run_dirs, snap[1])
    with pytest.raises(SystemExit, match="holds a test dump"):
        postfit_cli.main(run_dirs + ["--manifest", snap[1]])
    with pytest.raises(SystemExit, match="is a file"):
        postfit.candidate(os.path.join(snap[0], "splits", "test.jsonl"), snap[1])


# --- the Laya path is unchanged ----------------------------------------------------------------------------------

def test_laya_record_unchanged_by_the_dispatch(tmp_path, snap):
    run_dirs = [write_laya_candidate(tmp_path / "dumps", snap, lr, e) for lr, e in LAYA_GRID]
    assert [(lr, e) for lr, e in LAYA_GRID] == [(lr, e) for lr in postfit.LR_GRID
                                                for e in range(1, postfit.EPOCHS + 1)]
    implicit = postfit.record_bytes(postfit.fit(run_dirs, snap[1]))
    explicit = postfit.record_bytes(postfit.fit(run_dirs, snap[1], **postfit.protocol("laya")))
    constants = postfit.record_bytes(postfit.fit(run_dirs, snap[1], lr_grid=postfit.LR_GRID, epochs=postfit.EPOCHS,
                                                 head_lr=postfit.HEAD_LR, selection_seed=postfit.SELECTION_SEED,
                                                 final_seeds=postfit.FINAL_SEEDS))
    assert implicit == explicit == constants
    rec = json.loads(implicit.decode("utf-8"))
    assert rec["system_kind"] == "laya" and rec["rules"]["epoch_transfer"] == postfit.EPOCH_TRANSFER
    assert postfit.EPOCH_TRANSFER == postfit.EPOCH_TRANSFER_TEMPLATE % postfit.EPOCHS
    assert rec["protocol"]["head_lr"] == postfit.HEAD_LR and rec["protocol"]["epochs"] == postfit.EPOCHS
