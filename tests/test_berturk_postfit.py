"""BERTurk through Laya's own postfit (BASE-02): pilot.postfit.protocol("berturk") and fit() dispatch by system kind
over BERTurk val dumps (1 LR x epochs 1..30, pilot.berturk.EPOCHS since the second BASE-03 fix of the plan 06-16 gate;
the head LR fixed at Laya's HEAD_LR 1e-4 since the first; Laya's seeds, objective, tie order, T and θ rules), the
30-epoch schedule through the record and the printed final commands, the superseded 10-epoch protocols kept with their
own literals, the val-only refusals for BERTurk input, the Laya record unchanged by the dispatch, and the guarded prune
of BERTurk exports through postfit.py --prune.

Candidate val dumps are written with pilot.dumps.write_dump on the synthetic val split (conftest), with a sidecar shaped
like score.py --model berturk's (score.sidecar_base + pilot.berturk.training_block): no model, torch or Ray. Only the
prune tests train a tiny export with berturk_train.py on CPU and score it with score.py (seconds).
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
                            n=None, head_lr=postfit.HEAD_LR, kind="berturk", logits=None, epochs=EPOCHS):
    """A BERTurk candidate val dump under root/<dump run id>: random margins (or `logits`), a fake local BERTurk
    identity and the training block pilot.berturk.training_block builds from a berturk_train.json record (canonical
    with split, epochs (default EPOCHS = pilot.berturk.EPOCHS; a superseded-protocol candidate passes its protocol's
    epochs), encoder_lr lr, head_lr (default Laya's pilot.postfit.HEAD_LR = pilot.berturk.HEAD_LR, the BASE-03 fix; a
    test that wants the superseded tied head passes head_lr=lr), seed). Returns the run dir."""
    _, manifest, entry, uids = snap
    uids = uids if n is None else uids[:n]
    if logits is None:
        rng = np.random.default_rng(int(lr * 1e6) * 1000 + epoch * 10 + seed)
        logits = rng.normal(0.0, 3.0, size=(len(uids), 48))
    tag = "%s-%d-%s-%s-%s-%s-%s-%s-%d" % (lr, epoch, seed, split, calibrated, shipped, head_lr, kind, epochs)
    ident = {"local": True, "kind": "berturk", "model_sha256": sha256_bytes(("model " + tag).encode()),
             "config_sha256": sha256_bytes(("config " + tag).encode()),
             "record_sha256": sha256_bytes(("record " + tag).encode()),
             "files_sha256": {f: sha256_bytes(f.encode()) for f in berturk.TOKENIZER_FILES}}
    canon = berturk.canonical(_values(split=split, encoder_lr=lr, head_lr=head_lr, seed=seed, epochs=epochs))
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


def _all_epochs(tmp_path, snap, root="dumps"):
    """One candidate per epoch 1..EPOCHS (the current protocol's full grid)."""
    return [write_berturk_candidate(tmp_path / root, snap, e) for e in range(1, EPOCHS + 1)]


def _grid_kw(kind):
    return {k: v for k, v in postfit.protocol(kind).items() if k != "final_seeds"}


# --- the BERTurk record ------------------------------------------------------------------------------------------

def test_berturk_record_from_the_30_epoch_grid(tmp_path, snap, capsys):
    run_dirs = _all_epochs(tmp_path, snap)
    out = tmp_path / "postfit" / "berturk.json"
    postfit_cli.main(run_dirs + ["--manifest", snap[1], "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    rec = json.loads(text)
    assert json.loads(capsys.readouterr().out) == rec
    assert rec["schema"] == "postfit/1" and rec["system_kind"] == "berturk"
    assert rec["protocol"] == {"selection_split": "selection_train", "final_split": "final_train",
                               "lr_grid": [2e-05], "head_lr": 0.0001, "epochs": 30, "selection_seed": 0,
                               "final_seeds": [0, 1, 2]}
    assert "--epochs 30" in rec["rules"]["epoch_transfer"] and "--stop-after-epoch" in rec["rules"]["epoch_transfer"]
    assert rec["rules"]["epoch_transfer"] == postfit.EPOCH_TRANSFER_TEMPLATE % 30
    assert rec["rules"]["objective"] == postfit.OBJECTIVE and rec["rules"]["tie_order"] == postfit.TIE_ORDER
    assert rec["rules"]["temperature"] == postfit.TEMPERATURE_RULE and rec["rules"]["temperature_bounds"] == [0.5, 5.0]
    cands = rec["candidates"]
    assert [(c["encoder_lr"], c["epoch"]) for c in cands] == [(LR, e) for e in range(1, 31)]
    assert all(c["system_kind"] == "berturk" and c["canonical"]["head_lr"] == berturk.HEAD_LR for c in cands)
    chosen = rec["chosen"]
    assert chosen["dump_sha256"] in {c["dump_sha256"] for c in cands}
    best = max(postfit.score(c) for c in cands)
    assert postfit.score(next(c for c in cands if c["dump_sha256"] == chosen["dump_sha256"])) == best
    for c in cands + [chosen]:
        assert 0.5 <= c["temperature"] <= 5.0
    assert chosen["stop_after_epoch"] == (None if chosen["epoch"] == 30 else chosen["epoch"])
    assert rec["final"] == {"split": "final_train", "seeds": [0, 1, 2], "epochs": 30,
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
    assert p == {"lr_grid": (2e-05,), "epochs": 30, "head_lr": 0.0001, "selection_seed": postfit.SELECTION_SEED,
                 "final_seeds": postfit.FINAL_SEEDS}
    assert p["selection_seed"] == 0 and p["final_seeds"] == (0, 1, 2)
    assert p is not berturk.PROTOCOL                                  # a copy: a caller cannot edit the protocol
    assert postfit.protocol("laya") == {"lr_grid": postfit.LR_GRID, "epochs": postfit.EPOCHS,
                                        "head_lr": postfit.HEAD_LR, "selection_seed": 0, "final_seeds": (0, 1, 2)}


def test_berturk_and_laya_share_both_lrs():
    """Equal tuning: since the BASE-03 fix BERTurk and Laya share the encoder LR grid and the fixed head LR."""
    b, laya = postfit.protocol("berturk"), postfit.protocol("laya")
    assert b["lr_grid"] == laya["lr_grid"] == postfit.LR_GRID
    assert b["head_lr"] == laya["head_lr"] == postfit.HEAD_LR == berturk.HEAD_LR == 0.0001


def test_berturk_epoch_grid_is_the_second_base03_fix(tmp_path, snap):
    """The second BASE-03 fix (plan 06-16 gate reply "fix: extend epochs (Recommended)", R2-3): BERTurk's selection
    schedule is 30 epochs, exactly the delta the superseded head-LR run recorded; nothing else in the protocol moved,
    and a full 1..10 grid (the superseded protocol's) is no longer a complete grid."""
    last = berturk.SUPERSEDED[-1]
    assert berturk.EPOCHS == berturk.PROTOCOL["epochs"] == last["delta"]["epochs"] == 30
    assert postfit.protocol("berturk")["epochs"] == 30 and last["protocol"]["epochs"] == 10
    moved = dict(last["protocol"], **last["delta"])
    assert json.dumps(moved, sort_keys=True) == json.dumps(berturk.PROTOCOL, sort_keys=True)
    old = [write_berturk_candidate(tmp_path / "grid10", snap, e, epochs=last["protocol"]["epochs"])
           for e in range(1, last["protocol"]["epochs"] + 1)]
    with pytest.raises(SystemExit, match="not the selection grid \\[2e-05\\] x epochs 1..30: missing "
                                         "\\[\\(2e-05, 11\\), "):
        postfit.fit(old, snap[1])


def test_final_command_follows_the_30_epoch_schedule(tmp_path, snap):
    """The printed final-seed argv carries the 30-epoch selection schedule from the record (--epochs 30) and the
    chosen stop epoch: --stop-after-epoch 12 for a lone best epoch 12, no stop flag when epoch 30 is chosen."""
    cands = [postfit.candidate(d, snap[1]) for d in _all_epochs(tmp_path, snap)]

    def command(best):
        for c in cands:
            c["objective"] = {"tp": 9, "fp": 1, "fn": 1} if c["epoch"] == best else {"tp": 1, "fp": 3, "fn": 3}
        rec = postfit.build_record(cands, postfit.select(cands, **_grid_kw("berturk")), snap[1],
                                   **postfit.protocol("berturk"))
        argv = berturk.final_command(rec, 0, "results/postfit/berturk.json", "results/postfit/berturk-reference.json",
                                     "berturk_runs", "ckpt/berturk-final", base="tiny-base")   # a dir: no pin check
        return rec, argv

    rec, argv = command(12)
    assert argv[argv.index("--epochs") + 1] == "30"
    assert rec["chosen"]["epoch"] == 12 and argv[argv.index("--stop-after-epoch") + 1] == "12"
    assert rec["final"] == {"split": "final_train", "seeds": [0, 1, 2], "epochs": 30, "stop_after_epoch": 12}
    rec, argv = command(30)
    assert argv[argv.index("--epochs") + 1] == "30"
    assert rec["chosen"]["epoch"] == 30 and rec["chosen"]["stop_after_epoch"] is None
    assert "--stop-after-epoch" not in argv and rec["final"]["stop_after_epoch"] is None


def test_superseded_protocol_still_ties_the_head_lr(tmp_path, snap):
    """The superseded protocol keeps its tied head LR (head_lr None) and its own 10-epoch grid (a literal, never
    pilot.berturk.EPOCHS), so its history record still selects and reproduces; under it an untied candidate is
    refused."""
    kw = {k: v for k, v in berturk.SUPERSEDED[0]["protocol"].items() if k != "final_seeds"}
    assert kw["head_lr"] is None and kw["epochs"] == 10
    tied = [postfit.candidate(write_berturk_candidate(tmp_path / "tied", snap, e, head_lr=LR, epochs=kw["epochs"]),
                              snap[1]) for e in range(1, kw["epochs"] + 1)]
    assert postfit.select(tied, **kw)["dump_sha256"] in {c["dump_sha256"] for c in tied}
    untied = postfit.candidate(write_berturk_candidate(tmp_path / "untied", snap, 5, epochs=kw["epochs"]), snap[1])
    with pytest.raises(SystemExit, match="is not its encoder_lr"):
        postfit.select([c for c in tied if c["epoch"] != 5] + [untied], **kw)


def test_berturk_grid_must_be_complete(tmp_path, snap):
    run_dirs = _all_epochs(tmp_path, snap)
    with pytest.raises(SystemExit, match="missing \\[\\(2e-05, 30\\)\\], unexpected \\[\\]"):
        postfit.fit(run_dirs[:-1], snap[1])                           # 29 epochs
    extra = write_berturk_candidate(tmp_path / "more", snap, 31)
    with pytest.raises(SystemExit, match="missing \\[\\], unexpected \\[\\(2e-05, 31\\)\\]"):
        postfit.fit(run_dirs + [extra], snap[1])                      # 31 epochs
    other_lr = write_berturk_candidate(tmp_path / "more", snap, 1, lr=3e-05)
    with pytest.raises(SystemExit, match="unexpected \\[\\(3e-05, 1\\)\\]"):
        postfit.fit(run_dirs + [other_lr], snap[1])                   # a second LR
    with pytest.raises(SystemExit, match="selection grid \\[2e-05\\] x epochs 1..30"):
        postfit_cli.main(run_dirs[1:] + ["--manifest", snap[1]])      # the CLI refuses it the same way


def test_berturk_ties_go_to_fewer_epochs(tmp_path, snap):
    cands = [postfit.candidate(d, snap[1]) for d in _all_epochs(tmp_path, snap)]
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
    assert rec["final"]["stop_after_epoch"] == 3 and rec["final"]["epochs"] == 30
    cands[-1]["objective"] = {"tp": 9, "fp": 1, "fn": 1}              # the last epoch alone best: no stop epoch
    rec = postfit.build_record(cands, postfit.select(cands, **kw), snap[1], **postfit.protocol("berturk"))
    assert rec["chosen"]["epoch"] == 30 and rec["chosen"]["stop_after_epoch"] is None


def test_berturk_temperature_band_bounds(tmp_path, snap):
    """The T rule keeps its band for BERTurk: an out-of-band optimum returns exactly 5.0 or 0.5."""
    _, manifest, entry, uids = snap
    Y = dumps.read_gold(entry, manifest, uids).astype(np.float64)
    noise = np.random.default_rng(1).normal(0.0, 50.0, size=Y.shape)          # loud and uninformative: T -> high
    hi = write_berturk_candidate(tmp_path / "hi", snap, 1, logits=noise)
    lo = write_berturk_candidate(tmp_path / "lo", snap, 1, logits=(2.0 * Y - 1.0) * 0.01)   # faint, exact: T -> low
    assert postfit.candidate(hi, manifest)["temperature"] == postfit.T_MAX == 5.0
    assert postfit.candidate(lo, manifest)["temperature"] == postfit.T_MIN == 0.5


def test_berturk_head_lr_must_be_the_laya_head_lr(tmp_path, snap):
    """The BASE-03 fix: under the BERTurk protocol a candidate whose head LR is tied to its encoder LR (the
    superseded recipe) is refused."""
    run_dirs = _all_epochs(tmp_path, snap)
    tied = write_berturk_candidate(tmp_path / "tied", snap, 5, head_lr=LR)
    dirs = [d for k, d in enumerate(run_dirs, 1) if k != 5] + [tied]
    with pytest.raises(SystemExit, match="canonical head_lr 2e-05, expected 0.0001"):
        postfit.fit(dirs, snap[1])
    with pytest.raises(SystemExit, match="canonical head_lr 2e-05, expected 0.0001"):
        postfit_cli.main(dirs + ["--manifest", snap[1]])


def test_fit_refuses_mixed_kinds(tmp_path, snap):
    run_dirs = _all_epochs(tmp_path, snap)
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
        postfit.fit(_all_epochs(tmp_path, snap, "all")[:-1] + [bad], snap[1])


def test_berturk_postfit_refuses_a_test_dump(tmp_path, snap):
    run_dirs = _all_epochs(tmp_path, snap)
    with open(os.path.join(run_dirs[3], "test.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    with pytest.raises(SystemExit, match="holds a test dump"):
        postfit.fit(run_dirs, snap[1])
    with pytest.raises(SystemExit, match="holds a test dump"):
        postfit_cli.main(run_dirs + ["--manifest", snap[1]])
    with pytest.raises(SystemExit, match="is a file"):
        postfit.candidate(os.path.join(snap[0], "splits", "test.jsonl"), snap[1])


# --- guarded prune of BERTurk exports (a tiny export trained by berturk_train.py on CPU) ---------------------------

@pytest.fixture
def berturk_export(tmp_path, snap):
    """(export dir <storage>/<run_id>/exports/epoch-001, dump root, manifest, val.jsonl, tiny base dir): a 1-epoch
    berturk_train.py run on the tiny offline base, on CPU. Scoring sets torch's thread count; it is restored after."""
    import subprocess
    import sys
    import torch
    from tiny_berturk import tiny_base
    threads = torch.get_num_threads()
    out_dir, manifest = snap[0], snap[1]
    base = tiny_base(tmp_path / "base")
    storage = tmp_path / "storage"
    argv = [sys.executable, "berturk_train.py", "--data", os.path.join(out_dir, "splits", "selection_train.jsonl"),
            "--manifest", manifest, "--base", base, "--device", "cpu", "--epochs", "1", "--batch", "4", "--max-len",
            "32", "--seed", "0", "--storage", str(storage)]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run(argv, cwd=pilot.data.REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    (run,) = os.listdir(storage)
    path = str(storage / run / "exports" / "epoch-001")
    assert os.path.isfile(os.path.join(path, berturk.RECORD))
    yield path, str(tmp_path / "dumps"), manifest, os.path.join(out_dir, "splits", "val.jsonl"), base
    torch.set_num_threads(threads)


def _score_berturk_val(export_dir, dump_root, manifest, val_jsonl, *extra):
    return score.main([val_jsonl, "--model", "berturk", "--checkpoint", export_dir, "--workers", "0",
                       "--cpus-per-worker", "1", "--manifest", manifest, "--out-root", dump_root, *extra])


@pytest.mark.timeout(600)
def test_prune_deletes_a_berturk_export_only_after_its_val_dump(berturk_export, capsys):
    path, dump_root, manifest, val_jsonl, _ = berturk_export
    with pytest.raises(SystemExit, match="no verified full val dump"):
        postfit_cli.main(["--prune", path, "--dump-root", dump_root, "--manifest", manifest])
    assert os.path.isdir(path)
    d, _ = _score_berturk_val(path, dump_root, manifest, val_jsonl)
    want = berturk.dump_run_id(berturk.identity(path))
    assert os.path.basename(d) == want
    capsys.readouterr()
    rid = postfit_cli.main(["--prune", path, "--dump-root", dump_root, "--manifest", manifest])[0]
    assert rid == want and not os.path.exists(path)
    assert os.path.isdir(os.path.dirname(path))                    # only the epoch dir is deleted
    run = os.path.basename(os.path.dirname(os.path.dirname(path)))
    assert capsys.readouterr().out.strip() == "pruned %s/exports/epoch-001 (val dump %s)" % (run, rid)
    _, _, side, _ = dumps.load_dump(os.path.join(dump_root, rid), "val", manifest)   # the dump stays and verifies
    assert side["system"]["kind"] == "berturk"


@pytest.mark.timeout(600)
def test_prune_refuses_a_berturk_export_without_a_dump_or_through_a_symlink(tmp_path, berturk_export):
    path, dump_root, manifest, val_jsonl, base = berturk_export
    with pytest.raises(SystemExit, match="no verified full val dump"):
        berturk.prune_export(path, dump_root, manifest)
    link = tmp_path / "link-epoch-001"
    os.symlink(path, link)
    with pytest.raises(SystemExit, match="symlink"):
        berturk.prune_export(str(link), dump_root, manifest)
    with pytest.raises(SystemExit, match="symlink"):
        postfit_cli.main(["--prune", str(link), "--dump-root", dump_root, "--manifest", manifest])
    linked_exports = tmp_path / "storage2" / "berturk-y" / "exports"
    linked_exports.parent.mkdir(parents=True)
    os.symlink(os.path.dirname(path), linked_exports)
    with pytest.raises(SystemExit, match="symlink"):
        berturk.prune_export(str(linked_exports / "epoch-001"), dump_root, manifest)
    with pytest.raises(SystemExit, match="not a berturk_train.py export"):
        berturk.prune_export(base, dump_root, manifest)            # the tiny base, outside an exports/ dir
    with pytest.raises(SystemExit, match="not a ray_train export"):
        postfit_cli.main(["--prune", base, "--dump-root", dump_root, "--manifest", manifest])   # no RECORD: Laya's
    with pytest.raises(SystemExit, match="not a berturk_train.py export"):
        berturk.prune_export(os.path.join(path, berturk.RECORD), dump_root, manifest)
    _score_berturk_val(path, dump_root, manifest, val_jsonl, "--n", "2")   # a subset val dump at the export's run id
    with pytest.raises(SystemExit, match="subset dump or another checkpoint"):
        berturk.prune_export(path, dump_root, manifest)
    with pytest.raises(SystemExit, match="subset dump or another checkpoint"):
        postfit_cli.main(["--prune", path, "--dump-root", dump_root, "--manifest", manifest])
    for p in (path, link, base):
        assert os.path.isdir(p)                                    # every refusal left its target in place


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


# --- the BASE-03 reference check (fixed before any BERTurk val score) -----------------------------------------------

DIFFERENCES = [
    "a fixed split (selection train folds 0–13 with about 1,450 messages; final train folds 0–15 with about 1,658)",
    "a different test portion (folds 16–19)",
    "a fine-tuned BERTurk rather than a frozen encoder",
    "a single split rather than cross-validation averages",
    "this check itself reads val folds 14-15",
]


def _record(tmp_path, snap, counts=None):
    """(path, record, sha) of a BERTurk postfit record from the full grid of dumps; counts=(tp, fp, fn) replaces the chosen
    candidate's objective counts (the record is rewritten, so its sha is of the written bytes)."""
    rec = postfit.fit(_all_epochs(tmp_path, snap), snap[1])
    if counts is not None:
        c = next(c for c in rec["candidates"] if c["dump_sha256"] == rec["chosen"]["dump_sha256"])
        c["objective"] = dict(zip(("tp", "fp", "fn"), counts))
    path = str(tmp_path / "postfit" / "berturk.json")
    postfit.write_record(rec, path)
    record, sha = postfit.read_record(path)
    return path, record, sha


@pytest.mark.parametrize("counts, status, fraction", [
    ((707, 293, 293), "ok", [707, 1000]),                    # exactly 0.707 passes
    ((706, 294, 294), "investigate", [353, 500]),            # 0.706
    ((706999, 293001, 293001), "investigate", [706999, 1000000]),   # 0.706999: no rounding to 0.707
    ((708, 292, 292), "ok", [177, 250]),
])
def test_reference_threshold_is_exact_at_0_707(tmp_path, snap, counts, status, fraction):
    path, record, sha = _record(tmp_path, snap, counts)
    out = berturk.reference_check(record, path, sha)
    assert out["status"] == status and out["below_threshold"] is (status == "investigate")
    assert out["value"]["fraction"] == fraction
    assert out["threshold"] == 0.707 and berturk.THRESHOLD == __import__("fractions").Fraction(707, 1000)
    assert out["resolution"] is None


def test_reference_quotes_the_source_and_differences(tmp_path, snap):
    path, record, sha = _record(tmp_path, snap)
    out = berturk.reference_check(record, path, sha)
    assert out["schema"] == berturk.REFERENCE_SCHEMA == "berturk-reference/1"
    assert out["postfit"] == {"path": pilot.data.repo_relpath(path), "sha256": sha}
    ref = out["reference"]
    assert ref == berturk.REFERENCE and ref["url"] == "https://arxiv.org/abs/2212.10455"
    assert ref["metric"] == "micro-F1" and ref["lower_reference"] == 80.7
    t12 = ref["table_12"]
    assert "Turkish" in t12["language"] and t12["domain"] == "BANKING" and "9 of 10 folds" in t12["setting"]
    assert [(v["micro_f1"], v["theta"], v["encoder"]) for v in t12["values"]] == [(82.8, 0.3, "frozen"),
                                                                                    (80.7, 0.3, "frozen")]
    assert ["LaBSE" in t12["values"][0]["model"], "mpnet" in t12["values"][1]["model"]] == [True, True]
    assert ref["table_18"]["micro_f1"] == 90.1 and "mDeBERTa" in ref["table_18"]["model"]
    assert "Moghe" in ref["paper"] and "2023" in ref["venue"]
    assert out["threshold"] == 0.707 and "exactly 0.707 passes" in out["threshold_rule"]
    assert out["differences"] == DIFFERENCES == berturk.DIFFERENCES
    assert out["caveats"] == berturk.CAVEATS
    assert any("same val messages" in c and "in-sample" in c for c in out["caveats"])
    chosen = record["chosen"]
    cand = next(c for c in record["candidates"] if c["dump_sha256"] == chosen["dump_sha256"])
    v = out["value"]
    assert v["metric"] == postfit.OBJECTIVE
    assert v["val_micro_f1_at_global_theta"] == cand["val_micro_f1_at_global_theta"]
    assert v["val_micro_f1_at_0_5"] == cand["val_micro_f1_at_0_5"]
    assert (v["epoch"], v["encoder_lr"], v["temperature"], v["train_run_id"], v["dump_run_id"]) == (
        chosen["epoch"], chosen["encoder_lr"], chosen["temperature"], chosen["train_run_id"], chosen["dump_run_id"])
    o = cand["objective"]
    f = __import__("fractions").Fraction(2 * o["tp"], 2 * o["tp"] + o["fp"] + o["fn"])
    assert v["fraction"] == [f.numerator, f.denominator] and f == postfit.score(cand)
    assert out["status"] == ("investigate" if f < berturk.THRESHOLD else "ok")
    text = json.dumps(out, ensure_ascii=False)
    assert str(tmp_path) not in text and os.path.expanduser("~") not in text and "timestamp" not in text
    assert berturk.reference_check(record, path, sha) == out


def test_reference_refuses_a_laya_record(tmp_path, snap):
    run_dirs = [write_laya_candidate(tmp_path / "laya", snap, lr, e) for lr, e in LAYA_GRID]
    path = str(tmp_path / "postfit" / "laya.json")
    postfit.write_record(postfit.fit(run_dirs, snap[1]), path)
    record, sha = postfit.read_record(path)
    with pytest.raises(SystemExit, match="applies to BERTurk \\(BASE-03\\)"):
        berturk.reference_check(record, path, sha)
    with pytest.raises(SystemExit, match="applies to BERTurk"):
        postfit_cli.main(["--reference-check", path])
    bad = dict(record, system_kind="berturk", chosen=dict(record["chosen"], dump_sha256="0" * 64))
    with pytest.raises(SystemExit, match="not among the record's candidates"):
        berturk.reference_check(bad, path, sha)


def test_reference_resolution_rules(tmp_path, snap):
    path, record, sha = _record(tmp_path, snap, (707, 293, 293))
    with pytest.raises(SystemExit, match="nothing to resolve"):
        berturk.reference_check(record, path, sha, resolution="documented: lower LR")
    assert berturk.reference_check(record, path, sha)["resolution"] is None
    path, record, sha = _record(tmp_path, snap, (706, 294, 294))
    assert berturk.reference_check(record, path, sha)["resolution"] is None          # the shortfall is on record
    out = berturk.reference_check(record, path, sha, resolution="documented: the user accepts the shortfall")
    assert out["status"] == "investigate" and out["resolution"] == "documented: the user accepts the shortfall"


def test_reference_cli_writes_and_reproduces(tmp_path, snap, capsys):
    path, record, sha = _record(tmp_path, snap, (706, 294, 294))
    out = tmp_path / "postfit" / "berturk-reference.json"
    got = postfit_cli.main(["--reference-check", path, "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert json.loads(text) == got == berturk.reference_check(record, path, sha)
    assert capsys.readouterr().out == text
    assert text == json.dumps(got, ensure_ascii=False, indent=2) + "\n"
    again = tmp_path / "again.json"
    postfit_cli.main(["--reference-check", path, "--out", str(again)])
    assert again.read_bytes() == out.read_bytes()                 # byte-identical for the same inputs
    resolved = postfit_cli.main(["--reference-check", path, "--resolution", "fixed: see 06-10"])
    assert resolved["resolution"] == "fixed: see 06-10"
    ok_path, _, _ = _record(tmp_path / "ok", snap, (707, 293, 293))
    with pytest.raises(SystemExit, match="nothing to resolve"):
        postfit_cli.main(["--reference-check", ok_path, "--resolution", "x"])
    for bad in (["--resolution", "x", path],                                      # --resolution in RUN_DIR mode
                [path, "--reference-check", path],                                # two modes at once
                ["--reference-check", path, "--prune", path]):
        with pytest.raises(SystemExit) as e:
            postfit_cli.main(bad)
        assert e.value.code == 2
