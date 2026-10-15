"""pilot.postfit and postfit.py (LAYA-05, LAYA-08): the val temperature fit, the selection rule, the record and its
refusals of test input. No model, torch or Ray: candidate val dumps are written with pilot.dumps.write_dump on the
synthetic val split (conftest), with a sidecar shaped like score.py's (score.sidecar_base + score.training_block).
"""
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

import pilot.data
import postfit as postfit_cli
import score
from conftest import prepared
from pilot import dumps, postfit
from pilot.data import REPO_ROOT, sha256_bytes

pytestmark = pytest.mark.unit

GRID = [(lr, e) for lr in postfit.LR_GRID for e in range(1, postfit.EPOCHS + 1)]
SCORER = {"device": "mps", "dtype": "torch.float32", "amp_enabled": False}


# --- temperature ------------------------------------------------------------------------------------------------

def _planted(t, n=20000, seed=0):
    """(margins, labels) whose labels are drawn from sigmoid(m / t): the NLL-optimal temperature is about t."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0, 3.0, size=n)
    y = rng.random(n) < 1.0 / (1.0 + np.exp(-m / t))
    return m, y


def test_temperature_recovers_a_planted_t():
    m, y = _planted(1.7)
    T, nll_t, nll_1 = postfit.fit_temperature(m, y)
    assert abs(T - 1.7) < 0.1, T
    assert nll_t <= nll_1


def test_temperature_is_one_for_calibrated_margins():
    m, y = _planted(1.0, seed=3)
    T, nll_t, nll_1 = postfit.fit_temperature(m.reshape(-1, 40), y.reshape(-1, 40))   # any [n, k] shape: every pair
    assert abs(T - 1.0) < 0.1, T
    assert nll_t <= nll_1


def test_temperature_out_of_band_returns_the_exact_bound():
    assert postfit.fit_temperature(*_planted(20.0))[0] == postfit.T_MAX == 5.0
    assert postfit.fit_temperature(*_planted(0.1))[0] == postfit.T_MIN == 0.5


def test_temperature_refuses_empty_and_all_zero_margins():
    with pytest.raises(SystemExit, match="no \\(message, tag\\) pairs"):
        postfit.fit_temperature(np.zeros((0, 48)), np.zeros((0, 48), dtype=bool))
    with pytest.raises(SystemExit, match="not identifiable"):
        postfit.fit_temperature(np.zeros((3, 48)), np.eye(3, 48, dtype=bool))


def _laya_t(m, y):
    """laya's own noul T (pilot.laya_adapter.fit_temperature_map, the oracle) on the same (margin, label) pairs: one
    noul record (qtype 2, logits [0, m], target [1 - y, y], 2 options) per pair."""
    from pilot import laya_adapter as la
    records = [(2, [0.0, float(v)], [1.0 - float(t), float(t)], 2) for v, t in zip(m, y)]
    fitted = la.fit_temperature_map(records)
    return fitted["temperature_by_options"].get("noul:2", fitted["temperature"][2])


def test_temperature_matches_laya_fit():
    """LAYA-05: the scipy fit on every val pair is laya's own temperature rule, without torch (planning probe:
    0.0008-0.0011 apart for planted T 0.8, 1.0, 1.7); out of band both clamp to the exact bound."""
    for t in (0.8, 1.0, 1.7):
        m, y = _planted(t, n=2880, seed=11)
        mine, theirs = postfit.fit_temperature(m, y)[0], _laya_t(m, y)
        assert abs(mine - theirs) < 2e-3, (t, mine, theirs)
    m, y = _planted(20.0, n=2880, seed=11)
    assert postfit.fit_temperature(m, y)[0] == 5.0 and _laya_t(m, y) == 5.0


# --- selection on candidate dicts -------------------------------------------------------------------------------

def _canon(lr, **over):
    return dict({"split": "selection_train", "seed": 0, "neg_seed": 0, "epochs": 4, "encoder_lr": lr,
                 "head_lr": 0.0001, "device": "mps", "max_len": 128}, **over)


def _cand(lr, epoch, counts=(1, 3, 3), seed=0, kind="laya", scorer=None, canon=None):
    name = "laya-%s-%d" % (lr, epoch)
    return {"dump_run_id": name, "dump_sha256": sha256_bytes(name.encode()), "uids_sha256": "u" * 64,
            "system_kind": kind, "train_run_id": "laya-selection_train-neg8-s0-%s" % lr, "encoder_lr": lr,
            "epoch": epoch, "epochs_schedule": 4, "seed": seed, "rung": "none", "fallback": False, "weaker": False,
            "loss_by_epoch": [0.1] * epoch, "canonical": canon or _canon(lr), "scorer": dict(scorer or SCORER),
            "temperature": 1.0, "nll_raw": 0.1, "nll_fitted": 0.1,
            "objective": dict(zip(("tp", "fp", "fn"), counts)), "val_micro_f1_at_0_5": 0.0,
            "val_micro_f1_at_global_theta": 0.0, "theta": {}}


def _grid(**counts):
    """The 12 candidates, each scoring F 0.25 unless counts[(lr, epoch)] gives other (tp, fp, fn)."""
    return [_cand(lr, e, counts.get((lr, e), (1, 3, 3))) for lr, e in GRID]


def _pick(cands):
    c = postfit.select(cands)
    return c["encoder_lr"], c["epoch"]


def test_select_picks_the_highest_objective():
    cands = _grid()
    cands[GRID.index((3e-05, 2))]["objective"] = {"tp": 9, "fp": 1, "fn": 1}
    assert _pick(cands) == (3e-05, 2)


def test_select_ties_go_to_fewer_epochs_then_lower_lr():
    cands = _grid()                                          # (1, 1, 1) and (2, 2, 2) both score exactly 1/2
    cands[GRID.index((5e-05, 2))]["objective"] = {"tp": 1, "fp": 1, "fn": 1}
    cands[GRID.index((3e-05, 2))]["objective"] = {"tp": 2, "fp": 2, "fn": 2}
    assert _pick(cands) == (3e-05, 2)                        # same epoch: the lower LR
    cands[GRID.index((5e-05, 1))]["objective"] = {"tp": 3, "fp": 3, "fn": 3}
    assert _pick(cands) == (5e-05, 1)                        # fewer epochs before the lower LR
    assert postfit.score(cands[GRID.index((5e-05, 1))]) == postfit.score(cands[GRID.index((3e-05, 2))])


def test_select_is_independent_of_input_order():
    cands = _grid()
    for k, key in enumerate(((2e-05, 4), (5e-05, 3), (3e-05, 3))):
        cands[GRID.index(key)]["objective"] = {"tp": 4, "fp": 2, "fn": 2}   # a three-way tie at 2/3
    want = postfit.select(cands)["dump_sha256"]
    for seed in range(20):
        shuffled = list(cands)
        random.Random(seed).shuffle(shuffled)
        assert postfit.select(shuffled)["dump_sha256"] == want
    assert _pick(cands) == (3e-05, 3)


def test_select_refuses_an_incomplete_or_duplicated_grid():
    with pytest.raises(SystemExit, match="no candidates"):
        postfit.select([])
    with pytest.raises(SystemExit, match="missing \\[\\(5e-05, 4\\)\\]"):
        postfit.select(_grid()[:-1])
    with pytest.raises(SystemExit, match="unexpected \\[\\(0.0001, 1\\)\\]"):
        postfit.select(_grid() + [_cand(0.0001, 1)])
    dup = _cand(2e-05, 1)
    dup["dump_sha256"] = "x" * 64
    with pytest.raises(SystemExit, match="more than once"):
        postfit.select(_grid() + [dup])
    same = _grid()
    same[1]["dump_sha256"] = same[0]["dump_sha256"]
    with pytest.raises(SystemExit, match="same val dump"):
        postfit.select(same)
    with pytest.raises(SystemExit, match="epochs 1..3"):     # a smaller declared grid still has to match exactly
        postfit.select(_grid(), epochs=3)


@pytest.mark.parametrize("change, match", [
    (lambda c: c.update(seed=1), "seeds"),
    (lambda c: c.update(system_kind="berturk"), "system kinds"),
    (lambda c: c.update(uids_sha256="v" * 64), "unpaired"),
    (lambda c: c["scorer"].update(device="cpu"), "scorer settings"),
    (lambda c: c["canonical"].update(max_len=64), "differ in max_len"),
    (lambda c: c["canonical"].update(max_len=128.0), "differ in max_len"),      # exact JSON equality: 128 != 128.0
    (lambda c: c["canonical"].update(split="final_train"), "canonical split"),
    (lambda c: c["canonical"].update(head_lr=0.0002), "canonical head_lr"),
    (lambda c: c["canonical"].update(epochs=3), "canonical epochs"),
    (lambda c: c.update(train_run_id="another-run"), "two train runs"),
])
def test_select_refuses_mixed_seeds_scorers_or_recipes(change, match):
    cands = _grid()
    change(cands[5])
    with pytest.raises(SystemExit, match=match):
        postfit.select(cands)


# --- candidate val dumps on the synthetic snapshot -------------------------------------------------------------

@pytest.fixture
def snap(tmp_path, synthetic_raw):
    """(out_dir, manifest_path, val entry, val uids) of the synthetic snapshot prepared under tmp_path."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    path = os.path.join(out_dir, "splits", "val.jsonl")
    with open(path, "rb") as f:
        raw = f.read()
    entry = pilot.data.verify_split_file(path, manifest_path=manifest, roles=("val",), data=raw)
    return out_dir, manifest, entry, [r["uid"] for r in pilot.data.parse_jsonl(raw)]


def write_candidate(root, snap, lr, epoch, seed=0, split="selection_train", calibrated=False, shipped=1.0,
                    n=None, rng_seed=None):
    """A candidate val dump under root/<run_id>: random margins, a fake local identity and a training block shaped
    like score.training_block's. Returns the run dir."""
    _, manifest, entry, uids = snap
    rng = np.random.default_rng(rng_seed if rng_seed is not None else int(lr * 1e6) * 10 + epoch)
    uids = uids if n is None else uids[:n]
    logits = rng.normal(0.0, 3.0, size=(len(uids), 48)).astype(np.float32)
    tag = "%s-%d-%s-%s-%s-%s" % (lr, epoch, seed, split, calibrated, shipped)
    ident = {"local": True, "model_sha256": sha256_bytes(("model " + tag).encode()),
             "config_sha256": sha256_bytes(("config " + tag).encode()),
             "files_sha256": {"encoder/config.json": "e" * 64, "tokenizer/tokenizer_config.json#normalized": "t" * 64}}
    rid = dumps.run_id("laya", ident)
    canon = _canon(lr, split=split, seed=seed, neg_seed=seed)
    training = {"run_id": "laya-%s-neg8-s%d-%08x" % (split, seed, int(lr * 1e6)), "split": split, "epoch": epoch,
                "epochs_schedule": 4, "stop_after_epoch": None, "encoder_lr": lr, "head_lr": 0.0001,
                "neg_seed": seed, "rung": "none", "fallback": False, "weaker": False,
                "loss_by_epoch": [0.5 / (k + 1) for k in range(epoch)], "calibrated": calibrated,
                "postfit_sha256": None, "canonical": canon,
                "config_sha256_sans_seed": dumps.training_digest(canon, epoch)}
    fields = {"run_id": rid, "seed": seed, "training": training,
              "system": {"kind": "laya", "checkpoint": "ray_results/x/exports/epoch-%03d" % epoch,
                         "checkpoint_identity": ident, "baseline_source": None, "clip_eps": None},
              "temperature": {"shipped": shipped, "source": "test"},
              "scorer": dict(SCORER, workers=0), "timing": {"seconds": 0.0}}
    sidecar = dict(score.sidecar_base(entry, "val", manifest, None if n is None else len(uids)), **fields)
    run_dir = os.path.join(str(root), rid)
    dumps.write_dump(run_dir, "val", logits, uids, sidecar)
    return run_dir


def _twelve(tmp_path, snap):
    return [write_candidate(tmp_path / "dumps", snap, lr, e) for lr, e in GRID]


def test_record_from_twelve_dumps(tmp_path, snap, capsys):
    run_dirs = _twelve(tmp_path, snap)
    out = tmp_path / "postfit" / "laya.json"
    postfit_cli.main(run_dirs + ["--manifest", snap[1], "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    rec = json.loads(text)
    assert json.loads(capsys.readouterr().out) == rec
    assert rec["schema"] == postfit.SCHEMA and rec["system_kind"] == "laya"
    assert rec["split"]["name"] == "val" and rec["split"]["uids_sha256"] == snap[2]["uids_sha256"]
    assert rec["protocol"] == {"selection_split": "selection_train", "final_split": "final_train",
                               "lr_grid": [2e-05, 3e-05, 5e-05], "head_lr": 0.0001, "epochs": 4,
                               "selection_seed": 0, "final_seeds": [0, 1, 2]}
    assert rec["rules"]["temperature_bounds"] == [0.5, 5.0]
    cands = rec["candidates"]
    assert [(c["encoder_lr"], c["epoch"]) for c in cands] == GRID
    chosen = rec["chosen"]
    assert chosen["dump_sha256"] in {c["dump_sha256"] for c in cands}
    for c in cands + [chosen]:
        assert 0.5 <= c["temperature"] <= 5.0
        for variant in ("shipped", "raw"):
            th = c["theta"][variant]
            assert {"global", "per_tag", "eligible_tags"} <= set(th)
    for c in cands:
        assert c["nll_fitted"] <= c["nll_raw"] and set(c["objective"]) == {"tp", "fp", "fn"}
        assert c["loss_by_epoch"] and len(c["loss_by_epoch"]) == c["epoch"]
    best = max(postfit.score(c) for c in cands)
    assert postfit.score(next(c for c in cands if c["dump_sha256"] == chosen["dump_sha256"])) == best
    assert chosen["stop_after_epoch"] == (None if chosen["epoch"] == 4 else chosen["epoch"])
    assert rec["final"] == {"split": "final_train", "seeds": [0, 1, 2], "epochs": 4,
                            "stop_after_epoch": chosen["stop_after_epoch"]}
    assert os.path.expanduser("~") not in text and str(tmp_path) not in text
    record, sha = postfit.read_record(str(out))
    assert record == rec and sha == pilot.data.sha256_file(str(out))


def test_record_bytes_do_not_depend_on_argument_order(tmp_path, snap):
    run_dirs = _twelve(tmp_path, snap)
    first = postfit.record_bytes(postfit.fit(run_dirs, snap[1]))
    for seed in range(3):
        shuffled = list(run_dirs)
        random.Random(seed).shuffle(shuffled)
        assert postfit.record_bytes(postfit.fit(shuffled, snap[1])) == first
    for name, dirs in (("a", run_dirs), ("b", run_dirs[::-1])):
        postfit_cli.main(dirs + ["--manifest", snap[1], "--out", str(tmp_path / name / "laya.json")])
    assert (tmp_path / "a" / "laya.json").read_bytes() == (tmp_path / "b" / "laya.json").read_bytes() == first


def test_read_record_refuses_missing_invalid_or_other_schema(tmp_path):
    p = tmp_path / "r.json"
    with pytest.raises(SystemExit, match="not found"):
        postfit.read_record(str(p))
    p.write_text("{", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        postfit.read_record(str(p))
    p.write_text(json.dumps({"schema": "postfit/0"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="schema"):
        postfit.read_record(str(p))
    p.write_text(json.dumps({"schema": postfit.SCHEMA, "chosen": {}, "protocol": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="no final block"):
        postfit.read_record(str(p))


def test_postfit_refuses_test_input(tmp_path, snap):
    out_dir, manifest = snap[0], snap[1]
    run_dir = write_candidate(tmp_path / "dumps", snap, 2e-05, 1)
    postfit.candidate(run_dir, manifest)                     # a full selection val dump is accepted
    with pytest.raises(SystemExit, match="is a file"):
        postfit.candidate(os.path.join(out_dir, "splits", "test.jsonl"), manifest)
    with pytest.raises(SystemExit, match="is a file"):
        postfit.candidate(os.path.join(run_dir, "val.npz"), manifest)
    with open(os.path.join(run_dir, "test.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    with pytest.raises(SystemExit, match="holds a test dump"):
        postfit.candidate(run_dir, manifest)
    with pytest.raises(SystemExit, match="holds a test dump"):    # the CLI refuses it the same way
        postfit_cli.main([run_dir, "--manifest", manifest])
    subset = write_candidate(tmp_path / "subset", snap, 2e-05, 1, n=2)
    with pytest.raises(SystemExit, match="subset val dump"):
        postfit.candidate(subset, manifest)
    empty = tmp_path / "dumps" / "laya-empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="val.npz not found"):
        postfit.candidate(str(empty), manifest)
    with pytest.raises(SystemExit):
        postfit_cli.main(["--manifest", manifest])           # no run dir at all: argparse error


@pytest.mark.parametrize("kw, match", [
    ({"split": "final_train"}, "trained on final_train"),
    ({"calibrated": True}, "calibrated True"),
    ({"shipped": 1.3}, "shipped temperature 1.3"),
])
def test_postfit_refuses_in_sample_or_calibrated_candidates(tmp_path, snap, kw, match):
    run_dir = write_candidate(tmp_path / "dumps", snap, 2e-05, 1, **kw)
    with pytest.raises(SystemExit, match=match):
        postfit.candidate(run_dir, snap[1])


def test_postfit_module_is_torch_free():
    code = ("import sys, pilot.postfit, postfit; "
            "print(sorted(m for m in ('torch', 'ray', 'laya') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


# --- guarded export prune (the tiny checkpoint, scored in-process on the synthetic val) ---------------------------

@pytest.fixture
def export(tmp_path, snap):
    """(export dir at storage/laya-x/exports/epoch-001, dump root, manifest, val.jsonl): a tiny checkpoint saved
    where ray_train.py writes its epoch exports. Scoring sets torch's thread count; it is restored afterwards."""
    import torch
    from tiny_laya import tiny_checkpoint
    threads = torch.get_num_threads()
    path = tiny_checkpoint(tmp_path / "storage" / "laya-x" / "exports" / "epoch-001")
    yield path, str(tmp_path / "dumps"), snap[1], os.path.join(snap[0], "splits", "val.jsonl")
    torch.set_num_threads(threads)


def _score_val(export_dir, dump_root, manifest, val_jsonl, *extra):
    score.main([val_jsonl, "--checkpoint", export_dir, "--workers", "0", "--cpus-per-worker", "1",
                "--manifest", manifest, "--out-root", dump_root, *extra])


def test_prune_deletes_an_export_only_after_its_val_dump(tmp_path, export, capsys):
    path, dump_root, manifest, val_jsonl = export
    with pytest.raises(SystemExit, match="no verified full val dump"):
        postfit.prune_export(path, dump_root, manifest)
    assert os.path.isdir(path)
    _score_val(path, dump_root, manifest, val_jsonl)
    capsys.readouterr()
    rid = postfit_cli.main(["--prune", path, "--dump-root", dump_root, "--manifest", manifest])[0]
    assert not os.path.exists(path)
    assert os.path.isdir(os.path.dirname(path))                    # only the epoch dir is deleted
    assert capsys.readouterr().out.strip() == "pruned laya-x/exports/epoch-001 (val dump %s)" % rid
    dumps.load_dump(os.path.join(dump_root, rid), "val", manifest)  # the dump stays and still verifies


def test_prune_refuses_without_a_dump_symlinks_and_other_paths(tmp_path, export):
    from tiny_laya import tiny_checkpoint
    path, dump_root, manifest, val_jsonl = export
    link = tmp_path / "link-epoch-001"
    os.symlink(path, link)
    with pytest.raises(SystemExit, match="symlink"):
        postfit.prune_export(str(link), dump_root, manifest)
    linked_exports = tmp_path / "storage2" / "laya-y" / "exports"
    linked_exports.parent.mkdir(parents=True)
    os.symlink(os.path.dirname(path), linked_exports)
    with pytest.raises(SystemExit, match="symlink"):
        postfit.prune_export(str(linked_exports / "epoch-001"), dump_root, manifest)
    base = tiny_checkpoint(tmp_path / "base")
    with pytest.raises(SystemExit, match="not a ray_train export"):
        postfit.prune_export(base, dump_root, manifest)
    odd = tiny_checkpoint(tmp_path / "storage" / "laya-x" / "exports" / "epoch-1")
    with pytest.raises(SystemExit, match="not a ray_train export"):
        postfit.prune_export(odd, dump_root, manifest)
    with pytest.raises(SystemExit, match="not a ray_train export"):
        postfit.prune_export(os.path.join(path, "rl_agent_config.json"), dump_root, manifest)
    _score_val(path, dump_root, manifest, val_jsonl, "--n", "2")   # a subset val dump at the export's run id
    with pytest.raises(SystemExit, match="subset dump or another checkpoint"):
        postfit.prune_export(path, dump_root, manifest)
    with pytest.raises(SystemExit):
        postfit_cli.main(["--prune", path, "--out", str(tmp_path / "x.json"), "--manifest", manifest])
    with pytest.raises(SystemExit):
        postfit_cli.main([path, "--prune", path, "--manifest", manifest])   # two modes at once
    for p in (path, link, base, odd):
        assert os.path.isdir(p)                                    # every refusal left its target in place


# --- final seeds: the printed commands and the final manifest (LAYA-08) ------------------------------------------

def _committed_entry(name):
    with open(os.path.join(REPO_ROOT, "data", "splits.json"), encoding="utf-8") as f:
        return json.load(f)["splits"][name]


SELECTION_COMMANDS = [
    ["--device", "mps", "--workers", "1", "--micro-batch", "8", "--grad-accum", "4", "--max-len", "128",
     "--encoder-lr", "3e-05"],
    ["--device", "cuda", "--workers", "2", "--no-amp", "--freeze-embeddings", "--shuffle-options", "--neg", "0",
     "--head-max-len", "64", "--max-messages", "50", "--no-grad-checkpointing", "--loss", "soft-ce",
     "--encoder-lr", "5e-05"],
    ["--device", "cpu", "--workers", "2", "--amp", "--freeze-encoder", "--micro-batch", "4", "--grad-accum", "8"],
]


def _selection_record(ray_train, flags, tmp_path, stop=2):
    """(record path, record) whose chosen canonical is the selection canonical of `flags` on the committed
    selection_train entry and the pinned base (ray_train.parser, resolve, identity_values)."""
    from pilot import runs
    from pilot.weights import identity
    a = ray_train.parser().parse_args(["--data", "data/splits/selection_train.jsonl", "--out", "o"] + flags)
    canon = runs.canonical(ray_train.identity_values(a, ray_train.resolve(a), _committed_entry("selection_train"),
                                                     "multilingual", identity("multilingual")))
    theta = {"global": 0.42, "per_tag": {}, "eligible_tags": []}
    record = {"schema": postfit.SCHEMA,
              "chosen": {"train_run_id": runs.run_id(canon), "dump_run_id": "laya-x", "dump_sha256": "d" * 64,
                         "encoder_lr": canon["encoder_lr"], "epoch": stop or canon["epochs"],
                         "epochs_schedule": canon["epochs"], "stop_after_epoch": stop, "canonical": canon,
                         "temperature": 1.6109, "theta": {"shipped": theta, "raw": theta},
                         "val_micro_f1_at_global_theta": 0.8},
              "final": {"split": "final_train", "seeds": [0, 1, 2], "epochs": canon["epochs"], "stop_after_epoch": stop},
              "protocol": {}, "rules": {}}
    path = str(tmp_path / "laya.json")
    postfit.write_record(record, path)
    return path, record


@pytest.mark.parametrize("flags", SELECTION_COMMANDS)
def test_final_command_passes_the_binding(flags, tmp_path):
    """Each printed final command, parsed by ray_train.py on the final_train entry with the record's sha, is a canonical
    config check_final accepts: every hyperparameter, the AMP and checkpointing choices and the stop epoch transfer."""
    ray_train = pytest.importorskip("ray_train")
    from pilot import runs
    from pilot.weights import identity
    path, record = _selection_record(ray_train, flags, tmp_path)
    sha = pilot.data.sha256_file(path)
    for seed in (0, 1, 2):
        argv = postfit.final_command(record, seed, path, "ray_results", "ckpt/laya-final", manifest="m.json")
        assert argv[0] == "ray_train.py"
        a = ray_train.parser().parse_args(argv[1:])
        assert a.postfit == path and a.export_epochs is False and a.out == "ckpt/laya-final-s%d" % seed
        assert a.storage == "ray_results" and a.manifest == "m.json" and a.neg_seed is None
        assert a.data == "data/splits/final_train.jsonl" and a.stop_after_epoch == 2
        canon = runs.canonical(ray_train.identity_values(a, ray_train.resolve(a), _committed_entry("final_train"),
                                                         "multilingual", identity("multilingual"), sha))
        postfit.check_final(record, canon)
        assert canon["seed"] == canon["neg_seed"] == seed and canon["split"] == "final_train"
    with pytest.raises(SystemExit, match="seed 7 is not one of the record's final seeds"):
        postfit.final_command(record, 7, path, "ray_results", "ckpt/laya-final")
    other = dict(record, chosen=dict(record["chosen"], canonical=dict(record["chosen"]["canonical"],
                                                                      base_model_sha256="0" * 64)))
    with pytest.raises(SystemExit, match="trained another base"):
        postfit.final_command(other, 0, path, "ray_results", "ckpt/laya-final")
    assert postfit.final_command(other, 0, path, "s", "o", base="ckpt/base")[4] == "ckpt/base"   # a local base


def test_final_command_omits_the_stop_flag_for_the_last_epoch(tmp_path):
    ray_train = pytest.importorskip("ray_train")
    path, record = _selection_record(ray_train, SELECTION_COMMANDS[0], tmp_path, stop=None)
    argv = postfit.final_command(record, 1, path, "s", "o")
    assert "--stop-after-epoch" not in argv and "--manifest" not in argv
    assert ray_train.parser().parse_args(argv[1:]).stop_after_epoch is None


def test_print_final_commands_cli(tmp_path, capsys):
    import shlex
    ray_train = pytest.importorskip("ray_train")
    path, record = _selection_record(ray_train, SELECTION_COMMANDS[0], tmp_path)
    lines = postfit_cli.main(["--print-final-commands", path, "--storage", "ray_results", "--out-prefix", "ckpt/f"])
    assert capsys.readouterr().out.splitlines() == lines and len(lines) == 3
    assert [shlex.split(ln) for ln in lines] == [postfit.final_command(record, s, path, "ray_results", "ckpt/f")
                                                 for s in (0, 1, 2)]
    for bad in (["--print-final-commands", path, "--storage", "s"],                      # no --out-prefix
                ["--print-final-commands", path, "--storage", "s", "--out-prefix", "o", "--out", "x.json"],
                [path, "--storage", "s"]):                                                # --storage in another mode
        with pytest.raises(SystemExit) as e:
            postfit_cli.main(bad)
        assert e.value.code == 2
    r = subprocess.run([sys.executable, "postfit.py", "--print-final-commands", str(tmp_path / "nonexistent.json"),
                        "--storage", "s", "--out-prefix", "o"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "nonexistent.json not found" in r.stderr, r.stderr


@pytest.fixture
def finals(tmp_path):
    """(record path, record, sha, make) where make(name, seed, **change) writes a tiny checkpoint whose config carries
    the training block of a --postfit final seed of the record (final_train canonical, chosen epoch 2, T, record rows);
    change: postfit_sha (the bound record sha), split, epoch, temperature, data_sha256."""
    from tiny_laya import tiny_checkpoint
    chosen_canon = _canon(3e-05, data_sha256="s" * 64, data_role="train", max_messages=None)
    T = 1.6109375
    record = {"schema": postfit.SCHEMA,
              "chosen": {"train_run_id": "laya-selection_train-neg8-s0-00000003", "dump_run_id": "laya-y",
                         "encoder_lr": 3e-05, "epoch": 2, "epochs_schedule": 4, "stop_after_epoch": 2,
                         "canonical": chosen_canon, "temperature": T, "theta": {}},
              "final": {"split": "final_train", "seeds": [0, 1, 2], "epochs": 4, "stop_after_epoch": 2},
              "protocol": {}}
    path = str(tmp_path / "postfit" / "laya.json")
    postfit.write_record(record, path)
    sha = pilot.data.sha256_file(path)
    threads = __import__("torch").get_num_threads()

    def make(name, seed, postfit_sha=sha, split="final_train", epoch=2, temperature=T, data_sha256="f" * 64):
        d = tiny_checkpoint(tmp_path / "ckpt" / name, seed=seed)
        canon = dict(chosen_canon, split=split, seed=seed, neg_seed=seed, data_sha256=data_sha256,
                     stop_after_epoch=2, postfit_sha256=postfit_sha)
        rows = [{"epoch": e, "loss": 0.5 / e, "train_s": 10.0 * e, "save_s": 1.0, "optimizer_steps": 3, "n_items": 9,
                 "peak_total_bytes": 2**30, "peak_driver_bytes": None, "peak_rss_bytes": 2**30,
                 "current_allocated_bytes": None, "swap_delta_bytes": 0, "segment": 0, "resumed": False}
                for e in range(1, epoch + 1)]
        with open(os.path.join(d, "rl_agent_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.update(temperature=[1.0, 1.0, temperature], temperature_by_options={}, fine_tuned=True)
        cfg["training"] = {"epoch": epoch, "calibrated": True, "postfit": {"sha256": postfit_sha, "temperature": T},
                           "ray_train": {"run_id": "laya-%s-neg8-s%d-0000000%d" % (split, seed, seed),
                                         "canonical": canon,
                                         "record": {"rung": "none", "fallback": False, "weaker": False,
                                                    "code_changed_on_resume": False, "epochs": rows}}}
        with open(os.path.join(d, "rl_agent_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return d
    yield path, record, sha, make
    __import__("torch").set_num_threads(threads)


def test_final_manifest_lists_the_three_seeds(finals, tmp_path, capsys):
    from pilot.weights import identity
    path, record, sha, make = finals
    dirs = [make("laya-final-s%d" % s, s) for s in (2, 0, 1)]                   # any argument order
    out = tmp_path / "laya-final.json"
    man = postfit_cli.main(["--final-manifest", path] + dirs + ["--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert json.loads(text) == man and capsys.readouterr().out == text
    assert man["schema"] == "postfit-final/1" and man["final_split"] == "final_train"
    assert man["postfit"] == {"path": pilot.data.repo_relpath(path), "sha256": sha}
    assert man["chosen"] == {"train_run_id": record["chosen"]["train_run_id"], "encoder_lr": 3e-05, "epoch": 2,
                             "epochs_schedule": 4, "stop_after_epoch": 2, "temperature": 1.6109375}
    assert [e["seed"] for e in man["seeds"]] == [0, 1, 2]
    for e, d in zip(man["seeds"], sorted(dirs)):
        ident = identity(d)
        assert e["checkpoint_name"] == os.path.basename(d) and e["checkpoint_identity"] == ident
        assert e["dump_run_id"] == dumps.run_id("laya", ident) and e["temperature"] == 1.6109375
        assert e["loss_by_epoch"] == [0.5, 0.25] and [r["epoch"] for r in e["epochs"]] == [1, 2]
        assert set(e["epochs"][0]) == {"epoch", "loss", "train_s", "peak_total_bytes", "swap_delta_bytes"}
        assert e["rung"] == "none" and e["weaker"] is False and e["code_changed_on_resume"] is False
    canon0 = json.load(open(os.path.join(dirs[1], "rl_agent_config.json"), encoding="utf-8"))
    assert man["config_sha256_sans_seed"] == dumps.training_digest(canon0["training"]["ray_train"]["canonical"], 2)
    assert os.path.expanduser("~") not in text and str(tmp_path) not in text and "timestamp" not in text
    assert postfit.final_manifest(path, list(reversed(dirs))) == man                 # order-independent


@pytest.mark.parametrize("case, match", [
    ("another_sha", "bound to postfit record"),
    ("duplicate_seed", "seeds \\[0, 0, 1\\]"),
    ("two_seeds", "seeds \\[0, 1\\]"),
    ("another_t", "serves temperature"),
    ("another_epoch", "stopped at epoch 3"),
    ("selection_split", "trained on selection_train"),
    ("another_recipe", "differ in more than the seed"),
])
def test_final_manifest_refuses_a_wrong_set(finals, case, match):
    path, record, sha, make = finals
    s0, s1 = make("s0", 0), make("s1", 1)
    third = {"another_sha": lambda: make("s2", 2, postfit_sha="0" * 64),
             "duplicate_seed": lambda: make("s0b", 0),
             "two_seeds": lambda: None,
             "another_t": lambda: make("s2", 2, temperature=1.6109376),
             "another_epoch": lambda: make("s2", 2, epoch=3),
             "selection_split": lambda: make("s2", 2, split="selection_train"),
             "another_recipe": lambda: make("s2", 2, data_sha256="0" * 64)}[case]()
    with pytest.raises(SystemExit, match=match):
        postfit.final_manifest(path, [s0, s1] + ([third] if third else []))
