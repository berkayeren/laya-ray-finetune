"""berturk_train.py as a recipe (Phase 6, BASE-01): a seeded CPU run reproduces byte for byte, every export is a
complete, self-describing checkpoint, and done.json marks a finished run.

Two identical runs (tiny offline base, the conftest synthetic selection_train, --device cpu --batch 4 --max-len 32,
2 epochs) go to two storage dirs; their weights, config and tokenizer files must be equal byte for byte (06-RESEARCH
probe 1 and Pitfall 6: the seed is set right before the head is created; CPU is bit-reproducible, MPS is recorded as
not). The record's timings and memory peaks differ between runs by design, so berturk_train.json is checked by
content, not bytes.

The final-seed path (plan 06-08, BASE-01/02/03): --stop-after-epoch K at its boundaries and on the selection
trajectory (byte-identical to the uninterrupted run's epoch K on CPU), the --postfit binding and the BASE-03
--reference gate refusing every departure before any fetch, load or run dir (also with --print-run-id), and the final
checkpoint serving the record's T bit for bit. The record here is a hand-built postfit/1 record whose chosen canonical
is the tiny selection command's own (captured from berturk_train.py --print-run-id), chosen epoch 1 of 2.

No marker: it runs in `make test` (no Ray, no weights, no data download; seconds).
"""
import json
import math
import os
import re
import subprocess
import sys

import pytest

import pilot.data
from conftest import make_raw
from intents import INTENTS
from pilot import berturk, postfit
from pilot.data import sha256_file
from tiny_berturk import tiny_base

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600
EPOCHS, BATCH, MAX_LEN = 2, 4, 32
SAME_BYTES = ("model.safetensors", "config.json") + berturk.TOKENIZER_FILES


def _bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def train_argv(data, manifest, base, storage, *extra):
    """berturk_train.py arguments of the smoke recipe on the synthetic split and the tiny base."""
    return ["--data", data, "--manifest", manifest, "--base", base, "--device", "cpu", "--epochs", str(EPOCHS),
            "--batch", str(BATCH), "--max-len", str(MAX_LEN), "--seed", "0", "--storage", str(storage)] + list(extra)


def run_train(argv):
    """Run berturk_train.py as a subprocess with the test interpreter (no PYTHONPATH, like a user's command)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "berturk_train.py"] + [str(x) for x in argv], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0, "berturk_train.py failed (exit %d):\n%s" % (r.returncode, r.stderr[-4000:])
    return r


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """(selection_train path, manifest, base dir) of the synthetic snapshot and the tiny base, shared by the module."""
    root = tmp_path_factory.mktemp("inputs")
    raw_root, raw_sha256 = make_raw(str(root / "raw"))
    out_dir = str(root / "data")
    pilot.data.prepare(raw_root, out_dir, raw_sha256)
    return (os.path.join(out_dir, "splits", "selection_train.jsonl"), os.path.join(out_dir, "splits.json"),
            tiny_base(root / "base"))


@pytest.fixture(scope="module")
def two_runs(inputs, tmp_path_factory):
    """Two runs of the identical command into two storage dirs: (run id, [run dir a, run dir b])."""
    data, manifest, base = inputs
    root = tmp_path_factory.mktemp("runs")
    rid = run_train(train_argv(data, manifest, base, root / "a", "--print-run-id")).stdout.strip()
    dirs = []
    for name in ("a", "b"):
        run_train(train_argv(data, manifest, base, root / name))
        dirs.append(root / name / rid)
    return rid, dirs


def exports(run_dir):
    return [os.path.join(str(run_dir), "exports", "epoch-%03d" % k) for k in range(1, EPOCHS + 1)]


def strings(obj):
    """Every string (keys included) in a parsed JSON document."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)
    elif isinstance(obj, str):
        yield obj


@pytest.mark.timeout(TIMEOUT)
def test_cpu_runs_are_deterministic(two_runs):
    rid, (a, b) = two_runs
    assert os.path.basename(str(a)) == os.path.basename(str(b)) == rid      # the same command, the same run id
    for ea, eb in zip(exports(a), exports(b)):
        for f in SAME_BYTES:
            assert _bytes(os.path.join(ea, f)) == _bytes(os.path.join(eb, f)), "%s/%s differs between two CPU runs" \
                % (os.path.basename(ea), f)
    # the two epochs trained: epoch 2's weights are not epoch 1's
    assert _bytes(os.path.join(exports(a)[0], "model.safetensors")) \
        != _bytes(os.path.join(exports(a)[1], "model.safetensors"))


@pytest.mark.timeout(TIMEOUT)
def test_exports_hold_the_checkpoint_files(two_runs, inputs):
    _, _, base = inputs
    rid, (a, b) = two_runs
    for run_dir in (a, b):
        assert sorted(os.listdir(os.path.join(str(run_dir), "exports"))) == ["epoch-%03d" % k
                                                                            for k in range(1, EPOCHS + 1)]
        for k, exp in enumerate(exports(run_dir), 1):
            assert sorted(os.listdir(exp)) == sorted(berturk.CHECKPOINT_FILES)
            for f in berturk.TOKENIZER_FILES:                                # byte copies of the base
                assert _bytes(os.path.join(exp, f)) == _bytes(os.path.join(base, f))
            cfg = _json(os.path.join(exp, "config.json"))
            assert [cfg["id2label"][str(i)] for i in range(len(INTENTS))] == list(INTENTS)
            assert cfg["label2id"] == {t: i for i, t in enumerate(INTENTS)}
            assert cfg["problem_type"] == "multi_label_classification"
            rec = _json(os.path.join(exp, berturk.RECORD))
            assert rec["schema"] == "berturk-train/1" and rec["run_id"] == rid and rec["epoch"] == k
            assert rec["epochs_schedule"] == EPOCHS and rec["calibrated"] is False and rec["temperature"] == 1.0
            assert rec["postfit"] is None and rec["canonical"]["seed"] == 0
            r = rec["record"]
            assert r["device"] == r["param_device"] == "cpu" and r["dtype"] == "torch.float32" and r["amp"] is False
            assert r["bit_reproducible"] is True
            assert type(r["n_truncated"]) is int and 0 <= r["n_truncated"] <= r["n_messages"]
            assert type(r["n_messages"]) is int and r["n_messages"] > 0 and r["max_len"] == MAX_LEN
            assert [row["epoch"] for row in r["epochs"]] == list(range(1, k + 1))
            for row in r["epochs"]:
                assert tuple(row) == berturk.EPOCH_FIELDS
                assert math.isfinite(row["loss"]) and row["loss"] > 0
                assert row["optimizer_steps"] == math.ceil(r["n_messages"] / BATCH)
                assert row["n_messages"] == r["n_messages"] and row["train_s"] >= 0
                assert row["peak_driver_bytes"] is None                     # cpu: no driver memory, never 0
            local = [s for s in strings(rec) if os.path.isabs(s) or str(REPO) in s or os.path.expanduser("~") in s]
            assert not local, "absolute local paths in %s: %s" % (berturk.RECORD, local)


@pytest.mark.timeout(TIMEOUT)
def test_done_marker_written_after_the_last_epoch(two_runs):
    rid, (a, _) = two_runs
    done = os.path.join(str(a), berturk.DONE)
    assert sorted(os.listdir(str(a))) == sorted([berturk.DONE, "exports"])  # no Trainer checkpoint or pickle (A6)
    rec = _json(done)
    assert rec["schema"] == berturk.RECORD_SCHEMA and rec["run_id"] == rid and rec["epoch"] == EPOCHS
    assert [row["epoch"] for row in rec["record"]["epochs"]] == list(range(1, EPOCHS + 1))
    assert rec == _json(os.path.join(exports(a)[-1], berturk.RECORD))     # the last export's record, whole
    assert os.stat(done).st_mtime_ns >= os.stat(os.path.join(exports(a)[-1], berturk.RECORD)).st_mtime_ns


# ---- refusals before any model load or run dir (Task 2) ----

@pytest.fixture
def no_load(monkeypatch):
    """Make every tokenizer or model load fail loudly: a refusal test passes only if main stops before one."""
    import transformers

    def boom(*args, **kwargs):
        raise AssertionError("a model or tokenizer was loaded before the refusal")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)
    monkeypatch.setattr(transformers.BertForSequenceClassification, "from_pretrained", boom)
    monkeypatch.setattr(transformers.PreTrainedModel, "from_pretrained", boom)


def refused(argv, storage, match):
    """main(argv) raises SystemExit matching `match`, and the storage dir was not created."""
    import berturk_train
    with pytest.raises(SystemExit, match=match):
        berturk_train.main([str(x) for x in argv])
    assert not os.path.exists(str(storage)), "the refusal left %s behind" % storage


@pytest.mark.parametrize("name", ["val", "test", "hotels_probe"])
def test_refuses_a_non_train_split(inputs, tmp_path, no_load, name):
    data, manifest, base = inputs
    other = os.path.join(os.path.dirname(data), name + ".jsonl")
    refused(train_argv(other, manifest, base, tmp_path / "runs"), tmp_path / "runs", "has role")


def test_refuses_a_tampered_or_unregistered_split(tmp_path, synthetic_raw, inputs, no_load):
    from conftest import prepared
    _, _, base = inputs
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    data = os.path.join(out_dir, "splits", "selection_train.jsonl")
    copy = str(tmp_path / "copy.jsonl")
    with open(data, "rb") as f:
        raw = bytearray(f.read())
    with open(copy, "wb") as f:
        f.write(bytes(raw))
    refused(train_argv(copy, manifest, base, tmp_path / "runs"), tmp_path / "runs", "not a split registered")
    i = raw.index(b'"state"') + len(b'"state": "')         # one byte of the first message's text
    raw[i] = ord("X") if raw[i] != ord("X") else ord("Y")
    with open(data, "wb") as f:
        f.write(bytes(raw))
    refused(train_argv(data, manifest, base, tmp_path / "runs"), tmp_path / "runs", "sha256")


def test_refuses_a_row_missing_a_tag(inputs, tmp_path, monkeypatch, no_load):
    import berturk_train
    data, manifest, base = inputs
    seen = {}

    def parse_dropping_a_tag(raw):                          # a doctored file would fail its sha256 first
        rows = pilot.data.parse_jsonl(raw)
        del rows[0]["expected"][berturk.TAGS[5]]
        seen["uid"] = rows[0]["uid"]
        return rows
    monkeypatch.setattr(berturk_train, "parse_jsonl", parse_dropping_a_tag)
    with pytest.raises(SystemExit) as e:
        berturk_train.main([str(x) for x in train_argv(data, manifest, base, tmp_path / "runs")])
    assert seen["uid"] in str(e.value) and berturk.TAGS[5] in str(e.value), str(e.value)
    assert not os.path.exists(str(tmp_path / "runs"))


def test_refuses_an_empty_selection(inputs, tmp_path, monkeypatch, no_load):
    import berturk_train
    data, manifest, base = inputs
    monkeypatch.setattr(berturk_train, "parse_jsonl", lambda raw: [])
    refused(train_argv(data, manifest, base, tmp_path / "runs"), tmp_path / "runs", "no messages")


@pytest.mark.parametrize("tags,match", [("no_such_tag", "unknown tag"),
                                        ("%s,%s" % (berturk.TAGS[0], berturk.TAGS[0]), "more than once")])
def test_refuses_bad_mask_tags(inputs, tmp_path, no_load, tags, match):
    data, manifest, base = inputs
    refused(train_argv(data, manifest, base, tmp_path / "runs", "--mask-tags", tags), tmp_path / "runs", match)


@pytest.mark.parametrize("fault", list(berturk.BASE_FILES) + ["lowercasing"])
def test_refuses_an_incomplete_or_lowercasing_base(inputs, tmp_path, no_load, fault):
    import shutil
    data, manifest, base = inputs
    bad = str(tmp_path / "base")
    shutil.copytree(base, bad)
    if fault == "lowercasing":
        with open(os.path.join(bad, "tokenizer_config.json"), "wb") as f:
            f.write(json.dumps({"do_lower_case": True}).encode("utf-8"))
        match = "do_lower_case"
    else:
        os.remove(os.path.join(bad, fault))
        match = "lacks %s" % fault
    refused(train_argv(data, manifest, bad, tmp_path / "runs"), tmp_path / "runs", match)


@pytest.mark.parametrize("flag", ["--epochs", "--batch", "--max-len", "--max-messages", "--cpus", "--lr"])
def test_refuses_bad_numbers(inputs, tmp_path, no_load, capsys, flag):
    import berturk_train
    data, manifest, base = inputs
    with pytest.raises(SystemExit) as e:
        berturk_train.main([str(x) for x in train_argv(data, manifest, base, tmp_path / "runs", flag, "0")])
    assert e.value.code == 2 and flag in capsys.readouterr().err
    assert not os.path.exists(str(tmp_path / "runs"))


def test_refuses_mps_when_unavailable(inputs, tmp_path, monkeypatch, no_load):
    import torch
    from pilot import spike
    data, manifest, base = inputs
    for k in spike.spike_env("0", 4):                       # main sets the MPS env; monkeypatch restores it
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    threads = torch.get_num_threads()
    argv = train_argv(data, manifest, base, tmp_path / "runs")
    argv[argv.index("--device") + 1] = "mps"
    try:
        refused(argv, tmp_path / "runs", "MPS is not available")
    finally:
        torch.set_num_threads(threads)


def test_refuses_an_existing_run_dir(inputs, tmp_path, capsys, no_load):
    import berturk_train
    data, manifest, base = inputs
    storage = tmp_path / "runs"
    berturk_train.main([str(x) for x in train_argv(data, manifest, base, storage, "--print-run-id")])
    run_dir = storage / capsys.readouterr().out.strip()
    run_dir.mkdir(parents=True)                             # a stale, possibly half-written run
    with pytest.raises(SystemExit, match="never resumed") as e:
        berturk_train.main([str(x) for x in train_argv(data, manifest, base, storage)])
    assert str(run_dir) in str(e.value)
    assert os.listdir(str(run_dir)) == []                   # nothing was written into it


def test_print_run_id_starts_nothing(inputs, tmp_path, capsys, no_load):
    import re
    import berturk_train
    data, manifest, base = inputs
    storage = tmp_path / "runs"
    assert berturk_train.main([str(x) for x in train_argv(data, manifest, base, storage, "--print-run-id")]) is None
    out = capsys.readouterr().out
    assert re.match(r"^berturk-selection_train-s0-[0-9a-f]{8}\n$", out), out
    assert not storage.exists()


# ---- the final-seed path: --stop-after-epoch, --out, --postfit, --reference (plan 06-08) ----

T_EXACT = 1.2345678901234567                                 # 17 significant digits: any rounding would show


def set_flag(argv, flag, value=None):
    """argv with `flag` set to `value` (replaced or appended); value None removes the flag and its value."""
    argv = [str(x) for x in argv]
    if flag in argv:
        i = argv.index(flag)
        del argv[i:i + 2]
    return argv if value is None else argv + [flag, str(value)]


def exit_code(argv, capsys):
    """(exit code, stderr) of berturk_train.main(argv)."""
    import berturk_train
    with pytest.raises(SystemExit) as e:
        berturk_train.main([str(x) for x in argv])
    return e.value.code, capsys.readouterr().err


@pytest.mark.parametrize("k", [0, 3, 4])
def test_stop_after_epoch_boundaries_refused(inputs, tmp_path, capsys, no_load, k):
    data, manifest, base = inputs
    argv = set_flag(train_argv(data, manifest, base, tmp_path / "runs", "--print-run-id"), "--epochs", 3)
    code, err = exit_code(argv + ["--stop-after-epoch", str(k)], capsys)
    assert code == 2 and "--stop-after-epoch must be at least 1 and below --epochs (3)" in err, err
    assert not os.path.exists(str(tmp_path / "runs"))


@pytest.mark.parametrize("k", [1, 2])
def test_stop_after_epoch_boundaries_accepted(inputs, tmp_path, capsys, no_load, k):
    import berturk_train
    data, manifest, base = inputs
    argv = set_flag(train_argv(data, manifest, base, tmp_path / "runs", "--print-run-id"), "--epochs", 3)
    assert berturk_train.main(argv) is None
    full = capsys.readouterr().out.strip()
    assert berturk_train.main(argv + ["--stop-after-epoch", str(k)]) is None
    stopped = capsys.readouterr().out.strip()
    assert stopped.startswith("berturk-selection_train-s0-") and stopped != full    # its own trajectory, own run id
    assert not os.path.exists(str(tmp_path / "runs"))


@pytest.mark.timeout(TIMEOUT)
def test_stop_after_follows_the_schedule(inputs, tmp_path):
    """A4: --epochs 3 --stop-after-epoch 2 --out O is the 3-epoch run's epoch 2 (the linear schedule and its warm-up
    stay sized by --epochs), byte for byte on CPU."""
    data, manifest, base = inputs
    full = set_flag(train_argv(data, manifest, base, tmp_path / "a"), "--epochs", 3)
    rid_full = run_train(full + ["--print-run-id"]).stdout.strip()
    run_train(full)
    out = tmp_path / "ckpt" / "stop2"
    stopped = set_flag(train_argv(data, manifest, base, tmp_path / "b"), "--epochs", 3) + [
        "--stop-after-epoch", "2", "--out", str(out)]
    rid_stop = run_train(stopped + ["--print-run-id"]).stdout.strip()
    run_train(stopped)
    epoch2 = tmp_path / "a" / rid_full / "exports" / "epoch-002"
    assert _bytes(out / "model.safetensors") == _bytes(epoch2 / "model.safetensors")
    assert _bytes(out / "model.safetensors") != _bytes(tmp_path / "a" / rid_full / "exports" / "epoch-003"
                                                       / "model.safetensors")
    assert sorted(os.listdir(str(out))) == sorted(berturk.CHECKPOINT_FILES)
    assert sorted(os.listdir(str(tmp_path / "b" / rid_stop / "exports"))) == ["epoch-001", "epoch-002"]
    rec = _json(out / berturk.RECORD)
    assert rec["epoch"] == 2 and rec["epochs_schedule"] == 3 and rec["stop_after_epoch"] == 2
    assert rec["calibrated"] is False and rec["temperature"] == 1.0 and rec["postfit"] is None   # no --postfit
    assert rec["canonical"]["stop_after_epoch"] == 2 and rec["canonical"]["epochs"] == 3
    assert [row["epoch"] for row in rec["record"]["epochs"]] == [1, 2]


@pytest.fixture(scope="module")
def selection_canon(inputs):
    """(run id, canonical config) of the tiny selection command, captured from berturk_train.py --print-run-id."""
    import berturk_train
    data, manifest, base = inputs
    seen = []
    run_id = berturk.run_id

    def capture(canon):
        seen.append(canon)
        return run_id(canon)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(berturk, "run_id", capture)
        berturk_train.main([str(x) for x in train_argv(data, manifest, base, "unused-storage", "--print-run-id")])
    return run_id(seen[0]), seen[0]


def fake_record(rid, canon, temperature=T_EXACT):
    """A postfit/1 BERTurk record whose chosen candidate is epoch 1 of the 2-epoch schedule of `canon`."""
    cand = {"dump_sha256": "d" * 64, "objective": {"tp": 1, "fp": 9, "fn": 9}, "val_micro_f1_at_global_theta": 0.1,
            "val_micro_f1_at_0_5": 0.1}
    return {"schema": postfit.SCHEMA, "system_kind": "berturk",
            "protocol": {"final_split": postfit.FINAL_SPLIT, "final_seeds": [0, 1, 2], "epochs": EPOCHS},
            "rules": {"objective": postfit.OBJECTIVE, "tie_order": postfit.TIE_ORDER,
                      "epoch_transfer": postfit.EPOCH_TRANSFER_TEMPLATE % EPOCHS},
            "candidates": [cand],
            "chosen": {"train_run_id": rid, "dump_run_id": "berturk-test", "dump_sha256": "d" * 64,
                       "encoder_lr": canon["encoder_lr"], "epoch": 1, "epochs_schedule": EPOCHS, "stop_after_epoch": 1,
                       "canonical": canon, "temperature": temperature, "theta": {"shipped": {"global": 0.5}},
                       "val_micro_f1_at_global_theta": 0.1},
            "final": {"split": postfit.FINAL_SPLIT, "seeds": [0, 1, 2], "epochs": EPOCHS, "stop_after_epoch": 1}}


def write_json(path, obj):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def final_setup(inputs, selection_canon, tmp_path_factory):
    """Paths of a tiny record (chosen epoch 1 of 2) and its resolved reference check, a second record with its own
    resolved check, and the record's unresolved check: {record, reference, other_record, other_reference,
    unresolved, final_train}."""
    data, manifest, base = inputs
    rid, canon = selection_canon
    root = tmp_path_factory.mktemp("postfit")
    out = {"final_train": os.path.join(os.path.dirname(data), "final_train.jsonl")}
    for name, t in (("record", T_EXACT), ("other_record", 2.5)):
        path = write_json(root / (name + ".json"), fake_record(rid, canon, t))
        ref = name.replace("record", "reference")
        rec, sha = postfit.read_record(path)
        out[name] = path
        out[ref] = write_json(root / (ref + ".json"), berturk.reference_check(rec, path, sha, "tiny offline model"))
        if name == "record":
            out["unresolved"] = write_json(root / "unresolved.json", berturk.reference_check(rec, path, sha))
    return out


def final_argv(inputs, final_setup, storage, seed=0):
    """berturk_train.py arguments of the record's final seed, as pilot.berturk.final_command prints them."""
    data, manifest, base = inputs
    rec, _ = postfit.read_record(final_setup["record"])
    return berturk.final_command(rec, seed, final_setup["record"], final_setup["reference"], str(storage),
                                 str(storage) + "-ckpt/final", base=base, data=final_setup["final_train"],
                                 manifest=manifest)[1:]


BINDING = [  # (edit of the printed final command, the refusal that names the key)
    ("lr", lambda v, d: set_flag(v, "--lr", "3e-05"), "encoder_lr is 3e-05"),
    ("batch", lambda v, d: set_flag(v, "--batch", 8), "batch is 8"),
    ("max_len", lambda v, d: set_flag(v, "--max-len", 16), "max_len is 16"),
    ("epochs", lambda v, d: set_flag(v, "--epochs", 3), "epochs is 3"),
    ("seed", lambda v, d: set_flag(v, "--seed", 5), "--seed 5 is not one of the record's final seeds"),
    ("data", lambda v, d: set_flag(v, "--data", d), "trains the final seeds on final_train; --data is selection_train"),
    ("stop_missing", lambda v, d: set_flag(v, "--stop-after-epoch"), "pass --stop-after-epoch 1"),
    ("stop_wrong", lambda v, d: set_flag(set_flag(v, "--epochs", 3), "--stop-after-epoch", 2),
     "--stop-after-epoch is 2 here"),
]


@pytest.mark.parametrize("print_run_id", [False, True], ids=["train", "print_run_id"])
@pytest.mark.parametrize("name,edit,match", BINDING, ids=[b[0] for b in BINDING])
def test_postfit_binding_refuses_other_settings(inputs, final_setup, tmp_path, no_load, name, edit, match,
                                                print_run_id):
    data, _, _ = inputs
    storage = tmp_path / "runs"
    argv = edit(final_argv(inputs, final_setup, storage), data) + (["--print-run-id"] if print_run_id else [])
    refused(argv, storage, re.escape(match))
    assert not os.path.exists(str(storage) + "-ckpt")       # no --out either


def test_postfit_binding_accepts_the_printed_command(inputs, final_setup, tmp_path, capsys, no_load):
    import berturk_train
    storage = tmp_path / "runs"
    assert berturk_train.main(final_argv(inputs, final_setup, storage) + ["--print-run-id"]) is None
    assert capsys.readouterr().out.startswith("berturk-final_train-s0-")
    assert not os.path.exists(str(storage))


REFERENCE = [  # (the --reference given, the refusal)
    ("other_record", lambda s, d: s["other_reference"], "is of another postfit record"),
    ("investigate_unresolved", lambda s, d: s["unresolved"], "below 0.707 and no resolution is recorded"),
    ("missing", lambda s, d: str(d / "nonexistent.json"), "not found"),
    ("other_schema", lambda s, d: write_json(d / "schema.json", {"schema": "berturk-reference/0"}), "schema"),
    ("invalid_json", lambda s, d: d.joinpath("bad.json").write_text("{") and str(d / "bad.json"), "not valid JSON"),
    ("status_edited", lambda s, d: write_json(d / "edited.json", dict(_json(s["unresolved"]), status="ok")),
     "disagrees"),
]


@pytest.mark.parametrize("print_run_id", [False, True], ids=["train", "print_run_id"])
@pytest.mark.parametrize("name,ref,match", REFERENCE, ids=[r[0] for r in REFERENCE])
def test_reference_gate_refuses(inputs, final_setup, tmp_path, monkeypatch, no_load, name, ref, match, print_run_id):
    """BASE-03 (OQ2): refused before any fetch: the command uses --base berturk, whose fetch fails loudly."""
    import pilot.weights

    def no_fetch(*args, **kwargs):
        raise AssertionError("the pinned base was fetched before the BASE-03 refusal")
    monkeypatch.setattr(pilot.weights, "fetch", no_fetch)
    storage = tmp_path / "runs"
    argv = set_flag(set_flag(final_argv(inputs, final_setup, storage), "--base", "berturk"), "--reference",
                    ref(final_setup, tmp_path))
    refused(argv + (["--print-run-id"] if print_run_id else []), storage, re.escape(match))


@pytest.mark.parametrize("drop", ["--reference", "--out", "--postfit"])
def test_postfit_needs_out_and_reference(inputs, final_setup, tmp_path, capsys, no_load, drop):
    storage = tmp_path / "runs"
    code, err = exit_code(set_flag(final_argv(inputs, final_setup, storage), drop), capsys)
    want = {"--reference": "--reference", "--out": "--out", "--postfit": "--postfit"}[drop]
    assert code == 2 and want in err, err
    assert not os.path.exists(str(storage))


@pytest.mark.timeout(TIMEOUT)
def test_final_temperature_is_bit_exact(inputs, final_setup, tmp_path):
    """A9: the final berturk_train.json serves the record's T as the same JSON float (no rounding, no arithmetic) and
    carries pilot.postfit.final_block of the record; exports stay uncalibrated."""
    storage = tmp_path / "runs"
    argv = [x for x in final_argv(inputs, final_setup, storage) if x != "--no-export-epochs"]   # exports on
    run_train(argv)
    out = argv[argv.index("--out") + 1]
    rec_path = final_setup["record"]
    record, sha = postfit.read_record(rec_path)
    with open(os.path.join(out, berturk.RECORD), encoding="utf-8") as f:
        text = f.read()
    rec = json.loads(text)
    assert '"temperature": %s,' % repr(T_EXACT) in text
    assert rec["temperature"] == T_EXACT and json.dumps(rec["temperature"]) == json.dumps(T_EXACT)
    assert rec["calibrated"] is True and rec["epoch"] == 1 and rec["stop_after_epoch"] == 1
    assert rec["postfit"] == postfit.final_block(record, sha, rec_path) and rec["postfit"]["sha256"] == sha
    assert rec["postfit"]["sha256"] == sha256_file(rec_path)
    assert rec["canonical"]["postfit_sha256"] == sha and rec["canonical"]["stop_after_epoch"] == 1
    assert rec["canonical"]["split"] == "final_train" and rec["canonical"]["seed"] == 0
    run_dir = os.path.join(str(storage), rec["run_id"])
    export = _json(os.path.join(run_dir, "exports", "epoch-001", berturk.RECORD))
    assert export["calibrated"] is False and export["temperature"] == 1.0 and export["postfit"] is None


def test_selection_identity_has_no_final_keys(two_runs, selection_canon):
    rid, (a, _) = two_runs
    rec = _json(os.path.join(exports(a)[-1], berturk.RECORD))
    for canon in (rec["canonical"], selection_canon[1]):
        assert "stop_after_epoch" not in canon and "postfit_sha256" not in canon
    assert rec["run_id"] == rid == berturk.run_id(rec["canonical"]) == selection_canon[0]
    assert rec["stop_after_epoch"] is None and rec["postfit"] is None
