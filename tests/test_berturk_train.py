"""berturk_train.py as a recipe (Phase 6, BASE-01): a seeded CPU run reproduces byte for byte, every export is a
complete, self-describing checkpoint, and done.json marks a finished run.

Two identical runs (tiny offline base, the conftest synthetic selection_train, --device cpu --batch 4 --max-len 32,
2 epochs) go to two storage dirs; their weights, config and tokenizer files must be equal byte for byte (06-RESEARCH
probe 1 and Pitfall 6: the seed is set right before the head is created; CPU is bit-reproducible, MPS is recorded as
not). The record's timings and memory peaks differ between runs by design, so berturk_train.json is checked by
content, not bytes.

No marker: it runs in `make test` (no Ray, no weights, no data download; seconds).
"""
import json
import math
import os
import subprocess
import sys

import pytest

import pilot.data
from conftest import make_raw
from intents import INTENTS
from pilot import berturk
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
