"""berturk_train.py as a recipe: a seeded CPU run reproduces byte for byte, every export is a complete, self-describing
checkpoint, and done.json marks a finished run.

Two identical runs (tiny offline base, the conftest synthetic selection_train, --device cpu --batch 4 --max-len 32, 2
epochs) go to two storage dirs; their weights, config and tokenizer files must be equal byte for byte (the seed is set
right before the head is created, so the head is seeded too; CPU is bit-reproducible, MPS is recorded as not). The
record's timings and memory peaks differ between runs by design, so berturk_train.json is checked by content, not bytes.

The final-seed path: --stop-after-epoch K at its boundaries and on the selection trajectory (byte-identical to the
uninterrupted run's epoch K on CPU), the --postfit binding and the reference-check gate (--reference) refusing every
departure before any fetch, load or run dir (also with --print-run-id), and the final checkpoint serving the record's T
bit for bit. The record here is a hand-built postfit/1 record whose chosen canonical is the tiny selection command's own
(captured from berturk_train.py --print-run-id), chosen epoch 1 of 2.

The optimizer (the first reference-check fix): four AdamW groups with the fresh head at --head-lr and the encoder at
--lr, the Trainer's own class, kwargs and weight-decay rule, a constant head/encoder LR ratio under the Trainer's
linear schedule and warm-up, byte-identical weights to the Trainer's own optim adamw_torch when the head LR equals the
encoder LR, and the refusal of a base whose head is not freshly initialised.

The few-shot modes: every refusal of --fewshot-base and --fewshot-draw/--fewshot-k before a load, a fetch or a run
dir, under a synthetic frozen protocol.

TensorBoard (--tensorboard DIR): a third run of the identical command with the flag on must keep the run id, the
export bytes and the canonical config of the two flag-off runs, record neither the flag nor the logdir, and write one
event file (.pidN) under DIR/RUN_ID with every per-step and per-epoch scalar, read back with the crc-checked
tests/tb_events.py reader. A Trainer that stops clipping through trainer.accelerator is refused (the pre-clip norm
comes from that call), a DIR under results/, protocol.json or --storage is refused before any load, also with
--print-run-id, and the flag-off runs write no event file.

No marker: it runs in `make test` (no Ray, no weights, no data download; seconds), except the one weights-marked
check of the pinned BERTurk base's head (HF_HUB_OFFLINE=1 reuses the local Hub cache).
"""
import json
import math
import os
import re
import subprocess
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

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
Inputs = tuple[str, str, str]                     # the inputs fixture: (selection_train path, manifest, base dir)
TwoRuns = tuple[str, list[Path]]                  # the two_runs fixture: (run id, [run dir a, run dir b])
SyntheticRaw = Callable[..., tuple[str, dict[str, str]]]   # conftest's synthetic_raw fixture
FewshotEnv = types.SimpleNamespace                # the fewshot_env fixture


def _bytes(path: str) -> bytes:
    """The bytes of a file."""
    with open(path, "rb") as f:
        return f.read()


def _json(path: str) -> Any:
    """The parsed JSON at path."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def train_argv(data: str, manifest: str, base: str, storage: str | Path, *extra: Any) -> list[Any]:
    """berturk_train.py arguments of the smoke recipe on the synthetic split and the tiny base."""
    return ["--data", data, "--manifest", manifest, "--base", base, "--device", "cpu", "--epochs", str(EPOCHS),
            "--batch", str(BATCH), "--max-len", str(MAX_LEN), "--seed", "0", "--storage", str(storage)] + list(extra)


def run_train(argv: list[Any]) -> subprocess.CompletedProcess[str]:
    """Run berturk_train.py as a subprocess with the test interpreter (no PYTHONPATH, like a user's command)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "berturk_train.py"] + [str(x) for x in argv], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0, "berturk_train.py failed (exit %d):\n%s" % (r.returncode, r.stderr[-4000:])
    return r


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    """(selection_train path, manifest, base dir) of the synthetic snapshot and the tiny base, shared by the module."""
    root = tmp_path_factory.mktemp("inputs")
    raw_root, raw_sha256 = make_raw(str(root / "raw"))
    out_dir = str(root / "data")
    pilot.data.prepare(raw_root, out_dir, raw_sha256)
    return (os.path.join(out_dir, "splits", "selection_train.jsonl"), os.path.join(out_dir, "splits.json"),
            tiny_base(root / "base"))


@pytest.fixture(scope="module")
def two_runs(inputs: Inputs, tmp_path_factory: pytest.TempPathFactory) -> TwoRuns:
    """Two runs of the identical command into two storage dirs: (run id, [run dir a, run dir b])."""
    data, manifest, base = inputs
    root = tmp_path_factory.mktemp("runs")
    rid = run_train(train_argv(data, manifest, base, root / "a", "--print-run-id")).stdout.strip()
    dirs = []
    for name in ("a", "b"):
        run_train(train_argv(data, manifest, base, root / name))
        dirs.append(root / name / rid)
    return rid, dirs


def exports(run_dir: str | Path) -> list[str]:
    """The export dirs epoch-001..epoch-EPOCHS of a run dir."""
    return [os.path.join(str(run_dir), "exports", "epoch-%03d" % k) for k in range(1, EPOCHS + 1)]


def strings(obj: Any) -> Iterator[str]:
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
def test_cpu_runs_are_deterministic(two_runs: TwoRuns) -> None:
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
def test_exports_hold_the_checkpoint_files(two_runs: TwoRuns, inputs: Inputs) -> None:
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
            assert r["fewshot"] is None and "fewshot" not in rec["canonical"]   # no few-shot run: no proof, no key
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
def test_done_marker_written_after_the_last_epoch(two_runs: TwoRuns) -> None:
    rid, (a, _) = two_runs
    done = os.path.join(str(a), berturk.DONE)
    assert sorted(os.listdir(str(a))) == sorted([berturk.DONE, "exports"])  # no Trainer checkpoint or pickle
    rec = _json(done)
    assert rec["schema"] == berturk.RECORD_SCHEMA and rec["run_id"] == rid and rec["epoch"] == EPOCHS
    assert [row["epoch"] for row in rec["record"]["epochs"]] == list(range(1, EPOCHS + 1))
    assert rec == _json(os.path.join(exports(a)[-1], berturk.RECORD))     # the last export's record, whole
    assert os.stat(done).st_mtime_ns >= os.stat(os.path.join(exports(a)[-1], berturk.RECORD)).st_mtime_ns


# ---- refusals before any model load or run dir (Task 2) ----

@pytest.fixture
def no_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every tokenizer or model load fail loudly: a refusal test passes only if main stops before one."""
    import transformers

    def boom(*args: Any, **kwargs: Any) -> None:
        """Fail the test: a load was reached."""
        raise AssertionError("a model or tokenizer was loaded before the refusal")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)
    monkeypatch.setattr(transformers.BertForSequenceClassification, "from_pretrained", boom)
    monkeypatch.setattr(transformers.PreTrainedModel, "from_pretrained", boom)


def refused(argv: list[Any], storage: str | Path, match: str) -> None:
    """main(argv) raises SystemExit matching `match`, and the storage dir was not created."""
    import berturk_train
    with pytest.raises(SystemExit, match=match):
        berturk_train.main([str(x) for x in argv])
    assert not os.path.exists(str(storage)), "the refusal left %s behind" % storage


@pytest.mark.parametrize("name", ["val", "test", "hotels_probe"])
def test_refuses_a_non_train_split(inputs: Inputs, tmp_path: Path, no_load: None, name: str) -> None:
    data, manifest, base = inputs
    other = os.path.join(os.path.dirname(data), name + ".jsonl")
    refused(train_argv(other, manifest, base, tmp_path / "runs"), tmp_path / "runs", "has role")


def test_refuses_a_tampered_or_unregistered_split(tmp_path: Path, synthetic_raw: SyntheticRaw, inputs: Inputs,
                                                  no_load: None) -> None:
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


def test_refuses_a_row_missing_a_tag(inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                     no_load: None) -> None:
    import berturk_train
    data, manifest, base = inputs
    seen = {}

    def parse_dropping_a_tag(raw: bytes) -> list[dict[str, Any]]:   # a doctored file would fail its sha256 first
        """The parsed rows with tag TAGS[5] dropped from the first row's labels."""
        rows = pilot.data.parse_jsonl(raw)
        del rows[0]["expected"][berturk.TAGS[5]]
        seen["uid"] = rows[0]["uid"]
        return rows
    monkeypatch.setattr(berturk_train, "parse_jsonl", parse_dropping_a_tag)
    with pytest.raises(SystemExit) as e:
        berturk_train.main([str(x) for x in train_argv(data, manifest, base, tmp_path / "runs")])
    assert seen["uid"] in str(e.value) and berturk.TAGS[5] in str(e.value), str(e.value)
    assert not os.path.exists(str(tmp_path / "runs"))


def test_refuses_an_empty_selection(inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                    no_load: None) -> None:
    import berturk_train
    data, manifest, base = inputs
    monkeypatch.setattr(berturk_train, "parse_jsonl", lambda raw: [])
    refused(train_argv(data, manifest, base, tmp_path / "runs"), tmp_path / "runs", "no messages")


@pytest.mark.parametrize("tags", ["no_such_tag", "%s,%s" % (berturk.TAGS[0], berturk.TAGS[0]), "card"])
def test_refuses_bad_mask_tags(inputs: Inputs, tmp_path: Path, no_load: None, capsys: pytest.CaptureFixture[str],
                               tags: str) -> None:
    """--mask-tags masks a few-shot base only: alone it is an argument error naming --fewshot-base, before the
    split is read (pilot.berturk.check_mask_tags still refuses an unknown or repeated tag: test_berturk_unit)."""
    data, manifest, base = inputs
    code, err = exit_code(train_argv(data, manifest, base, tmp_path / "runs", "--mask-tags", tags), capsys)
    assert code == 2 and "--mask-tags masks a few-shot base only" in err and "--fewshot-base" in err, err
    assert not os.path.exists(str(tmp_path / "runs"))


@pytest.mark.parametrize("fault", list(berturk.BASE_FILES) + ["lowercasing"])
def test_refuses_an_incomplete_or_lowercasing_base(inputs: Inputs, tmp_path: Path, no_load: None, fault: str) -> None:
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


@pytest.mark.parametrize("flag", ["--epochs", "--batch", "--max-len", "--max-messages", "--cpus", "--lr",
                                  "--head-lr"])
def test_refuses_bad_numbers(inputs: Inputs, tmp_path: Path, no_load: None, capsys: pytest.CaptureFixture[str],
                             flag: str) -> None:
    import berturk_train
    data, manifest, base = inputs
    with pytest.raises(SystemExit) as e:
        berturk_train.main([str(x) for x in train_argv(data, manifest, base, tmp_path / "runs", flag, "0")])
    assert e.value.code == 2 and flag in capsys.readouterr().err
    assert not os.path.exists(str(tmp_path / "runs"))


def test_refuses_mps_when_unavailable(inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      no_load: None) -> None:
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


def test_refuses_an_existing_run_dir(inputs: Inputs, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                     no_load: None) -> None:
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


def test_print_run_id_starts_nothing(inputs: Inputs, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                     no_load: None) -> None:
    import re
    import berturk_train
    data, manifest, base = inputs
    storage = tmp_path / "runs"
    assert berturk_train.main([str(x) for x in train_argv(data, manifest, base, storage, "--print-run-id")]) is None
    out = capsys.readouterr().out
    assert re.match(r"^berturk-selection_train-s0-[0-9a-f]{8}\n$", out), out
    assert not storage.exists()


# ---- the final-seed path: --stop-after-epoch, --out, --postfit, --reference ----

T_EXACT = 1.2345678901234567                                 # 17 significant digits: any rounding would show


def set_flag(argv: list[Any], flag: str, value: Any = None) -> list[str]:
    """argv with `flag` set to `value` (replaced or appended); value None removes the flag and its value."""
    argv = [str(x) for x in argv]
    if flag in argv:
        i = argv.index(flag)
        del argv[i:i + 2]
    return argv if value is None else argv + [flag, str(value)]


def exit_code(argv: list[Any], capsys: pytest.CaptureFixture[str]) -> tuple[Any, str]:
    """(exit code, stderr) of berturk_train.main(argv)."""
    import berturk_train
    with pytest.raises(SystemExit) as e:
        berturk_train.main([str(x) for x in argv])
    return e.value.code, capsys.readouterr().err


@pytest.mark.parametrize("k", [0, 3, 4])
def test_stop_after_epoch_boundaries_refused(inputs: Inputs, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                             no_load: None, k: int) -> None:
    data, manifest, base = inputs
    argv = set_flag(train_argv(data, manifest, base, tmp_path / "runs", "--print-run-id"), "--epochs", 3)
    code, err = exit_code(argv + ["--stop-after-epoch", str(k)], capsys)
    assert code == 2 and "--stop-after-epoch must be at least 1 and below --epochs (3)" in err, err
    assert not os.path.exists(str(tmp_path / "runs"))


@pytest.mark.parametrize("k", [1, 2])
def test_stop_after_epoch_boundaries_accepted(inputs: Inputs, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                              no_load: None, k: int) -> None:
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
def test_stop_after_follows_the_schedule(inputs: Inputs, tmp_path: Path) -> None:
    """--epochs 3 --stop-after-epoch 2 --out O is the 3-epoch run's epoch 2 (the linear schedule and its warm-up
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
def selection_canon(inputs: Inputs) -> tuple[str, dict[str, Any]]:
    """(run id, canonical config) of the tiny selection command, captured from berturk_train.py --print-run-id."""
    import berturk_train
    data, manifest, base = inputs
    seen = []
    run_id = berturk.run_id

    def capture(canon: dict[str, Any]) -> str:
        """pilot.berturk.run_id, recording the canonical config."""
        seen.append(canon)
        return run_id(canon)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(berturk, "run_id", capture)
        berturk_train.main([str(x) for x in train_argv(data, manifest, base, "unused-storage", "--print-run-id")])
    return run_id(seen[0]), seen[0]


def fake_record(rid: str, canon: dict[str, Any], temperature: float = T_EXACT) -> dict[str, Any]:
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


def write_json(path: str | Path, obj: Any) -> str:
    """Write obj as indented JSON at path (parent dirs made); returns str(path)."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def final_setup(inputs: Inputs, selection_canon: tuple[str, dict[str, Any]],
                tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
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


def final_argv(inputs: Inputs, final_setup: dict[str, str], storage: str | Path, seed: int = 0) -> list[str]:
    """berturk_train.py arguments of the record's final seed, as pilot.berturk.final_command prints them."""
    data, manifest, base = inputs
    rec, _ = postfit.read_record(final_setup["record"])
    return berturk.final_command(rec, seed, final_setup["record"], final_setup["reference"], str(storage),
                                 str(storage) + "-ckpt/final", base=base, data=final_setup["final_train"],
                                 manifest=manifest)[1:]


BINDING = [  # (edit of the printed final command, the refusal that names the key)
    ("lr", lambda v, d: set_flag(v, "--lr", "3e-05"), "encoder_lr is 3e-05"),
    ("head_lr", lambda v, d: set_flag(v, "--head-lr", "2e-05"), "head_lr is 2e-05"),
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
def test_postfit_binding_refuses_other_settings(inputs: Inputs, final_setup: dict[str, str], tmp_path: Path,
                                                no_load: None, name: str, edit: Callable[[list[str], str], list[str]],
                                                match: str, print_run_id: bool) -> None:
    data, _, _ = inputs
    storage = tmp_path / "runs"
    argv = edit(final_argv(inputs, final_setup, storage), data) + (["--print-run-id"] if print_run_id else [])
    refused(argv, storage, re.escape(match))
    assert not os.path.exists(str(storage) + "-ckpt")       # no --out either


def test_final_command_prints_the_head_lr(inputs: Inputs, final_setup: dict[str, str], tmp_path: Path) -> None:
    """The head LR is bound, not transcribed: final_command prints the chosen canonical head_lr after --head-lr."""
    rec, _ = postfit.read_record(final_setup["record"])
    head_lr = rec["chosen"]["canonical"]["head_lr"]
    assert head_lr == berturk.HEAD_LR
    argv = final_argv(inputs, final_setup, tmp_path / "runs")
    assert argv.count("--head-lr") == 1 and argv[argv.index("--head-lr") + 1] == repr(head_lr) == "0.0001"
    assert argv[argv.index("--lr") + 1] == repr(rec["chosen"]["canonical"]["encoder_lr"])


def test_postfit_binding_accepts_the_printed_command(inputs: Inputs, final_setup: dict[str, str], tmp_path: Path,
                                                     capsys: pytest.CaptureFixture[str], no_load: None) -> None:
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
    # a status and fraction edited together agree with each other, but not with the record
    ("value_edited", lambda s, d: write_json(d / "value.json", dict(
        _json(s["unresolved"]), status="ok", value=dict(_json(s["unresolved"])["value"], fraction=[800, 1000]))),
     "is not pilot.berturk.reference_check of the postfit record"),
]


@pytest.mark.parametrize("print_run_id", [False, True], ids=["train", "print_run_id"])
@pytest.mark.parametrize("name,ref,match", REFERENCE, ids=[r[0] for r in REFERENCE])
def test_reference_gate_refuses(inputs: Inputs, final_setup: dict[str, str], tmp_path: Path,
                                monkeypatch: pytest.MonkeyPatch, no_load: None, name: str,
                                ref: Callable[[dict[str, str], Path], str], match: str, print_run_id: bool) -> None:
    """The reference-check gate refuses before any fetch: the command uses --base berturk, whose fetch fails loudly."""
    import pilot.weights

    def no_fetch(*args: Any, **kwargs: Any) -> None:
        """Fail the test: the pinned fetch was reached."""
        raise AssertionError("the pinned base was fetched before the reference-check refusal")
    monkeypatch.setattr(pilot.weights, "fetch", no_fetch)
    storage = tmp_path / "runs"
    argv = set_flag(set_flag(final_argv(inputs, final_setup, storage), "--base", "berturk"), "--reference",
                    ref(final_setup, tmp_path))
    refused(argv + (["--print-run-id"] if print_run_id else []), storage, re.escape(match))


def test_a_leftover_out_tmp_is_refused_never_deleted(inputs: Inputs, final_setup: dict[str, str], tmp_path: Path,
                                                     no_load: None) -> None:
    """A <out>.tmp sibling this process did not make is refused before training, and by
    _write_checkpoint itself (an export's or --out's temp dir), never silently deleted."""
    import berturk_train
    storage = tmp_path / "runs"
    argv = final_argv(inputs, final_setup, storage)
    out = argv[argv.index("--out") + 1]
    os.makedirs(out + ".tmp")
    keep = os.path.join(out + ".tmp", "keep.txt")
    with open(keep, "w", encoding="utf-8") as f:
        f.write("not ours")
    refused(argv, storage, re.escape("%s.tmp exists" % os.path.abspath(out)))
    with pytest.raises(SystemExit, match=re.escape("%s.tmp exists" % out)):
        berturk_train._write_checkpoint(None, None, out, None)
    assert os.path.isfile(keep) and not os.path.exists(out)


@pytest.mark.parametrize("drop", ["--reference", "--out", "--postfit"])
def test_postfit_needs_out_and_reference(inputs: Inputs, final_setup: dict[str, str], tmp_path: Path,
                                         capsys: pytest.CaptureFixture[str], no_load: None, drop: str) -> None:
    storage = tmp_path / "runs"
    code, err = exit_code(set_flag(final_argv(inputs, final_setup, storage), drop), capsys)
    want = {"--reference": "--reference", "--out": "--out", "--postfit": "--postfit"}[drop]
    assert code == 2 and want in err, err
    assert not os.path.exists(str(storage))


@pytest.mark.timeout(TIMEOUT)
def test_final_temperature_is_bit_exact(inputs: Inputs, final_setup: dict[str, str], tmp_path: Path) -> None:
    """The final berturk_train.json serves the record's T as the same JSON float (no rounding, no arithmetic) and
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


def test_selection_identity_has_no_final_keys(two_runs: TwoRuns, selection_canon: tuple[str, dict[str, Any]]) -> None:
    rid, (a, _) = two_runs
    rec = _json(os.path.join(exports(a)[-1], berturk.RECORD))
    for canon in (rec["canonical"], selection_canon[1]):
        assert "stop_after_epoch" not in canon and "postfit_sha256" not in canon
    assert rec["run_id"] == rid == berturk.run_id(rec["canonical"]) == selection_canon[0]
    assert rec["stop_after_epoch"] is None and rec["postfit"] is None


# ---- the optimizer: the head at --head-lr, the encoder at --lr (the first reference-check fix) ----

ENC_LR, HEAD_LR = 2e-05, berturk.HEAD_LR


def test_optimizer_groups_split_head_and_encoder() -> None:
    """Torch-free, on names: four groups in the fixed order, the head is exactly the classifier.* names, decay follows
    the given names, every name in exactly one group; refused unless the head is exactly the fresh names."""
    names = ["bert.embeddings.word_embeddings.weight", "bert.embeddings.LayerNorm.weight",
             "bert.encoder.layer.0.attention.self.query.weight", "bert.encoder.layer.0.attention.self.query.bias",
             "bert.pooler.dense.weight", "bert.pooler.dense.bias", "classifier.weight", "classifier.bias"]
    decay = [n for n in names if n.endswith(".weight") and "LayerNorm" not in n]
    fresh = ["classifier.bias", "classifier.weight"]
    groups = berturk.optimizer_groups(names, decay, fresh, ENC_LR, HEAD_LR, 0.01)
    assert [(g["group"], g["decay"], g["lr"], g["weight_decay"]) for g in groups] == [
        ("encoder", True, ENC_LR, 0.01), ("encoder", False, ENC_LR, 0.0), ("head", True, HEAD_LR, 0.01),
        ("head", False, HEAD_LR, 0.0)]
    assert groups[0]["params"] == ["bert.embeddings.word_embeddings.weight",
                                   "bert.encoder.layer.0.attention.self.query.weight", "bert.pooler.dense.weight"]
    assert groups[1]["params"] == ["bert.embeddings.LayerNorm.weight", "bert.encoder.layer.0.attention.self.query.bias",
                                   "bert.pooler.dense.bias"]
    assert groups[2]["params"] == ["classifier.weight"] and groups[3]["params"] == ["classifier.bias"]
    placed = [n for g in groups for n in g["params"]]
    assert sorted(placed) == sorted(names) and len(placed) == len(names)          # each name exactly once
    for bad in (fresh + ["bert.pooler.dense.weight"],    # a pretrained parameter reported as newly initialised
                ["classifier.weight"],                   # a head parameter that was loaded, not fresh
                []):                                     # nothing fresh: the base already holds a head
        with pytest.raises(SystemExit, match="the classifier head must be exactly the newly initialised parameters"):
            berturk.optimizer_groups(names, decay, bad, ENC_LR, HEAD_LR, 0.01)
    with pytest.raises(SystemExit, match="head none"):
        berturk.optimizer_groups([n for n in names if not n.startswith("classifier.")], decay, [], ENC_LR, HEAD_LR,
                                 0.01)


def _rows(data: str, tok: Any, n: int | None = None) -> list[dict[str, Any]]:
    """The tokenised training rows berturk_train.main builds from the synthetic selection split."""
    with open(data, "rb") as f:
        rows = pilot.data.parse_jsonl(f.read())
    out = []
    for r in rows[:n]:
        enc = tok(r["state"], truncation=True, max_length=MAX_LEN)
        out.append({"input_ids": enc["input_ids"], "token_type_ids": enc["token_type_ids"],
                    "attention_mask": enc["attention_mask"], "labels": berturk.label_vector(r)})
    return out


def _load(base: str, seed: int = 0) -> tuple[Any, list[str]]:
    """(model, newly initialised names) of the tiny base, loaded exactly as berturk_train.main loads a base."""
    import torch
    import transformers
    transformers.set_seed(seed)
    model, info = transformers.BertForSequenceClassification.from_pretrained(
        base, num_labels=len(berturk.TAGS), problem_type=berturk.PROBLEM_TYPE, id2label=berturk.id2label(),
        label2id=berturk.label2id(), dtype=torch.float32, local_files_only=True, output_loading_info=True)
    return model, info["missing_keys"]


def _args(out: str | Path, epochs: int = 1) -> Any:
    """The TrainingArguments berturk_train.main builds (the recipe constants, the smoke batch, CPU)."""
    import transformers
    return transformers.TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=BATCH, gradient_accumulation_steps=1, num_train_epochs=epochs,
        learning_rate=ENC_LR, weight_decay=berturk.WEIGHT_DECAY, warmup_steps=berturk.WARMUP_RATIO,
        lr_scheduler_type=berturk.LR_SCHEDULER, max_grad_norm=berturk.MAX_GRAD_NORM, optim=berturk.OPTIM, seed=0,
        data_seed=0, use_cpu=True, fp16=False, bf16=False, save_strategy="no", eval_strategy="no",
        logging_strategy="epoch", report_to="none", disable_tqdm=True, dataloader_num_workers=0,
        dataloader_pin_memory=False)


@pytest.fixture
def keep_threads() -> Iterator[None]:
    """Restore torch's thread count after the test."""
    import torch
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


def test_head_lr_param_groups_in_the_trainer(inputs: Inputs, tmp_path: Path, keep_threads: None) -> None:
    import berturk_train
    import torch
    import transformers
    data, _, base = inputs
    model, fresh = _load(base)
    assert sorted(fresh) == ["classifier.bias", "classifier.weight"]
    args = _args(tmp_path / "t")
    optimizer, groups = berturk_train.build_optimizer(model, args, HEAD_LR, fresh)
    pg = optimizer.param_groups
    assert [(g["group"], g["decay"]) for g in groups] == [("encoder", True), ("encoder", False), ("head", True),
                                                          ("head", False)]
    assert [g["lr"] for g in pg] == [ENC_LR, ENC_LR, HEAD_LR, HEAD_LR]
    assert [g["weight_decay"] for g in pg] == [0.01, 0.0, 0.01, 0.0]
    assert pg[2]["params"][0] is model.classifier.weight and len(pg[2]["params"]) == 1
    assert pg[3]["params"][0] is model.classifier.bias and len(pg[3]["params"]) == 1
    named = dict(model.named_parameters())
    assert all(p is named[n] for g, q in zip(groups, pg) for n, p in zip(g["params"], q["params"]))
    tok = transformers.AutoTokenizer.from_pretrained(base, local_files_only=True)
    trainer = transformers.Trainer(model=model, args=args, train_dataset=_rows(data, tok),
                                   data_collator=transformers.DataCollatorWithPadding(tok),
                                   compute_loss_func=berturk_train.masked_bce, optimizers=(optimizer, None))
    decay = set(trainer.get_decay_parameter_names(model))              # the Trainer's own rule, on an instance
    assert {n for g in groups if g["decay"] for n in g["params"]} == decay
    assert {n for g in groups if not g["decay"] for n in g["params"]} == set(named) - decay
    cls, kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(args)
    assert type(optimizer) is cls is torch.optim.AdamW
    assert all(g["betas"] == kwargs["betas"] and g["eps"] == kwargs["eps"] for g in pg)
    assert trainer.optimizer is optimizer
    n_steps = 20
    sched = trainer.create_scheduler(num_training_steps=n_steps)
    warm = args.get_warmup_steps(n_steps)
    assert warm == 2
    enc, head = [], []
    for _ in range(n_steps):
        enc.append(pg[0]["lr"])
        head.append(pg[2]["lr"])
        assert pg[1]["lr"] == pg[0]["lr"] and pg[3]["lr"] == pg[2]["lr"]
        assert math.isclose(head[-1], enc[-1] * (HEAD_LR / ENC_LR), rel_tol=1e-12, abs_tol=0.0)
        optimizer.step()                                                # no gradients: no parameter moves
        sched.step()
    assert enc[0] == 0.0 and enc[1] < ENC_LR and enc[warm] == ENC_LR and head[warm] == HEAD_LR   # the warm-up
    assert all(enc[k] > enc[k + 1] for k in range(warm, n_steps - 1))                          # then linear decay
    assert math.isclose(HEAD_LR / ENC_LR, 5.0, rel_tol=1e-12)


def _train_one_epoch(base: str, data: str, out: str | Path, own_optimizer: bool) -> dict[str, Any]:
    """The tiny base trained for one epoch with berturk_train's optimizer at head LR = encoder LR (own_optimizer
    False) or with the Trainer's own optim adamw_torch (True); the trained state_dict."""
    import berturk_train
    import transformers
    model, fresh = _load(base)
    args = _args(out)
    tok = transformers.AutoTokenizer.from_pretrained(base, local_files_only=True)
    opt = (None, None) if own_optimizer else (berturk_train.build_optimizer(model, args, ENC_LR, fresh)[0], None)
    trainer = transformers.Trainer(model=model, args=args, train_dataset=_rows(data, tok),
                                   data_collator=transformers.DataCollatorWithPadding(tok),
                                   compute_loss_func=berturk_train.masked_bce, optimizers=opt)
    trainer.train()
    if own_optimizer:
        assert len(trainer.optimizer.param_groups) == 2              # the Trainer's own decay / no-decay split
    else:
        assert len(trainer.optimizer.param_groups) == 4
    return {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}


@pytest.mark.timeout(TIMEOUT)
def test_tied_head_lr_equals_the_trainers_own_adamw(inputs: Inputs, tmp_path: Path, keep_threads: None) -> None:
    """The param groups change nothing but the head LR: tied, they train to exactly the Trainer's own weights."""
    import torch
    data, _, base = inputs
    ours = _train_one_epoch(base, data, tmp_path / "ours", own_optimizer=False)
    own = _train_one_epoch(base, data, tmp_path / "own", own_optimizer=True)
    assert sorted(ours) == sorted(own)
    differ = [k for k in ours if not torch.equal(ours[k], own[k])]
    assert not differ, "tensors differ from the Trainer's own adamw_torch: %s" % differ
    model, _ = _load(base)
    start = model.state_dict()
    assert not torch.equal(ours["classifier.weight"], start["classifier.weight"])               # it trained


@pytest.mark.timeout(TIMEOUT)
def test_head_lr_reaches_the_record_and_the_weights(inputs: Inputs, two_runs: TwoRuns, tmp_path: Path) -> None:
    data, manifest, base = inputs
    rid, (a, _) = two_runs
    default = _json(os.path.join(exports(a)[0], berturk.RECORD))
    assert default["canonical"]["head_lr"] == HEAD_LR
    assert [g["lr"] for g in default["record"]["optimizer_groups"]] == [ENC_LR, ENC_LR, HEAD_LR, HEAD_LR]
    argv = train_argv(data, manifest, base, tmp_path / "tied", "--head-lr", "2e-05")
    tied_rid = run_train(argv + ["--print-run-id"]).stdout.strip()
    assert tied_rid != rid and tied_rid.startswith("berturk-selection_train-s0-")
    run_train(argv)
    tied = _json(os.path.join(str(tmp_path / "tied" / tied_rid), "exports", "epoch-001", berturk.RECORD))
    assert tied["canonical"]["head_lr"] == ENC_LR and tied["canonical"]["encoder_lr"] == ENC_LR
    groups = tied["record"]["optimizer_groups"]
    assert [g["lr"] for g in groups] == [ENC_LR] * 4
    assert [g["n_tensors"] for g in groups] == [g["n_tensors"] for g in default["record"]["optimizer_groups"]]
    assert set(groups[0]) == {"group", "decay", "lr", "weight_decay", "n_tensors"}            # counts, never names
    assert _bytes(os.path.join(str(tmp_path / "tied" / tied_rid), "exports", "epoch-001", "model.safetensors")) \
        != _bytes(os.path.join(exports(a)[0], "model.safetensors"))


def test_refuses_a_base_whose_head_is_not_fresh(inputs: Inputs, tmp_path: Path, keep_threads: None) -> None:
    """A base that already holds a classifier (saved as BertForSequenceClassification) is refused before the run dir:
    its head would not be the freshly initialised parameters the head LR is meant for."""
    import shutil
    import torch
    import transformers
    data, manifest, base = inputs
    bad = str(tmp_path / "base-with-head")
    model, _ = _load(base)
    model.save_pretrained(bad)
    for f in berturk.TOKENIZER_FILES:
        shutil.copyfile(os.path.join(base, f), os.path.join(bad, f))
    loaded, info = transformers.BertForSequenceClassification.from_pretrained(
        bad, num_labels=len(berturk.TAGS), dtype=torch.float32, local_files_only=True, output_loading_info=True)
    assert not info["missing_keys"]                                    # nothing is newly initialised (a set in 5.19)
    refused(train_argv(data, manifest, bad, tmp_path / "runs"), tmp_path / "runs",
            "the classifier head must be exactly the newly initialised parameters")


@pytest.mark.weights
def test_pinned_base_head_is_the_fresh_classifier(keep_threads: None) -> None:
    """The pinned BERTurk base: from_pretrained newly initialises exactly classifier.weight and classifier.bias, and
    optimizer_groups accepts them as the head."""
    import torch
    import transformers
    from pilot import pins
    from pilot.weights import fetch
    model, info = transformers.BertForSequenceClassification.from_pretrained(
        fetch(pins.BERTURK), num_labels=len(berturk.TAGS), problem_type=berturk.PROBLEM_TYPE,
        id2label=berturk.id2label(), label2id=berturk.label2id(), dtype=torch.float32, local_files_only=True,
        output_loading_info=True)
    assert sorted(info["missing_keys"]) == ["classifier.bias", "classifier.weight"]
    names = [n for n, _ in model.named_parameters()]
    decay = transformers.Trainer.get_decay_parameter_names(None, model)
    groups = berturk.optimizer_groups(names, decay, info["missing_keys"], ENC_LR, HEAD_LR, berturk.WEIGHT_DECAY)
    assert groups[2]["params"] == ["classifier.weight"] and groups[3]["params"] == ["classifier.bias"]
    assert sum(len(g["params"]) for g in groups[:2]) == len(names) - 2


# ---- the few-shot modes: every refusal before a load, a fetch or a run dir ----

@pytest.fixture
def fewshot_env(tmp_path: Path, synthetic_raw: SyntheticRaw, monkeypatch: pytest.MonkeyPatch,
                no_load: None) -> types.SimpleNamespace:
    """tests/test_fewshot_berturk.py's synthetic frozen protocol installed in this process (tag card, k 1 and 2, draw
    0, the tiny recipes, a postfit record whose chosen canonical is the tiny unmasked command's), with every model load
    (no_load) and the pinned fetch failing the test if reached. Returns a namespace: ns (synthetic_protocol's),
    base(*extra, data=None) and cont(base_dir, *extra) argv builders, base_canon (the masked base command's canonical
    config) and plant(canonical=..., record=True, name=...) -> a tiny BERTurk checkpoint dir (the base's files, a
    config.json scoring INTENTS in order and a berturk_train.json naming the base run), as a continued run's --base."""
    import shutil
    import berturk_train
    import pilot.weights
    from conftest import prepared
    from fewshot_synthetic import postfit_systems, synthetic_protocol
    from test_fewshot_berturk import COMMON, rules

    def no_fetch(*a: Any, **kw: Any) -> None:
        """Fail the test: the pinned fetch was reached."""
        raise AssertionError("the pinned fetch reached; a few-shot refusal must come first")
    monkeypatch.setattr(pilot.weights, "fetch", no_fetch)
    tiny = tiny_base(tmp_path / "tiny")
    seen, run_id = [], berturk.run_id

    def canon_of(argv: list[Any]) -> dict[str, Any]:
        """The canonical config main gives argv (through --print-run-id)."""
        def capture(canon: dict[str, Any]) -> str:
            """pilot.berturk.run_id, recording the canonical config."""
            seen.append(canon)
            return run_id(canon)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(berturk, "run_id", capture)
            berturk_train.main([str(x) for x in argv] + ["--print-run-id"])
        return seen[-1]
    pre_dir, pre_manifest = prepared(tmp_path, synthetic_raw, name="pre")
    chosen = canon_of(["--data", os.path.join(pre_dir, "splits", "selection_train.jsonl"), "--manifest", pre_manifest,
                       "--base", tiny, "--storage", tmp_path / "unused", "--epochs", "2"] + COMMON)
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, rules=rules(),
                            systems=postfit_systems(tmp_path, {"berturk": chosen}))

    def split(name: str) -> str:
        """The path of a synthetic split file."""
        return os.path.join(ns.out_dir, "splits", name + ".jsonl")

    def base(*extra: Any, data: str | None = None) -> list[str]:
        """argv of the frozen masked-base command, `extra` appended."""
        return [str(x) for x in ["--data", data or split("selection_train"), "--manifest", ns.manifest, "--base", tiny,
                                 "--out", tmp_path / "o", "--storage", tmp_path / "runs", "--epochs", "2",
                                 "--stop-after-epoch", "1", "--fewshot-base"] + COMMON + list(extra)]

    def cont(base_dir: str, *extra: Any) -> list[str]:
        """argv of the continued command (draw 0, k 2) from base_dir, `extra` appended."""
        return [str(x) for x in ["--data", split("selection_train"), "--manifest", ns.manifest, "--base", base_dir,
                                 "--out", tmp_path / "o", "--storage", tmp_path / "runs", "--epochs", "1",
                                 "--fewshot-draw", "0", "--fewshot-k", "2"] + COMMON + list(extra)]

    base_canon = canon_of(base())

    def plant(canonical: dict[str, Any] | None = None, record: bool = True, name: str = "planted") -> str:
        """A tiny BERTurk checkpoint dir naming a base run (see the fixture docstring); returns it."""
        d = str(tmp_path / name)
        shutil.copytree(tiny, d)
        cfg = _json(os.path.join(d, "config.json"))
        cfg.update(id2label={str(i): t for i, t in berturk.id2label().items()}, label2id=berturk.label2id(),
                   problem_type=berturk.PROBLEM_TYPE)
        write_json(os.path.join(d, "config.json"), cfg)
        canon = base_canon if canonical is None else canonical
        if record:
            write_json(os.path.join(d, berturk.RECORD), {"schema": berturk.RECORD_SCHEMA, "run_id": run_id(canon),
                                                        "canonical": canon, "epoch": 1, "temperature": 1.0})
        return d
    return types.SimpleNamespace(ns=ns, base=base, cont=cont, plant=plant, base_canon=base_canon, chosen=chosen,
                                 tmp=tmp_path, storage=tmp_path / "runs", out=tmp_path / "o")


def fewshot_refused(env: types.SimpleNamespace, argv: list[str], match: str) -> None:
    """main(argv) raises SystemExit matching `match`; no run dir, no --out and no load (no_load) followed."""
    import berturk_train
    with pytest.raises(SystemExit, match=match):
        berturk_train.main(argv)
    assert not env.storage.exists() and not env.out.exists()


def test_fewshot_base_and_continued_commands_print_their_run_ids_without_a_load(
        fewshot_env: FewshotEnv, capsys: pytest.CaptureFixture[str]) -> None:
    """The positive path the refusals below depart from: the frozen base command (with and without the frozen
    --mask-tags spelled out) and a continued command from a masked base of this protocol print a run id; no model,
    tokenizer or fetch is reached and nothing is created."""
    import berturk_train
    berturk_train.main(fewshot_env.base("--print-run-id"))
    rid = capsys.readouterr().out.strip()
    assert re.fullmatch(r"berturk-selection_train-s0-[0-9a-f]{8}", rid), rid
    berturk_train.main(fewshot_env.base("--mask-tags", "card", "--print-run-id"))
    assert capsys.readouterr().out.strip() == rid
    assert fewshot_env.base_canon["fewshot"] == {"protocol_sha256": fewshot_env.ns.pin, "stage": "base"}
    assert fewshot_env.base_canon["mask_tags"] == ["card"]
    berturk_train.main(fewshot_env.cont(fewshot_env.plant(), "--print-run-id"))
    cont = capsys.readouterr().out.strip()
    assert re.fullmatch(r"berturk-selection_train-s0-[0-9a-f]{8}", cont) and cont != rid
    assert not fewshot_env.storage.exists() and not fewshot_env.out.exists()


@pytest.mark.parametrize("case", ["missing", "unpinned", "changed"])
def test_fewshot_protocol_is_required_before_the_split_is_read(fewshot_env: FewshotEnv, monkeypatch: pytest.MonkeyPatch,
                                                               case: str) -> None:
    """require_frozen() runs first: a missing, unpinned or changed protocol is refused before verify_split_file
    reads --data (monkeypatched to fail the test if reached), for the base and the continued command."""
    import berturk_train
    from pilot import protocol as pr

    def no_split(*a: Any, **kw: Any) -> None:
        """Fail the test: the split was read."""
        raise AssertionError("verify_split_file reached before the protocol check")
    monkeypatch.setattr(berturk_train, "verify_split_file", no_split)
    base = fewshot_env.plant()
    if case == "missing":
        monkeypatch.setattr(pr, "PROTOCOL_PATH", str(fewshot_env.tmp / "absent.json"))
        match = "absent.json not found; the decision protocol must be frozen"
    elif case == "unpinned":
        monkeypatch.setattr(pr, "PROTOCOL_SHA256", None)
        match = "the decision protocol is not frozen"
    else:
        with open(fewshot_env.ns.protocol_path, "ab") as f:
            f.write(b" ")
        match = "the protocol may not change after the freeze"
    for argv in (fewshot_env.base(), fewshot_env.cont(base)):
        fewshot_refused(fewshot_env, argv, match)


def test_fewshot_decision_record_is_required(fewshot_env: FewshotEnv) -> None:
    """The projection's compute decision record names the mode; without it no few-shot command trains."""
    os.remove(fewshot_env.ns.decision_path)
    fewshot_refused(fewshot_env, fewshot_env.base("--print-run-id"), "run `python fewshot.py --projection")


def test_fewshot_protocol_mask_label_and_dtype_must_be_the_trainers(fewshot_env: FewshotEnv,
                                                                     tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """The frozen rules.base_runs.berturk name the mask label and dtype berturk_train.py trains with; a protocol
    naming another is refused before the split is read."""
    from fewshot_synthetic import postfit_systems, synthetic_protocol
    from test_fewshot_berturk import rules
    for key, value in (("mask_label", -1.0), ("dtype", "float16")):
        r = rules()
        r["base_runs"]["berturk"][key] = value
        sub = tmp_path / key                                 # a protocol is frozen once per path
        sub.mkdir()
        synthetic_protocol(sub, synthetic_raw, monkeypatch, rules=r,
                           systems=postfit_systems(sub, {"berturk": fewshot_env.chosen}))
        fewshot_refused(fewshot_env, fewshot_env.base("--print-run-id"), "cannot run this protocol unchanged")


def test_mask_tags_other_than_the_protocols_are_refused_with_fewshot_base(fewshot_env: FewshotEnv) -> None:
    """--mask-tags duplicates a frozen value, so any other list (another tag, an extra tag, a repeated tag) is
    refused naming the frozen tags."""
    for tags in ("loan", "card,loan", "card,card"):
        fewshot_refused(fewshot_env, fewshot_env.base("--mask-tags", tags, "--print-run-id"),
                        "--mask-tags %s is not the frozen protocol's held-out tags card; pass them" % tags)


FLAG_ERRORS = [  # (argv edit, the argument error it gives)
    ("draw_without_k", lambda e: set_flag(e.cont(e.plant()), "--fewshot-k"), "--fewshot-draw and --fewshot-k go"),
    ("k_without_draw", lambda e: set_flag(e.cont(e.plant()), "--fewshot-draw"), "--fewshot-draw and --fewshot-k go"),
    ("base_and_draw", lambda e: e.cont(e.plant(), "--fewshot-base"), "pass one of them"),
    ("stop_with_draw", lambda e: e.cont(e.plant(), "--stop-after-epoch", "1", "--epochs", "2"),
     "--fewshot-draw takes no --stop-after-epoch"),
    ("mask_with_draw", lambda e: e.cont(e.plant(), "--mask-tags", "card"), "--mask-tags masks a few-shot base only"),
    ("base_with_postfit", lambda e: e.base("--postfit", "p.json", "--reference", "r.json"),
     "--fewshot-base takes no --postfit or --reference"),
    ("base_with_reference", lambda e: e.base("--reference", "r.json"), "--fewshot-base takes no --postfit"),
    ("draw_with_postfit", lambda e: e.cont(e.plant(), "--postfit", "p.json"), "--fewshot-draw takes no --postfit"),
    ("base_with_max_messages", lambda e: e.base("--max-messages", "5"), "--fewshot-base takes no --max-messages"),
    ("draw_with_max_messages", lambda e: e.cont(e.plant(), "--max-messages", "5"),
     "--fewshot-draw takes no --max-messages"),
]


@pytest.mark.parametrize("name,edit,match", FLAG_ERRORS, ids=[f[0] for f in FLAG_ERRORS])
def test_fewshot_flag_combinations_are_argument_errors(fewshot_env: FewshotEnv,
                                                       capsys: pytest.CaptureFixture[str], name: str,
                                                       edit: Callable[[types.SimpleNamespace], list[str]],
                                                       match: str) -> None:
    """A few-shot mode is one frozen-protocol command; every other flag combination exits 2 naming the fix."""
    code, err = exit_code(edit(fewshot_env), capsys)
    assert code == 2 and match in err, err
    assert not fewshot_env.storage.exists() and not fewshot_env.out.exists()


def test_fewshot_base_on_another_split_is_refused(fewshot_env: FewshotEnv) -> None:
    """The few-shot source is the frozen selection_train; final_train (a train split) is refused by
    check_source."""
    final = os.path.join(fewshot_env.ns.out_dir, "splits", "final_train.jsonl")
    fewshot_refused(fewshot_env, fewshot_env.base("--print-run-id", data=final),
                    "the source entry has sha256 .* the frozen few-shot source")


def test_fewshot_base_with_another_recipe_is_refused(fewshot_env: FewshotEnv) -> None:
    """The base recipe is rules.base_runs.berturk: --epochs 10 is refused naming epochs, a seed outside the base
    seeds and another batch likewise."""
    fewshot_refused(fewshot_env, fewshot_env.base("--epochs", "10", "--print-run-id"),
                    "epochs is 10 here but 2 in the frozen rules.base_runs.berturk")
    fewshot_refused(fewshot_env, fewshot_env.base("--seed", "5", "--print-run-id"),
                    "seed 5 is not one of the base seeds \\[0\\]")
    fewshot_refused(fewshot_env, fewshot_env.base("--batch", "8", "--print-run-id"),
                    "batch is 8 here but 4 in the frozen rules.base_runs.berturk")


def test_fewshot_masked_base_refuses_a_row_missing_an_old_tag(fewshot_env: FewshotEnv,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """Masked-base edge: label_vector never reads a missing label as a negative, so a masked base whose row lacks an
    old-tag label is refused naming the uid and the tag (a missing held-out label is masked anyway)."""
    import berturk_train
    seen = {}

    def parse_dropping_a_tag(raw: bytes) -> list[dict[str, Any]]:   # a doctored file would fail its sha256 first
        """The parsed rows with tag loan dropped from the first row's labels."""
        rows = pilot.data.parse_jsonl(raw)
        del rows[0]["expected"]["loan"]
        seen["uid"] = rows[0]["uid"]
        return rows
    monkeypatch.setattr(berturk_train, "parse_jsonl", parse_dropping_a_tag)
    fewshot_refused(fewshot_env, fewshot_env.base("--print-run-id"), "does not label tag loan")
    assert seen["uid"]


def test_fewshot_continued_refuses_a_selected_uid_missing_from_the_rows(fewshot_env: FewshotEnv,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Continued-run edge: a continued run builds its rows from the frozen lists only (pilot.fewshot.berturk_rows), so a
    support uid missing from the rows is refused before any load, never trained on a shortened set (check_source,
    which would refuse first, is bypassed here to reach that refusal)."""
    import berturk_train
    from pilot import fewshot as fs
    support = sorted(fs.selection(fewshot_env.ns.block, 0, 2)["support"])

    def parse_dropping_a_support_row(raw: bytes) -> list[dict[str, Any]]:
        """The parsed rows without the first support uid."""
        return [r for r in pilot.data.parse_jsonl(raw) if r["uid"] != support[0]]
    monkeypatch.setattr(berturk_train, "parse_jsonl", parse_dropping_a_support_row)
    monkeypatch.setattr(berturk_train.fewshot, "check_source", lambda block, entry, rows: None)
    fewshot_refused(fewshot_env, fewshot_env.cont(fewshot_env.plant(), "--print-run-id"),
                    "selected uids are not in the rows")


def test_fewshot_continued_from_the_pinned_base_is_refused(fewshot_env: FewshotEnv) -> None:
    """A continued run starts from a masked base checkpoint, never the pinned weights (refused before the split
    is read and before the fetch, which fails the test if reached)."""
    fewshot_refused(fewshot_env, fewshot_env.cont("berturk"), "pass a berturk_train.py --fewshot-base checkpoint as "
                                                              "--base, not berturk")


def test_fewshot_continued_from_a_checkpoint_without_a_base_binding_is_refused(fewshot_env: FewshotEnv) -> None:
    """The base names itself by its own record bytes: a dir without berturk_train.json (the tiny base
    itself), a record whose run id is not its canonical config's, a run that is not a masked base of this protocol,
    and a config.json that does not score INTENTS in order are each refused before any load."""
    fewshot_refused(fewshot_env, fewshot_env.cont(fewshot_env.plant(record=False, name="bare"), "--print-run-id"),
                    "lacks berturk_train.json")
    unbound = {k: v for k, v in fewshot_env.base_canon.items() if k not in ("fewshot", "mask_tags")}
    fewshot_refused(fewshot_env, fewshot_env.cont(fewshot_env.plant(unbound, name="unbound"), "--print-run-id"),
                    "carries fewshot binding <absent>, not a masked base of this protocol")
    d = fewshot_env.plant(name="edited")
    rec = _json(os.path.join(d, berturk.RECORD))
    rec["run_id"] = "berturk-selection_train-s0-00000000"
    write_json(os.path.join(d, berturk.RECORD), rec)
    fewshot_refused(fewshot_env, fewshot_env.cont(d, "--print-run-id"),
                    "names run berturk-selection_train-s0-00000000, but its canonical config")
    d = fewshot_env.plant(name="unlabelled")
    cfg = _json(os.path.join(d, "config.json"))
    del cfg["id2label"]
    write_json(os.path.join(d, "config.json"), cfg)
    fewshot_refused(fewshot_env, fewshot_env.cont(d, "--print-run-id"), "does not score the 48 INTENTS tags")


def test_fewshot_continued_seed_must_be_the_draw_seed(fewshot_env: FewshotEnv) -> None:
    """The training seed of a continued run is the draw seed."""
    fewshot_refused(fewshot_env, fewshot_env.cont(fewshot_env.plant(), "--seed", "1", "--print-run-id"),
                    "seed is 1 here but draw 0's seed is 0; the training seed is the draw seed")


def test_fewshot_continued_k_draw_and_epochs_outside_the_frozen_values_are_refused(fewshot_env: FewshotEnv) -> None:
    """k is one of the frozen k_values (1, 2 here), the draw one of the frozen draws and the epochs the
    frozen continued_training's."""
    base = fewshot_env.plant()
    fewshot_refused(fewshot_env, fewshot_env.cont(base, "--fewshot-k", "3", "--print-run-id"),
                    "few-shot k 3 is not one of the frozen k values \\[1, 2\\]")
    fewshot_refused(fewshot_env, fewshot_env.cont(base, "--fewshot-draw", "1", "--print-run-id"),
                    "few-shot draw 1 is not one of the frozen draws 0..0")
    fewshot_refused(fewshot_env, fewshot_env.cont(base, "--epochs", "2", "--print-run-id"),
                    "epochs is 2 here but 1 in the frozen continued_training")


# ---- --tensorboard DIR: NOT_IDENTITY, never recorded, the event files ----

TbRun = tuple[str, Path, Path]                    # the tb_run fixture: (run id, run dir, tensorboard root)


@pytest.fixture(scope="module")
def tb_run(inputs: Inputs, two_runs: TwoRuns, tmp_path_factory: pytest.TempPathFactory) -> TbRun:
    """The two_runs command with --tensorboard DIR into a third storage dir: (run id, run dir, DIR). Its
    --print-run-id is two_runs' run id and creates neither the storage nor DIR."""
    data, manifest, base = inputs
    rid, _ = two_runs
    root = tmp_path_factory.mktemp("tb")
    tb = root / "tb"
    argv = train_argv(data, manifest, base, root / "c", "--tensorboard", tb)
    assert run_train(argv + ["--print-run-id"]).stdout.strip() == rid
    assert not tb.exists() and not (root / "c").exists()
    run_train(argv)
    return rid, root / "c" / rid, tb


def _done_epochs(run_dir: str | Path) -> list[dict[str, Any]]:
    """The epoch rows of a finished run's done.json."""
    return _json(os.path.join(str(run_dir), berturk.DONE))["record"]["epochs"]


@pytest.mark.timeout(TIMEOUT)
def test_tensorboard_keeps_the_run_id_and_the_weights(tb_run: TbRun, two_runs: TwoRuns) -> None:
    rid, run_dir, _ = tb_run
    assert os.path.basename(str(run_dir)) == rid == two_runs[0]
    for et, ea in zip(exports(run_dir), exports(two_runs[1][0])):
        for f in SAME_BYTES:
            assert _bytes(os.path.join(et, f)) == _bytes(os.path.join(ea, f)), \
                "%s/%s differs with --tensorboard" % (os.path.basename(et), f)


@pytest.mark.timeout(TIMEOUT)
def test_tensorboard_never_enters_a_record(tb_run: TbRun, two_runs: TwoRuns) -> None:
    _, run_dir, tb = tb_run
    a = two_runs[1][0]
    pairs = [(os.path.join(e, berturk.RECORD), os.path.join(f, berturk.RECORD))
             for e, f in zip(exports(run_dir), exports(a))]
    pairs.append((os.path.join(str(run_dir), berturk.DONE), os.path.join(str(a), berturk.DONE)))
    for path, ref in pairs:
        rec = _json(path)
        assert rec["canonical"] == _json(ref)["canonical"]
        found = [s for s in strings(rec) if str(tb) in s or s == "tensorboard"]
        assert not found, "%s holds %s" % (path, found)


@pytest.mark.timeout(TIMEOUT)
def test_tensorboard_events_hold_loss_and_grad_norm_per_step(tb_run: TbRun) -> None:
    from tb_events import event_files, read_events, scalars
    rid, run_dir, tb = tb_run
    files = event_files(str(tb / rid))
    assert len(files) == 1 and re.search(r"\.pid\d+$", files[0]), files
    ev = read_events(str(tb / rid))
    rows = _done_epochs(run_dir)
    total = sum(r["optimizer_steps"] for r in rows)
    loss, gn = scalars(ev, "train/loss"), scalars(ev, "train/grad_norm")
    assert sorted(loss) == sorted(gn) == list(range(1, total + 1))
    assert all(math.isfinite(v) and v > 0 for v in gn.values())
    epoch_loss = scalars(ev, "epoch/loss")
    end = 0
    for r in rows:
        start, end = end, end + r["optimizer_steps"]
        assert epoch_loss[end] == pytest.approx(r["loss"], rel=1e-6)
        step_losses = [loss[s] for s in range(start + 1, end + 1)]
        assert sum(step_losses) / len(step_losses) == pytest.approx(epoch_loss[end], rel=1e-5)
    assert sorted(epoch_loss) == [sum(r["optimizer_steps"] for r in rows[:k]) for k in range(1, len(rows) + 1)]


TB_STEP_TAGS_CPU = {"train/loss", "train/lr_encoder", "train/lr_head", "train/grad_norm", "train/messages_per_s",
                    "memory/rss_gib", "memory/swap_used_gib"}    # no device or MPS driver memory on cpu, no sigma
TB_EPOCH_TAGS = {"epoch/loss", "epoch/train_s", "epoch/wall_s", "epoch/peak_total_gib", "epoch/swap_delta_mib"}


@pytest.mark.timeout(TIMEOUT)
def test_tensorboard_events_hold_every_scalar(tb_run: TbRun) -> None:
    from tb_events import read_events, scalars
    rid, run_dir, tb = tb_run
    ev = read_events(str(tb / rid))
    rows = _done_epochs(run_dir)
    total = sum(r["optimizer_steps"] for r in rows)
    tags = {kind for _, _, kind, _ in ev if kind not in ("session_log", "file_version")}
    assert tags == TB_STEP_TAGS_CPU | TB_EPOCH_TAGS, sorted(tags)
    for tag in TB_STEP_TAGS_CPU:
        assert sorted(scalars(ev, tag)) == list(range(1, total + 1)), tag
    enc, head = scalars(ev, "train/lr_encoder"), scalars(ev, "train/lr_head")
    warm = [s for s in enc if enc[s] > 0]
    assert warm and enc[1] == 0.0                                   # the linear warm-up starts at LR 0
    for s in warm:                                                  # one multiplier for both LRs
        assert head[s] / enc[s] == pytest.approx(HEAD_LR / ENC_LR, rel=1e-5)
    assert all(v > 0 for v in scalars(ev, "train/messages_per_s").values())
    ends = [sum(r["optimizer_steps"] for r in rows[:k]) for k in range(1, len(rows) + 1)]
    train_s, wall_s = scalars(ev, "epoch/train_s"), scalars(ev, "epoch/wall_s")
    for tag in TB_EPOCH_TAGS:
        assert sorted(scalars(ev, tag)) == ends, tag
    for r, end in zip(rows, ends):
        assert train_s[end] == pytest.approx(r["train_s"], rel=1e-6)
        assert wall_s[end] >= train_s[end]                          # the export write included


def test_tensorboard_refuses_a_trainer_that_skips_the_accelerator_clip(inputs: Inputs, tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch,
                                                                       capsys: pytest.CaptureFixture[str],
                                                                       keep_threads: None) -> None:
    """The pre-clip norm comes from trainer.accelerator.clip_grad_norm_: a Trainer that clips without it is refused at
    the first step, and the event file is still closed (the writer's close ran; the file reads back, crc-checked) with
    no train/ scalar in it."""
    import berturk_train
    import torch
    import transformers
    from pilot import tensorboard
    from tb_events import event_files, read_events
    data, manifest, base = inputs
    opened = []
    open_writer = tensorboard.open_writer

    def noting_open_writer(logdir: str, purge_step: int, suffix: str) -> Any:
        """pilot.tensorboard.open_writer, noting the writer it returns."""
        opened.append(open_writer(logdir, purge_step, suffix))
        return opened[-1]

    def clip_without_the_accelerator(self: Any, model: Any) -> Any:
        """The Trainer's clip, bypassing trainer.accelerator."""
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
    monkeypatch.setattr(transformers.Trainer, "_clip_grad_norm", clip_without_the_accelerator)
    monkeypatch.setattr(tensorboard, "open_writer", noting_open_writer)
    argv = [str(x) for x in train_argv(data, manifest, base, tmp_path / "s", "--tensorboard", tmp_path / "tb")]
    berturk_train.main(argv + ["--print-run-id"])
    rid = capsys.readouterr().out.strip()
    with pytest.raises(SystemExit, match="clip_grad_norm_ was not called.*check the transformers version"):
        berturk_train.main(argv)
    assert len(opened) == 1 and opened[0]["open"] is False         # pilot.tensorboard.close ran
    assert len(event_files(str(tmp_path / "tb" / rid))) == 1
    ev = read_events(str(tmp_path / "tb" / rid))
    assert not [kind for _, _, kind, _ in ev if kind.startswith("train/")]


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a dir the user cannot write")
@pytest.mark.timeout(TIMEOUT)
def test_tensorboard_unwritable_dir_leaves_no_run_dir(inputs: Inputs, tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str], keep_threads: None) -> None:
    """WR-02: an unwritable --tensorboard DIR passes the path guard and is refused when the writer opens, before the run
    dir exists (a BERTurk run dir is never resumed and refuses a rerun), so once the DIR is writable the same command
    trains."""
    import berturk_train
    data, manifest, base = inputs
    ro = tmp_path / "ro"
    ro.mkdir()
    argv = [str(x) for x in train_argv(data, manifest, base, tmp_path / "s", "--tensorboard", ro / "tb")]
    berturk_train.main(argv + ["--print-run-id"])
    rid = capsys.readouterr().out.strip()
    os.chmod(ro, 0o555)
    try:
        with pytest.raises(SystemExit, match="^--tensorboard: cannot write event files under "):
            berturk_train.main(argv)
    finally:
        os.chmod(ro, 0o755)
    assert not (tmp_path / "s" / rid).exists() and not (ro / "tb").exists()
    run_dir = berturk_train.main(argv)                      # the same command, now that DIR is writable
    assert os.path.basename(run_dir) == rid and os.path.isfile(os.path.join(run_dir, berturk.DONE))
    assert len(list((ro / "tb" / rid).glob("events.out.tfevents.*"))) == 1


def test_tensorboard_print_run_id_creates_nothing(inputs: Inputs, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                  no_load: None) -> None:
    import berturk_train
    data, manifest, base = inputs
    argv = [str(x) for x in train_argv(data, manifest, base, tmp_path / "runs", "--print-run-id")]
    berturk_train.main(argv)
    plain = capsys.readouterr().out
    berturk_train.main(argv + ["--tensorboard", str(tmp_path / "tb")])
    assert capsys.readouterr().out == plain and plain.strip()
    assert not (tmp_path / "runs").exists() and not (tmp_path / "tb").exists()


@pytest.mark.parametrize("print_run_id", [False, True], ids=["train", "print_run_id"])
@pytest.mark.parametrize("where", ["results", "protocol.json", "storage"])
def test_tensorboard_refused_before_any_load(inputs: Inputs, tmp_path: Path, no_load: None, where: str,
                                             print_run_id: bool) -> None:
    """A --tensorboard DIR under results/ or protocol.json of the repo, or equal to --storage, is refused before the
    split is read, any load or the run dir, also with --print-run-id; nothing is created."""
    import berturk_train
    data, manifest, base = inputs
    storage = tmp_path / "runs"
    tb = {"results": os.path.join(REPO, "results", "tb"), "protocol.json": os.path.join(REPO, "protocol.json"),
          "storage": str(storage)}[where]
    argv = [str(x) for x in train_argv(data, manifest, base, storage, "--tensorboard", tb)]
    with pytest.raises(SystemExit) as e:
        berturk_train.main(argv + (["--print-run-id"] if print_run_id else []))
    assert str(e.value).startswith("--tensorboard "), str(e.value)
    assert not storage.exists()
    assert not os.path.exists(os.path.join(REPO, "results", "tb"))


@pytest.mark.timeout(TIMEOUT)
def test_tensorboard_flag_off_writes_no_event_file(two_runs: TwoRuns) -> None:
    root = two_runs[1][0].parent.parent                             # the storage dirs a and b
    assert not list(root.rglob("events.out.tfevents*"))
