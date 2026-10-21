"""score.py --model berturk refuses every wrong input before a model load and writes the sidecar postfit and report
read (BASE-02; ROADMAP Phase 6 criteria 2 and 4).

Refusals, each before transformers loads anything and without a dump: a split with role test (before the checkpoint
is hashed), the hotels_probe split (role gate), a config.json that does not score INTENTS in order as a multi-label
classifier, a checkpoint missing any of the five files, a berturk_train.json changed after its identity was taken, an
existing dump without --force, and the argument errors (--workers other than 0, no --checkpoint, --checkpoint
multilingual). check_bert refuses a float16 model, a model on another device and autocast, after the load and after
every batch. report.binds_every_file accepts a complete BERTurk identity, so a moved checkpoint dir stays the same
source.

The tiny export comes from berturk_train.py on CPU (1 epoch), built once per module; each test works on its own copy.
Offline, no Ray, no weights: runs in `make test`.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import pilot.data
import report
import score
from conftest import make_raw
from intents import INTENTS
from pilot import berturk, dumps
from pilot.data import sha256_file
from tiny_berturk import tiny_base

pytestmark = pytest.mark.unit

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """(splits dir, manifest path, a 1-epoch CPU export of the tiny base), built once for the module."""
    tmp = tmp_path_factory.mktemp("berturk_score")
    root, raw_sha256 = make_raw(str(tmp / "raw"))
    out_dir = str(tmp / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    manifest = os.path.join(out_dir, "splits.json")
    splits = os.path.join(out_dir, "splits")
    base = tiny_base(tmp / "base")
    storage = tmp / "runs"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [sys.executable, "berturk_train.py", "--data", os.path.join(splits, "selection_train.jsonl"),
            "--manifest", manifest, "--base", base, "--device", "cpu", "--epochs", "1", "--batch", "4",
            "--max-len", "32", "--seed", "0", "--storage", str(storage)]
    rid = subprocess.run(argv + ["--print-run-id"], cwd=REPO, env=env, capture_output=True, text=True,
                         timeout=TIMEOUT, check=True).stdout.strip()
    r = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0, r.stderr[-4000:]
    return splits, manifest, str(storage / rid / "exports" / "epoch-001")


@pytest.fixture
def ckpt(built, tmp_path):
    """A private copy of the module's export (tests edit or delete its files)."""
    dst = str(tmp_path / "epoch-001")
    shutil.copytree(built[2], dst)
    return dst


@pytest.fixture(autouse=True)
def _threads():
    import torch
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


@pytest.fixture
def no_load(monkeypatch):
    """Fail the test if transformers loads a tokenizer or model (a refusal must come before any load)."""
    import transformers

    def boom(*a, **k):
        raise AssertionError("a model or tokenizer was loaded before the refusal")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)
    monkeypatch.setattr(transformers.BertForSequenceClassification, "from_pretrained", boom)


def _score(built, ckpt, root, split="val", *extra):
    splits, manifest, _ = built
    return score.main([os.path.join(splits, split + ".jsonl"), "--model", "berturk", "--checkpoint", ckpt,
                       "--workers", "0", "--manifest", manifest, "--out-root", str(root)] + list(extra))


def _no_dump(root):
    """True when nothing was written under root (no run dir, no npz, no sidecar)."""
    return not os.path.exists(str(root)) or not any(files for _, _, files in os.walk(str(root)))


def _edit_json(path, change):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    change(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---- refusals ----

def test_berturk_test_role_refused_before_hashing(built, ckpt, tmp_path, monkeypatch, no_load):
    def hashed(*a, **k):
        raise AssertionError("the BERTurk checkpoint was hashed before the test-role refusal")
    monkeypatch.setattr(berturk, "identity", hashed)
    monkeypatch.setattr(berturk, "require_files", hashed)
    with pytest.raises(SystemExit, match="Phase 8"):
        _score(built, ckpt, tmp_path / "dumps", "test")
    assert _no_dump(tmp_path / "dumps")


def _relabelled(built, dst, lines):
    """A manifest copy under dst whose val entry registers `lines` (raw JSONL lines) as splits/val.jsonl; every other
    split keeps its bytes. Returns (val path, manifest path)."""
    splits, manifest, _ = built
    shutil.copytree(splits, str(dst / "splits"))
    raw = b"".join(lines)
    (dst / "splits" / "val.jsonl").write_bytes(raw)
    with open(manifest, encoding="utf-8") as f:
        m = json.load(f)
    rows = pilot.data.parse_jsonl(raw)
    m["splits"]["val"].update(sha256=pilot.data.sha256_bytes(raw), n_messages=len(rows),
                              n_no_intent=sum(not any(r["expected"].values()) for r in rows),
                              uids_sha256=pilot.data.sha256_bytes("\n".join(r["uid"] for r in rows).encode("utf-8")))
    (dst / "splits.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(dst / "splits" / "val.jsonl"), str(dst / "splits.json")


def test_berturk_interlock_is_keyed_on_the_test_content(built, ckpt, tmp_path, monkeypatch, no_load):
    """06-REVIEW CR-01: a manifest copy that registers the committed test bytes (or a part of them) as val is refused
    before any load, as for Laya (05-REVIEW CR-02). The module's synthetic data stands in for the committed split."""
    splits, manifest, _ = built
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", manifest)
    with open(os.path.join(splits, "test.jsonl"), "rb") as f:
        lines = f.read().splitlines(keepends=True)
    for name, part in (("whole", lines), ("part", lines[1:])):   # the entry's hashes, then the (uid, message) pairs
        val, other = _relabelled(built, tmp_path / name, part)
        assert pilot.data.verify_split_file(val, manifest_path=other, roles=("val",))["role"] == "val"
        with pytest.raises(SystemExit, match="committed test content"):
            score.main([val, "--model", "berturk", "--checkpoint", ckpt, "--workers", "0", "--manifest", other,
                        "--out-root", str(tmp_path / "dumps")])
        assert _no_dump(tmp_path / "dumps")


def test_berturk_val_under_a_manifest_copy_still_scores(built, ckpt, tmp_path, monkeypatch):
    """The content key leaves val alone: the module's val bytes under a byte-different manifest copy score."""
    splits, manifest, _ = built
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", manifest)
    with open(os.path.join(splits, "val.jsonl"), "rb") as f:
        val, other = _relabelled(built, tmp_path / "copy", f.read().splitlines(keepends=True))
    with open(other, "ab") as f:
        f.write(b"\n")
    run_dir, _ = score.main([val, "--model", "berturk", "--checkpoint", ckpt, "--workers", "0", "--manifest", other,
                             "--out-root", str(tmp_path / "dumps")])
    assert os.path.isfile(os.path.join(run_dir, "val.npz"))


def test_berturk_ood_probe_refused(built, ckpt, tmp_path, no_load):
    with pytest.raises(SystemExit, match="role ood_probe"):
        _score(built, ckpt, tmp_path / "dumps", "hotels_probe")
    assert _no_dump(tmp_path / "dumps")


def _swap_id2label(cfg):
    cfg["id2label"]["0"], cfg["id2label"]["1"] = cfg["id2label"]["1"], cfg["id2label"]["0"]


def _swap_label2id(cfg):
    a, b = berturk.TAGS[0], berturk.TAGS[1]
    cfg["label2id"][a], cfg["label2id"][b] = cfg["label2id"][b], cfg["label2id"][a]


def _single_label(cfg):
    cfg["problem_type"] = "single_label_classification"


@pytest.mark.parametrize("change", [_swap_id2label, _swap_label2id, _single_label],
                         ids=["id2label", "label2id", "problem_type"])
def test_berturk_permuted_label_order_refused_before_load(built, ckpt, tmp_path, no_load, change):
    _edit_json(os.path.join(ckpt, "config.json"), change)
    with pytest.raises(SystemExit, match="INTENTS order"):
        _score(built, ckpt, tmp_path / "dumps")
    assert _no_dump(tmp_path / "dumps")


@pytest.mark.parametrize("name", berturk.CHECKPOINT_FILES)
def test_berturk_missing_file_refused(built, ckpt, tmp_path, no_load, name):
    os.remove(os.path.join(ckpt, name))
    with pytest.raises(SystemExit, match="lacks %s" % name.replace(".", r"\.")):
        _score(built, ckpt, tmp_path / "dumps")
    assert _no_dump(tmp_path / "dumps")


@pytest.mark.parametrize("args", [["--checkpoint", "CKPT", "--workers", "1"], ["--checkpoint", "CKPT"],
                                  ["--workers", "0"], ["--checkpoint", "multilingual", "--workers", "0"]],
                         ids=["workers-1", "workers-default", "no-checkpoint", "multilingual"])
def test_berturk_argument_errors(built, ckpt, tmp_path, no_load, args, capsys):
    splits, manifest, _ = built
    argv = [os.path.join(splits, "val.jsonl"), "--model", "berturk", "--manifest", manifest,
            "--out-root", str(tmp_path / "dumps")] + [ckpt if x == "CKPT" else x for x in args]
    with pytest.raises(SystemExit) as e:
        score.main(argv)
    assert e.value.code == 2                                # ap.error
    assert "--model berturk" in capsys.readouterr().err
    assert _no_dump(tmp_path / "dumps")


def test_berturk_record_changed_after_hash_refused(built, ckpt, tmp_path, monkeypatch, no_load):
    real = berturk.identity

    def identity_then_edit(d):
        ident = real(d)
        with open(os.path.join(d, berturk.RECORD), "ab") as f:   # the record changes after its identity was taken
            f.write(b" ")
        return ident
    monkeypatch.setattr(berturk, "identity", identity_then_edit)
    with pytest.raises(SystemExit, match="changed after it was hashed"):
        _score(built, ckpt, tmp_path / "dumps")
    assert _no_dump(tmp_path / "dumps")


def test_berturk_config_changed_after_hash_refused(built, ckpt, tmp_path, monkeypatch, no_load):
    real = berturk.identity

    def identity_then_edit(d):
        ident = real(d)
        _edit_json(os.path.join(d, "config.json"), _swap_id2label)
        return ident
    monkeypatch.setattr(berturk, "identity", identity_then_edit)
    with pytest.raises(SystemExit, match="config.json changed after it was hashed"):
        _score(built, ckpt, tmp_path / "dumps")
    assert _no_dump(tmp_path / "dumps")


def test_berturk_existing_dump_needs_force(built, ckpt, tmp_path, monkeypatch):
    root = tmp_path / "dumps"
    run_dir, name = _score(built, ckpt, root)
    first = sha256_file(os.path.join(run_dir, name + ".json"))
    import transformers
    real = transformers.BertForSequenceClassification.from_pretrained.__func__
    loads = []

    def counted(cls, *a, **k):
        loads.append(1)
        return real(cls, *a, **k)
    monkeypatch.setattr(transformers.BertForSequenceClassification, "from_pretrained", classmethod(counted))
    with pytest.raises(SystemExit, match="already exists; pass --force"):
        _score(built, ckpt, root)
    assert loads == [] and sha256_file(os.path.join(run_dir, name + ".json")) == first
    again, _ = _score(built, ckpt, root, "val", "--force")
    assert again == run_dir and loads == [1]
    dumps.load_dump(run_dir, "val", built[1])


# ---- device, dtype and autocast ----

def _tiny_classifier():
    import torch
    import transformers
    torch.manual_seed(0)
    cfg = transformers.BertConfig(vocab_size=16, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=16, max_position_embeddings=16, num_labels=len(INTENTS),
                                  problem_type=berturk.PROBLEM_TYPE)
    return transformers.BertForSequenceClassification(cfg).eval()


def test_check_bert_refuses_wrong_dtype_device_or_autocast():
    import torch
    model = _tiny_classifier()
    assert score.check_bert(model, "cpu") == {"device": "cpu", "dtype": "torch.float32", "amp_enabled": False}
    with pytest.raises(SystemExit, match="BERTurk parameters on cpu, requested mps"):
        score.check_bert(model, "mps")
    with pytest.raises(SystemExit, match="BERTurk parameters on cpu, requested cuda"):
        score.check_bert(model, "cuda")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(SystemExit, match="autocast True on cpu; scoring is fp32"):
            score.check_bert(model, "cpu")
    with pytest.raises(SystemExit, match="torch.float16 with autocast False"):
        score.check_bert(_tiny_classifier().half(), "cpu")
    mixed = _tiny_classifier()
    mixed.classifier.half()                                 # one non-float32 parameter is enough
    with pytest.raises(SystemExit, match="torch.float16, torch.float32"):
        score.check_bert(mixed, "cpu")


def test_check_bert_runs_after_the_load_and_every_batch(built, ckpt, tmp_path, monkeypatch):
    calls = []
    real = score.check_bert

    def counted(model, device):
        calls.append(device)
        return real(model, device)
    monkeypatch.setattr(score, "check_bert", counted)
    monkeypatch.setattr(score, "BATCH", 1)                  # one message per batch: 1 + n calls
    run_dir, _ = _score(built, ckpt, tmp_path / "dumps")
    n = dumps.load_dump(run_dir, "val", built[1])[0].shape[0]
    assert n >= 2 and calls == ["cpu"] * (1 + n)


# ---- the sidecar postfit and report read ----

def _laya_training_keys(tmp_path):
    """The keys score.training_block writes for a ray_train.py checkpoint (the ones pilot.postfit.candidate reads)."""
    laya = tmp_path / "laya"
    laya.mkdir()
    cfg = {"training": {"epoch": 1, "calibrated": False,
                        "ray_train": {"run_id": "laya-x", "canonical": {"seed": 0}, "record": {"epochs": []}}}}
    (laya / "rl_agent_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return set(score.training_block(str(laya), {"config_sha256": sha256_file(str(laya / "rl_agent_config.json"))})[1])


def test_berturk_sidecar_matches_the_postfit_contract(built, ckpt, tmp_path):
    _edit_json(os.path.join(ckpt, berturk.RECORD), lambda r: r.update(temperature=1.3))   # T is read from the record
    with open(os.path.join(ckpt, berturk.RECORD), encoding="utf-8") as f:
        rec = json.load(f)
    run_dir, _ = _score(built, ckpt, tmp_path / "dumps")
    logits, uids, side, entry = dumps.load_dump(run_dir, "val", built[1])
    ident = berturk.identity(ckpt)
    assert os.path.basename(run_dir) == side["run_id"] == berturk.dump_run_id(ident)
    assert side["system"]["kind"] == "berturk" and side["system"]["checkpoint_identity"] == ident
    assert side["temperature"]["shipped"] == rec["temperature"] == 1.3
    assert side["logit"] == berturk.LOGIT_NOTE
    assert (side["scorer"]["device"], side["scorer"]["dtype"], side["scorer"]["amp_enabled"]) \
        == ("cpu", "torch.float32", False)
    tr = side["training"]
    assert set(tr) == _laya_training_keys(tmp_path)
    assert side["seed"] == rec["canonical"]["seed"] == 0 and tr["canonical"] == rec["canonical"]
    assert tr["config_sha256_sans_seed"] == dumps.training_digest(rec["canonical"], rec["epoch"])
    assert tr["epoch"] == 1 and tr["run_id"] == rec["run_id"]
    assert logits.dtype == np.float32 and logits.shape == (entry["n_messages"], len(INTENTS))


def test_report_binds_a_complete_berturk_identity(built, ckpt, tmp_path):
    assert report.BERTURK_FILES == berturk.TOKENIZER_FILES   # report.py is numpy-only: the names are spelled out
    ident = berturk.identity(ckpt)
    assert report.binds_every_file(ident)
    for name in berturk.TOKENIZER_FILES:
        assert not report.binds_every_file(dict(ident, files_sha256={k: v for k, v in ident["files_sha256"].items()
                                                                     if k != name}))
        assert not report.binds_every_file(dict(ident, files_sha256=dict(ident["files_sha256"], **{name: None})))
    for key in ("model_sha256", "config_sha256", "record_sha256"):
        assert not report.binds_every_file(dict(ident, **{key: None}))
    assert not report.binds_every_file(dict(ident, files_sha256=None))
    assert not report.binds_every_file(dict(ident, kind="laya"))   # a Laya identity still needs LOCAL_REQUIRED_FILES

    # the real sidecar: the same checkpoint scored from a moved dir is the same source; an incomplete identity is not
    run_dir, _ = _score(built, ckpt, tmp_path / "dumps")
    side = dumps.load_dump(run_dir, "val", built[1])[2]
    moved = dict(side, system=dict(side["system"], checkpoint="elsewhere/epoch-001"))
    report.same_source(moved, side, "moved")
    partial = dict(ident, files_sha256={"vocab.txt": ident["files_sha256"]["vocab.txt"]})
    side_p = dict(side, system=dict(side["system"], checkpoint_identity=partial))
    with pytest.raises(SystemExit, match=r"system\.checkpoint "):
        report.same_source(dict(side_p, system=dict(side_p["system"], checkpoint="elsewhere/epoch-001")), side_p,
                           "moved")
