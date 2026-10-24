"""score.py's protocol gate (REP-01; ROADMAP Phase 8 criteria 1 and 4): the one place a model runs opens test and
HOTELS to the protocol-bound final seeds, once, and to nothing else.

The gate replaced the Phase 5 Laya and Phase 6 BERTurk test interlocks (refuse_finetuned_test, refuse_berturk_test).
It triggers for a fine-tuned checkpoint on the same content as they did, extended to HOTELS (score.gate_trigger), and
admits a run only when pilot.protocol.require_frozen passes, no --n, --force or --out-root other than results/dumps is
given, the manifest is the committed one and names a gated split, the device and worker count are the protocol's, and
the hashed dump run id is one of the protocol's final seeds of that system or a listed few-shot checkpoint on test
(score.require_admitted). A listed few-shot checkpoint must re-derive its runs.json entry (score.rederive_entry,
09-REVIEW WR-01): a well-formed entry swapped to point at a retrained run or at a checkpoint without a Phase 9 binding
is refused, a byte-identical copy under another dir name is admitted. An admitted dump carries protocol_sha256; an
existing gated dump, or an npz an interrupted gated run left without its sidecar, is refused.

Every test runs on the synthetic snapshot with pilot.data.DEFAULT_MANIFEST, pilot.protocol.PROTOCOL_PATH and
PROTOCOL_SHA256 and pilot.dumps.DUMP_ROOT monkeypatched to temporary stand-ins (fake_protocol writes a minimal
protocol/1 file holding every REQUIRED_KEYS entry), with the tiny offline Laya checkpoint (tests/tiny_laya.py) and the
tiny BERTurk export of tests/test_score_berturk.py. Offline, no Ray: runs in `make test`.
"""
import json
import os
import re
import shutil

import pytest

import pilot.data
import pilot.weights
import score
from conftest import prepared
from pilot import dumps
from pilot import fewshot as pfs
from pilot import protocol as pr
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.unit

TINY_T = 1.3


def fake_protocol(path, manifest, systems, device="cpu", workers=0):
    """Write a minimal protocol/1 file to `path` and return (path, sha256): every pilot.protocol.REQUIRED_KEYS entry,
    the splits of `manifest` (pilot.protocol.splits_block: the manifest must be the monkeypatched committed one),
    systems {kind: [dump run id, ...]} as final seeds 0, 1, ... and a scoring block on `device` with `workers`."""
    p = {k: {} for k in pr.REQUIRED_KEYS}
    p.update(schema=pr.SCHEMA, splits=pr.splits_block(manifest),
             systems={kind: {"final_seeds": [{"seed": i, "dump_run_id": r} for i, r in enumerate(ids)],
                             "dump_run_ids": list(ids)} for kind, ids in systems.items()},
             scoring={"splits": list(pr.GATED), "device": device, "workers": workers})
    raw = pr.protocol_bytes(p)
    with open(str(path), "wb") as f:
        f.write(raw)
    return str(path), pilot.data.sha256_bytes(raw)


def finetuned(path, temperature=TINY_T):
    """A tiny Laya checkpoint marked fine-tuned (rl_agent_config.json fine_tuned true); returns its dir."""
    ckpt = tiny_checkpoint(path, temperature=temperature)
    cfg_path = os.path.join(ckpt, "rl_agent_config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["fine_tuned"] = True
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return ckpt


class Snap:
    """The gate's test bench: synthetic splits, the committed-manifest stand-in, a fine-tuned tiny checkpoint, its
    dump run id and a temp protocol binding it under systems.laya."""


def no_fewshot_runs(monkeypatch, tmp_path):
    """The Phase 8 benches have no few-shot allow-list: pilot.fewshot.RUNS_PATH points at a temp path that does not
    exist, so a refused final-seed checkpoint is refused as unlisted, never by a check of the committed
    results/fewshot/runs.json (written under the real pin, 09-11) against the synthetic protocol."""
    monkeypatch.setattr(pfs, "RUNS_PATH", str(tmp_path / "no-fewshot" / "runs.json"))


@pytest.fixture
def snap(tmp_path, synthetic_raw, monkeypatch):
    import torch
    threads = torch.get_num_threads()
    s = Snap()
    s.out_dir, s.manifest = prepared(tmp_path, synthetic_raw)
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", s.manifest)
    s.root = str(tmp_path / "dumps")
    monkeypatch.setattr(dumps, "DUMP_ROOT", s.root)
    no_fewshot_runs(monkeypatch, tmp_path)
    s.ckpt = finetuned(tmp_path / "ckpt")
    s.rid = dumps.run_id("laya", pilot.weights.identity(s.ckpt))
    s.tmp = tmp_path
    s.bind = lambda systems=None, device="cpu", workers=0: _bind(monkeypatch, s, systems, device, workers)
    s.path, s.sha = s.bind()
    yield s
    torch.set_num_threads(threads)


def _bind(monkeypatch, s, systems, device, workers):
    path, sha = fake_protocol(s.tmp / "protocol.json", s.manifest, systems or {"laya": [s.rid]}, device, workers)
    monkeypatch.setattr(pr, "PROTOCOL_PATH", path)
    monkeypatch.setattr(pr, "PROTOCOL_SHA256", sha)
    return path, sha


def split(s, name):
    return os.path.join(s.out_dir, "splits", name + ".jsonl")


def laya_argv(s, name, *extra, ckpt=None):
    return [split(s, name), "--model", "laya", "--checkpoint", ckpt or s.ckpt, "--workers", "0", "--device", "cpu",
            "--manifest", s.manifest] + list(extra)


def files(root):
    """Every file under root (relative paths), for 'nothing was written' checks."""
    out = []
    for d, _, names in os.walk(str(root)):
        out += [os.path.relpath(os.path.join(d, n), str(root)) for n in names]
    return sorted(out)


def refused(argv, *needles):
    """Run score.main(argv); assert a one-line gate refusal holding every needle; return its text."""
    with pytest.raises(SystemExit) as e:
        score.main(argv)
    text = str(e.value.code)
    assert text.startswith(score.GATE), text
    assert "\n" not in text, text
    for n in needles:
        assert n in text, (n, text)
    return text


def no_identity(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the checkpoint was hashed before the gate refused")
    monkeypatch.setattr(pilot.weights, "identity", boom)


def no_laya_load(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the checkpoint was loaded before the gate refused")
    monkeypatch.setattr(score, "load_agent", boom)


def sidecar(run_dir, name):
    with open(os.path.join(run_dir, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


# ---- admitted, once ----

def test_gate_admits_a_bound_final_seed_once(snap, monkeypatch):
    run_dir, name = score.main(laya_argv(snap, "test"))
    assert (run_dir, name) == (os.path.join(snap.root, snap.rid), "test")
    assert sidecar(run_dir, "test")["protocol_sha256"] == snap.sha
    run_dir, name = score.main(laya_argv(snap, "hotels_probe"))
    assert name == "hotels_probe"
    side = sidecar(run_dir, "hotels_probe")
    assert side["protocol_sha256"] == snap.sha and side["split"]["role"] == "ood_probe"
    logits, uids, _, entry = dumps.load_dump(run_dir, "hotels_probe", snap.manifest)
    assert logits.shape == (entry["n_messages"], 48) and all(u.startswith("hotel_") for u in uids)
    before = files(snap.root)
    refused(laya_argv(snap, "test"), "already exists", "once")
    refused(laya_argv(snap, "test", "--force"), "--force")
    refused(laya_argv(snap, "hotels_probe", "--n", "3"), "--n")
    refused(laya_argv(snap, "test", "--out-root", str(snap.tmp / "elsewhere")), "--out-root")
    assert files(snap.root) == before and not (snap.tmp / "elsewhere").exists()


def test_gate_accepts_the_dump_root_spelled_out(snap):
    run_dir, _ = score.main(laya_argv(snap, "test", "--out-root", snap.root))
    assert sidecar(run_dir, "test")["protocol_sha256"] == snap.sha


# ---- refused before hashing: the protocol itself ----

@pytest.mark.parametrize("how", ["missing", "unpinned", "changed"])
def test_gate_refuses_without_the_frozen_protocol_before_hashing(snap, monkeypatch, how):
    if how == "missing":
        os.remove(snap.path)
    elif how == "unpinned":
        monkeypatch.setattr(pr, "PROTOCOL_SHA256", None)
    else:
        with open(snap.path, "rb") as f:
            raw = f.read()
        with open(snap.path, "wb") as f:
            f.write(raw.replace(b'"workers": 0', b'"workers": 1'))
    no_identity(monkeypatch)
    for name in ("test", "hotels_probe"):
        refused(laya_argv(snap, name), "fine-tuned laya checkpoint", "protocol")
    assert files(snap.root) == []


@pytest.mark.parametrize("device, workers, flags, needle", [
    ("mps", 0, [], "device"),
    ("cpu", 0, ["--workers", "2"], "workers"),
], ids=["device", "workers"])
def test_gate_refuses_another_device_or_worker_count_before_hashing(snap, monkeypatch, device, workers, flags,
                                                                     needle):
    snap.bind(device=device, workers=workers)
    no_identity(monkeypatch)
    argv = laya_argv(snap, "test")
    argv[argv.index("--workers"):argv.index("--workers") + 2] = []
    refused(argv + (flags or ["--workers", "0"]), needle)
    assert files(snap.root) == []


def test_gate_refuses_a_manifest_copy_of_the_committed_splits(snap, monkeypatch):
    """Committed test content under a byte-different manifest triggers the gate (content key) and is refused: only
    the committed manifest admits a gated run."""
    shutil.copytree(os.path.join(snap.out_dir, "splits"), str(snap.tmp / "copy" / "splits"))
    with open(snap.manifest, "rb") as f:
        (snap.tmp / "copy" / "splits.json").write_bytes(f.read() + b"\n")
    no_identity(monkeypatch)
    argv = laya_argv(snap, "test")
    argv[0] = str(snap.tmp / "copy" / "splits" / "test.jsonl")
    argv[argv.index("--manifest") + 1] = str(snap.tmp / "copy" / "splits.json")
    refused(argv, "holds committed test content", "committed")


# ---- refused after hashing, before any load: not a final seed ----

def test_gate_refuses_a_checkpoint_outside_the_final_seeds(snap, monkeypatch):
    other = finetuned(snap.tmp / "other", temperature=0.7)
    rid = dumps.run_id("laya", pilot.weights.identity(other))
    assert rid != snap.rid
    no_laya_load(monkeypatch)
    text = refused(laya_argv(snap, "test", ckpt=other), rid, snap.rid)
    assert "never scored on test or HOTELS" in text
    assert files(snap.root) == []


def test_gate_refuses_an_orphan_npz(snap, monkeypatch):
    os.makedirs(os.path.join(snap.root, snap.rid))
    with open(os.path.join(snap.root, snap.rid, "test.npz"), "wb") as f:
        f.write(b"half")
    no_laya_load(monkeypatch)
    text = refused(laya_argv(snap, "test"), "without its sidecar")
    assert "load_dump never read it" in text and "delete it" in text


# ---- ungated: val, zero-shot, the trigger texts ----

def test_val_of_the_same_checkpoint_is_ungated(snap, monkeypatch):
    monkeypatch.setattr(pr, "PROTOCOL_SHA256", None)           # val never consults the protocol
    run_dir, _ = score.main(laya_argv(snap, "val", "--out-root", str(snap.tmp / "val-dumps")))
    assert "protocol_sha256" not in sidecar(run_dir, "val")


def _entry_cases(s, name, manifest=None):
    path = split(s, name)
    with open(path, "rb") as f:
        raw = f.read()
    entry = pilot.data.verify_split_file(path, manifest_path=manifest or s.manifest, data=raw)
    return entry, pilot.data.parse_jsonl(raw)


def test_gate_trigger_texts(snap):
    t_entry, t_cases = _entry_cases(snap, "test")
    h_entry, h_cases = _entry_cases(snap, "hotels_probe")
    v_entry, v_cases = _entry_cases(snap, "val")
    m = snap.manifest
    assert score.gate_trigger("laya", "multilingual", h_entry, m, h_cases) is None
    assert score.gate_trigger("laya", "multilingual", t_entry, m, t_cases) is None
    assert score.gate_trigger("laya", snap.ckpt, v_entry, m, v_cases) is None
    assert score.gate_trigger("laya", snap.ckpt, t_entry, m, t_cases) == "is a test split under the committed manifest"
    assert score.gate_trigger("laya", snap.ckpt, h_entry, m, h_cases) == \
        "is a ood_probe split under the committed manifest"
    for kind in ("majority", "prior"):
        assert score.gate_trigger(kind, None, t_entry, m, t_cases) is None
    assert score.gate_trigger("berturk", "x", t_entry, m, t_cases) == "is a split with role test"
    assert score.gate_trigger("berturk", "x", h_entry, m, h_cases) == "is a split with role ood_probe"
    assert score.gate_trigger("berturk", "x", v_entry, m, v_cases) is None
    # committed content under a role that does not trigger: the content texts
    assert score.gate_trigger("laya", snap.ckpt, dict(v_entry, sha256=t_entry["sha256"]), m, []) == \
        "holds committed test content"
    assert score.gate_trigger("berturk", "x", v_entry, m, v_cases + h_cases[:1]) == \
        "holds committed hotels_probe content"


def test_gate_keys_on_the_identity_not_the_checkpoint_claim(snap, monkeypatch):
    """08-REVIEW WR-02: a local Laya checkpoint whose rl_agent_config.json claims nothing (no fine_tuned, no
    training.ray_train: the keys an edit could strip) is gated on test and HOTELS like any fine-tuned one, and refused
    before any load unless it is a protocol final seed; on val it stays ungated."""
    plain = tiny_checkpoint(snap.tmp / "plain", temperature=TINY_T)
    with open(os.path.join(plain, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    tr = cfg.get("training")
    assert cfg.get("fine_tuned") is not True and not (isinstance(tr, dict) and tr.get("ray_train")), cfg
    for name, role in (("test", "test"), ("hotels_probe", "ood_probe")):
        entry, cases = _entry_cases(snap, name)
        assert score.gate_trigger("laya", plain, entry, snap.manifest, cases) == \
            "is a %s split under the committed manifest" % role
    v_entry, v_cases = _entry_cases(snap, "val")
    assert score.gate_trigger("laya", plain, v_entry, snap.manifest, v_cases) is None
    rid = dumps.run_id("laya", pilot.weights.identity(plain))
    no_laya_load(monkeypatch)
    for name in ("test", "hotels_probe"):
        refused(laya_argv(snap, name, ckpt=plain), rid, "never scored on test or HOTELS")
    assert files(snap.root) == []


def test_gate_trigger_fails_closed_without_the_committed_files(snap):
    """08-REVIEW WR-01: under a non-committed manifest of the pinned dataset (the same dataset.raw_sha256), a split
    whose hashes differ from the committed entry's (prepare_data.py --no-include-no-intent --out OTHER drops rows)
    keys only on the (uid, message) pairs of the committed test.jsonl and hotels_probe.jsonl. While either file is
    missing the trigger refuses (pilot.data.require_committed_files) instead of returning None, before any hashing."""
    t_entry, t_cases = _entry_cases(snap, "test")
    v_entry, v_cases = _entry_cases(snap, "val")
    other = snap.tmp / "other.json"
    with open(snap.manifest, "rb") as f:
        other.write_bytes(f.read() + b"\n")                    # byte-different, same pinned dataset
    dropped = dict(t_entry, sha256="0" * 64, uids_sha256="1" * 64)
    assert score.gate_trigger("laya", snap.ckpt, dropped, str(other), t_cases) == "holds committed test content"
    for name in ("test", "hotels_probe"):
        os.remove(split(snap, name))
        for kind, ckpt, entry, cases in (("laya", snap.ckpt, dropped, t_cases), ("berturk", "x", v_entry, v_cases)):
            with pytest.raises(SystemExit) as e:
                score.gate_trigger(kind, ckpt, entry, str(other), cases)
            text = str(e.value.code)
            assert text.startswith(score.GATE) and "\n" not in text, text
            assert "is missing" in text and "cannot be content-checked" in text and "make" in text, text
    # the committed manifest needs no split file (its splits are the committed bytes); ungated kinds never read them
    assert score.gate_trigger("laya", snap.ckpt, v_entry, snap.manifest, v_cases) is None
    assert score.gate_trigger("prior", None, dropped, str(other), t_cases) is None
    assert score.gate_trigger("laya", "multilingual", dropped, str(other), t_cases) is None


# ---- BERTurk through the same gate ----

from test_score_berturk import built, ckpt  # noqa: E402,F401  (pytest fixtures: the tiny BERTurk export)


@pytest.fixture
def bsnap(built, ckpt, tmp_path, monkeypatch):
    """The BERTurk bench: the module's synthetic splits as the committed stand-in, a private export copy, its dump run
    id and a temp protocol binding it under systems.berturk."""
    import torch
    from pilot import berturk
    threads = torch.get_num_threads()
    s = Snap()
    s.splits, s.manifest, _ = built
    s.out_dir = os.path.dirname(s.splits)
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", s.manifest)
    s.root = str(tmp_path / "dumps")
    monkeypatch.setattr(dumps, "DUMP_ROOT", s.root)
    no_fewshot_runs(monkeypatch, tmp_path)
    s.ckpt, s.tmp = ckpt, tmp_path
    s.rid = berturk.dump_run_id(berturk.identity(ckpt))
    s.path, s.sha = _bind(monkeypatch, s, {"berturk": [s.rid]}, "cpu", 0)
    yield s
    torch.set_num_threads(threads)


def bert_argv(s, name, *extra):
    return [split(s, name), "--model", "berturk", "--checkpoint", s.ckpt, "--workers", "0", "--device", "cpu",
            "--manifest", s.manifest] + list(extra)


def test_gate_admits_a_bound_berturk_final_seed_once(bsnap):
    for name in ("test", "hotels_probe"):
        run_dir, got = score.main(bert_argv(bsnap, name))
        assert (run_dir, got) == (os.path.join(bsnap.root, bsnap.rid), name)
        side = sidecar(run_dir, name)
        assert side["protocol_sha256"] == bsnap.sha and side["system"]["kind"] == "berturk"
        logits, _, _, entry = dumps.load_dump(run_dir, name, bsnap.manifest)
        assert logits.shape == (entry["n_messages"], 48)
    before = files(bsnap.root)
    refused(bert_argv(bsnap, "test"), "fine-tuned berturk checkpoint", "already exists")
    refused(bert_argv(bsnap, "hotels_probe", "--force"), "--force")
    assert files(bsnap.root) == before


def test_gate_refuses_a_berturk_export_outside_the_final_seeds(bsnap, monkeypatch):
    other = "berturk-000000000000-00000000-00000000"
    _bind(monkeypatch, bsnap, {"berturk": [other], "laya": [bsnap.rid]}, "cpu", 0)

    def boom(*a, **k):
        raise AssertionError("the BERTurk checkpoint was parsed or loaded before the gate refused")
    monkeypatch.setattr(score, "check_berturk", boom)
    monkeypatch.setattr(score, "load_bert", boom)
    for name in ("test", "hotels_probe"):
        text = refused(bert_argv(bsnap, name), bsnap.rid, other)
        assert "never scored on test or HOTELS" in text
    assert files(bsnap.root) == []


def test_gate_refuses_berturk_without_the_frozen_protocol_before_hashing(bsnap, monkeypatch):
    from pilot import berturk
    monkeypatch.setattr(pr, "PROTOCOL_SHA256", None)

    def boom(*a, **k):
        raise AssertionError("the BERTurk checkpoint was hashed before the gate refused")
    monkeypatch.setattr(berturk, "identity", boom)
    monkeypatch.setattr(berturk, "require_files", boom)
    for name in ("test", "hotels_probe"):
        refused(bert_argv(bsnap, name), "is a split with role", "PROTOCOL_SHA256 is unset")
    assert files(bsnap.root) == []


def test_berturk_val_is_ungated(bsnap, monkeypatch):
    monkeypatch.setattr(pr, "PROTOCOL_SHA256", None)
    run_dir, _ = score.main(bert_argv(bsnap, "val", "--out-root", str(bsnap.tmp / "val-dumps")))
    assert "protocol_sha256" not in sidecar(run_dir, "val")


# ---- HOTELS: closed to the baselines and to evaluate.py ----

@pytest.mark.parametrize("model", ["majority", "prior"])
def test_baselines_are_refused_on_hotels_naming_the_role(snap, model):
    with pytest.raises(SystemExit, match="role ood_probe") as e:
        score.main([split(snap, "hotels_probe"), "--model", model, "--manifest", snap.manifest,
                    "--out-root", str(snap.tmp / "b")])
    assert "TRAINED_ON" in str(e.value.code) and "only --model laya or berturk" in str(e.value.code)
    assert not (snap.tmp / "b").exists()


def test_evaluate_still_refuses_hotels_before_any_load(snap, monkeypatch):
    import evaluate

    def boom(*a, **k):
        raise AssertionError("evaluate.py reached a checkpoint resolve on hotels_probe")
    monkeypatch.setattr(score, "resolve_checkpoint", boom)
    no_identity(monkeypatch)
    with pytest.raises(SystemExit, match="role ood_probe; expected one of test, val"):
        evaluate.main([split(snap, "hotels_probe"), "--checkpoint", snap.ckpt, "--workers", "0",
                       "--manifest", snap.manifest, "--out-root", str(snap.tmp / "e")])
    assert not (snap.tmp / "e").exists()


# ---- Phase 9: few-shot checkpoints through the runs-manifest allow-list (09-05, D-09) ----

@pytest.fixture
def fsnap(tmp_path, synthetic_raw, monkeypatch):
    """The few-shot bench: the synthetic frozen protocol with a fewshot block (tests/fewshot_synthetic.bench), a
    planted set, its runs manifest at a temp pilot.fewshot.RUNS_PATH and a temp DUMP_ROOT."""
    import torch

    import fewshot as runner
    from fewshot_synthetic import bench, plant_fewshot_set
    from pilot import fewshot as fs
    threads = torch.get_num_threads()
    ns = bench(tmp_path, synthetic_raw, monkeypatch)
    ns.ckpts = plant_fewshot_set(ns, str(tmp_path / "ckpt"))
    ns.runs_path = str(tmp_path / "fewshot" / "runs.json")
    monkeypatch.setattr(fs, "RUNS_PATH", ns.runs_path)
    ns.root = str(tmp_path / "dumps")
    monkeypatch.setattr(dumps, "DUMP_ROOT", ns.root)
    ns.runs = runner.main(["--runs-manifest"] + sorted(ns.ckpts.values()) + ["--out", ns.runs_path])
    ns.tmp, ns.fs = tmp_path, fs
    ns.cont = ns.ckpts["laya-d0-k2"]
    ns.rid = dumps.run_id("laya", pilot.weights.identity(ns.cont))
    yield ns
    torch.set_num_threads(threads)


def fs_argv(ns, name, *extra, ckpt=None):
    return [os.path.join(ns.out_dir, "splits", name + ".jsonl"), "--model", "laya", "--checkpoint", ckpt or ns.cont,
            "--workers", "0", "--device", "cpu", "--manifest", ns.manifest] + list(extra)


def rewrite_runs(ns, edit):
    """Apply edit(runs dict) to the runs manifest on disk."""
    with open(ns.runs_path, encoding="utf-8") as f:
        runs = json.load(f)
    edit(runs)
    with open(ns.runs_path, "w", encoding="utf-8") as f:
        json.dump(runs, f)


def test_fewshot_gate_refuses_a_listed_checkpoint_on_hotels(fsnap, monkeypatch):
    no_laya_load(monkeypatch)
    text = refused(fs_argv(fsnap, "hotels_probe"), "laya-d0-k2", "test only", "HOTELS is never opened")
    assert "fewshot.rules.test_admission" in text
    assert files(fsnap.root) == []


def test_fewshot_gate_refuses_an_unlisted_or_edited_checkpoint(fsnap, monkeypatch):
    """A few-shot checkpoint outside the allow-list (an edited copy of a listed one: its dump run id changes) is
    refused on test, naming results/fewshot/runs.json."""
    edited = str(fsnap.tmp / "edited")
    shutil.copytree(fsnap.cont, edited)
    path = os.path.join(edited, "rl_agent_config.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["note"] = "edited"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    rid = dumps.run_id("laya", pilot.weights.identity(edited))
    assert rid != fsnap.rid
    no_laya_load(monkeypatch)
    text = refused(fs_argv(fsnap, "test", ckpt=edited), rid, "results/fewshot/runs.json")
    assert "never scored on test or HOTELS" in text
    assert files(fsnap.root) == []


@pytest.mark.parametrize("case", ["other_protocol", "changed_digest", "other_mode", "not_json"])
def test_fewshot_gate_refuses_a_runs_manifest_it_cannot_check(fsnap, monkeypatch, case):
    if case == "other_protocol":
        rewrite_runs(fsnap, lambda r: r.update(protocol_sha256="f" * 64))
        needles = ("runs.json is refused", "protocol sha256 " + "f" * 64)
    elif case == "changed_digest":
        def edit(r):
            for e in r["entries"]:
                if e["stage"] == "continued":
                    e["digest"] = "0" * 64
        rewrite_runs(fsnap, edit)
        needles = ("runs.json is refused", "digest", "frozen draw 0")
    elif case == "other_mode":
        rewrite_runs(fsnap, lambda r: r.update(mode="fallback"))
        needles = ("runs.json is refused", "'fallback' protocol")
    else:
        with open(fsnap.runs_path, "w", encoding="utf-8") as f:
            f.write("{not json")
        needles = ("runs.json is refused", "not valid JSON")
    no_laya_load(monkeypatch)
    refused(fs_argv(fsnap, "test"), *needles)
    assert files(fsnap.root) == []


@pytest.mark.parametrize("flags, needle", [
    (["--force"], "--force"),
    (["--n", "3"], "--n"),
    (["--out-root", "ELSEWHERE"], "--out-root"),
], ids=["force", "n", "out_root"])
def test_fewshot_gate_refuses_force_n_and_out_root(fsnap, monkeypatch, flags, needle):
    flags = [str(fsnap.tmp / "elsewhere") if f == "ELSEWHERE" else f for f in flags]
    no_identity(monkeypatch)
    refused(fs_argv(fsnap, "test", *flags), needle)
    assert files(fsnap.root) == [] and not (fsnap.tmp / "elsewhere").exists()


def test_fewshot_gate_refuses_another_device_or_worker_count(fsnap, monkeypatch):
    no_identity(monkeypatch)
    argv = fs_argv(fsnap, "test")
    argv[argv.index("--workers") + 1] = "2"
    refused(argv, "workers")
    assert files(fsnap.root) == []


def test_fewshot_gate_scores_a_listed_checkpoint_once(fsnap):
    run_dir, _ = score.main(fs_argv(fsnap, "test"))
    side = sidecar(run_dir, "test")
    with open(fsnap.runs_path, "rb") as f:
        assert side["fewshot_runs_sha256"] == pilot.data.sha256_bytes(f.read())
    before = files(fsnap.root)
    text = refused(fs_argv(fsnap, "test"), "already exists", "once per split")
    assert "listed few-shot checkpoint" in text
    assert files(fsnap.root) == before


def _copy_with_config(ns, name, edit=None):
    """A copy of the listed continued checkpoint under `name`, its rl_agent_config.json changed by edit(cfg)."""
    dst = str(ns.tmp / name)
    shutil.copytree(ns.cont, dst)
    if edit is not None:
        path = os.path.join(dst, "rl_agent_config.json")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        edit(cfg)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
    return dst, dumps.run_id("laya", pilot.weights.identity(dst))


def _list_as_the_continued_entry(ns, rid):
    """Swap the listed laya d0-k2 entry's dump_run_id for `rid`: the entry stays well formed (check_runs passes)."""
    def edit(runs):
        for e in runs["entries"]:
            if (e["kind"], e["stage"], e["draw"], e["k"]) == ("laya", "continued", 0, 2):
                e["dump_run_id"] = rid
    rewrite_runs(ns, edit)
    assert ns.fs.admitted(ns.protocol, ns.pin, "laya", rid) is not None


def test_fewshot_gate_refuses_a_swapped_entry_of_a_retrained_run(fsnap, monkeypatch):
    """09-REVIEW WR-01: a well-formed runs.json entry whose dump_run_id was swapped for another run of the same recipe
    (its own record gives other losses) is refused before any load: the gate re-derives the entry from the checkpoint
    (fewshot.entry_of) and names the keys that differ."""
    def retrain(cfg):
        for row in cfg["training"]["ray_train"]["record"]["epochs"]:
            row["loss"] = row["loss"] + 1.0
    ckpt, rid = _copy_with_config(fsnap, "retrained", retrain)
    assert rid != fsnap.rid
    _list_as_the_continued_entry(fsnap, rid)
    no_laya_load(monkeypatch)
    text = refused(fs_argv(fsnap, "test", ckpt=ckpt), "its own record gives another results/fewshot/runs.json entry",
                   "loss_by_epoch", "laya-d0-k2")
    differ = re.search(r"\(([^()]*) differ\)", text).group(1).split(", ")
    assert "loss_by_epoch" in differ and "dump_run_id" not in differ, differ   # the swapped id is the checkpoint's
    assert files(fsnap.root) == []


def test_fewshot_gate_refuses_a_swapped_entry_of_a_non_fewshot_checkpoint(fsnap, monkeypatch):
    """09-REVIEW WR-01: an entry swapped to point at a checkpoint without a Phase 9 binding (here its canonical config
    lost the few-shot binding, as a selection export has none) is refused: the checkpoint re-derives no entry."""
    def unbind(cfg):
        cfg["training"]["ray_train"]["canonical"].pop("fewshot")
    ckpt, rid = _copy_with_config(fsnap, "unbound", unbind)
    _list_as_the_continued_entry(fsnap, rid)
    no_laya_load(monkeypatch)
    refused(fs_argv(fsnap, "test", ckpt=ckpt), "its own record does not re-derive a results/fewshot/runs.json entry")
    assert files(fsnap.root) == []


def test_fewshot_gate_admits_a_listed_checkpoint_under_another_dir_name(fsnap):
    """The re-derived entry may differ from the listed one only in checkpoint_name (score.REDERIVE_IGNORED): a
    byte-identical copy has the same dump run id, so it is the listed checkpoint."""
    assert score.REDERIVE_IGNORED == ("checkpoint_name",)
    ckpt, rid = _copy_with_config(fsnap, "renamed")
    assert rid == fsnap.rid
    run_dir, _ = score.main(fs_argv(fsnap, "test", ckpt=ckpt))
    assert os.path.basename(run_dir) == fsnap.rid


def test_fewshot_gate_without_a_runs_manifest_refuses_as_in_phase_8(fsnap, monkeypatch):
    os.remove(fsnap.runs_path)
    assert fsnap.fs.admitted(fsnap.protocol, fsnap.pin, "laya", fsnap.rid) is None
    no_laya_load(monkeypatch)
    text = refused(fs_argv(fsnap, "test"), fsnap.rid, "final seeds (none)", "results/fewshot/runs.json")
    assert "never scored on test or HOTELS" in text
    assert files(fsnap.root) == []


def test_fewshot_masked_base_on_val_is_ungated(fsnap):
    run_dir, _ = score.main(fs_argv(fsnap, "val", ckpt=fsnap.ckpts["laya-base-s0"]))
    side = sidecar(run_dir, "val")
    assert "protocol_sha256" not in side and "fewshot_runs_sha256" not in side
