"""Few-shot test admission (FEW-01, FEW-02; D-01, D-04, D-09, D-14; 09-05): planted few-shot checkpoints ->
`fewshot.py --inspect` -> `fewshot.py --runs-manifest` -> score.py's protocol gate scores a listed checkpoint on test
exactly once; every allow-list refusal; the guarded deletion modes.

tests/fewshot_synthetic.bench installs the synthetic frozen protocol (tag card, k 1 and 2, one draw, base seed 0, tiny
recipes, scoring on cpu with workers 0, a D-08 decision record) and plant_fewshot_set plants, without training, one
consistent masked base and continued set per kind (tiny Laya checkpoints with a finished ray_train.py run record,
tiny BERTurk files with berturk_train.json). pilot.fewshot.RUNS_PATH and pilot.dumps.DUMP_ROOT point at temp paths.
Offline, no Ray: runs in `make test`.
"""
import json
import os

import pytest

import fewshot as runner
import pilot.data
import score
from fewshot_synthetic import bench, plant_fewshot_set
from pilot import dumps
from pilot import fewshot as fs

pytestmark = pytest.mark.unit


@pytest.fixture
def keep_threads():
    import torch
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


@pytest.fixture
def planted(tmp_path, synthetic_raw, monkeypatch, keep_threads):
    """The synthetic protocol, a planted set, a temp RUNS_PATH and DUMP_ROOT."""
    ns = bench(tmp_path, synthetic_raw, monkeypatch)
    ns.ckpts = plant_fewshot_set(ns, str(tmp_path / "ckpt"))
    ns.runs_path = str(tmp_path / "fewshot" / "runs.json")
    monkeypatch.setattr(fs, "RUNS_PATH", ns.runs_path, raising=False)
    ns.root = str(tmp_path / "dumps")
    monkeypatch.setattr(dumps, "DUMP_ROOT", ns.root)
    ns.tmp = tmp_path
    return ns


def split(ns, name):
    return os.path.join(ns.out_dir, "splits", name + ".jsonl")


def laya_argv(ns, name, ckpt, *extra):
    return [split(ns, name), "--model", "laya", "--checkpoint", ckpt, "--workers", "0", "--device", "cpu",
            "--manifest", ns.manifest] + list(extra)


def sidecar(run_dir, name):
    with open(os.path.join(run_dir, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def test_runs_manifest_admits_a_listed_checkpoint_on_test_once(planted, capsys):
    ns = planted
    names = sorted(ns.ckpts)
    assert names == ["berturk-base-s0", "berturk-d0-k1", "berturk-d0-k2", "laya-base-s0", "laya-d0-k1", "laya-d0-k2"]
    dirs = [ns.ckpts[n] for n in names]

    # --inspect: each checkpoint re-validated on its own, one entry each, its dump run id and proof printed
    capsys.readouterr()
    shown = runner.main(["--inspect"] + dirs)
    printed = json.loads(capsys.readouterr().out)
    assert printed == shown and len(shown) == len(dirs)
    by_name = {e["checkpoint_name"]: e for e in shown}
    assert sorted(by_name) == names
    lc = by_name["laya-d0-k2"]
    assert lc["kind"] == "laya" and lc["stage"] == "continued" and (lc["draw"], lc["k"]) == (0, 2)
    assert lc["digest"] == ns.block["draws"][0]["digest"] and lc["base_seed"] == 0
    assert lc["base_run_id"] == by_name["laya-base-s0"]["train_run_id"]
    assert lc["dump_run_id"].startswith("laya-")
    assert lc["proof"]["new_tag_items"] == {"card": lc["proof"]["support_rows"]}
    assert by_name["laya-base-s0"]["proof"]["held_out_items"] == {"card": 0}
    assert by_name["berturk-base-s0"]["proof"]["unmasked_held_out_labels"] == {"card": 0}
    assert by_name["berturk-d0-k1"]["dump_run_id"].startswith("berturk-")
    assert lc["wall_clock_s"] == pytest.approx(2.0 + 0.5 + 1.0 + 0.5)      # WALL_RULE: train_s + save_s + load + save
    assert by_name["berturk-d0-k1"]["wall_clock_s"] == pytest.approx(2.0 + 1.0)
    assert set(lc) == set(fs.ENTRY_KEYS)
    for text in (json.dumps(shown),):
        assert "bank_" not in text and "hotel_" not in text                    # no uid list or text

    # --runs-manifest: written once, check_runs accepts it, entries sorted by (kind, stage, seed, draw, k)
    runs = runner.main(["--runs-manifest"] + list(reversed(dirs)) + ["--out", ns.runs_path])
    with open(ns.runs_path, "rb") as f:
        raw = f.read()
    assert json.loads(raw) == runs and raw.endswith(b"\n")
    assert runs["schema"] == fs.RUNS_SCHEMA and runs["protocol_sha256"] == ns.pin and runs["mode"] == "primary"
    assert runs["wall_rule"] == fs.WALL_RULE
    with open(ns.decision_path, "rb") as f:
        assert runs["decision"]["sha256"] == pilot.data.sha256_bytes(f.read())
    key = [(e["kind"], e["stage"], e["seed"], -1 if e["draw"] is None else e["draw"], -1 if e["k"] is None else e["k"])
           for e in runs["entries"]]
    assert key == sorted(key) and len(key) == 6
    assert fs.check_runs(runs, ns.protocol, ns.pin, "primary") == runs["entries"]
    got, sha = fs.read_runs(ns.protocol, ns.pin)
    assert got == runs and sha == pilot.data.sha256_bytes(raw)
    assert fs.admitted(ns.protocol, ns.pin, "laya", lc["dump_run_id"]) == (by_name["laya-d0-k2"], sha)
    assert fs.admitted(ns.protocol, ns.pin, "berturk", lc["dump_run_id"]) is None

    # score.py: the listed continued checkpoint on test is admitted once, bound to the pin and the allow-list
    run_dir, name = score.main(laya_argv(ns, "test", ns.ckpts["laya-d0-k2"]))
    assert (run_dir, name) == (os.path.join(ns.root, lc["dump_run_id"]), "test")
    side = sidecar(run_dir, "test")
    assert side["protocol_sha256"] == ns.pin and side["fewshot_runs_sha256"] == sha
    logits, _, _, entry = dumps.load_dump(run_dir, "test", ns.manifest)
    assert logits.shape == (entry["n_messages"], 48)
    with pytest.raises(SystemExit) as e:
        score.main(laya_argv(ns, "test", ns.ckpts["laya-d0-k2"]))
    text = str(e.value.code)
    assert text.startswith(score.GATE) and "already exists" in text and "\n" not in text

    # the masked base on val: ungated, as every val dump is (D-05)
    run_dir, _ = score.main(laya_argv(ns, "val", ns.ckpts["laya-base-s0"]))
    side = sidecar(run_dir, "val")
    assert "protocol_sha256" not in side and "fewshot_runs_sha256" not in side
    assert os.path.basename(run_dir) == by_name["laya-base-s0"]["dump_run_id"]
