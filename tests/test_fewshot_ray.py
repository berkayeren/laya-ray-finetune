"""The Laya few-shot tracer (FEW-01, FEW-02; 09-02): a label-masked base, then a continued run from it, end to end
through ray_train.py and Ray Train on CPU, under a synthetic frozen protocol.

tests/fewshot_synthetic.synthetic_protocol installs a protocol whose fewshot block holds the tag card, k 1 and 2, one
draw (seed 0) and tiny recipes (base: --epochs 2 --stop-after-epoch 1, neg 2, micro-batch 2, grad-accum 1, max_len 32,
cpu, one worker; continued: 1 epoch at the base LRs). Its postfit record's chosen canonical is the canonical config of
the same tiny command without the few-shot flags, so check_base binds the masked base to it. Each ray_train.py command
runs in a subprocess (a DRIVER that pins the synthetic protocol, manifest and D-08 decision record, then calls
ray_train.main), like a user's command:

- `--fewshot-base` deletes card from every row before neg sampling, keeps every message, builds no item of the card
  question and records the proof (fewshot.stage base, held_out_items {card: 0}, items_per_question summing to the items
  built, which equal pilot.runs.neg_items of the masked rows);
- `--fewshot-draw 0 --fewshot-k 2 --base <masked base>` trains on the frozen support and replay rows of that draw by
  the frozen Laya item rule (pilot.fewshot.laya_rows): exactly one card item per support row, the item count of
  pilot.fewshot.laya_item_count, and the base named by its run id and canonical config;
- `--print-run-id` of each command prints the run id the run then trains under, and no uid of the frozen lists reaches
  a checkpoint config.
Runs in `make test` under the ray marker (CPU only, no weights or data).
"""
import json
import os
import subprocess
import sys

import pytest

import pilot.data
from conftest import prepared
from fewshot_synthetic import postfit_systems, synthetic_protocol
from pilot import dumps, runs
from pilot import fewshot as fs
from pilot.weights import identity
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.ray

REPO = pilot.data.REPO_ROOT
TIMEOUT = 900

# the frozen rules of the synthetic protocol: the base recipe and the continued training, at the tiny model's sizes
LAYA_RULE = {"trainer": "ray_train.py", "epochs": 2, "stop_after_epoch": 1, "neg": 2, "encoder_lr": 2e-05,
             "head_lr": 0.0001, "device": "cpu", "workers": 1, "micro_batch": 2, "grad_accum": 1, "max_len": 32}
CONTINUED = {"epochs": 1, "encoder_lr": 2e-05, "head_lr": 0.0001, "rule": "the synthetic continued training"}
COMMON = ["--device", "cpu", "--workers", "1", "--micro-batch", "2", "--grad-accum", "1", "--max-len", "32",
          "--neg", "2", "--encoder-lr", "2e-05", "--head-lr", "0.0001", "--seed", "0"]

DRIVER = """
import json, sys
import pilot.data, pilot.fewshot, pilot.protocol
import ray_train
p = json.loads(sys.argv[1])
pilot.protocol.PROTOCOL_PATH = p["protocol_path"]
pilot.protocol.PROTOCOL_SHA256 = p["pin"]
pilot.data.DEFAULT_MANIFEST = p["manifest"]
pilot.fewshot.DECISION_PATH = p["decision_path"]
ray_train.main(p["argv"])
"""


def chosen_canonical(argv):
    """The canonical config ray_train.py gives `argv` (no few-shot flag): the synthetic chosen selection run's."""
    import ray_train
    a = ray_train.parser().parse_args(argv)
    entry = pilot.data.verify_split_file(a.data, manifest_path=a.manifest, roles=("train",))
    return runs.canonical(ray_train.identity_values(a, ray_train.resolve(a), entry, a.base, identity(a.base)))


def rules():
    base_runs = json.loads(json.dumps(fs.RULES["base_runs"]))
    base_runs.update(seeds=[0], laya=dict(LAYA_RULE))
    return {"base_runs": base_runs, "continued_training": dict(CONTINUED)}


def drive(ns, argv):
    """ray_train.main(argv) in a fresh interpreter (no PYTHONPATH) pinned to the synthetic protocol; its stdout."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    arg = json.dumps({"protocol_path": ns.protocol_path, "pin": ns.pin, "manifest": ns.manifest,
                      "decision_path": ns.decision_path, "argv": [str(x) for x in argv]})
    r = subprocess.run([sys.executable, "-c", DRIVER, arg], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TIMEOUT)
    assert r.returncode == 0, "ray_train.py %s failed (exit %d):\n%s" % (" ".join(argv[-6:]), r.returncode,
                                                                         r.stderr[-4000:])
    return r.stdout


def config(ckpt):
    with open(os.path.join(ckpt, "rl_agent_config.json"), encoding="utf-8") as f:
        raw = f.read()
    return raw, json.loads(raw)


@pytest.mark.timeout(TIMEOUT)
def test_laya_masked_base_then_continued_run(tmp_path, synthetic_raw, monkeypatch):
    tiny = tiny_checkpoint(tmp_path / "tiny")
    pre_dir, pre_manifest = prepared(tmp_path, synthetic_raw, name="pre")
    chosen = chosen_canonical(["--data", os.path.join(pre_dir, "splits", "selection_train.jsonl"),
                               "--manifest", pre_manifest, "--base", tiny, "--out", "o", "--epochs", "2"] + COMMON)
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, rules=rules(),
                            systems=postfit_systems(tmp_path, {"laya": chosen}))
    assert fs.chosen_canonical(ns.protocol, "laya") == chosen
    sel = os.path.join(ns.out_dir, "splits", "selection_train.jsonl")
    storage, base_out, cont_out = tmp_path / "ray", tmp_path / "ckpt-base", tmp_path / "ckpt-cont"
    tags = ns.block["tags"]
    assert tags == ["card"]

    # ---- FEW-01: the label-masked base ----
    base_argv = ["--data", sel, "--manifest", ns.manifest, "--base", tiny, "--out", base_out, "--storage", storage,
                 "--epochs", "2", "--stop-after-epoch", "1", "--fewshot-base"] + COMMON
    base_rid = drive(ns, base_argv + ["--print-run-id"]).strip().splitlines()[-1]
    assert not storage.exists()                                     # --print-run-id never starts Ray
    drive(ns, base_argv)
    raw, cfg = config(str(base_out))
    rec = runs.read_record(str(base_out))
    assert rec["run_id"] == base_rid and cfg["training"]["ray_train"]["run_id"] == base_rid
    canon = rec["canonical"]
    assert canon == cfg["training"]["ray_train"]["canonical"]
    assert canon["mask_tags"] == ["card"] and canon["fewshot"] == {"protocol_sha256": ns.pin, "stage": "base"}
    assert {k for k in set(canon) | set(chosen) if canon.get(k) != chosen.get(k)} <= set(fs.BASE_FREE)
    masked = fs.masked_rows(ns.rows, tags)
    assert rec["n_train_messages"] == len(ns.rows)                  # every message is kept
    assert rec["n_train_items"] == rec["n_train_items_counted"] == runs.neg_items(masked, 2) > 0
    proof = rec["fewshot"]
    assert proof["stage"] == "base" and proof["tags"] == ["card"] and proof["protocol_sha256"] == ns.pin
    assert proof["held_out_items"] == {"card": 0}
    assert proof["items_per_question"]["card"] == 0
    assert sum(proof["items_per_question"].values()) == rec["n_train_items"]
    assert "bank_" not in raw and "fewshot_rows" not in cfg["training"]["ray_train"]

    # ---- FEW-02: the continued run from that base on support plus replay ----
    base_ident = identity(str(base_out))
    cont_argv = ["--data", sel, "--manifest", ns.manifest, "--base", base_out, "--out", cont_out, "--storage", storage,
                 "--epochs", "1", "--fewshot-draw", "0", "--fewshot-k", "2"] + COMMON
    cont_rid = drive(ns, cont_argv + ["--print-run-id"]).strip().splitlines()[-1]
    drive(ns, cont_argv)
    raw, cfg = config(str(cont_out))
    rec = runs.read_record(str(cont_out))
    assert rec["run_id"] == cont_rid != base_rid
    canon = rec["canonical"]
    d = ns.block["draws"][0]
    assert canon["fewshot"] == {"protocol_sha256": ns.pin, "stage": "continued", "draw": 0, "k": 2,
                                "digest": d["digest"], "base_run_id": base_rid}
    assert "mask_tags" not in canon and canon["epochs"] == 1 and canon["seed"] == d["seed"] == canon["neg_seed"]
    assert canon["base_model_sha256"] == base_ident["model_sha256"]
    assert canon["base_config_sha256"] == base_ident["config_sha256"]
    assert canon["base_files_sha256"] == dumps.files_digest(base_ident["files_sha256"])
    sel_k = fs.selection(ns.block, 0, 2)
    assert rec["n_train_messages"] == len(sel_k["support"]) + len(sel_k["replay"])
    assert rec["n_train_items"] == rec["n_train_items_counted"] == fs.laya_item_count(ns.rows, sel_k, tags, 2)
    proof = rec["fewshot"]
    assert proof["stage"] == "continued" and proof["draw"] == 0 and proof["k"] == 2
    assert proof["digest"] == d["digest"] and proof["tags"] == ["card"] and proof["protocol_sha256"] == ns.pin
    assert proof["support_rows"] == len(sel_k["support"]) and proof["replay_rows"] == len(sel_k["replay"])
    assert proof["new_tag_items"] == {"card": len(sel_k["support"])}     # replay rows contributed none
    assert proof["base"] == {"run_id": base_rid, "canonical": runs.read_record(str(base_out))["canonical"]}
    assert "bank_" not in raw and "fewshot_rows" not in cfg["training"]["ray_train"]
