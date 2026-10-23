"""The few-shot results record (09-06, FEW-02, D-12): pilot.fewshot_report and fewshot.py --report / --render.

Every test runs on the synthetic frozen protocol (tests/fewshot_synthetic.synthetic_protocol: tag card, k 1 and 2, two
draws, primary metric k = 2, a bootstrap block with B 200), a runs manifest planted straight from its slots (plant_runs)
and the dumps it lists (plant_fewshot_dumps: seeded logits, bases scored on val ungated and on test gated, continued
runs on test gated). No checkpoint and no model.

- primary: per draw the Laya and BERTurk macro-AP at the primary k equal pilot.metrics.new_tag_ap on the test dumps at
  T = 1 and sklearn's average_precision_score (test-only oracle); delta = Laya - BERTurk; mean_delta is
  float(np.mean(draws)), exactly what pilot.verdict.criterion_d checks, and its few-shot path is MET exactly when
  mean_delta >= 0.05
- fewshot.py --report --out writes build_record's bytes; refused without the frozen protocol (before any dump is read)
  and without a runs manifest
- build_record refuses a test dump of another pin or runs manifest, a val dump carrying protocol_sha256, a missing
  dump, a sidecar of another training run or seed, and test dumps that score different uid orders
"""
import copy
import json
import os

import numpy as np
import pytest

import fewshot as fewshot_cli
from fewshot_synthetic import plant_fewshot_dumps, plant_rules, plant_runs, split_gold, synthetic_protocol
from intents import INTENTS
from pilot import dumps, metrics, verdict
from pilot import fewshot as fs
from pilot import protocol as pr

pytestmark = pytest.mark.unit

TAGS = list(INTENTS)
BOOT = {"B": 200, "seed": 0, "level": 0.95, "coverage": 80, "unit": "message", "method": metrics.BOOT_METHOD}
PRIMARY_K = 2


def report_rules(seeds=(0, 1)):
    rules = plant_rules(seeds)
    rules["primary_metric"] = {"k": PRIMARY_K, "rule": "the synthetic primary metric at k = %d" % PRIMARY_K}
    return rules


def install(tmp_path, synthetic_raw, monkeypatch, mode="primary", logits=None, seed=0):
    """The synthetic protocol, a planted runs manifest at fs.RUNS_PATH and every dump it lists under dumps.DUMP_ROOT;
    returns (ns, runs, runs sha256, dump root)."""
    seeds = (0, 1) if mode == "primary" else (0,)
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, draw_seeds=(0, 1), rules=report_rules(seeds),
                            mode=mode, extra={"bootstrap": BOOT})
    path = plant_runs(ns, str(tmp_path / "fewshot" / "runs.json"))
    monkeypatch.setattr(fs, "RUNS_PATH", path)
    runs, sha = fs.read_runs(ns.protocol, ns.pin)
    root = str(tmp_path / "dumps")
    monkeypatch.setattr(dumps, "DUMP_ROOT", root)
    plant_fewshot_dumps(ns, runs, sha, root, seed=seed, logits=logits)
    return ns, runs, sha, root


@pytest.fixture
def planted(tmp_path, synthetic_raw, monkeypatch):
    return install(tmp_path, synthetic_raw, monkeypatch)


def _entry(runs, kind, stage, draw=None, k=None, seed=None):
    for e in runs["entries"]:
        if (e["kind"], e["stage"]) == (kind, stage) and (stage == "base" and e["seed"] == seed
                                                         or stage == "continued" and (e["draw"], e["k"]) == (draw, k)):
            return e
    raise AssertionError("no %s %s entry" % (kind, stage))


def _test_ap(ns, root, e):
    logits, uids, _, entry = dumps.load_dump(os.path.join(root, e["dump_run_id"]), "test", ns.manifest)
    Y = dumps.read_gold(entry, ns.manifest, uids)
    return Y, metrics.probabilities(logits, 1.0)


def _report(tmp_path, *extra):
    out = str(tmp_path / "out" / "fewshot.json")
    fewshot_cli.main(["--report", "--out", out] + list(extra))
    with open(out, encoding="utf-8") as f:
        return out, json.load(f)


# ---- the primary metric and criterion (d) (Task 1, tracer) ----

def test_record_primary_matches_criterion_d(planted, tmp_path):
    """fewshot.py --report writes the record; its primary block is the k = 2 per-draw Laya - BERTurk macro-AP and its
    mean, which pilot.verdict.criterion_d accepts as is; the few-shot path is MET exactly when mean_delta >= 0.05."""
    ns, runs, sha, root = planted
    _, rec = _report(tmp_path)
    p = rec["primary"]
    assert p["k"] == PRIMARY_K and p["metric"] == "new_tag_macro_ap"
    assert [d["draw"] for d in p["per_draw"]] == [0, 1]
    assert p["draws"] == [d["delta"] for d in p["per_draw"]]
    assert all(d["delta"] == d["laya"] - d["berturk"] for d in p["per_draw"])
    assert p["mean_delta"] == float(np.mean(np.asarray(p["draws"], dtype=np.float64)))
    rule = copy.deepcopy(verdict.DEFAULT_CRITERIA["d"])
    cal, abst = {"delta": 0.01, "ci": [0.0, 0.02]}, {"delta": 0.0, "ci": [-0.01, 0.01]}   # neither path MET
    res = verdict.criterion_d(cal, abst, {"mean_delta": p["mean_delta"], "draws": p["draws"]}, rule)
    few = res["paths"][2]
    assert few["path"] == "fewshot" and few["mean_delta"] == p["mean_delta"]
    assert few["status"] == (verdict.MET if p["mean_delta"] >= 0.05 else verdict.NOT_MET)
    assert res["status"] == (verdict.PASS if p["mean_delta"] >= 0.05 else verdict.FAIL)


def _separating(good):
    """A logits callback: `good` kind ranks every positive above every negative, the other kind inverts the ranking."""
    def make(e, split, Y, rng):
        sign = 1.0 if e["kind"] == good else -1.0
        return np.where(Y, 4.0, -4.0) * sign + rng.normal(0.0, 0.1, size=Y.shape)
    return make


@pytest.mark.parametrize("good, status", [("laya", "MET"), ("berturk", "NOT_MET")])
def test_criterion_d_fewshot_path_follows_the_mean_delta(tmp_path, synthetic_raw, monkeypatch, good, status):
    """A Laya that ranks the new tag perfectly against a BERTurk that inverts it is MET; the reverse is NOT_MET."""
    from pilot import fewshot_report as fr
    ns, runs, sha, root = install(tmp_path, synthetic_raw, monkeypatch, logits=_separating(good))
    p = fr.build_record(ns.protocol, ns.pin)["primary"]
    assert all(d["laya"] == (1.0 if good == "laya" else d["laya"]) for d in p["per_draw"])
    res = verdict.criterion_d({"delta": 0.01, "ci": [0.0, 0.02]}, {"delta": 0.0, "ci": [-0.01, 0.01]},
                              {"mean_delta": p["mean_delta"], "draws": p["draws"]}, verdict.DEFAULT_CRITERIA["d"])
    assert res["paths"][2]["status"] == getattr(verdict, status)
    assert (p["mean_delta"] > 0) == (good == "laya")


def test_primary_equals_new_tag_ap_and_the_sklearn_oracle(planted):
    """Per draw, each model's macro-AP is new_tag_ap of its continued run's test dump at T = 1, and the per-tag AP is
    sklearn's average_precision_score (a test-only oracle)."""
    from sklearn.metrics import average_precision_score

    from pilot import fewshot_report as fr
    ns, runs, sha, root = planted
    rec = fr.build_record(ns.protocol, ns.pin)
    for d in rec["primary"]["per_draw"]:
        for kind in ("laya", "berturk"):
            e = _entry(runs, kind, "continued", d["draw"], PRIMARY_K)
            Y, P = _test_ap(ns, root, e)
            want = metrics.new_tag_ap(Y, P, ns.block["tags"])
            assert d[kind] == want["macro"]
            j = TAGS.index("card")
            assert abs(want["per_tag"]["card"] - average_precision_score(Y[:, j], P[:, j])) < 1e-12
            assert rec["ap"][kind]["continued"]["d%d-k%d" % (d["draw"], PRIMARY_K)]["macro"] == want["macro"]
    direct = fr.primary(ns.protocol, ns.pin, runs, sha, root, ns.manifest)
    assert direct == rec["primary"]


def test_report_cli_writes_the_record_build_record_returns(planted, tmp_path):
    ns, runs, sha, root = planted
    from pilot import fewshot_report as fr
    out, rec = _report(tmp_path)
    assert rec == fr.build_record(ns.protocol, ns.pin)
    with open(out, "rb") as f:
        assert f.read() == fr.record_bytes(rec)
    assert rec["schema"] == fr.RESULTS_SCHEMA == "fewshot-results/1"
    assert rec["protocol_sha256"] == ns.pin and rec["runs"]["sha256"] == sha
    assert rec["decision"]["mode"] == "primary" and rec["decision"]["disclosure"] is None
    assert fr.read_record(out) == (rec, fr.sha256_bytes(fr.record_bytes(rec)))
    _report(tmp_path)                                          # a rebuild overwrites the same path


def test_report_refuses_without_the_frozen_protocol_before_any_dump(planted, tmp_path, monkeypatch):
    ns, runs, sha, root = planted
    read = []
    monkeypatch.setattr(dumps, "load_dump", lambda *a, **k: read.append(a))
    monkeypatch.setattr(pr, "PROTOCOL_SHA256", "0" * 64)
    with pytest.raises(SystemExit) as e:
        _report(tmp_path)
    assert "frozen protocol" in str(e.value.code) and not read
    assert not os.path.exists(str(tmp_path / "out" / "fewshot.json"))


def test_report_refuses_without_a_runs_manifest(planted, tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "RUNS_PATH", str(tmp_path / "absent" / "runs.json"))
    with pytest.raises(SystemExit) as e:
        _report(tmp_path)
    assert "runs manifest" in str(e.value.code) and "--runs-manifest" in str(e.value.code)


def _rewrite_sidecar(run_dir, split, **changes):
    path = os.path.join(run_dir, split + ".json")
    with open(path, encoding="utf-8") as f:
        side = json.load(f)
    for key, value in changes.items():
        if value is None:
            side.pop(key, None)
        else:
            side[key] = value
    with open(path, "wb") as f:
        f.write(dumps._sidecar_bytes(side))


@pytest.mark.parametrize("split, changes, needle", [
    ("test", {"protocol_sha256": "1" * 64}, "protocol_sha256"),
    ("test", {"fewshot_runs_sha256": "2" * 64}, "fewshot_runs_sha256"),
    ("test", {"fewshot_runs_sha256": None}, "fewshot_runs_sha256"),
    ("val", {"protocol_sha256": "3" * 64}, "ungated"),
    ("test", {"seed": 7}, "seed"),
    ("test", {"training": {"run_id": "laya-train-other"}}, "training run"),
])
def test_build_record_refuses_an_unbound_dump(planted, split, changes, needle):
    from pilot import fewshot_report as fr
    ns, runs, sha, root = planted
    e = _entry(runs, "laya", "base", seed=0)
    _rewrite_sidecar(os.path.join(root, e["dump_run_id"]), split, **changes)
    with pytest.raises(SystemExit) as x:
        fr.build_record(ns.protocol, ns.pin)
    assert needle in str(x.value.code) and e["dump_run_id"] in str(x.value.code)


def test_build_record_refuses_a_missing_dump(planted):
    from pilot import fewshot_report as fr
    ns, runs, sha, root = planted
    e = _entry(runs, "berturk", "continued", 1, 1)
    os.remove(os.path.join(root, e["dump_run_id"], "test.npz"))
    with pytest.raises(SystemExit) as x:
        fr.build_record(ns.protocol, ns.pin)
    assert "test.npz not found" in str(x.value.code)


def test_build_record_refuses_unpaired_uid_orders(planted):
    """A test dump scoring the split in another row order (a subset dump of every uid, permuted) is refused: every
    model and draw must score one uid order (the bootstrap pairs messages)."""
    from pilot import fewshot_report as fr
    ns, runs, sha, root = planted
    e = _entry(runs, "berturk", "continued", 1, 2)
    run_dir = os.path.join(root, e["dump_run_id"])
    logits, uids, side, _ = dumps.load_dump(run_dir, "test", ns.manifest)
    for name in ("test.npz", "test.json"):
        os.remove(os.path.join(run_dir, name))
    side = {k: v for k, v in side.items() if k not in ("schema_version", "dump_sha256")}
    side["n_subset"] = len(uids)
    dumps.write_dump(run_dir, "test", logits[::-1].copy(), uids[::-1], side)
    with pytest.raises(SystemExit) as x:
        fr.build_record(ns.protocol, ns.pin)
    assert "uid order" in str(x.value.code)


def test_split_gold_helper_reads_the_synthetic_test(planted):
    ns, runs, sha, root = planted
    entry, uids, Y = split_gold(ns.manifest, "test")
    assert entry["role"] == "test" and len(uids) == Y.shape[0] and Y[:, TAGS.index("card")].sum() >= 1
