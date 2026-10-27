"""Unit contracts of pilot/tune.py (the partition, the dev objective, search points, attempts and the tune/1 record)
and of the Tune objects tune.py builds (the generator-level concurrency limit, ASHA, the param space).

Synthetic rows built in the tests; no data file, model or Ray cluster.
"""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from intents import INTENTS
from pilot import metrics
from pilot import tune as pt
from pilot.data import REPO_ROOT

pytestmark = pytest.mark.unit

TAGS = list(INTENTS)
LAYA_POINT = {"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 4}


def _row(uid: str, text: str = "x", tags: tuple[str, ...] = ("card",)) -> dict[str, Any]:
    return {"state": text, "uid": uid, "questions": {}, "expected": {t: t in tags for t in TAGS}}


def _rows() -> list[dict[str, Any]]:
    """bank_<fold>_<i> for folds 0..13, two rows each, in ascending folds."""
    return [_row("bank_%d_%d" % (f, i), "text %d %d" % (f, i)) for f in range(14) for i in range(2)]


def _digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(r["uid"] for r in rows).encode("utf-8")).hexdigest()


# ---- the partition ----

def test_partition_splits_by_uid_fold() -> None:
    rows = _rows()
    train, dev, block = pt.partition(rows)
    assert [r["uid"] for r in train] == ["bank_%d_%d" % (f, i) for f in range(12) for i in range(2)]
    assert [r["uid"] for r in dev] == ["bank_12_0", "bank_12_1", "bank_13_0", "bank_13_1"]
    assert list(block) == ["source", "train_folds", "dev_folds", "train_uids_sha256", "dev_uids_sha256"]
    assert block["source"] == "selection_train"
    assert block["train_folds"] == list(range(12)) and block["dev_folds"] == [12, 13]
    assert block["train_uids_sha256"] == _digest(train) and block["dev_uids_sha256"] == _digest(dev)
    assert not {r["uid"] for r in train} & {r["uid"] for r in dev}
    assert not {(r["uid"], r["state"]) for r in train} & {(r["uid"], r["state"]) for r in dev}
    assert pt.fold_of("bank_13_7") == 13


@pytest.mark.parametrize("edit, match", [
    (lambda rows: rows.__setitem__(3, _row("hotel_1_0")), "row 3: uid 'hotel_1_0' is not bank_<fold>_<index>"),
    (lambda rows: rows.__setitem__(3, _row("bank_x_1")), "row 3: uid 'bank_x_1' is not bank_<fold>_<index>"),
    (lambda rows: rows.append(_row("bank_14_0")), "fold 14, outside selection_train's folds 0-13"),
    (lambda rows: rows.insert(2, _row("bank_1_0")), "row 3: duplicate uid bank_1_0"),
    (lambda rows: rows.insert(4, _row("bank_0_9")), "row 4: fold 0 follows fold 1"),
    (lambda rows: rows.__delitem__(slice(24, None)), "the tuning-dev side .* is empty"),
    (lambda rows: rows.__delitem__(slice(0, 24)), "the tuning-train side .* is empty"),
], ids=["hotel-uid", "bad-fold", "fold-14", "duplicate", "decreasing", "no-dev", "no-train"])
def test_partition_refuses(edit: Any, match: str) -> None:
    rows = _rows()
    edit(rows)
    with pytest.raises(SystemExit, match=match):
        pt.partition(rows)


def test_split_rows_refuses_another_block() -> None:
    rows = _rows()
    train, dev, block = pt.partition(rows)
    assert pt.split_rows(rows, block) == (train, dev)
    with pytest.raises(SystemExit, match=r"differs from the search's \(key dev_uids_sha256\)"):
        pt.split_rows(rows, dict(block, dev_uids_sha256="0" * 64))
    with pytest.raises(SystemExit, match=r"\(key extra\)"):
        pt.split_rows(rows, dict(block, extra=1))


def test_read_partition_refuses_other_bytes(tmp_path: Path) -> None:
    path = tmp_path / "selection_train.jsonl"
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _rows()).encode("utf-8")
    path.write_bytes(raw)
    _train, _dev, block = pt.partition(_rows())
    train, dev = pt.read_partition(str(path), hashlib.sha256(raw).hexdigest(), block)
    assert len(train) == 24 and len(dev) == 4
    path.write_bytes(b"not json at all\n")                 # refused before parsing: the digest differs
    with pytest.raises(SystemExit, match="verified on the driver"):
        pt.read_partition(str(path), hashlib.sha256(raw).hexdigest(), block)


def test_shared_texts_counts_dev_rows_whose_text_trains() -> None:
    rows = _rows()
    rows[-1] = dict(rows[-1], state=rows[0]["state"])    # one tuning-dev row repeats a tuning-train text
    train, dev, _block = pt.partition(rows)
    assert pt.shared_texts(train, dev) == 1
    assert pt.shared_texts(*pt.partition(_rows())[:2]) == 0


def test_labels_refuse_a_missing_tag() -> None:
    rows = _rows()[:2]
    Y = pt.labels(rows)
    assert Y.dtype == bool and Y.shape == (2, 48) and Y[:, TAGS.index("card")].all() and Y.sum() == 2
    del rows[1]["expected"]["deny"]
    with pytest.raises(SystemExit, match="uid bank_0_1.* does not label tag deny"):
        pt.labels(rows)


def test_dev_objective_matches_pilot_metrics() -> None:
    rng = np.random.default_rng(0)
    Y = rng.random((30, 48)) < 0.1
    logits = rng.normal(0, 2, (30, 48)) + 2.5 * Y
    out = pt.dev_objective(Y, logits)
    assert list(out) == ["dev_micro_f1", "dev_theta", "dev_macro_f1", "dev_tp", "dev_fp", "dev_fn"]
    P = metrics.probabilities(logits, 1.0)
    theta = metrics.select_theta(Y, P)["global"]
    m = metrics.metrics(Y, P, theta)
    assert out["dev_theta"] == theta and out["dev_micro_f1"] == m["micro_f1"] and out["dev_macro_f1"] == m["macro_f1"]
    tp, fp, fn = out["dev_tp"], out["dev_fp"], out["dev_fn"]
    assert all(type(v) is int for v in (tp, fp, fn))
    assert tp + fn == int(Y.sum())
    assert out["dev_micro_f1"] == 2 * tp / (2 * tp + fp + fn)


# ---- search constants and points ----

def test_max_t_and_rungs() -> None:
    assert pt.max_t("laya") == 8 and pt.rungs("laya") == [1, 3]
    assert pt.max_t("berturk") == 30 and pt.rungs("berturk") == [1, 3, 9, 27]


@pytest.mark.parametrize("edit, match", [
    (lambda p: p.pop("neg"), "missing neg"),
    (lambda p: p.update(sigma=0.4), "unexpected sigma"),
    (lambda p: p.update(encoder_lr=2e-4), "encoder_lr 0.0002 is not a float in"),
    (lambda p: p.update(neg=5), "neg 5 is not one of"),
    (lambda p: p.update(epochs=5), "epochs 5 is not one of"),
    (lambda p: p.update(encoder_lr="1e-5"), "encoder_lr '1e-5' is not a float in"),
    (lambda p: p.update(neg=8.0), "neg 8.0 is not one of"),
], ids=["missing", "extra", "lr-high", "neg-5", "epochs-5", "string-lr", "float-neg"])
def test_check_point_refuses(edit: Any, match: str) -> None:
    p = dict(LAYA_POINT)
    edit(p)
    with pytest.raises(SystemExit, match=match):
        pt.check_point("laya", p)


def test_check_point_returns_the_space_order() -> None:
    p = {"epochs": 6, "neg": 16, "head_lr": 3e-05, "encoder_lr": 5e-06}
    assert list(pt.check_point("laya", p)) == list(pt.SPACES["laya"])
    with pytest.raises(SystemExit, match="not one of laya, berturk"):
        pt.check_point("bert", p)


def test_trial_index_and_dirname() -> None:
    assert pt.trial_index("7245d_00012") == 12
    assert pt.trial_dirname_of(12) == "trial_00012"
    for bad in ("abc", "7245d_12", "7245d_000012", "laya_trial_7245d_00012"):
        with pytest.raises(SystemExit, match="is not <hex>_<5 digits>"):
            pt.trial_index(bad)


def test_attempts_bump_and_read(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="has no attempts.json"):
        pt.read_attempts(str(tmp_path))
    assert pt.bump_attempts(str(tmp_path)) == 1
    assert pt.bump_attempts(str(tmp_path)) == 2
    assert pt.read_attempts(str(tmp_path)) == 2
    doc = json.loads((tmp_path / pt.ATTEMPTS_FILE).read_text(encoding="utf-8"))
    assert doc["attempts"] == 2 and len(doc["started"]) == 2
    (tmp_path / pt.ATTEMPTS_FILE).write_text('{"attempts": 3, "started": []}', encoding="utf-8")
    with pytest.raises(SystemExit, match="no valid attempt count"):
        pt.read_attempts(str(tmp_path))


def _result(epoch: int, tp: int, fp: int, fn: int, theta: float = 0.5) -> dict[str, Any]:
    return {"epoch": epoch, "dev_tp": tp, "dev_fp": fp, "dev_fn": fn, "dev_theta": theta, "dev_macro_f1": 0.25,
            "loss": 0.5, "train_s": 1.5, "dev_s": 0.25, "peak_total_bytes": 1000, "swap_delta_bytes": -4}


def test_curve_row_keeps_only_the_curve_keys() -> None:
    res = dict(_result(1, 2, 2, 2), hostname="h", pid=1, timestamp=2, trial_id="a_00000", dev_micro_f1=0.5)
    assert pt.curve_row(res) == _result(1, 2, 2, 2)
    for key, bad in (("dev_theta", 0.505), ("epoch", 0), ("dev_tp", -1), ("dev_fp", 1.0), ("loss", float("nan")),
                     ("dev_micro_f1", 0.4), ("peak_total_bytes", True)):
        with pytest.raises(SystemExit, match=key):
            pt.curve_row(dict(res, **{key: bad}))


def test_a_diverged_epoch_carries_a_null_loss(tmp_path: Path) -> None:
    """A diverged epoch (a non-finite training loss) is reported as loss None: curve_row keeps it, the record builds,
    writes and reads back byte for byte, and the render names the diverged trial; NaN itself is still refused."""
    res = dict(_result(1, 0, 0, 5), loss=None)
    assert pt.curve_row(res) == res
    with pytest.raises(SystemExit, match="loss"):
        pt.curve_row(dict(res, loss=float("inf")))
    c = _canonical()
    c["trials"][1]["curve"] = [dict(_result(1, 0, 0, 5), loss=None)]
    rec = pt.build_record(c)
    got, _ = pt.read_record(pt.write_record(rec, str(tmp_path / "laya.json")))
    assert got == rec and got["trials"][1]["curve"][0]["loss"] is None
    assert pt.DIVERGED_LINE % "1" in pt.render_markdown([rec]).splitlines()
    assert "Diverged" not in pt.render_markdown([pt.build_record(_canonical())])


# ---- the tune/1 record ----

def _canonical() -> dict[str, Any]:
    """A synthetic canonical laya record of two driver starts: trial 0 with 3 epochs (best F1 1/2 first at epoch 2),
    trial 1 with 1 epoch and 2 attempts (F1 1/2 too: the tie goes to the lower index)."""
    h = "a" * 64
    return {"schema": pt.SCHEMA, "system": "laya",
            "search": {"seed": 0, "n_trials": 2, "search_alg": "BasicVariantGenerator", "max_concurrent": 1,
                       "points_to_evaluate": None, "space": copy.deepcopy(pt.SPACES["laya"]),
                       "scheduler": dict(pt.ASHA, max_t=8), "objective": pt.OBJECTIVE},
            "data": {"split": "selection_train", "sha256": h, "manifest_is_committed": True,
                     "train_folds": list(range(12)), "dev_folds": [12, 13], "train_uids_sha256": "b" * 64,
                     "dev_uids_sha256": "c" * 64, "n_train": 1244, "n_dev": 206, "dev_rows_with_train_text": 4},
            "fixed": copy.deepcopy(pt.FIXED["laya"]),
            "base": {"name": "multilingual", "model_sha256": "d" * 64, "config_sha256": "e" * 64},
            "device": "mps", "cpus": 4, "versions": {"ray": "2.59.0", "torch": "2.14.1", "laya": "0.4.6"},
            "segments": [{"git_sha": "f" * 40, "git_dirty": False, "git_diff_sha256": "0" * 64, "preflight": None},
                         {"git_sha": "f" * 40, "git_dirty": False, "git_diff_sha256": "0" * 64, "preflight": None}],
            "notes": list(pt.NOTES),
            "trials": [{"index": 0, "config": dict(LAYA_POINT), "attempts": 1,
                        "curve": [_result(1, 1, 5, 3), _result(2, 2, 2, 2), _result(3, 3, 3, 3)]},
                       {"index": 1, "config": dict(LAYA_POINT, neg=4, epochs=6), "attempts": 2,
                        "curve": [_result(1, 1, 1, 1)]}]}


def test_record_rebuilds_byte_for_byte(tmp_path: Path) -> None:
    canonical = _canonical()
    rec = pt.build_record(canonical)
    assert canonical == _canonical()                        # the input is never mutated
    assert pt.canonical_part(rec) == canonical
    path = pt.write_record(rec, str(tmp_path / "tune" / "laya.json"))
    got, sha = pt.read_record(path)
    raw = Path(path).read_bytes()
    assert got == rec and raw == pt.record_bytes(rec) and sha == hashlib.sha256(raw).hexdigest()
    pt.check_record(got)
    assert pt.record_bytes(pt.build_record(pt.canonical_part(got))) == raw
    t0, t1 = rec["trials"]
    assert [r["dev_micro_f1"] for r in t0["curve"]] == [0.2, 0.5, 0.5]
    assert list(t0["curve"][0])[:5] == ["epoch", "dev_tp", "dev_fp", "dev_fn", "dev_micro_f1"]
    assert (t0["best_epoch"], t0["best_dev_micro_f1"], t0["stop_epoch"], t0["asha_stopped"]) == (2, 0.5, 3, True)
    assert (t1["best_epoch"], t1["stop_epoch"], t1["asha_stopped"]) == (1, 1, True)
    assert rec["best"] == {"index": 0, "epoch": 2, "dev_micro_f1": 0.5, "config": LAYA_POINT}
    assert rec["n_asha_stopped"] == 2
    full = _canonical()                                     # a trial that ran its whole schedule is not ASHA-stopped
    full["trials"][0]["curve"].append(_result(4, 4, 0, 0))
    rec2 = pt.build_record(full)
    assert rec2["trials"][0]["asha_stopped"] is False and rec2["n_asha_stopped"] == 1
    assert rec2["best"] == {"index": 0, "epoch": 4, "dev_micro_f1": 1.0, "config": LAYA_POINT}
    later = _canonical()                                    # a strictly better later trial wins
    later["trials"][1]["curve"] = [_result(1, 5, 1, 1)]
    assert pt.build_record(later)["best"]["index"] == 1


def test_build_record_refuses_what_no_asha_search_produces() -> None:
    """A curve cut short of its schedule off an ASHA rung (Laya rungs 1 and 3), and more attempts than driver starts
    (Tune starts a trial at most once per start), are refused; a cut at a rung and attempts up to the starts pass."""
    off = _canonical()
    off["trials"][0]["curve"] = off["trials"][0]["curve"][:2]   # epoch 2 of 4: not a rung
    with pytest.raises(SystemExit, match=r"trials\[0\] stops at epoch 2, below its schedule 4 but not an ASHA rung"):
        pt.build_record(off)
    many = _canonical()
    many["trials"][1]["attempts"] = 3
    with pytest.raises(SystemExit, match=r"trials\[1\].attempts 3 exceeds the 2 driver starts"):
        pt.build_record(many)
    one = _canonical()
    one["segments"] = one["segments"][:1]                   # trial 1's 2 attempts need 2 starts
    with pytest.raises(SystemExit, match="exceeds the 1 driver starts"):
        pt.build_record(one)
    ok = _canonical()
    ok["trials"][0]["curve"] = ok["trials"][0]["curve"][:1]     # epoch 1: a rung
    assert pt.build_record(ok)["trials"][0]["asha_stopped"] is True


def _edit_best(rec: dict[str, Any]) -> None:
    rec["best"] = dict(rec["best"], index=1, config=rec["trials"][1]["config"])


def _edit_f1(rec: dict[str, Any]) -> None:
    rec["trials"][0]["curve"][1]["dev_micro_f1"] = 0.75


def _edit_long_curve(rec: dict[str, Any]) -> None:
    curve = rec["trials"][0]["curve"]
    for e in range(4, 6):
        curve.append(dict(curve[-1], epoch=e))


def _edit_stop(rec: dict[str, Any]) -> None:
    rec["trials"][1]["stop_epoch"] = 5


def _edit_extra(rec: dict[str, Any]) -> None:
    rec["extra"] = 1


def _edit_notes(rec: dict[str, Any]) -> None:
    rec["notes"] = rec["notes"][:-1]


def _edit_config(rec: dict[str, Any]) -> None:
    rec["trials"][0]["config"]["neg"] = 5


def _edit_index(rec: dict[str, Any]) -> None:
    rec["trials"][1]["index"] = 2


@pytest.mark.parametrize("edit", [_edit_best, _edit_f1, _edit_long_curve, _edit_stop, _edit_extra, _edit_notes,
                                  _edit_config, _edit_index],
                         ids=["best", "curve-f1", "long-curve", "stop-epoch", "extra-key", "notes", "config", "index"])
def test_check_record_refuses(edit: Any, tmp_path: Path) -> None:
    rec = pt.build_record(_canonical())
    edit(rec)
    with pytest.raises(SystemExit):
        pt.check_record(rec)
    path = tmp_path / "laya.json"
    path.write_bytes(pt.record_bytes(rec))
    with pytest.raises(SystemExit):
        pt.read_record(str(path))


def test_read_record_refuses_another_layout(tmp_path: Path) -> None:
    rec = pt.build_record(_canonical())
    path = tmp_path / "laya.json"
    path.write_bytes(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
    with pytest.raises(SystemExit, match="byte layout"):
        pt.read_record(str(path))
    path.write_bytes(pt.record_bytes(dict(rec, schema="tune/0")))
    with pytest.raises(SystemExit, match="schema 'tune/0'"):
        pt.read_record(str(path))
    with pytest.raises(SystemExit, match="not found"):
        pt.read_record(str(tmp_path / "missing.json"))


def test_import_budget() -> None:
    """pilot/tune.py stays torch-free: tune.py's driver half and the tests import it without a model stack."""
    banned = ("torch", "scipy", "laya", "ray", "transformers", "sklearn")
    code = "import sys, pilot.tune; print(sorted(m for m in %r if m in sys.modules))" % (banned,)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


# ---- the Tune objects tune.py builds ----

def test_tune_config_limits_concurrency_on_the_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tune ignores TuneConfig(max_concurrent_trials) for a SearchGenerator, so the one-at-a-time limit must live on
    the BasicVariantGenerator; ASHA gets its schedule from the constructor (metric and mode come from TuneConfig)."""
    import ray.tune.schedulers
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.basic_variant import BasicVariantGenerator

    import tune as tune_cli
    seen = []

    class Spy(ASHAScheduler):
        def __init__(self, **kw: Any) -> None:
            seen.append(kw)
            super().__init__(**kw)

    monkeypatch.setattr(ray.tune.schedulers, "ASHAScheduler", Spy)
    for system, trials, max_t in (("laya", 20, 8), ("berturk", 20, 30)):
        cfg = tune_cli.tune_config(system, 0, trials, None)
        assert cfg.metric == "dev_micro_f1" and cfg.mode == "max" and cfg.num_samples == trials
        assert cfg.max_concurrent_trials is None
        assert isinstance(cfg.search_alg, BasicVariantGenerator) and cfg.search_alg.max_concurrent == 1
        assert isinstance(cfg.scheduler, Spy)
        assert seen[-1] == {"time_attr": "epoch", "grace_period": 1, "reduction_factor": 3, "max_t": max_t}
        assert cfg.trial_dirname_creator is tune_cli.trial_dirname


def test_trial_dirname_names_the_creation_index() -> None:
    import types

    import tune as tune_cli
    assert tune_cli.trial_dirname(types.SimpleNamespace(trial_id="cb463_00007")) == "trial_00007"


def test_param_space_matches_spaces() -> None:
    import tune as tune_cli
    space = tune_cli.param_space("laya")
    assert list(space) == list(pt.SPACES["laya"])
    assert {space["neg"].sample() for _ in range(60)} <= {4, 8, 16}
    assert {space["epochs"].sample() for _ in range(60)} <= {4, 6, 8}
    for k in ("encoder_lr", "head_lr"):
        _kind, low, high = pt.SPACES["laya"][k]
        assert (space[k].lower, space[k].upper) == (low, high)
        assert all(low <= space[k].sample() <= high for _ in range(60))
    bert = tune_cli.param_space("berturk")
    assert list(bert) == list(pt.SPACES["berturk"])
    assert {bert["weight_decay"].sample() for _ in range(60)} <= {0.0, 0.01}
