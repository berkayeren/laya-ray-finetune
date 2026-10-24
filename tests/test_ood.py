"""The out-of-domain false-positive rate on HOTELS end to end on the synthetic snapshot, with no model, torch or Ray.

No scorer runs here: a hotels_probe dump (role ood_probe) is written by pilot.dumps.write_dump with a
score.sidecar_base sidecar of the synthetic manifest, next to a prior baseline's val dump (score.py --model prior) of
the same source. pilot.dumps.load_dump reads it with every check, gold_valid_tags names its valid columns (the
manifest's ood_scoring_tags; the 22 pins tags under the committed manifest), and report.py --confidence RUN_DIR
--split hotels_probe reports fp_rate on those columns at θ 0.5 and at the banking-val θ. report.run_report and
report.compare refuse the dump naming its role. score.py scores hotels_probe for --model laya or berturk only, a
fine-tuned checkpoint only through its protocol gate (tests/test_score_gate.py), and refuses it for the baselines and
evaluate.py (tests/test_score_report.py, tests/test_consumers.py).
"""
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import pilot.data
import report
import score
from conftest import prepared
from intents import INTENTS
from pilot import dumps, metrics, pins

pytestmark = pytest.mark.unit


@pytest.fixture
def snap(tmp_path: Path, synthetic_raw: Callable[..., tuple[str, dict[str, str]]]) -> tuple[str, str]:
    """(out_dir, manifest_path) of the synthetic snapshot prepared under tmp_path."""
    return prepared(tmp_path, synthetic_raw)


def _hotels(manifest: str) -> tuple[dict[str, Any], list[str]]:
    """(manifest entry, uids in file order) of the verified synthetic hotels_probe split."""
    path = os.path.join(os.path.dirname(manifest), "splits", "hotels_probe.jsonl")
    with open(path, "rb") as f:
        raw = f.read()
    entry = pilot.data.verify_split_file(path, manifest_path=manifest, roles=("ood_probe",), data=raw)
    return entry, [r["uid"] for r in pilot.data.parse_jsonl(raw)]


def write_hotels_run(root: Path, snap: tuple[str, str], seed: int = 3, val: bool = True) -> tuple[str, np.ndarray]:
    """A run dir holding the prior baseline's val dump (score.py) and a synthetic hotels_probe dump of the same source:
    the val sidecar's run_id, system, temperature, scorer, seed and training on score.sidecar_base of the hotels
    entry, logits from a seeded generator so some banking-only columns fire at θ 0.5. Returns (run_dir, logits)."""
    out_dir, manifest = snap
    score.main([os.path.join(out_dir, "splits", "val.jsonl"), "--model", "prior", "--manifest", manifest,
                "--out-root", str(root)])
    run_dir = os.path.join(str(root), "prior")
    _, _, vside, _ = dumps.load_dump(run_dir, "val", manifest)
    entry, uids = _hotels(manifest)
    side = score.sidecar_base(entry, "hotels_probe", manifest, None)
    side.update({k: vside[k] for k in ("run_id", "system", "temperature", "scorer", "seed", "training")})
    logits = np.random.default_rng(seed).normal(-2.5, 1.5, size=(len(uids), len(INTENTS))).astype(np.float32)
    dumps.write_dump(run_dir, "hotels_probe", logits, uids, side)
    if not val:
        for ext in ("npz", "json"):
            os.remove(os.path.join(run_dir, "val.%s" % ext))
    return run_dir, logits


@pytest.fixture
def run(tmp_path: Path, snap: tuple[str, str], capsys: pytest.CaptureFixture[str]) -> tuple[str, np.ndarray]:
    """(run_dir, hotels logits) of write_hotels_run under tmp_path/dumps."""
    d = write_hotels_run(tmp_path / "dumps", snap)
    capsys.readouterr()
    return d


def test_hotels_dump_loads_with_role_ood_probe(snap: tuple[str, str], run: tuple[str, np.ndarray]) -> None:
    run_dir, logits = run
    entry, uids = _hotels(snap[1])
    got, got_uids, side, got_entry = dumps.load_dump(run_dir, "hotels_probe", snap[1])
    assert got_entry["role"] == "ood_probe" and got_entry["gold_valid_tags"] == "ood_scoring_tags"
    assert got_uids == uids and all(u.startswith("hotel_") for u in uids) and (got == logits).all()
    assert side["split"]["role"] == "ood_probe" and side["n_subset"] is None
    Y = dumps.read_gold(got_entry, snap[1], got_uids)
    assert Y.shape == (len(uids), len(INTENTS)) and not Y.any()     # every hotels row is banking-negative
    assert dumps.EVAL_ROLES == ("val", "test") and dumps.READ_ROLES == ("val", "test", "ood_probe")


def test_hotels_dump_keeps_every_load_check(tmp_path: Path, snap: tuple[str, str], run: tuple[str, np.ndarray]) -> None:
    run_dir, logits = run
    _, uids, side, _ = dumps.load_dump(run_dir, "hotels_probe", snap[1])
    planted = {k: v for k, v in side.items() if k not in ("dump_sha256", "schema_version")}
    dumps.write_dump(run_dir, "hotels_probe", logits[::-1].copy(), uids[::-1], planted)   # reordered rows
    with pytest.raises(SystemExit, match="uids_sha256 mismatch"):
        dumps.load_dump(run_dir, "hotels_probe", snap[1])
    with open(os.path.join(run_dir, "hotels_probe.npz"), "ab") as f:
        f.write(b"x")
    with pytest.raises(SystemExit, match="npz sha256"):
        dumps.load_dump(run_dir, "hotels_probe", snap[1])


def test_ood_columns_are_the_manifest_banking_only_tags(snap: tuple[str, str]) -> None:
    manifest = snap[1]
    entry, _ = _hotels(manifest)
    with open(manifest, encoding="utf-8") as f:
        listed = json.load(f)["ood_scoring_tags"]
    cols = dumps.gold_valid_tags(entry, manifest)
    assert cols == [t for t in INTENTS if t in listed] and sorted(cols) == sorted(listed)
    assert "affirm" not in cols                                      # a synthetic HOTELS tag is never banking-only
    with open(manifest, encoding="utf-8") as f:
        val_entry = json.load(f)["splits"]["val"]
    assert dumps.gold_valid_tags(val_entry, manifest) == list(INTENTS)
    committed = dumps.gold_valid_tags({"gold_valid_tags": "ood_scoring_tags"}, dumps.DEFAULT_MANIFEST)
    assert len(committed) == 22 and sorted(committed) == sorted(pins.OOD_SCORING_TAGS)
    assert committed == [t for t in INTENTS if t in pins.OOD_SCORING_TAGS]


def _manifest_with(tmp_path: Path, snap: tuple[str, str], tags: Any) -> str:
    """A copy of the snapshot's manifest whose ood_scoring_tags is `tags` (any value, to test the refusals)."""
    with open(snap[1], encoding="utf-8") as f:
        m = json.load(f)
    m["ood_scoring_tags"] = tags
    path = tmp_path / "bad" / ("splits-%d.json" % len(os.listdir(tmp_path)))
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_gold_valid_tags_refuses_a_bad_list(tmp_path: Path, snap: tuple[str, str],
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    entry = {"gold_valid_tags": "ood_scoring_tags"}
    for tags in ([], ["card", "card"], ["card", "booking"], "card", None):
        with pytest.raises(SystemExit, match="non-empty list of distinct INTENTS tags"):
            dumps.gold_valid_tags(entry, _manifest_with(tmp_path, snap, tags))
    with pytest.raises(SystemExit, match="gold_valid_tags 'all' is unknown"):
        dumps.gold_valid_tags({"gold_valid_tags": "all"}, snap[1])
    monkeypatch.setattr(dumps, "manifest_is_committed", lambda path: True)    # the synthetic list is not the pins'
    with pytest.raises(SystemExit, match="drifted from the pins"):
        dumps.gold_valid_tags(entry, snap[1])
    assert len(dumps.gold_valid_tags(entry, dumps.DEFAULT_MANIFEST)) == 22


def test_ood_fp_report_uses_val_theta_on_banking_only_columns(snap: tuple[str, str],
                                                              run: tuple[str, np.ndarray]) -> None:
    run_dir, logits = run
    manifest = snap[1]
    entry, uids = _hotels(manifest)
    rep = report.confidence_report(run_dir, "hotels_probe", manifest)
    cols = dumps.gold_valid_tags(entry, manifest)
    val_logits, val_uids, val_side, val_entry = dumps.load_dump(run_dir, "val", manifest)
    Y_val = dumps.read_gold(val_entry, manifest, val_uids)
    idx = [dumps.TAGS.index(t) for t in cols]
    assert rep["schema"] == report.CONFIDENCE_SCHEMA and rep["split"]["name"] == "hotels_probe"
    assert rep["binding"]["kind"] == "val_selected" and rep["binding"]["theta_source"]["split"] == "val"
    assert rep["in_sample"] is False and rep["caveat"] is None and rep["n_messages"] == len(uids)
    assert rep["definitions"] == {"false_positive": metrics.FP_RULE, "wilson": report.WILSON_TEXT,
                                  "ood_columns": report.OOD_COLUMNS_TEXT}
    fired = 0
    for name, T_val in (("headline", val_side["temperature"]["shipped"]), ("raw", 1.0)):
        v = rep["variants"][name]
        assert sorted(v) == ["ood_fp", "temperature", "temperature_source", "theta"]   # no calibration, coverage
        assert v["theta"] == metrics.select_theta(Y_val, metrics.probabilities(val_logits, T_val))
        P = metrics.probabilities(logits, v["temperature"])
        for key, th in (("0.5", np.full(len(INTENTS), 0.5)), ("theta", metrics.theta_vector(v["theta"]))):
            f = v["ood_fp"][key]
            pred = P[:, idx] >= th[idx]                          # hand count on the same logits, banking-only columns
            assert f["cols"] == cols and f["n"] == len(uids)    # every hotels message: no gold among the columns
            assert f["k"] == int(pred.any(axis=1).sum()) and f["rate"] == f["k"] / f["n"]
            assert f["wilson95"] == metrics.wilson(f["k"], f["n"])
            assert f["per_tag_fp"] == {t: int(c) for t, c in zip(cols, pred.sum(axis=0)) if c}
            assert set(f["per_tag_fp"]) <= set(cols)
            fired += f["k"]
    assert fired > 0                                             # the seeded logits do fire some columns


def test_ood_report_refuses_without_a_val_theta_source(tmp_path: Path, snap: tuple[str, str],
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    run_dir, _ = write_hotels_run(tmp_path / "noval", snap, val=False)
    capsys.readouterr()
    with pytest.raises(SystemExit, match="no full val dump in .*θ is chosen on val only"):
        report.confidence_report(run_dir, "hotels_probe", snap[1])


def test_ood_report_refuses_a_val_dump_of_another_source(snap: tuple[str, str], run: tuple[str, np.ndarray]) -> None:
    run_dir, logits = run
    _, uids, side, _ = dumps.load_dump(run_dir, "hotels_probe", snap[1])
    planted = {k: v for k, v in side.items() if k not in ("dump_sha256", "schema_version")}
    planted["temperature"] = dict(side["temperature"], shipped=2.0)
    dumps.write_dump(run_dir, "hotels_probe", logits, uids, planted)
    with pytest.raises(SystemExit, match="differ in temperature.shipped"):
        report.confidence_report(run_dir, "hotels_probe", snap[1])


def test_run_report_refuses_hotels_probe_role(snap: tuple[str, str], run: tuple[str, np.ndarray]) -> None:
    with pytest.raises(SystemExit, match="split hotels_probe has role ood_probe"):
        report.run_report(run[0], "hotels_probe", snap[1])
    with pytest.raises(SystemExit, match="role ood_probe"):
        report.main([run[0], "--split", "hotels_probe", "--manifest", snap[1]])


def test_compare_refuses_hotels_probe_role(tmp_path: Path, snap: tuple[str, str], run: tuple[str, np.ndarray],
                                           capsys: pytest.CaptureFixture[str]) -> None:
    other, _ = write_hotels_run(tmp_path / "other", snap, seed=11)
    capsys.readouterr()
    with pytest.raises(SystemExit, match="role ood_probe; report.py --compare"):
        report.compare([run[0]], [other], "hotels_probe", snap[1], B=10)


def _sections(text: str) -> dict[str, str]:
    """{heading: section text} of a render_confidence output, split at its level-3 headings."""
    return {p.split("\n", 1)[0]: p for p in ("\n" + text).split("\n### ")[1:]}


def test_render_shows_ood_table_only_with_ood_records(snap: tuple[str, str], run: tuple[str, np.ndarray]) -> None:
    val = report.confidence_report(run[0], "val", snap[1])
    ood = report.confidence_report(run[0], "hotels_probe", snap[1])
    alone = report.render_confidence([val])
    assert "### Out-of-domain" not in alone
    both = report.render_confidence([val, ood])
    heading = "Out-of-domain false positives (HOTELS, %d banking-only tags)" % len(dumps.gold_valid_tags(
        _hotels(snap[1])[0], snap[1]))
    got, want = _sections(both), _sections(alone)
    assert heading in got and set(got) - set(want) == {heading}
    assert {k: v for k, v in got.items() if k != heading} == want   # every other section renders the val record only
    ood_rows = [l for l in got[heading].splitlines() if l.startswith("| prior")]
    assert len(ood_rows) == 4                                         # 2 variants x (θ 0.5, banking-val θ)
    f = ood["variants"]["headline"]["ood_fp"]["0.5"]
    assert "%d/%d" % (f["k"], f["n"]) in ood_rows[0] and "banking-val, global" in ood_rows[1]
    assert "- ood_columns: %s" % report.OOD_COLUMNS_TEXT in got[heading]
    only = report.render_confidence([ood])                            # an OOD record alone: just its section
    assert list(_sections(only)) == [heading]


def test_ood_out_writes_json_without_svg(tmp_path: Path, snap: tuple[str, str], run: tuple[str, np.ndarray],
                                         capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "conf" / "prior-hotels_probe.json"
    rep = report.main(["--confidence", run[0], "--split", "hotels_probe", "--manifest", snap[1], "--out", str(out)])
    capsys.readouterr()
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(rep, ensure_ascii=False))
    assert sorted(os.listdir(str(tmp_path / "conf"))) == ["prior-hotels_probe.json"]
    with pytest.raises(SystemExit, match="no reliability diagram"):
        report.render_svg(rep)
    shutil.rmtree(str(tmp_path / "conf"))
