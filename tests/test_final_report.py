"""The final scoring's per-seed reports and verdict, offline on synthetic final seeds: no model, torch or Ray.

A final seed (a dump whose training.postfit_sha256 is set) was trained on final_train, which holds the val folds, so
report.py never selects θ on its own val, and never on test. `report.py RUN_DIR --split
test --postfit RECORD` reports it at its postfit record's θ instead: theta.shipped for the shipped variant (at the
sidecar's shipped T, which must be the record's chosen T) and theta.raw for raw (T = 1), through
report.final_seed_binding, the one helper the --confidence final_seed binding uses too (record sha256 ==
training.postfit_sha256, the same system kind, refuse_held_out_record, T within T_MATCH), so the two report modes
refuse a wrong record with the same message. The val_selected rows name their source, theta_source {postfit, chosen,
split: val}; without --postfit a final seed's val_selected rows are null with FINAL_SEED_REASON; --postfit on any
other dump (a selection candidate, a baseline) is refused. The record θ was selected on one selection-run seed; that
is disclosed with the verdict, never re-selected.

The invariant (theta_source promoted): every val_selected row of every report mode names a θ source
on val: run_report's theta_source, each --confidence binding's theta_source, and each --compare side's val dumps
(dump_sha256.val; the compare output is left as it is, so the committed results/reports/compare-*.json still rebuild
byte for byte).

render_confidence takes svg_dir (default CONFIDENCE_SVG_DIR, results/confidence, so the committed val README block
renders byte for byte as before), and --confidence-table derives it from the inputs' one common directory
(confidence_svg_dir), so the README's test/HOTELS block renders from results/final/confidence with matching links.

The synthetic snapshot, the postfit record over a synthetic grid and the final-seed and split-dump builders are
tests/test_confidence.py's.

The verdict (the results table and the GO block) is proven end to end on a synthetic repo, independent of any real test
number (synthetic_final, one module fixture): a fitted Laya and BERTurk postfit record over synthetic grids, three final
seeds per kind whose test dumps carry protocol_sha256 == the synthetic protocol's own sha256 and training.postfit_sha256
== their record's, the label prior's and majority's val and test dumps with their test reports as the reference rows,
two latency-like reports and a protocol dict (pilot.protocol's blocks where they apply to any root, B 200 for speed)
that lists every input with its sha256. report.verdict_from checks every input's sha256, loads exactly the protocol's
final-seed test dumps (pin, seed, postfit binding, pairing, the protocol's test uids), seed-averages each model at its
record T and θ, recomputes the prior's frozen fractions and hands the inputs to pilot.verdict.build_verdict, whose
criteria and verdict the tests recompute from the record's own numbers. Every refusal is exercised on a copy of the
dumps or of the protocol dict, so the module fixture is never changed. The renders (render_final_table, render_go) and
the CLI modes (--verdict, --final-table, --go-block) are tested on the synthetic verdict and on build_verdict of
hand-made inputs (INCONCLUSIVE, PROVISIONAL, GO, equal seed values).
"""
import copy as copy_module
import json
import os
import shutil
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import pilot.data
import postfit as postfit_cli
import report
import score
from conftest import make_raw
from fewshot_synthetic import (decision_record, fewshot_block, plant_fewshot_dumps, plant_rules, plant_runs,
                               write_decision)
from pilot import dumps, metrics
from pilot import fewshot as fs
from pilot import fewshot_report
from pilot import protocol as pilot_protocol
from pilot import verdict
from pilot.data import repo_relpath
from test_berturk_postfit import EPOCHS as BERTURK_EPOCHS
from test_berturk_postfit import write_berturk_candidate
from test_confidence import (Fitted, Snap, _drop_val, _final_seed, _prior, _write_json, _write_split_dump,  # noqa: F401
                             fitted, snap)
from test_postfit import GRID, write_candidate

pytestmark = pytest.mark.unit


def _final(root: Path, snap: Snap, record: dict[str, Any], sha: str, **kw: Any) -> str:
    """A final seed's run dir holding only a test dump (the final seeds' dump dirs hold no val dump)."""
    return _drop_val(_write_split_dump(_final_seed(root, snap, record, sha, **kw), snap[1], "test"))


def _expected_row(run_dir: str, manifest: str, T: float, theta: dict[str, Any]) -> dict[str, Any]:
    """The metric row of run_dir's test dump at T and θ, as run_report's val_selected row of a final seed."""
    logits, uids, _, entry = dumps.load_dump(run_dir, "test", manifest)
    Y = dumps.read_gold(entry, manifest, uids)
    P = metrics.probabilities(logits, T)
    return dict(metrics.metrics(Y, P, metrics.theta_vector(theta)), theta=theta, in_sample=False)


# --- the final-seed θ binding in run_report ----------------------------------------------------------------------

def test_final_seed_report_uses_the_record_theta(tmp_path: Path, snap: Snap, fitted: Fitted) -> None:
    """The shipped row at the record's theta.shipped (sidecar T == record T), the raw row at theta.raw (T = 1), both
    held-out, and the source named: the record (path, sha256, kind), its chosen candidate, split val."""
    _, path, record, sha = fitted
    chosen = record["chosen"]
    final = _final(tmp_path / "final", snap, record, sha)
    rep = report.run_report(final, "test", snap[1], postfit=path)
    assert rep["theta_source"] == {"postfit": {"path": repo_relpath(path), "sha256": sha, "system_kind": "laya"},
                                   "chosen": chosen["dump_run_id"], "split": "val"}
    shipped, raw = rep["variants"]["shipped"], rep["variants"]["raw"]
    assert shipped["temperature"] == chosen["temperature"] and raw["temperature"] == 1.0
    assert shipped["rows"]["val_selected"] == _expected_row(final, snap[1], chosen["temperature"],
                                                            chosen["theta"]["shipped"])
    assert raw["rows"]["val_selected"] == _expected_row(final, snap[1], 1.0, chosen["theta"]["raw"])
    assert "val_selected_reason" not in shipped and "val_selected_reason" not in raw
    no_post = report.run_report(final, "test", snap[1])
    for name in ("shipped", "raw"):                         # the θ 0.5 rows do not depend on the binding
        assert rep["variants"][name]["rows"]["0.5"] == no_post["variants"][name]["rows"]["0.5"]


def test_final_seed_report_without_postfit_names_the_reason(tmp_path: Path, snap: Snap, fitted: Fitted) -> None:
    """θ is never self-selected for a final seed: without --postfit its val_selected rows are null, with the reason."""
    _, _, record, sha = fitted
    rep = report.run_report(_final(tmp_path / "final", snap, record, sha), "test", snap[1])
    assert rep["theta_source"] is None
    for v in rep["variants"].values():
        assert v["rows"]["val_selected"] is None and v["val_selected_reason"] == report.FINAL_SEED_REASON
    assert "--postfit RECORD" in report.FINAL_SEED_REASON


def _message(call: Callable[[], Any]) -> str:
    """The SystemExit message call() raises."""
    with pytest.raises(SystemExit) as e:
        call()
    return str(e.value)


def test_final_seed_postfit_refusals_match_confidence(tmp_path: Path, snap: Snap, fitted: Fitted) -> None:
    """A copy of the record (same content, other bytes), another kind's record and a shipped T off by 0.5 are refused
    by run_report with exactly the message --confidence gives: one binding, final_seed_binding."""
    _, path, record, sha = fitted
    final = _final(tmp_path / "final", snap, record, sha, epoch=3)
    copy = tmp_path / "copy.json"
    copy.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    other = _write_json(tmp_path / "other" / "berturk.json", dict(record, system_kind="berturk"))
    kinded = _final(tmp_path / "kinded", snap, record, pilot.data.sha256_file(other), epoch=3)
    off = _final(tmp_path / "off", snap, record, sha, shipped=record["chosen"]["temperature"] + 0.5, epoch=3)
    for run_dir, rec, want in ((final, str(copy), "final seed was bound to record"),
                               (kinded, other, "is a berturk record, the dump is laya"),
                               (off, path, "record's chosen T")):
        got = _message(lambda: report.run_report(run_dir, "test", snap[1], postfit=rec))
        assert want in got
        assert got == _message(lambda: report.confidence_report(run_dir, "test", snap[1], rec))


def test_postfit_refused_for_a_dump_that_is_not_a_final_seed(tmp_path: Path, snap: Snap, fitted: Fitted,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    """--postfit binds a final seed only: a selection candidate and a baseline take θ from their run dir's val dump;
    a final seed reported on val stays refused (its val is training data)."""
    run_dirs, path, record, sha = fitted
    base = _prior(snap, tmp_path / "base")
    capsys.readouterr()
    for d, split in ((run_dirs[0], "val"), (base, "test"), (base, "val")):
        with pytest.raises(SystemExit, match="--postfit applies to a final seed's dump"):
            report.run_report(d, split, snap[1], postfit=path)
    final = _final_seed(tmp_path / "final", snap, record, sha)
    for args in ({"postfit": path}, {}):
        with pytest.raises(SystemExit, match="trained on final_train"):
            report.run_report(final, "val", snap[1], **args)


def test_cli_run_dir_postfit_writes_the_run_report(tmp_path: Path, snap: Snap, fitted: Fitted,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """`report.py RUN_DIR --split test --postfit RECORD --out X.json` writes exactly run_report's dict; --postfit with
    --compare or --table is an argparse error."""
    _, path, record, sha = fitted
    final = _final(tmp_path / "final", snap, record, sha)
    out = tmp_path / "rep" / "final.json"
    report.main([final, "--split", "test", "--manifest", snap[1], "--postfit", path, "--out", str(out)])
    capsys.readouterr()
    want = report.run_report(final, "test", snap[1], postfit=path)
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(want, ensure_ascii=False))
    for argv in (["--compare", final, final, "--postfit", path], ["--table", str(out.parent), "--postfit", path],
                 ["--confidence-table", str(out.parent), "--postfit", path]):
        with pytest.raises(SystemExit) as e:
            report.main(argv)
        assert e.value.code == 2 and "--postfit applies only to RUN_DIR or --confidence" in capsys.readouterr().err


def _theta_sources(rep: dict[str, Any], manifest: str, dirs: tuple[str, ...] = ()) -> list[tuple[str, Any]]:
    """[(where, θ-source split)] of every val_selected row of one report: run_report's theta_source, a confidence
    record's binding.theta_source, each --compare side's val dumps (dump_sha256.val, read back by load_dump from the
    one of `dirs` named by the side's run id)."""
    if rep.get("kind") == "paired_bootstrap":
        out = []
        for side in ("a", "b"):
            for rid, sha in zip(rep[side]["run_ids"], rep[side]["dump_sha256"]["val"]):
                found = [d for d in dirs if os.path.basename(d) == rid]
                _, _, vside, _ = dumps.load_dump(found[0], "val", manifest)
                assert vside["dump_sha256"] == sha
                out.append(("compare %s %s" % (side, rid), vside["split"]["name"]))
        return out
    if rep.get("schema") == report.CONFIDENCE_SCHEMA:
        return [("confidence %s" % rep["binding"]["kind"], rep["binding"]["theta_source"]["split"])]
    rows = [v["rows"]["val_selected"] for v in rep["variants"].values()]
    if all(r is None for r in rows):
        return []
    return [("run_report %s" % rep["run_id"], (rep["theta_source"] or {}).get("split"))]


def test_every_val_selected_row_names_a_val_theta_source(tmp_path: Path, snap: Snap, fitted: Fitted,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    """The invariant of the promoted theta_source: in every report mode, every val_selected row names where θ came
    from, and that is always val (a baseline's own val dump, a selection candidate's record, a final seed's record)."""
    run_dirs, path, record, sha = fitted
    base = _prior(snap, tmp_path / "base")
    major = _prior(snap, tmp_path / "base", model="majority")
    capsys.readouterr()
    final = _drop_val(_write_split_dump(_write_split_dump(_final_seed(tmp_path / "final", snap, record, sha), snap[1],
                                                          "test"), snap[1], "hotels_probe"))
    reps = [report.run_report(base, "val", snap[1]), report.run_report(base, "test", snap[1]),
            report.run_report(run_dirs[0], "val", snap[1]), report.run_report(final, "test", snap[1], postfit=path),
            report.confidence_report(base, "val", snap[1]), report.confidence_report(base, "test", snap[1]),
            report.confidence_report(run_dirs[0], "val", snap[1], path),
            report.confidence_report(final, "test", snap[1], path),
            report.confidence_report(final, "hotels_probe", snap[1], path),
            report.compare([base], [major], "test", snap[1], B=10)]
    seen = [s for r in reps for s in _theta_sources(r, snap[1], (base, major))]
    assert len(seen) == 11
    assert all(split == "val" for _, split in seen), seen


# --- render_confidence's svg_dir (the README's test/HOTELS block renders from results/final/confidence) ------------

CONFIDENCE_DIR = os.path.join(pilot.data.REPO_ROOT, "results", "confidence")


def _committed_confidence() -> tuple[list[str], list[dict[str, Any]]]:
    """(paths, parsed records) of the committed val confidence records, in sorted path order."""
    paths = report.table_inputs([CONFIDENCE_DIR])
    assert paths, "no committed confidence record under results/confidence/"
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return paths, out


def test_render_confidence_svg_dir_default_is_the_val_block() -> None:
    """The default svg_dir is results/confidence, so the committed val README block renders as before."""
    _, records = _committed_confidence()
    assert report.CONFIDENCE_SVG_DIR == "results/confidence"
    assert report.render_confidence(records) == report.render_confidence(records, svg_dir="results/confidence")


def test_render_confidence_svg_dir_changes_only_the_links() -> None:
    """Another svg_dir moves every diagram link there and changes nothing else."""
    _, records = _committed_confidence()
    base = report.render_confidence(records).split("\n")
    moved = report.render_confidence(records, svg_dir="results/final/confidence").split("\n")
    assert len(base) == len(moved)
    links = ["![reliability diagram %s (%s)](results/final/confidence/%s-%s.svg)"
             % (r["run_id"], r["split"]["name"], r["run_id"], r["split"]["name"]) for r in records]
    changed = [(a, b) for a, b in zip(base, moved) if a != b]
    assert [b for _, b in changed] == links
    assert [a for a, _ in changed] == [x.replace("results/final/confidence/", "results/confidence/") for x in links]


def test_confidence_table_svg_dir_keeps_the_committed_block(capsys: pytest.CaptureFixture[str]) -> None:
    """`report.py --confidence-table results/confidence` derives svg_dir from the inputs' directory: the committed
    records give exactly render_confidence's default text (the README val confidence block)."""
    paths, records = _committed_confidence()
    assert report.confidence_svg_dir(paths) == report.CONFIDENCE_SVG_DIR
    text = report.main(["--confidence-table", CONFIDENCE_DIR])
    assert capsys.readouterr().out == text == report.render_confidence(records)


def test_confidence_svg_dir_follows_one_common_directory_under_root(tmp_path: Path) -> None:
    """One common directory under root gives its forward-slash relative path; two directories, root itself or a
    record outside root give CONFIDENCE_SVG_DIR (never an absolute or outside-repo link)."""
    root = tmp_path / "repo"
    for rel in ("a/b/x-test.json", "a/b/y-test.json", "a/c/z-test.json", "top-test.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}", encoding="utf-8")
    outside = tmp_path / "elsewhere" / "w-test.json"
    outside.parent.mkdir()
    outside.write_text("{}", encoding="utf-8")
    one = [str(root / "a" / "b" / "x-test.json"), str(root / "a" / "b" / "y-test.json")]
    assert report.confidence_svg_dir(one, root=str(root)) == "a/b"
    assert report.confidence_svg_dir(one + [str(root / "a" / "c" / "z-test.json")], root=str(root)) \
        == report.CONFIDENCE_SVG_DIR
    assert report.confidence_svg_dir([str(root / "top-test.json")], root=str(root)) == report.CONFIDENCE_SVG_DIR
    assert report.confidence_svg_dir([str(outside)], root=str(root)) == report.CONFIDENCE_SVG_DIR
    assert report.confidence_svg_dir([], root=str(root)) == report.CONFIDENCE_SVG_DIR


# --- the verdict: a synthetic protocol, three Laya and three BERTurk final seeds ------------------------------------

KINDS = ("laya", "berturk")
SEEDS = (0, 1, 2)
SEVEN = ("micro_f1", "macro_f1", "exact_match", "top1_hit", "brier", "classwise_ece", "precision_at_80")
LATENCY = {"laya": {"system": "laya", "device": "mps", "pooled": {"n": 21, "p50": 0.33, "p95": 0.41}},
           "berturk": {"system": "berturk", "device": "mps", "pooled": {"n": 21, "p50": 0.012, "p95": 0.015}}}
# the synthetic few-shot block: tag card, k 1 and 2, two draws on base seeds 0 and 1, primary metric at k = 2,
# which the synthetic criteria (d) fewshot path names too (the real protocol has k = 10 in both places)
FEWSHOT_K = 2
FEWSHOT_RULES = dict(plant_rules((0, 1)), primary_metric={"k": FEWSHOT_K, "rule": "the synthetic primary metric at "
                                                                                 "k = %d" % FEWSHOT_K})


def _rewrite(run_dir: str, manifest: str, split: str, reverse: bool = False, **over: Any) -> str:
    """Rewrite a dump of run_dir with sidecar keys replaced by `over` (a planted case); reverse=True writes the rows in
    reverse order as a --n subset of the full count (the only way load_dump admits another order)."""
    logits, uids, side, _ = dumps.load_dump(run_dir, split, manifest)
    side = {k: v for k, v in side.items() if k not in ("dump_sha256", "schema_version")}
    side.update(over)
    if reverse:
        logits, uids, side["n_subset"] = logits[::-1].copy(), uids[::-1], len(uids)
    dumps.write_dump(run_dir, split, logits, uids, side)
    return run_dir


def _gated_test_dump(run_dir: str, manifest: str, pin: str, rng_seed: int) -> str:
    """The test dump score.py's protocol gate writes for a final seed: the val sidecar's run_id, system,
    temperature, scorer, seed and training, protocol_sha256 == pin, seeded float32 logits."""
    _, _, vside, _ = dumps.load_dump(run_dir, "val", manifest)
    path = os.path.join(os.path.dirname(manifest), "splits", "test.jsonl")
    with open(path, "rb") as f:
        raw = f.read()
    entry = pilot.data.verify_split_file(path, manifest_path=manifest, roles=dumps.READ_ROLES, data=raw)
    uids = [r["uid"] for r in pilot.data.parse_jsonl(raw)]
    side = score.sidecar_base(entry, "test", manifest, None)
    side.update({k: vside[k] for k in ("run_id", "system", "temperature", "scorer", "seed", "training")})
    side["protocol_sha256"] = pin
    logits = np.random.default_rng(rng_seed).normal(0.0, 3.0, size=(len(uids), 48)).astype(np.float32)
    dumps.write_dump(run_dir, "test", logits, uids, side)
    return run_dir


def _final_seed_dir(dump_root: Path, snap: Snap, kind: str, record: dict[str, Any], rsha: str, seed: int) -> str:
    """A final seed's val dump (dropped once its gated test dump exists): calibrated, at the record's T, trained on
    final_train, training.postfit_sha256 == rsha. BERTurk's builder has no postfit argument, so its sidecar is
    rewritten with the binding."""
    c = record["chosen"]
    if kind == "laya":
        return write_candidate(dump_root, snap, c["encoder_lr"], c["epoch"], seed=seed, split="final_train",
                               calibrated=True, shipped=c["temperature"], postfit_sha256=rsha)
    d = write_berturk_candidate(dump_root, snap, c["epoch"], seed=seed, split="final_train", calibrated=True,
                                shipped=c["temperature"])
    _, _, side, _ = dumps.load_dump(d, "val", snap[1])
    return _rewrite(d, snap[1], "val", training=dict(side["training"], postfit_sha256=rsha))


@pytest.fixture(scope="module")
def synthetic_final(tmp_path_factory: pytest.TempPathFactory) -> types.SimpleNamespace:
    """A synthetic repo (root): the snapshot under data/, a fitted laya and berturk postfit record and their final
    manifests under results/postfit/, three gated final-seed test dumps per kind, the prior's and majority's dumps and
    test reports, two latency reports, and the protocol dict listing every input with its sha256, written to
    root/protocol.json; sha is the sha256 of its bytes and the pin every final-seed test dump carries."""
    tmp = tmp_path_factory.mktemp("verdict")
    root = tmp / "repo"
    raw_root, raw_sha = make_raw(str(tmp / "raw"))
    out_dir = str(root / "data")
    pilot.data.prepare(raw_root, out_dir, raw_sha)
    manifest = os.path.join(out_dir, "splits.json")
    vpath = os.path.join(out_dir, "splits", "val.jsonl")
    with open(vpath, "rb") as f:
        raw = f.read()
    entry = pilot.data.verify_split_file(vpath, manifest_path=manifest, roles=("val",), data=raw)
    snap = (out_dir, manifest, entry, [r["uid"] for r in pilot.data.parse_jsonl(raw)])
    dump_root = root / "results" / "dumps"
    grids = {"laya": [write_candidate(dump_root, snap, lr, e) for lr, e in GRID],
             "berturk": [write_berturk_candidate(dump_root, snap, e) for e in range(1, BERTURK_EPOCHS + 1)]}
    records, finals = {}, {}
    for kind in KINDS:
        out = root / "results" / "postfit" / ("%s.json" % kind)
        postfit_cli.main(grids[kind] + ["--manifest", manifest, "--out", str(out)])
        record, rsha = json.loads(out.read_text(encoding="utf-8")), pilot.data.sha256_file(str(out))
        records[kind] = (record, rsha, str(out))
        finals[kind] = [_final_seed_dir(dump_root, snap, kind, record, rsha, s) for s in SEEDS]
        sides = [dumps.load_dump(d, "val", manifest)[2] for d in finals[kind]]
        c = record["chosen"]
        _write_json(root / "results" / "postfit" / ("%s-final.json" % kind), {
            "schema": "postfit-final/1", "system_kind": kind,
            "postfit": {"path": "results/postfit/%s.json" % kind, "sha256": rsha}, "final_split": "final_train",
            "chosen": {"train_run_id": c["train_run_id"], "encoder_lr": c["encoder_lr"], "epoch": c["epoch"],
                       "epochs_schedule": c["epochs_schedule"], "stop_after_epoch": c["epoch"],
                       "temperature": c["temperature"]},
            "config_sha256_sans_seed": sides[0]["training"]["config_sha256_sans_seed"],
            "seeds": [{"seed": s, "train_run_id": side["training"]["run_id"],
                       "checkpoint_name": "ckpt-%s-final-s%d" % (kind, s), "dump_run_id": side["run_id"],
                       "temperature": side["temperature"]["shipped"], "loss_by_epoch": side["training"]["loss_by_epoch"]}
                      for s, side in zip(SEEDS, sides)]})
    refs = {}
    for model in ("majority", "prior"):
        d = _prior(snap, dump_root, model=model)
        _write_json(root / "results" / "reports" / ("%s.json" % model), report.run_report(d, "test", manifest))
        refs[model] = {"run_id": model, "report": "results/reports/%s.json" % model, "gated": False,
                       "hotels_probe": None}
    for kind in KINDS:
        _write_json(root / "results" / "latency" / ("%s-mps.json" % kind), dict(LATENCY[kind], schema="latency/1"))
    rels = (["data/splits.json"] + ["results/postfit/%s.json" % k for k in KINDS]
            + ["results/postfit/%s-final.json" % k for k in KINDS]
            + ["results/dumps/prior/val.json", "results/dumps/prior/test.json"]
            + ["results/reports/%s.json" % m for m in refs] + ["results/latency/%s-mps.json" % k for k in KINDS]
            + ["results/dumps/%s/val.json" % records[k][0]["chosen"]["dump_run_id"] for k in KINDS])
    with open(manifest, encoding="utf-8") as f:
        msplits = json.load(f)["splits"]
    test = msplits["test"]
    systems = pilot_protocol.systems_block(str(root))
    criteria = copy_module.deepcopy(verdict.DEFAULT_CRITERIA)
    for p in criteria["d"]["paths"]:
        if p["path"] == "fewshot":
            p["k"] = FEWSHOT_K                              # criteria (d)'s fewshot k is the few-shot block's primary k
    protocol = {
        "schema": pilot_protocol.SCHEMA, "title": "synthetic", "units": pilot_protocol.UNITS,
        "frozen": dict(pilot_protocol.FROZEN),
        "inputs": {rel: pilot.data.sha256_file(str(root / rel)) for rel in rels},
        "splits": {name: dict({"path": "data/" + msplits[name]["path"]},
                              **{k: msplits[name][k] for k in pilot_protocol.SPLIT_KEYS}) for name in ("test", "val")},
        "systems": systems,
        "references": {"rule": pilot_protocol.REFERENCES_RULE, "rows": refs},
        "scoring": copy_module.deepcopy(pilot_protocol.SCORING),
        "variant": dict(pilot_protocol.VARIANT), "seed_averaging": dict(pilot_protocol.SEED_AVERAGING),
        "bootstrap": dict(pilot_protocol.bootstrap_block(), B=200),
        "criteria": criteria,
        "collapse": pilot_protocol.collapse_block(manifest, str(root)),
        "verdict": pilot_protocol.verdict_block(), "definitions": pilot_protocol.definitions_block(),
        "fewshot": fewshot_block(manifest, draw_seeds=(0, 1), rules=FEWSHOT_RULES),
        "known_at_freeze": {"note": "synthetic"},
        "power_context": copy_module.deepcopy(pilot_protocol.POWER_CONTEXT),
        "assumptions": copy_module.deepcopy(pilot_protocol.ASSUMPTIONS),
        "delegation": copy_module.deepcopy(pilot_protocol.DELEGATION),
        "code": {"git_head": "0" * 40, "dirty": False}}
    path = str(root / "protocol.json")
    sha = pilot_protocol.write_protocol(json.loads(json.dumps(protocol)), path)
    for i, kind in enumerate(KINDS):                        # scored once, through the gate, after the freeze
        for s, d in zip(SEEDS, finals[kind]):
            _drop_val(_gated_test_dump(d, manifest, sha, 100 + 10 * i + s))
    protocol, _ = pilot_protocol.require_frozen(path, sha)
    return types.SimpleNamespace(root=str(root), manifest=manifest, dump_root=str(dump_root), protocol=protocol,
                                 sha=sha, path=path, records=records, finals=finals, snap=snap)


def _verdict(sf: types.SimpleNamespace, protocol: dict[str, Any] | None = None, sha: str | None = None,
             dump_root: str | None = None) -> dict[str, Any]:
    """report.verdict_from of the synthetic repo, with the protocol, pin or dump root replaced when given."""
    return report.verdict_from(sf.protocol if protocol is None else protocol, sf.sha if sha is None else sha,
                               sf.manifest, sf.root, sf.dump_root if dump_root is None else dump_root)


@pytest.fixture(scope="module")
def synthetic_verdict(synthetic_final: types.SimpleNamespace) -> dict[str, Any]:
    """The verdict of the synthetic repo without the few-shot record (the few-shot path PENDING)."""
    return _verdict(synthetic_final)


def test_verdict_end_to_end_on_synthetic_final_seeds(synthetic_final: types.SimpleNamespace) -> None:
    """verdict/1 from protocol-bound final-seed dumps: its criteria are pilot.verdict's functions of its own deltas,
    collapse rows and latency rows under the protocol's criteria block, its verdict is decide of those statuses, the
    per-seed rows are run_report's at the record θ, and the settings are the records'."""
    sf = synthetic_final
    v = _verdict(sf)
    crit = sf.protocol["criteria"]
    assert v["schema"] == verdict.VERDICT_SCHEMA == "verdict/1"
    assert list(v) == ["schema", "protocol", "split", "systems", "references", "seed_averaged", "collapse", "latency",
                       "fewshot", "criteria", "verdict"]
    assert v["protocol"] == {"path": "protocol.json", "sha256": sf.sha}
    test = sf.protocol["splits"]["test"]
    assert {k: v["split"][k] for k in ("name", "sha256", "uids_sha256", "n")} == {
        "name": "test", "sha256": test["sha256"], "uids_sha256": test["uids_sha256"], "n": test["n_messages"]}
    assert v["split"]["n_no_intent"] == test["n_no_intent"]
    h = v["seed_averaged"]["headline"]["metrics"]
    a = verdict.criterion_a(h["micro_f1"], h["macro_f1"], crit["a"])
    alt = verdict.alternative_go(h["micro_f1"], crit["alternative_go"])
    b = verdict.criterion_b(v["collapse"]["rows"], {k: [r["seed"] for r in v["systems"][k]["per_seed"]] for k in KINDS},
                            crit["b"])
    c = verdict.criterion_c(v["latency"], crit["c"])
    d = verdict.criterion_d(h["brier"], h["precision_at_80"], None, crit["d"])
    for key, want in (("a", a), ("b", b), ("c", c), ("d", d), ("alternative_go", alt)):
        got = v["criteria"][key]
        assert {k: x for k, x in got.items() if k != "name"} == json.loads(json.dumps(want)), key
        assert got["name"] == crit[key]["name"]
    assert v["verdict"] == verdict.decide(a["status"], b["status"], d["status"], alt["status"])
    assert v["criteria"]["c"]["status"] == verdict.REPORTED and d["paths"][2]["status"] == verdict.PENDING
    assert v["latency"] == [{"system": k, "device": "mps", "n": LATENCY[k]["pooled"]["n"],
                             "p50": LATENCY[k]["pooled"]["p50"], "p95": LATENCY[k]["pooled"]["p95"]} for k in KINDS]
    assert [r["name"] for r in v["references"]] == ["majority", "prior"]
    for kind in KINDS:
        record, rsha, rpath = sf.records[kind]
        s = v["systems"][kind]
        assert s["postfit"] == {"path": "results/postfit/%s.json" % kind, "sha256": rsha}
        assert s["settings"]["temperature"] == record["chosen"]["temperature"]
        assert s["settings"]["theta"]["shipped"] == {"global": record["chosen"]["theta"]["shipped"]["global"],
                                                     "n_per_tag": len(record["chosen"]["theta"]["shipped"]["per_tag"])}
        assert [r["seed"] for r in s["per_seed"]] == list(SEEDS) == s["seeds"]
        for row, d_ in zip(s["per_seed"], sf.finals[kind]):
            sel = report.run_report(d_, "test", sf.manifest, postfit=rpath)["variants"]["shipped"]["rows"]
            for m in verdict.TABLE_METRICS:
                assert row[m] == sel["val_selected"][m]
            assert row["dump_run_id"] == os.path.basename(d_)
        for m in verdict.TABLE_METRICS:
            mean, std = verdict.mean_std([r[m] for r in s["per_seed"]])
            assert s["summary"][m] == {"mean": mean, "std": std, "values": [r[m] for r in s["per_seed"]]}


def test_verdict_seed_averages_at_the_record_t_and_theta(synthetic_final: types.SimpleNamespace,
                                                         synthetic_verdict: dict[str, Any]) -> None:
    """The seed-averaged systems: per seed sigmoid(logit / T) at the record T, averaged, thresholded at the
    record's theta.shipped (raw: T 1, theta.raw), compared by system_deltas with the protocol's bootstrap block; the
    seven metrics of system_deltas, Laya - BERTurk."""
    sf, v = synthetic_final, synthetic_verdict
    P, th = {}, {}
    for variant, key in (("headline", "shipped"), ("raw", "raw")):
        for kind in KINDS:
            record = sf.records[kind][0]
            T = record["chosen"]["temperature"] if variant == "headline" else 1.0
            loaded = [dumps.load_dump(d, "test", sf.manifest) for d in sf.finals[kind]]
            P[kind] = metrics.seed_average([metrics.probabilities(x[0], T) for x in loaded])
            th[kind] = metrics.theta_vector(record["chosen"]["theta"][key])
        Y = dumps.read_gold(loaded[0][3], sf.manifest, loaded[0][1])
        want = verdict.system_deltas(Y, P["laya"], th["laya"], P["berturk"], th["berturk"], sf.protocol["bootstrap"])
        got = v["seed_averaged"][variant]
        assert tuple(got["metrics"]) == SEVEN
        assert {k: got[k] for k in want} == json.loads(json.dumps(want))
        assert got["a"] == "laya" and got["b"] == "berturk" and got["n"] == Y.shape[0] and got["B"] == 200


def test_verdict_collapse_rows_in_order(synthetic_final: types.SimpleNamespace,
                                        synthetic_verdict: dict[str, Any]) -> None:
    """Collapse rows: selection laya, selection berturk, then final laya s0..s2 and final berturk s0..s2; the
    selection runs by the val floor and the loss, the final seeds by the loss and the test floor, against the
    protocol's frozen prior fractions."""
    sf, v = synthetic_final, synthetic_verdict
    rows = v["collapse"]["rows"]
    assert [(r["run"], r["kind"], r["seed"]) for r in rows] == (
        [("selection", k, 0) for k in KINDS] + [("final", k, s) for k in KINDS for s in SEEDS])
    col = sf.protocol["collapse"]
    assert {k: v["collapse"][k] for k in ("prior_val_fraction", "prior_test_fraction", "prior_theta_global")} == {
        k: col[k] for k in ("prior_val_fraction", "prior_test_fraction", "prior_theta_global")}
    for r in rows:
        names = [c["component"] for c in r["components"]]
        assert names == (["val_floor", "loss"] if r["run"] == "selection" else ["loss", "test_floor"])
        assert r["collapsed"] == any(c["collapsed"] for c in r["components"])
    sel = [r for r in rows if r["run"] == "selection"]
    for r, kind in zip(sel, KINDS):
        record = sf.records[kind][0]
        cand = [c for c in record["candidates"] if c["dump_run_id"] == record["chosen"]["dump_run_id"]][0]
        o = cand["objective"]
        assert r["components"][0] == verdict.collapse_val(verdict.micro_fraction(o["tp"], o["fp"], o["fn"]),
                                                          col["prior_val_fraction"])
        assert r["components"][1] == verdict.collapse_loss(cand["loss_by_epoch"])
    for r in rows[2:]:
        seed_row = v["systems"][r["kind"]]["per_seed"][r["seed"]]
        n = seed_row["test_counts"]
        assert r["components"][1] == verdict.collapse_test_floor(verdict.micro_fraction(n["tp"], n["fp"], n["fn"]),
                                                                 col["prior_test_fraction"])


def test_verdict_is_deterministic(synthetic_final: types.SimpleNamespace, synthetic_verdict: dict[str, Any]) -> None:
    """No timestamp, no rounding: the same inputs give identical bytes."""
    again = _verdict(synthetic_final)
    assert json.dumps(again, ensure_ascii=False, indent=2) == json.dumps(synthetic_verdict, ensure_ascii=False,
                                                                         indent=2)


def test_verdict_refuses_a_changed_protocol_input(synthetic_final: types.SimpleNamespace) -> None:
    """An input whose bytes changed after the freeze is refused, named, before anything is computed."""
    sf = synthetic_final
    path = os.path.join(sf.root, "results", "latency", "laya-mps.json")
    with open(path, "rb") as f:
        raw = f.read()
    try:
        with open(path, "ab") as f:
            f.write(b" ")
        with pytest.raises(SystemExit, match="protocol input results/latency/laya-mps.json changed"):
            _verdict(sf)
    finally:
        with open(path, "wb") as f:
            f.write(raw)


def _dump_copy(sf: types.SimpleNamespace, tmp_path: Path) -> str:
    """A copy of the synthetic dump root under tmp_path (the module fixture's dumps are never changed)."""
    out = str(tmp_path / "dumps")
    shutil.copytree(sf.dump_root, out)
    return out


def test_verdict_refuses_a_missing_final_seed_test_dump(synthetic_final: types.SimpleNamespace, tmp_path: Path) -> None:
    sf = synthetic_final
    root = _dump_copy(sf, tmp_path)
    rid = os.path.basename(sf.finals["berturk"][1])
    os.remove(os.path.join(root, rid, "test.npz"))
    with pytest.raises(SystemExit, match="berturk final seed 1 .*no test dump"):
        _verdict(sf, dump_root=root)


def test_verdict_refuses_a_dump_bound_to_another_protocol(synthetic_final: types.SimpleNamespace) -> None:
    """A sidecar protocol_sha256 other than the frozen protocol's sha256 is refused."""
    with pytest.raises(SystemExit, match="protocol_sha256"):
        _verdict(synthetic_final, sha="0" * 64)


@pytest.mark.parametrize("plant, want", [
    (lambda side: {"seed": 7}, "seed 7"),
    (lambda side: {"training": dict(side["training"], postfit_sha256="9" * 64)}, "postfit_sha256 " + "9" * 64),
])
def test_verdict_refuses_a_seed_or_postfit_binding_mismatch(synthetic_final: types.SimpleNamespace, tmp_path: Path,
                                                            plant: Callable[[dict[str, Any]], dict[str, Any]],
                                                            want: str) -> None:
    sf = synthetic_final
    root = _dump_copy(sf, tmp_path)
    d = os.path.join(root, os.path.basename(sf.finals["laya"][2]))
    side = dumps.load_dump(d, "test", sf.manifest)[2]
    _rewrite(d, sf.manifest, "test", **plant(side))
    with pytest.raises(SystemExit, match=want):
        _verdict(sf, dump_root=root)


def test_verdict_refuses_an_unpaired_uid_order(synthetic_final: types.SimpleNamespace, tmp_path: Path) -> None:
    """Every final-seed test dump must score the protocol's test uids in one order (the bootstrap's pairing)."""
    sf = synthetic_final
    root = _dump_copy(sf, tmp_path)
    _rewrite(os.path.join(root, os.path.basename(sf.finals["berturk"][0])), sf.manifest, "test", reverse=True)
    with pytest.raises(SystemExit, match="unpaired"):
        _verdict(sf, dump_root=root)


@pytest.mark.parametrize("key, value", [("prior_val_fraction", "1/3"), ("prior_test_fraction", "1/3"),
                                        ("prior_theta_global", 0.37)])
def test_verdict_refuses_a_prior_fraction_off_the_frozen_one(synthetic_final: types.SimpleNamespace, key: str,
                                                             value: Any) -> None:
    """The prior's floors are recomputed from its dumps (pilot.protocol.prior_floors) and must equal the frozen ones."""
    protocol = copy_module.deepcopy(synthetic_final.protocol)
    protocol["collapse"][key] = value
    with pytest.raises(SystemExit, match=key):
        _verdict(synthetic_final, protocol=protocol)


@pytest.mark.parametrize("where, key", [("definitions", "brier"), ("definitions", "bootstrap_method"),
                                        ("bootstrap", "procedure"), ("bootstrap", "extra")])
def test_verdict_refuses_code_texts_off_the_frozen_ones(synthetic_final: types.SimpleNamespace,
                                                        monkeypatch: pytest.MonkeyPatch, where: str, key: str) -> None:
    """The verdict is computed by the frozen code only. A metric or bootstrap text of the live code
    that differs from the protocol's (here: the live block changed after the freeze) is refused naming the key, before
    any input is read; the bootstrap parameters (B, seed, level, coverage) are the protocol's to set."""
    fn = "definitions_block" if where == "definitions" else "bootstrap_block"
    live = getattr(pilot_protocol, fn)
    monkeypatch.setattr(pilot_protocol, fn, lambda: dict(live(), **{key: "changed after the freeze"}))
    with pytest.raises(SystemExit) as e:
        _verdict(synthetic_final)
    text = str(e.value.code)
    assert "\n" not in text and where in text and key in text and "frozen code" in text, text
    monkeypatch.setattr(pilot_protocol, fn, live)
    assert synthetic_final.protocol["bootstrap"]["B"] != pilot_protocol.bootstrap_block()["B"]   # parameters differ
    report.frozen_code(synthetic_final.protocol)                                                # and still pass


def test_verdict_refuses_a_system_with_fewer_final_seeds_than_listed(synthetic_final: types.SimpleNamespace) -> None:
    protocol = copy_module.deepcopy(synthetic_final.protocol)
    protocol["systems"]["laya"]["final_seeds"].pop()
    with pytest.raises(SystemExit, match="laya: .*final seeds \\[0, 1\\].*\\[0, 1, 2\\]"):
        _verdict(synthetic_final, protocol=protocol)


# --- the verdict with the few-shot record: criterion (d)'s few-shot path at its frozen threshold ---------------------

def _separating(good: str) -> Callable[[dict[str, Any], str, np.ndarray, np.random.Generator], np.ndarray]:
    """A logits callback: `good` kind ranks every positive above every negative, the other kind inverts the ranking."""
    def make(e: dict[str, Any], split: str, Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sign = 1.0 if e["kind"] == good else -1.0
        return np.where(Y, 4.0, -4.0) * sign + rng.normal(0.0, 0.1, size=Y.shape)
    return make


@pytest.fixture(scope="module")
def synthetic_fewshot(synthetic_final: types.SimpleNamespace) -> types.SimpleNamespace:
    """The few-shot results of the synthetic protocol, planted after its freeze under its root: the compute decision
    record (primary mode), the complete runs manifest, every dump it lists in the verdict's dump root (Laya ranks the
    new tag perfectly, BERTurk inverts it, so the few-shot path is MET) and the record
    pilot.fewshot_report.build_record writes, with repo-relative paths under the synthetic root. fewshot.DECISION_PATH
    must point at `decision` while a test reads it (monkeypatch)."""
    sf = synthetic_final
    ns = types.SimpleNamespace(manifest=sf.manifest, pin=sf.sha, block=sf.protocol["fewshot"], protocol=sf.protocol)
    base = os.path.join(sf.root, "results", "fewshot")
    decision = write_decision(os.path.join(base, "projection.json"), decision_record(sf.sha))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fs, "DECISION_PATH", decision)
        with pytest.MonkeyPatch.context() as rooted:        # repo-relative paths under the synthetic root
            rooted.setattr(pilot.data, "REPO_ROOT", sf.root)
            runs_path = plant_runs(ns, os.path.join(base, "runs.json"))
        runs, runs_sha = fs.read_runs(sf.protocol, sf.sha, runs_path)
        plant_fewshot_dumps(ns, runs, runs_sha, sf.dump_root, logits=_separating("laya"))
        with pytest.MonkeyPatch.context() as rooted:
            rooted.setattr(pilot.data, "REPO_ROOT", sf.root)
            rec = fewshot_report.build_record(sf.protocol, sf.sha, runs_path, sf.dump_root, sf.manifest)
    path = fewshot_report.write_record(rec, os.path.join(base, "fewshot.json"))
    return types.SimpleNamespace(path=path, record=rec, sha=pilot.data.sha256_file(path), runs_path=runs_path,
                                 runs_sha=runs_sha, decision=decision)


def _fewshot_verdict(sf: types.SimpleNamespace, fw: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
                     path: str | None = None, protocol: dict[str, Any] | None = None,
                     dump_root: str | None = None) -> dict[str, Any]:
    """report.verdict_from of the synthetic repo with the few-shot record (`path`, default the planted one)."""
    monkeypatch.setattr(fs, "DECISION_PATH", fw.decision)
    return report.verdict_from(sf.protocol if protocol is None else protocol, sf.sha, sf.manifest, sf.root,
                               sf.dump_root if dump_root is None else dump_root,
                               fewshot=fw.path if path is None else path)


def test_verdict_with_the_fewshot_record(synthetic_final: types.SimpleNamespace,
                                         synthetic_fewshot: types.SimpleNamespace, synthetic_verdict: dict[str, Any],
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """verdict_from(..., fewshot=RECORD) recomputes the record's primary block from the dumps and gives criterion
    (d)'s few-shot path its status (MET: Laya - BERTurk mean macro-AP at k = 2 is far above 0.05); (d) is PASS, the
    verdict is not provisional and names no few-shot branch; the fewshot block holds the path's numbers, the record's
    path and sha256 and the mode; check_record accepts it and render_go shows the mean delta and the few-shot line.
    Everything but fewshot, (d) and the verdict is the PENDING verdict's, which verdict_from without fewshot still
    gives byte for byte."""
    sf, fw = synthetic_final, synthetic_fewshot
    v = _fewshot_verdict(sf, fw, monkeypatch)
    p = fw.record["primary"]
    assert p["k"] == FEWSHOT_K and p["mean_delta"] >= 0.05
    assert v["fewshot"] == {"status": verdict.MET, "phase": 9, "k": FEWSHOT_K, "mean_delta": p["mean_delta"],
                            "draws": p["draws"], "record": {"path": "results/fewshot/fewshot.json", "sha256": fw.sha},
                            "mode": "primary", "disclosure": None}
    assert list(v["fewshot"]) == list(verdict.FEWSHOT_KEYS)
    d = v["criteria"]["d"]
    assert d["paths"][2] == {"path": "fewshot", "metric": "new_tag_macro_ap", "status": verdict.MET,
                             "mean_delta": p["mean_delta"], "draws": p["draws"]}
    assert d["paths"][:2] == synthetic_verdict["criteria"]["d"]["paths"][:2]
    assert d["status"] == verdict.PASS
    s = {k: v["criteria"][k]["status"] for k in ("a", "b", "d", "alternative_go")}
    assert v["verdict"] == json.loads(json.dumps(verdict.decide(s["a"], s["b"], s["d"], s["alternative_go"])))
    assert v["verdict"]["provisional"] is False
    assert v["verdict"]["if_fewshot_met"] is None and v["verdict"]["if_fewshot_not_met"] is None
    for key in verdict.VERDICT_KEYS:
        if key not in ("fewshot", "criteria", "verdict"):
            assert v[key] == synthetic_verdict[key], key
    for key in ("a", "b", "c", "alternative_go"):
        assert v["criteria"][key] == synthetic_verdict["criteria"][key], key
    assert verdict.check_record(v, sf.protocol, sf.sha) is v
    text = verdict.render_go(v)
    d_row = [r for r in _table_rows(text) if r.startswith("| (d) ")][0]
    assert "fewshot (new_tag_macro_ap): MET, mean delta %+.2f points" % (p["mean_delta"] * 100) in d_row
    assert "fewshot (new_tag_macro_ap): %s" % verdict.PENDING not in text and "PROVISIONAL" not in text
    assert ("- Few-shot path: k = %d, mean over %d draws, mode primary; record results/fewshot/fewshot.json "
            "sha256 `%s`." % (FEWSHOT_K, len(p["draws"]), fw.sha[:12])) in text
    assert _verdict(sf) == synthetic_verdict                # without the record: the PENDING verdict, unchanged
    assert verdict.render_go(synthetic_verdict).count("Few-shot path") == 0


def _record_copy(fw: types.SimpleNamespace, tmp_path: Path, edit: Callable[[dict[str, Any]], Any]) -> str:
    """A copy of the planted few-shot record (the module fixture's file is never changed) with one edit applied to
    its parsed JSON, written as fewshot_report.record_bytes would; returns its path."""
    rec = copy_module.deepcopy(fw.record)
    edit(rec)
    path = str(tmp_path / "fewshot.json")
    with open(path, "wb") as f:
        f.write(fewshot_report.record_bytes(rec))
    return path


def _bump_primary(rec: dict[str, Any]) -> None:
    """Move every primary draw delta by 0.01 and recompute the mean: an edit that keeps the block self-consistent."""
    rec["primary"]["draws"] = [x + 0.01 for x in rec["primary"]["draws"]]
    rec["primary"]["mean_delta"] = fewshot_report.mean_delta(rec["primary"]["draws"])


@pytest.mark.parametrize("edit, want", [
    (lambda r: r.update(protocol_sha256="0" * 64), "built under protocol sha256 " + "0" * 64),
    (lambda r: r["primary"].update(k=1), "primary k 1 is not criteria \\(d\\)'s fewshot k 2"),
    (lambda r: r["runs"].update(sha256="0" * 64), "names runs manifest sha256 " + "0" * 64),
    (lambda r: r["runs"].update(path="../outside/runs.json"), "outside the repo"),
    (lambda r: r["runs"].update(path="/tmp/runs.json"), "outside the repo"),
    (lambda r: r["decision"].update(mode="fallback"), "decision .* is not the runs manifest's"),
    (_bump_primary, "primary block does not recompute"),
    (lambda r: r.update(schema="fewshot-results/0"), "schema"),
])
def test_verdict_refuses_an_edited_fewshot_record(synthetic_final: types.SimpleNamespace,
                                                  synthetic_fewshot: types.SimpleNamespace, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch,
                                                  edit: Callable[[dict[str, Any]], Any], want: str) -> None:
    """A few-shot record is read into criterion (d) only when it is of the pin, its primary k is criteria
    (d)'s, the runs manifest it names is under the repo and is its own (sha256), its decision is that manifest's and
    its primary block recomputes from the dumps; an edited copy is refused with one line, before any verdict."""
    sf, fw = synthetic_final, synthetic_fewshot
    path = _record_copy(fw, tmp_path, edit)
    with pytest.raises(SystemExit, match=want) as e:
        _fewshot_verdict(sf, fw, monkeypatch, path=path)
    assert "\n" not in str(e.value.code)


def test_verdict_refuses_a_fewshot_k_off_criteria_d(synthetic_final: types.SimpleNamespace,
                                                    synthetic_fewshot: types.SimpleNamespace,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """The few-shot path is applied at criteria (d)'s k only: a protocol (copy) whose criteria (d) fewshot k is not the
    record's primary k is refused naming both."""
    protocol = copy_module.deepcopy(synthetic_final.protocol)
    for p in protocol["criteria"]["d"]["paths"]:
        if p["path"] == "fewshot":
            p["k"] = 10
    with pytest.raises(SystemExit, match="primary k 2 is not criteria \\(d\\)'s fewshot k 10"):
        _fewshot_verdict(synthetic_final, synthetic_fewshot, monkeypatch, protocol=protocol)


def test_verdict_refuses_a_fewshot_record_whose_dumps_changed(synthetic_final: types.SimpleNamespace,
                                                              synthetic_fewshot: types.SimpleNamespace, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """The record is recomputed from the dumps at verdict time: a listed test dump rewritten with other logits (same
    sidecar, a copy of the dump root) makes the unchanged record refused."""
    sf, fw = synthetic_final, synthetic_fewshot
    root = _dump_copy(sf, tmp_path)
    runs = json.loads(open(fw.runs_path, encoding="utf-8").read())
    e = [x for x in runs["entries"] if (x["kind"], x["stage"], x["k"]) == ("laya", "continued", FEWSHOT_K)][0]
    d = os.path.join(root, e["dump_run_id"])
    logits, uids, side, _ = dumps.load_dump(d, "test", sf.manifest)
    side = {k: x for k, x in side.items() if k not in ("dump_sha256", "schema_version")}
    dumps.write_dump(d, "test", -logits, uids, side)
    with pytest.raises(SystemExit, match="primary block does not recompute"):
        _fewshot_verdict(sf, fw, monkeypatch, dump_root=root)


def test_cli_verdict_with_fewshot(synthetic_final: types.SimpleNamespace, synthetic_fewshot: types.SimpleNamespace,
                                  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture[str]) -> None:
    """`report.py --verdict --fewshot RECORD --out X.json` writes verdict_from(..., fewshot=RECORD) (JSON plus a
    newline), and --go-block renders it after check_record; --fewshot with any other mode is an argparse error."""
    sf, fw = synthetic_final, synthetic_fewshot
    _frozen(monkeypatch, sf)
    want = _fewshot_verdict(sf, fw, monkeypatch)
    out = tmp_path / "final" / "verdict.json"
    got = report.main(["--verdict", "--fewshot", fw.path, "--manifest", sf.manifest, "--out", str(out)])
    capsys.readouterr()
    assert got == want and out.read_text(encoding="utf-8") == json.dumps(want, ensure_ascii=False, indent=2) + "\n"
    assert report.main(["--go-block", str(out)]) == verdict.render_go(want)
    capsys.readouterr()
    for argv in (["--go-block", str(out), "--fewshot", fw.path], ["--final-table", str(out), "--fewshot", fw.path],
                 ["--table", str(out.parent), "--fewshot", fw.path], [str(tmp_path), "--fewshot", fw.path]):
        with pytest.raises(SystemExit) as e:
            report.main(argv)
        assert e.value.code == 2 and "--fewshot applies only to --verdict" in capsys.readouterr().err


# --- build_verdict on hand-made inputs ------------------------------------------------------------------------------

def _delta(d: float, lo: float, hi: float, a: float = 0.5) -> dict[str, Any]:
    """One bootstrap metric block: A = a + d, B = a, delta d, CI [lo, hi]."""
    return {"a": a + d, "b": a, "delta": d, "ci": [lo, hi], "width": hi - lo, "n_undefined": 0}


def _hand_deltas(micro: tuple[float, float, float] = (0.0, -0.01, 0.01),
                 macro: tuple[float, float, float] = (0.0, -0.02, 0.02),
                 brier: tuple[float, float, float] = (0.001, 0.0005, 0.002),
                 p80: tuple[float, float, float] = (0.0, -0.02, 0.02)) -> dict[str, Any]:
    """A seed_averaged variant with the given (delta, lo, hi) of micro-F1, macro-F1, Brier and precision at 80 %."""
    m = {"micro_f1": _delta(*micro), "macro_f1": _delta(*macro), "exact_match": _delta(0.01, -0.02, 0.04),
         "top1_hit": _delta(-0.01, -0.03, 0.01), "brier": _delta(*brier, a=0.02),
         "classwise_ece": _delta(0.002, 0.001, 0.004, a=0.03), "precision_at_80": _delta(*p80, a=0.8)}
    return {"n": 413, "B": 10000, "seed": 0, "level": 0.95, "method": metrics.BOOT_METHOD, "coverage": 80,
            "metrics": m, "a": "laya", "b": "berturk", "temperature": {"laya": 4.77, "berturk": 0.743},
            "theta": {"laya": "theta.shipped", "berturk": "theta.shipped"}}


def _hand_seed(kind: str, seed: int, values: dict[str, float] | None = None) -> dict[str, Any]:
    """One per-seed row of `kind` with fixed metrics, `values` overriding them."""
    v = dict({"micro_f1": 0.8, "macro_f1": 0.7, "exact_match": 0.5, "top1_hit": 0.9}, **(values or {}))
    return dict({"seed": seed, "dump_run_id": "%s-final-%d" % (kind, seed), "dump_sha256": "d" * 64,
                 "train_run_id": "%s-final_train-s%d" % (kind, seed), "temperature": 1.5, "n_top1": 398,
                 "test_counts": {"tp": 400, "fp": 100, "fn": 100}, "loss_by_epoch": [0.8, 0.5, 0.3]}, **v)


def _hand_inputs(per_seed: dict[str, list[dict[str, Any]]] | None = None,
                 **deltas: tuple[float, float, float]) -> dict[str, Any]:
    """verdict_from's inputs with chosen numbers: no collapsed run, three seeds per model, the headline deltas as
    given (the raw variant equal to them)."""
    systems = {}
    for kind in KINDS:
        rows = (per_seed or {}).get(kind) or [_hand_seed(kind, s) for s in SEEDS]
        systems[kind] = {
            "kind": kind, "postfit": {"path": "results/postfit/%s.json" % kind, "sha256": "p" * 64},
            "final_manifest": {"path": "results/postfit/%s-final.json" % kind, "sha256": "f" * 64},
            "settings": {"chosen": "%s-chosen" % kind, "train_run_id": "%s-selection_train-s0" % kind,
                         "temperature": 1.5, "theta": {"shipped": {"global": 0.6, "n_per_tag": 20},
                                                       "raw": {"global": 0.4, "n_per_tag": 19}},
                         "epoch": 3, "epochs_schedule": 4, "encoder_lr": 2e-05, "head_lr": 0.0001,
                         "config_sha256_sans_seed": "c" * 64},
            "seeds": list(SEEDS), "per_seed": rows}
    h = _hand_deltas(**deltas)
    return {"split": {"name": "test", "path": "data/splits/test.jsonl", "sha256": "t" * 64, "uids_sha256": "u" * 64,
                      "n": 413, "n_no_intent": 15},
            "systems": systems,
            "references": [{"name": "majority", "run_id": "majority", "report": {"path": "results/reports/majority.json",
                                                                               "sha256": "m" * 64},
                            "kind": "majority", "checkpoint": None, "baseline_split": "final_train",
                            "temperature": 1.0, "theta": {"global": 0.5, "n_per_tag": 13}, "micro_f1": 0.14,
                            "macro_f1": 0.05, "exact_match": 0.0, "top1_hit": 0.2, "n": 413, "n_top1": 398}],
            "seed_averaged": {"headline": h, "raw": dict(copy_module.deepcopy(h), temperature={k: 1.0 for k in KINDS},
                                                         theta={k: "theta.raw" for k in KINDS})},
            "collapse": {"selection": [{"kind": k, "seed": 0, "train_run_id": "%s-selection_train-s0" % k,
                                        "dump_run_id": "%s-chosen" % k, "objective": {"tp": 426, "fp": 142, "fn": 57},
                                        "loss_by_epoch": [0.8, 0.5, 0.3]} for k in KINDS]},
            "latency": [{"system": k, "device": "mps", "n": 624, "p50": 0.3, "p95": 0.4} for k in KINDS]}


def _hand_protocol() -> dict[str, Any]:
    """The protocol blocks build_verdict reads: the default criteria and collapse, the verdict and bootstrap blocks."""
    collapse = dict(copy_module.deepcopy(verdict.DEFAULT_COLLAPSE), prior_val_fraction="10/41",
                    prior_test_fraction="520/2163", prior_theta_global=0.12)
    return {"criteria": copy_module.deepcopy(verdict.DEFAULT_CRITERIA), "collapse": collapse,
            "verdict": pilot_protocol.verdict_block(), "bootstrap": dict(pilot_protocol.bootstrap_block())}


def test_build_verdict_marks_the_fewshot_path_pending() -> None:
    """With no few-shot result, (d)'s few-shot path is PENDING and the record says so; a (d) with neither
    other path met is PENDING, and the verdict is decide's."""
    v = verdict.build_verdict(_hand_protocol(), "a" * 64, _hand_inputs())
    assert v["fewshot"] == {"status": verdict.PENDING, "phase": 9}
    paths = {p["path"]: p for p in v["criteria"]["d"]["paths"]}
    assert paths["fewshot"]["status"] == verdict.PENDING and paths["fewshot"]["mean_delta"] is None
    assert v["criteria"]["d"]["status"] == verdict.PENDING
    s = {k: v["criteria"][k]["status"] for k in ("a", "b", "d", "alternative_go")}
    assert v["verdict"] == verdict.decide(s["a"], s["b"], s["d"], s["alternative_go"])


def test_build_verdict_refuses_rules_other_than_the_code_it_runs() -> None:
    """The frozen verdict rules must be the ones pilot.verdict implements, else the verdict would not follow them."""
    protocol = _hand_protocol()
    protocol["verdict"]["rules"][0] = "(b) FAIL -> GO"
    with pytest.raises(SystemExit, match="verdict rules"):
        verdict.build_verdict(protocol, "a" * 64, _hand_inputs())


def test_build_verdict_refuses_per_seed_rows_off_the_listed_seeds() -> None:
    inputs = _hand_inputs()
    inputs["systems"]["berturk"]["per_seed"] = inputs["systems"]["berturk"]["per_seed"][:2]
    with pytest.raises(SystemExit, match="berturk.*seeds \\[0, 1\\].*\\[0, 1, 2\\]"):
        verdict.build_verdict(_hand_protocol(), "a" * 64, inputs)


def test_mean_std_is_the_sample_std_and_exact_for_equal_values() -> None:
    """The sample std (ddof 1); equal values give exactly 0.0 and their own value as the mean; fewer than 2 values are
    refused."""
    mean, std = verdict.mean_std([0.5, 0.5, 0.7])
    assert mean == pytest.approx(0.5666666666666667, abs=1e-15)
    assert std == pytest.approx(float(np.std([0.5, 0.5, 0.7], ddof=1)), abs=1e-15)
    assert verdict.mean_std([0.6, 0.6, 0.6]) == (0.6, 0.0)
    assert verdict.mean_std([0.1, 0.1]) == (0.1, 0.0)
    for bad in ([0.5], [], [0.5, None], [0.5, float("nan")]):
        with pytest.raises(SystemExit):
            verdict.mean_std(bad)


# --- the renders (the results table, the GO block) and the CLI modes ----------------------------------------------

def _cell(values: list[float]) -> str:
    """The results-table cell of per-seed values: mean ± sample std (the values in seed order)."""
    mean, std = verdict.mean_std(values)
    return "%.3f ± %.3f (%s)" % (mean, std, " / ".join("%.3f" % x for x in values))


def _table_rows(text: str) -> list[str]:
    """The body rows of the markdown tables in text (a header row is the one followed by the |---| line)."""
    lines = text.split("\n")
    return [x for i, x in enumerate(lines) if x.startswith("| ")
            and not (i + 1 < len(lines) and lines[i + 1].startswith("|---"))]


def test_render_final_table_rows_in_the_fixed_order(synthetic_verdict: dict[str, Any]) -> None:
    """References (majority, prior, zero-shot Laya), then each fine-tuned model as mean ± sample std with its
    per-seed values in seed order, then its seed-averaged system; the order does not follow the record's list
    order; a settings table with T, θ, epoch, LRs, seeds and train run ids."""
    v = synthetic_verdict
    text = verdict.render_final_table(v)
    assert text.startswith(verdict.FINAL_HEADING + "\n")
    rows = _table_rows(text.split("Settings")[0])
    labels = [r.split(" | ")[0][2:] for r in rows]
    assert labels == ["majority (train: final_train)", "prior (train: final_train)",
                      "laya fine-tuned (final seeds 0, 1, 2)", "laya fine-tuned, seed-averaged",
                      "berturk fine-tuned (final seeds 0, 1, 2)", "berturk fine-tuned, seed-averaged"]
    for kind, row in (("laya", rows[2]), ("berturk", rows[4])):
        cells = row.strip("| ").split(" | ")
        per = v["systems"][kind]["per_seed"]
        assert cells[2:] == [_cell([r[m] for r in per]) for m in verdict.TABLE_METRICS]
    side = {"laya": "a", "berturk": "b"}
    for kind, row in (("laya", rows[3]), ("berturk", rows[5])):
        cells = row.strip("| ").split(" | ")
        h = v["seed_averaged"]["headline"]["metrics"]
        assert cells[2:] == ["%.3f" % h[m][side[kind]] for m in verdict.TABLE_METRICS]
    for kind in KINDS:
        st = v["systems"][kind]["settings"]
        assert "%.4f" % st["temperature"] in text
        for r in v["systems"][kind]["per_seed"]:
            assert r["train_run_id"] in text
        assert "%d / %d" % (st["epoch"], st["epochs_schedule"]) in text
    shuffled = dict(v, references=list(reversed(v["references"])))
    assert verdict.render_final_table(shuffled) == text           # the fixed order, not the list order
    assert verdict.render_final_table(v) == text                  # deterministic


def test_render_final_table_keeps_equal_seed_values_and_a_zero_std() -> None:
    """Adjacency edge: two equal seed values stay two values; equal values give std exactly 0.000."""
    per = {"laya": [_hand_seed("laya", s, {"micro_f1": x}) for s, x in zip(SEEDS, (0.81, 0.81, 0.83))],
           "berturk": [_hand_seed("berturk", s, {"micro_f1": 0.8}) for s in SEEDS]}
    v = verdict.build_verdict(_hand_protocol(), "a" * 64, _hand_inputs(per_seed=per))
    assert v["systems"]["berturk"]["summary"]["micro_f1"] == {"mean": 0.8, "std": 0.0, "values": [0.8, 0.8, 0.8]}
    rows = _table_rows(verdict.render_final_table(v).split("Settings")[0])
    laya = [r for r in rows if r.startswith("| laya fine-tuned (")][0].strip("| ").split(" | ")
    berturk = [r for r in rows if r.startswith("| berturk fine-tuned (")][0].strip("| ").split(" | ")
    assert laya[2] == _cell([0.81, 0.81, 0.83]) and "(0.810 / 0.810 / 0.830)" in laya[2]
    assert berturk[2] == "0.800 ± 0.000 (0.800 / 0.800 / 0.800)"


def test_render_final_table_refuses_a_dropped_seed_or_missing_references(synthetic_verdict: dict[str, Any]) -> None:
    """Empty edge: fewer per-seed rows than the listed seeds, no reference row, or another schema is refused."""
    dropped = copy_module.deepcopy(synthetic_verdict)
    dropped["systems"]["laya"]["per_seed"].pop()
    with pytest.raises(SystemExit, match="laya"):
        verdict.render_final_table(dropped)
    with pytest.raises(SystemExit, match="reference"):
        verdict.render_final_table(dict(synthetic_verdict, references=[]))
    for render in (verdict.render_final_table, verdict.render_go):
        with pytest.raises(SystemExit, match="verdict/1"):
            render(dict(synthetic_verdict, schema="confidence/1"))


def test_render_go_shows_every_criterion_and_the_verdict(synthetic_verdict: dict[str, Any]) -> None:
    """One row per criterion and the alternative GO with its status and numbers, (c) REPORTED (not gated), the
    macro-F1 CI label, the verdict line, the protocol sha256's first 12 hex characters, and a closing descriptive line;
    deterministic."""
    v = synthetic_verdict
    text = verdict.render_go(v)
    assert text.startswith("### GO block (mechanical)\n") and text == verdict.render_go(v)
    rows = _table_rows(text)
    assert [r.split(" | ")[0][2:] for r in rows] == ["(a) non-inferiority", "(b) seeds and no collapsed run",
                                                     "(c) latency",
                                                     "(d) a clear edge on calibration, abstention or few-shot learning",
                                                     "alternative GO"]
    for key, row in zip(("a", "b", "c", "d", "alternative_go"), rows):
        assert row.split(" | ")[1] == v["criteria"][key]["status"]
    assert "(c) REPORTED (not gated)" in text
    assert v["criteria"]["a"]["macro_ci_label"] in text
    micro = v["criteria"]["a"]["micro"]
    assert "%+.2f points" % (micro["delta"] * 100) in rows[0]
    assert "[%+.2f, %+.2f]" % (micro["ci"][0] * 100, micro["ci"][1] * 100) in rows[0]
    assert "**Verdict: %s**" % v["verdict"]["verdict"] in text
    assert v["protocol"]["sha256"][:12] in text and v["protocol"]["sha256"] not in text
    last = text.rstrip("\n").split("\n")[-1]
    assert last.startswith("Descriptive, not gating:")
    raw = v["seed_averaged"]["raw"]["metrics"]["micro_f1"]["delta"]
    for name in ("classwise_ece", "exact_match", "top1_hit"):
        assert name in last
    assert "%+.2f points" % (raw * 100) in last


def test_render_go_inconclusive_counts_as_no_go() -> None:
    """(a) INCONCLUSIVE, the alternative GO not met: the verdict is INCONCLUSIVE and the block says what it means."""
    v = verdict.build_verdict(_hand_protocol(), "a" * 64, _hand_inputs(micro=(-0.005, -0.03, 0.02)))
    assert v["criteria"]["a"]["status"] == verdict.INCONCLUSIVE and v["verdict"]["verdict"] == verdict.INCONCLUSIVE
    text = verdict.render_go(v)
    assert "counts as NO-GO; collect bank data" in text
    assert "PROVISIONAL" not in text


def test_render_go_provisional_names_both_outcomes() -> None:
    """(a) PASS, (d) PENDING (neither Brier nor abstention met), (b) PASS: the verdict depends on the few-shot path
    and the block shows the PROVISIONAL sentence; with the Brier path met it is a plain GO."""
    v = verdict.build_verdict(_hand_protocol(), "a" * 64, _hand_inputs())
    assert v["verdict"]["provisional"] is True
    text = verdict.render_go(v)
    assert "PROVISIONAL: GO if the few-shot path of (d) is met, else NO-GO" in text
    assert verdict.INCONCLUSIVE_MEANS not in text
    go = verdict.build_verdict(_hand_protocol(), "a" * 64, _hand_inputs(brier=(-0.001, -0.002, -0.0005)))
    assert go["verdict"]["verdict"] == verdict.GO and go["verdict"]["provisional"] is False
    text = verdict.render_go(go)
    assert "**Verdict: GO**" in text and "PROVISIONAL" not in text


def _frozen(monkeypatch: pytest.MonkeyPatch, sf: types.SimpleNamespace) -> None:
    """Point pilot.protocol's path and pin and report.REPO_ROOT at the synthetic repo."""
    monkeypatch.setattr(pilot_protocol, "PROTOCOL_PATH", sf.path)
    monkeypatch.setattr(pilot_protocol, "PROTOCOL_SHA256", sf.sha)
    monkeypatch.setattr(report, "REPO_ROOT", sf.root)


def test_cli_verdict_final_table_and_go_block(synthetic_final: types.SimpleNamespace, synthetic_verdict: dict[str, Any],
                                              tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    """`report.py --verdict --out X.json` writes the verdict of the frozen protocol (JSON plus a newline);
    `--final-table X.json` and `--go-block X.json` print the renders of the saved record."""
    sf = synthetic_final
    _frozen(monkeypatch, sf)
    out = tmp_path / "final" / "verdict.json"
    got = report.main(["--verdict", "--manifest", sf.manifest, "--out", str(out)])
    capsys.readouterr()
    text = out.read_text(encoding="utf-8")
    assert text == json.dumps(synthetic_verdict, ensure_ascii=False, indent=2) + "\n"
    assert got == synthetic_verdict
    assert report.main(["--final-table", str(out)]) == verdict.render_final_table(synthetic_verdict)
    assert capsys.readouterr().out == verdict.render_final_table(synthetic_verdict)
    assert report.main(["--go-block", str(out)]) == verdict.render_go(synthetic_verdict)
    assert capsys.readouterr().out == verdict.render_go(synthetic_verdict)


def _edit_verdict(v: dict[str, Any], how: str) -> dict[str, Any]:
    """A copy of verdict record v with one hand edit."""
    v = copy_module.deepcopy(v)
    if how == "verdict":
        v["verdict"]["verdict"] = verdict.GO if v["verdict"]["verdict"] != verdict.GO else verdict.NO_GO
    elif how == "status":
        v["criteria"]["a"]["status"] = verdict.PASS if v["criteria"]["a"]["status"] != verdict.PASS else verdict.FAIL
    elif how == "number":
        v["criteria"]["a"]["micro"]["delta"] += 0.05
    elif how == "protocol":
        v["protocol"]["sha256"] = "0" * 64
    elif how == "collapse":
        row = v["collapse"]["rows"][0]
        row["components"][0]["collapsed"] = not row["components"][0]["collapsed"]
        row["collapsed"] = any(c["collapsed"] for c in row["components"])
    elif how == "seeds":
        v["criteria"]["b"]["seeds"]["laya"] = v["criteria"]["b"]["seeds"]["laya"][:1]
    elif how == "fewshot":
        v["fewshot"]["status"] = verdict.MET
    return v


@pytest.mark.parametrize("how", ["verdict", "status", "number", "protocol", "collapse", "seeds", "fewshot"])
def test_cli_renders_refuse_an_edited_verdict(synthetic_final: types.SimpleNamespace, synthetic_verdict: dict[str, Any],
                                              tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str], how: str) -> None:
    """--go-block and --final-table render only a record bound to the frozen protocol, whose every
    criterion, collapse flag and verdict recompute from its own numbers (pilot.verdict.check_record); a hand-edited
    status, verdict, number or protocol sha256 is refused with one line, nothing printed."""
    sf = synthetic_final
    _frozen(monkeypatch, sf)
    verdict.check_record(synthetic_verdict, sf.protocol, sf.sha)        # the unedited record passes
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(_edit_verdict(synthetic_verdict, how), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    for mode in ("--go-block", "--final-table"):
        with pytest.raises(SystemExit) as e:
            report.main([mode, str(path)])
        text = str(e.value.code)
        assert "\n" not in text and ("protocol" in text or "rebuild" in text), (how, text)
        assert capsys.readouterr().out == "", (how, mode)


def test_cli_renders_refuse_while_the_protocol_is_not_frozen(synthetic_final: types.SimpleNamespace,
                                                             synthetic_verdict: dict[str, Any], tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the frozen protocol (pin unset) no verdict record is rendered."""
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(synthetic_verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(pilot_protocol, "PROTOCOL_SHA256", None)
    for mode in ("--go-block", "--final-table"):
        with pytest.raises(SystemExit, match="protocol"):
            report.main([mode, str(path)])


def test_cli_verdict_modes_are_exclusive(synthetic_final: types.SimpleNamespace, tmp_path: Path,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    path = str(tmp_path / "v.json")
    for argv in (["--verdict", "--final-table", path], ["--final-table", path, "--go-block", path],
                 ["--go-block", path, "--table", path], ["--verdict", "--confidence-table", path]):
        with pytest.raises(SystemExit) as e:
            report.main(argv)
        assert e.value.code == 2 and "give exactly one of" in capsys.readouterr().err


def test_cli_verdict_refuses_while_the_protocol_is_not_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the pin unset (the repo before the freeze commit), --verdict is refused naming the protocol, before any
    dump is read."""
    monkeypatch.setattr(pilot_protocol, "PROTOCOL_SHA256", None)
    with pytest.raises(SystemExit, match="protocol"):
        report.main(["--verdict"])
