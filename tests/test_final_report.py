"""The per-seed half of REP-02 (Phase 8, 08-04), offline on synthetic final seeds: no model, torch or Ray.

A final seed (a dump whose training.postfit_sha256 is set) was trained on final_train, which holds the val folds, so
report.py never selects θ on its own val, and never on test (08-RESEARCH A5, Pattern 3). `report.py RUN_DIR --split
test --postfit RECORD` reports it at its postfit record's θ instead: theta.shipped for the shipped variant (at the
sidecar's shipped T, which must be the record's chosen T) and theta.raw for raw (T = 1), through
report.final_seed_binding, the one helper the --confidence final_seed binding uses too (record sha256 ==
training.postfit_sha256, the same system kind, refuse_held_out_record, T within T_MATCH), so the two report modes
refuse a wrong record with the same message. The val_selected rows name their source, theta_source {postfit, chosen,
split: val}; without --postfit a final seed's val_selected rows are null with FINAL_SEED_REASON; --postfit on any
other dump (a selection candidate, a baseline) is refused. Pitfall 3 (the record θ was selected on one selection-run
seed) is disclosed in Phase 8, never re-selected.

The invariant (assumption delta, theta_source promoted): every val_selected row of every report mode names a θ source
on val: run_report's theta_source, each --confidence binding's theta_source, and each --compare side's val dumps
(dump_sha256.val; the compare output is left as it is, so the committed results/reports/compare-*.json still rebuild
byte for byte).

render_confidence takes svg_dir (default CONFIDENCE_SVG_DIR, results/confidence, so the committed Phase 7 README block
renders byte for byte as before), and --confidence-table derives it from the inputs' one common directory
(confidence_svg_dir), so 08-12 renders the test/HOTELS block from results/final/confidence with matching links.

The synthetic snapshot, the postfit record over a synthetic grid and the final-seed and split-dump builders are
tests/test_confidence.py's.

The verdict (08-09, REP-02 and REP-03) is proven end to end on a synthetic repo before any real test number exists
(synthetic_final, one module fixture): a fitted Laya and BERTurk postfit record over synthetic grids, three final seeds
per kind whose test dumps carry protocol_sha256 == the synthetic protocol's own sha256 and training.postfit_sha256 ==
their record's, the label prior's and majority's val and test dumps with their test reports as the reference rows, two
latency-like reports and a protocol dict (pilot.protocol's blocks where they apply to any root, B 200 for speed) that
lists every input with its sha256. report.verdict_from checks every input's sha256, loads exactly the protocol's
final-seed test dumps (pin, seed, postfit binding, pairing, the protocol's test uids), seed-averages each model at its
record T and θ (A5), recomputes the prior's frozen fractions and hands the inputs to pilot.verdict.build_verdict, whose
criteria and verdict the tests recompute from the record's own numbers. Every refusal is exercised on a copy of the
dumps or of the protocol dict, so the module fixture is never changed.
"""
import copy as copy_module
import json
import os
import shutil
import types

import numpy as np
import pytest

import pilot.data
import postfit as postfit_cli
import report
import score
from conftest import make_raw
from pilot import dumps, metrics
from pilot import protocol as pilot_protocol
from pilot import verdict
from pilot.data import repo_relpath
from test_berturk_postfit import EPOCHS as BERTURK_EPOCHS
from test_berturk_postfit import write_berturk_candidate
from test_confidence import _drop_val, _final_seed, _prior, _write_json, _write_split_dump, fitted, snap  # noqa: F401
from test_postfit import GRID, write_candidate

pytestmark = pytest.mark.unit


def _final(root, snap, record, sha, **kw):
    """A final seed's run dir holding only a test dump (Phase 8's final-seed dirs hold no val dump)."""
    return _drop_val(_write_split_dump(_final_seed(root, snap, record, sha, **kw), snap[1], "test"))


def _expected_row(run_dir, manifest, T, theta):
    logits, uids, _, entry = dumps.load_dump(run_dir, "test", manifest)
    Y = dumps.read_gold(entry, manifest, uids)
    P = metrics.probabilities(logits, T)
    return dict(metrics.metrics(Y, P, metrics.theta_vector(theta)), theta=theta, in_sample=False)


# --- the final-seed θ binding in run_report ----------------------------------------------------------------------

def test_final_seed_report_uses_the_record_theta(tmp_path, snap, fitted):
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


def test_final_seed_report_without_postfit_names_the_reason(tmp_path, snap, fitted):
    """θ is never self-selected for a final seed: without --postfit its val_selected rows are null, with the reason."""
    _, _, record, sha = fitted
    rep = report.run_report(_final(tmp_path / "final", snap, record, sha), "test", snap[1])
    assert rep["theta_source"] is None
    for v in rep["variants"].values():
        assert v["rows"]["val_selected"] is None and v["val_selected_reason"] == report.FINAL_SEED_REASON
    assert "--postfit RECORD" in report.FINAL_SEED_REASON


def _message(call):
    with pytest.raises(SystemExit) as e:
        call()
    return str(e.value)


def test_final_seed_postfit_refusals_match_confidence(tmp_path, snap, fitted):
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


def test_postfit_refused_for_a_dump_that_is_not_a_final_seed(tmp_path, snap, fitted, capsys):
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


def test_cli_run_dir_postfit_writes_the_run_report(tmp_path, snap, fitted, capsys):
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


def _theta_sources(rep, manifest, dirs=()):
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


def test_every_val_selected_row_names_a_val_theta_source(tmp_path, snap, fitted, capsys):
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


# --- render_confidence's svg_dir (08-12 renders the test/HOTELS block from results/final/confidence) ----------------

CONFIDENCE_DIR = os.path.join(pilot.data.REPO_ROOT, "results", "confidence")


def _committed_confidence():
    paths = report.table_inputs([CONFIDENCE_DIR])
    assert paths, "no committed confidence record under results/confidence/"
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return paths, out


def test_render_confidence_svg_dir_default_is_the_phase7_block():
    """The default svg_dir is results/confidence, so the committed Phase 7 README block renders as before."""
    _, records = _committed_confidence()
    assert report.CONFIDENCE_SVG_DIR == "results/confidence"
    assert report.render_confidence(records) == report.render_confidence(records, svg_dir="results/confidence")


def test_render_confidence_svg_dir_changes_only_the_links():
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


def test_confidence_table_svg_dir_keeps_the_committed_block(capsys):
    """`report.py --confidence-table results/confidence` derives svg_dir from the inputs' directory: the committed
    records give exactly render_confidence's default text (the README Phase 7 block)."""
    paths, records = _committed_confidence()
    assert report.confidence_svg_dir(paths) == report.CONFIDENCE_SVG_DIR
    text = report.main(["--confidence-table", CONFIDENCE_DIR])
    assert capsys.readouterr().out == text == report.render_confidence(records)


def test_confidence_svg_dir_follows_one_common_directory_under_root(tmp_path):
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


# --- the verdict (08-09): a synthetic protocol, three Laya and three BERTurk final seeds ----------------------------

KINDS = ("laya", "berturk")
SEEDS = (0, 1, 2)
SEVEN = ("micro_f1", "macro_f1", "exact_match", "top1_hit", "brier", "classwise_ece", "precision_at_80")
LATENCY = {"laya": {"system": "laya", "device": "mps", "pooled": {"n": 21, "p50": 0.33, "p95": 0.41}},
           "berturk": {"system": "berturk", "device": "mps", "pooled": {"n": 21, "p50": 0.012, "p95": 0.015}}}


def _rewrite(run_dir, manifest, split, reverse=False, **over):
    """Rewrite a dump of run_dir with sidecar keys replaced by `over` (a planted case); reverse=True writes the rows in
    reverse order as a --n subset of the full count (the only way load_dump admits another order)."""
    logits, uids, side, _ = dumps.load_dump(run_dir, split, manifest)
    side = {k: v for k, v in side.items() if k not in ("dump_sha256", "schema_version")}
    side.update(over)
    if reverse:
        logits, uids, side["n_subset"] = logits[::-1].copy(), uids[::-1], len(uids)
    dumps.write_dump(run_dir, split, logits, uids, side)
    return run_dir


def _gated_test_dump(run_dir, manifest, pin, rng_seed):
    """The test dump score.py's protocol gate writes for a final seed (08-07): the val sidecar's run_id, system,
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


def _final_seed_dir(dump_root, snap, kind, record, rsha, seed):
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
def synthetic_final(tmp_path_factory):
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
        test = json.load(f)["splits"]["test"]
    systems = pilot_protocol.systems_block(str(root))
    protocol = {
        "schema": pilot_protocol.SCHEMA, "title": "synthetic", "units": pilot_protocol.UNITS,
        "frozen": dict(pilot_protocol.FROZEN),
        "inputs": {rel: pilot.data.sha256_file(str(root / rel)) for rel in rels},
        "splits": {"test": dict({"path": "data/" + test["path"]},
                                **{k: test[k] for k in pilot_protocol.SPLIT_KEYS})},
        "systems": systems,
        "references": {"rule": pilot_protocol.REFERENCES_RULE, "rows": refs},
        "scoring": copy_module.deepcopy(pilot_protocol.SCORING),
        "variant": dict(pilot_protocol.VARIANT), "seed_averaging": dict(pilot_protocol.SEED_AVERAGING),
        "bootstrap": dict(pilot_protocol.bootstrap_block(), B=200),
        "criteria": copy_module.deepcopy(verdict.DEFAULT_CRITERIA),
        "collapse": pilot_protocol.collapse_block(manifest, str(root)),
        "verdict": pilot_protocol.verdict_block(), "definitions": pilot_protocol.definitions_block(),
        "fewshot": {"note": "synthetic"}, "known_at_freeze": {"note": "synthetic"},
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


def _verdict(sf, protocol=None, sha=None, dump_root=None):
    return report.verdict_from(sf.protocol if protocol is None else protocol, sf.sha if sha is None else sha,
                               sf.manifest, sf.root, sf.dump_root if dump_root is None else dump_root)


@pytest.fixture(scope="module")
def synthetic_verdict(synthetic_final):
    return _verdict(synthetic_final)


def test_verdict_end_to_end_on_synthetic_final_seeds(synthetic_final):
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


def test_verdict_seed_averages_at_the_record_t_and_theta(synthetic_final, synthetic_verdict):
    """The seed-averaged systems (A5): per seed sigmoid(logit / T) at the record T, averaged, thresholded at the
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


def test_verdict_collapse_rows_in_order(synthetic_final, synthetic_verdict):
    """Collapse rows: selection laya, selection berturk, then final laya s0..s2 and final berturk s0..s2; the
    selection runs by the val floor and the loss, the final seeds by the loss and the test floor (A6), against the
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


def test_verdict_is_deterministic(synthetic_final, synthetic_verdict):
    """No timestamp, no rounding: the same inputs give identical bytes."""
    again = _verdict(synthetic_final)
    assert json.dumps(again, ensure_ascii=False, indent=2) == json.dumps(synthetic_verdict, ensure_ascii=False,
                                                                         indent=2)


def test_verdict_refuses_a_changed_protocol_input(synthetic_final):
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


def _dump_copy(sf, tmp_path):
    out = str(tmp_path / "dumps")
    shutil.copytree(sf.dump_root, out)
    return out


def test_verdict_refuses_a_missing_final_seed_test_dump(synthetic_final, tmp_path):
    sf = synthetic_final
    root = _dump_copy(sf, tmp_path)
    rid = os.path.basename(sf.finals["berturk"][1])
    os.remove(os.path.join(root, rid, "test.npz"))
    with pytest.raises(SystemExit, match="berturk final seed 1 .*no test dump"):
        _verdict(sf, dump_root=root)


def test_verdict_refuses_a_dump_bound_to_another_protocol(synthetic_final):
    """A sidecar protocol_sha256 other than the frozen protocol's sha256 is refused."""
    with pytest.raises(SystemExit, match="protocol_sha256"):
        _verdict(synthetic_final, sha="0" * 64)


@pytest.mark.parametrize("plant, want", [
    (lambda side: {"seed": 7}, "seed 7"),
    (lambda side: {"training": dict(side["training"], postfit_sha256="9" * 64)}, "postfit_sha256 " + "9" * 64),
])
def test_verdict_refuses_a_seed_or_postfit_binding_mismatch(synthetic_final, tmp_path, plant, want):
    sf = synthetic_final
    root = _dump_copy(sf, tmp_path)
    d = os.path.join(root, os.path.basename(sf.finals["laya"][2]))
    side = dumps.load_dump(d, "test", sf.manifest)[2]
    _rewrite(d, sf.manifest, "test", **plant(side))
    with pytest.raises(SystemExit, match=want):
        _verdict(sf, dump_root=root)


def test_verdict_refuses_an_unpaired_uid_order(synthetic_final, tmp_path):
    """Every final-seed test dump must score the protocol's test uids in one order (EVAL-07 pairing)."""
    sf = synthetic_final
    root = _dump_copy(sf, tmp_path)
    _rewrite(os.path.join(root, os.path.basename(sf.finals["berturk"][0])), sf.manifest, "test", reverse=True)
    with pytest.raises(SystemExit, match="unpaired"):
        _verdict(sf, dump_root=root)


@pytest.mark.parametrize("key, value", [("prior_val_fraction", "1/3"), ("prior_test_fraction", "1/3"),
                                        ("prior_theta_global", 0.37)])
def test_verdict_refuses_a_prior_fraction_off_the_frozen_one(synthetic_final, key, value):
    """The prior's floors are recomputed from its dumps (pilot.protocol.prior_floors) and must equal the frozen ones."""
    protocol = copy_module.deepcopy(synthetic_final.protocol)
    protocol["collapse"][key] = value
    with pytest.raises(SystemExit, match=key):
        _verdict(synthetic_final, protocol=protocol)


def test_verdict_refuses_a_system_with_fewer_final_seeds_than_listed(synthetic_final):
    protocol = copy_module.deepcopy(synthetic_final.protocol)
    protocol["systems"]["laya"]["final_seeds"].pop()
    with pytest.raises(SystemExit, match="laya: .*final seeds \\[0, 1\\].*\\[0, 1, 2\\]"):
        _verdict(synthetic_final, protocol=protocol)


# --- build_verdict on hand-made inputs ------------------------------------------------------------------------------

def _delta(d, lo, hi, a=0.5):
    return {"a": a + d, "b": a, "delta": d, "ci": [lo, hi], "width": hi - lo, "n_undefined": 0}


def _hand_deltas(micro=(0.0, -0.01, 0.01), macro=(0.0, -0.02, 0.02), brier=(0.001, 0.0005, 0.002),
                 p80=(0.0, -0.02, 0.02)):
    m = {"micro_f1": _delta(*micro), "macro_f1": _delta(*macro), "exact_match": _delta(0.01, -0.02, 0.04),
         "top1_hit": _delta(-0.01, -0.03, 0.01), "brier": _delta(*brier, a=0.02),
         "classwise_ece": _delta(0.002, 0.001, 0.004, a=0.03), "precision_at_80": _delta(*p80, a=0.8)}
    return {"n": 413, "B": 10000, "seed": 0, "level": 0.95, "method": metrics.BOOT_METHOD, "coverage": 80,
            "metrics": m, "a": "laya", "b": "berturk"}


def _hand_seed(kind, seed, values=None):
    v = dict({"micro_f1": 0.8, "macro_f1": 0.7, "exact_match": 0.5, "top1_hit": 0.9}, **(values or {}))
    return dict({"seed": seed, "dump_run_id": "%s-final-%d" % (kind, seed), "dump_sha256": "d" * 64,
                 "train_run_id": "%s-final_train-s%d" % (kind, seed), "temperature": 1.5, "n_top1": 398,
                 "test_counts": {"tp": 400, "fp": 100, "fn": 100}, "loss_by_epoch": [0.8, 0.5, 0.3]}, **v)


def _hand_inputs(per_seed=None, **deltas):
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
            "seed_averaged": {"headline": h, "raw": copy_module.deepcopy(h)},
            "collapse": {"selection": [{"kind": k, "seed": 0, "train_run_id": "%s-selection_train-s0" % k,
                                        "dump_run_id": "%s-chosen" % k, "objective": {"tp": 426, "fp": 142, "fn": 57},
                                        "loss_by_epoch": [0.8, 0.5, 0.3]} for k in KINDS]},
            "latency": [{"system": k, "device": "mps", "n": 624, "p50": 0.3, "p95": 0.4} for k in KINDS]}


def _hand_protocol():
    collapse = dict(copy_module.deepcopy(verdict.DEFAULT_COLLAPSE), prior_val_fraction="10/41",
                    prior_test_fraction="520/2163", prior_theta_global=0.12)
    return {"criteria": copy_module.deepcopy(verdict.DEFAULT_CRITERIA), "collapse": collapse,
            "verdict": pilot_protocol.verdict_block(), "bootstrap": dict(pilot_protocol.bootstrap_block())}


def test_build_verdict_marks_the_fewshot_path_pending():
    """With no few-shot result (Phase 9), (d)'s few-shot path is PENDING and the record says so; a (d) with neither
    other path met is PENDING, and the verdict is decide's."""
    v = verdict.build_verdict(_hand_protocol(), "a" * 64, _hand_inputs())
    assert v["fewshot"] == {"status": verdict.PENDING, "phase": 9}
    paths = {p["path"]: p for p in v["criteria"]["d"]["paths"]}
    assert paths["fewshot"]["status"] == verdict.PENDING and paths["fewshot"]["mean_delta"] is None
    assert v["criteria"]["d"]["status"] == verdict.PENDING
    s = {k: v["criteria"][k]["status"] for k in ("a", "b", "d", "alternative_go")}
    assert v["verdict"] == verdict.decide(s["a"], s["b"], s["d"], s["alternative_go"])


def test_build_verdict_refuses_rules_other_than_the_code_it_runs():
    """The frozen verdict rules must be the ones pilot.verdict implements, else the verdict would not follow them."""
    protocol = _hand_protocol()
    protocol["verdict"]["rules"][0] = "(b) FAIL -> GO"
    with pytest.raises(SystemExit, match="verdict rules"):
        verdict.build_verdict(protocol, "a" * 64, _hand_inputs())


def test_build_verdict_refuses_per_seed_rows_off_the_listed_seeds():
    inputs = _hand_inputs()
    inputs["systems"]["berturk"]["per_seed"] = inputs["systems"]["berturk"]["per_seed"][:2]
    with pytest.raises(SystemExit, match="berturk.*seeds \\[0, 1\\].*\\[0, 1, 2\\]"):
        verdict.build_verdict(_hand_protocol(), "a" * 64, inputs)


def test_mean_std_is_the_sample_std_and_exact_for_equal_values():
    """A16: ddof 1; equal values give exactly 0.0 and their own value as the mean; fewer than 2 values are refused."""
    mean, std = verdict.mean_std([0.5, 0.5, 0.7])
    assert mean == pytest.approx(0.5666666666666667, abs=1e-15)
    assert std == pytest.approx(float(np.std([0.5, 0.5, 0.7], ddof=1)), abs=1e-15)
    assert verdict.mean_std([0.6, 0.6, 0.6]) == (0.6, 0.0)
    assert verdict.mean_std([0.1, 0.1]) == (0.1, 0.0)
    for bad in ([0.5], [], [0.5, None], [0.5, float("nan")]):
        with pytest.raises(SystemExit):
            verdict.mean_std(bad)
