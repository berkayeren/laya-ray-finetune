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
"""
import json
import os

import pytest

import pilot.data
import report
from pilot import dumps, metrics
from pilot.data import repo_relpath
from test_confidence import _drop_val, _final_seed, _prior, _write_json, _write_split_dump, fitted, snap  # noqa: F401

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
