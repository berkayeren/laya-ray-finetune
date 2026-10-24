"""The committed Phase 9 few-shot records (FEW-02 and later): bindings, byte rebuilds and the rules they apply.

This file grows through Phase 9: 09-04 adds the D-08 compute projection and decision record, 09-11 to 09-13 add the
runs manifest, the few-shot result record and the README block.

09-04 (D-08, ROADMAP Phase 9 criterion 1):
- results/fewshot/projection.json (fewshot-projection/1) is tracked, bound to the frozen protocol's sha256 and to the
  sha256 of results/compute/compute_checkpoint.json, mode primary with projection_fired false and no ladder failure,
  a total between 8.0 and 8.6 h whose lines sum to it, and pilot.fewshot.read_mode reads primary from it; it holds no
  uid string and no local path.
- the projection and the decision recompute from the committed compute rates and the record's own counts
  (pilot.compute.fewshot_projection, fewshot_decision), offline; the Laya lines use calib_s None (LAYA-05) and the
  notes say so.
- the frozen trigger at its boundary: a projection of exactly 36,000 s (10 h) does not fire it ("exceeds 10 h"), one
  microsecond more does, and the fallback carries the frozen disclosure; a recorded ladder failure gives the fallback
  too (base seed 0 only); read_mode accepts every such record.
- `fewshot.py --projection` (synthetic protocol, temp paths): a fresh run refuses an existing --out without writing;
  --ladder-failure TEXT refuses a missing --out, a fallback record, a record of another protocol, a record without a
  projection and an empty text, and rewrites a primary record of this protocol into the fallback record naming the
  failure, everything else unchanged.
- data-marked: `fewshot.py --projection` rebuilds the record byte for byte from the committed data.

09-11 (D-04, D-05, D-09, D-11, FEW-01, FEW-02):
- results/fewshot/runs.json (fewshot-runs/1) is tracked and pilot.fewshot.read_runs accepts it under the frozen pin and
  the recorded D-08 mode; per kind it lists one masked base per allowed base seed and one continued run per frozen
  (draw, k) (pilot.fewshot.slots), every base proof reads 0 for each held-out tag (Laya held_out_items, BERTurk
  unmasked_held_out_labels), every continued proof counts exactly its frozen support rows per tag and the frozen
  replay size (BERTurk replay_unmasked_held_out_labels 0); its bytes hold no bank_ or hotel_ uid and no local path.
- `make fewshot-scoring` is exactly the command list derived from the committed runs.json: in entry order, per base
  its val line (ungated) then its test line (gated), per continued run its test line, each `score.py ... --model
  <kind> --checkpoint $(FEWSHOT_ROOT)/<checkpoint_name> --device mps --workers 0`; no line holds --force, --n or
  --out-root; the header line, `FEWSHOT_ROOT ?= .` and the .PHONY entry exist.
Offline except the `data` tests.
"""
import json
import os
import re
import subprocess

import pytest

import fewshot as runner
from fewshot_synthetic import decision_record, synthetic_protocol, write_decision
from pilot import compute
from pilot import fewshot as fs
from pilot import protocol as pr
from pilot.data import REPO_ROOT, sha256_bytes

pytestmark = pytest.mark.unit

PROJECTION = os.path.join(REPO_ROOT, "results", "fewshot", "projection.json")
COMPUTE = os.path.join(REPO_ROOT, "results", "compute", "compute_checkpoint.json")
LINE_IDS = ["fewshot_laya_bases", "fewshot_berturk_bases", "fewshot_laya_continued", "fewshot_berturk_continued",
            "fewshot_scoring"]


def committed():
    with open(PROJECTION, "rb") as f:
        raw = f.read()
    return raw, json.loads(raw.decode("utf-8"))


def frozen():
    return pr.require_frozen()


def measured():
    with open(COMPUTE, encoding="utf-8") as f:
        return json.load(f)["measured"]


def n_scored(proto):
    return {s: proto["splits"][s]["n_messages"] for s in ("val", "test")}


def test_projection_record_is_committed_and_bound():
    out = subprocess.run(["git", "ls-files", "--error-unmatch", "results/fewshot/projection.json"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, "results/fewshot/projection.json is not tracked: %s" % out.stderr
    raw, rec = committed()
    assert rec["schema"] == fs.DECISION_SCHEMA == "fewshot-projection/1"
    assert rec["protocol_sha256"] == pr.PROTOCOL_SHA256
    with open(COMPUTE, "rb") as f:
        assert rec["compute"] == {"path": "results/compute/compute_checkpoint.json", "sha256": sha256_bytes(f.read())}
    assert rec["counts"]["protocol_sha256"] == pr.PROTOCOL_SHA256
    assert rec["decision"]["mode"] == "primary" and rec["decision"]["disclosure"] is None
    assert rec["trigger"]["projection_fired"] is False and rec["trigger"]["ladder_failure"] is False
    assert rec["trigger"]["ladder_failure_text"] is None
    p = rec["projection"]
    assert [line["id"] for line in p["lines"]] == LINE_IDS
    assert 8.0 <= p["total_h"] <= 8.6, p["total_h"]
    assert sum(line["seconds"] for line in p["lines"]) == pytest.approx(p["total_s"], rel=1e-12)
    assert p["fallback_total_s"] < p["total_s"]
    assert fs.read_mode(pr.PROTOCOL_SHA256) == "primary"
    text = raw.decode("utf-8")
    assert "bank_" not in text and "hotel_" not in text
    assert os.path.expanduser("~") not in text and REPO_ROOT not in text
    assert raw.endswith(b"\n") and raw == (json.dumps(rec, ensure_ascii=False, indent=2, sort_keys=True)
                                          + "\n").encode("utf-8")


def test_projection_recomputes_from_the_committed_rates_and_counts():
    proto, _ = frozen()
    _, rec = committed()
    block = proto["fewshot"]
    p = compute.fewshot_projection(measured(), rec["counts"], block, n_scored(proto))
    assert json.loads(json.dumps(p)) == rec["projection"]
    d = compute.fewshot_decision(p, block)
    assert {"trigger": rec["trigger"], "decision": rec["decision"]} == json.loads(json.dumps(d))


def test_projection_lines_use_the_measured_terms_without_calibration():
    proto, _ = frozen()
    _, rec = committed()
    m = measured()
    assert m["laya"]["calib_s"] is not None                   # the Phase 4 probe calibrated; Phase 9 runs do not
    p = compute.fewshot_projection(m, rec["counts"], proto["fewshot"], n_scored(proto))
    lines = {line["id"]: line for line in p["lines"]}
    for line in p["lines"]:
        assert line["phase"] == 9 and line["basis"] == "measured" and line["source"] and line["formula"]
    rules = proto["fewshot"]["rules"]
    laya_terms = dict(m["laya"], calib_s=None)
    base = lines["fewshot_laya_bases"]
    assert base["runs"] == 3 and base["epochs"] == rules["base_runs"]["laya"]["stop_after_epoch"] == 3
    assert base["units_per_epoch"] == rec["counts"]["base"]["laya_steps_per_epoch"] == 456
    assert base["seconds"] == pytest.approx(3 * compute.laya_run_s(14580, 3, laya_terms))
    assert "calibration" not in base["formula"] and "calibration" not in base["source"]
    bert = lines["fewshot_berturk_bases"]
    assert bert["runs"] == 3 and bert["epochs"] == 29 and bert["units_per_epoch"] == 46
    assert bert["seconds"] == pytest.approx(3 * 29 * 46 * m["berturk"]["s_per_step"])
    cont = lines["fewshot_laya_continued"]
    assert [(q["draw"], q["k"]) for q in cont["parts"]] == [(c["draw"], c["k"]) for c in rec["counts"]["continued"]]
    assert [q["units_per_epoch"] for q in cont["parts"]] == [c["laya_steps_per_epoch"]
                                                             for c in rec["counts"]["continued"]]
    assert cont["runs"] == 6 and cont["epochs"] == rules["continued_training"]["epochs"] == 10
    bcont = lines["fewshot_berturk_continued"]
    assert [q["units_per_epoch"] for q in bcont["parts"]] == [c["berturk_steps_per_epoch"]
                                                              for c in rec["counts"]["continued"]]
    sc = {(q["model"], q["split"]): q for q in lines["fewshot_scoring"]["parts"]}
    assert {key: (q["runs"], q["units_per_epoch"]) for key, q in sc.items()} == {
        ("laya", "val"): (3, 208), ("laya", "test"): (9, 413), ("berturk", "val"): (3, 208),
        ("berturk", "test"): (9, 413)}
    assert any("LAYA-05" in n and "calib" in n for n in rec["notes"])
    # the fallback: one base per model, the continued runs and their scoring unchanged
    one = 2 * compute.laya_run_s(14580, 3, laya_terms) + 2 * 29 * 46 * m["berturk"]["s_per_step"] + \
        2 * (208 + 413) * (m["scoring"]["s_per_msg"] + m["berturk"]["s_per_msg_infer"])
    assert p["fallback_total_s"] == pytest.approx(p["total_s"] - one)


@pytest.mark.data
def test_projection_rebuilds(tmp_path):
    raw, _ = committed()
    out = str(tmp_path / "projection.json")
    runner.main(["--projection", "--out", out])
    with open(out, "rb") as f:
        assert f.read() == raw


# ---- the trigger at its boundary, the fallback disclosure and the ladder-failure branch (09-04 Task 2) ----

SPLITS = {"val": {"n_messages": 5}, "test": {"n_messages": 7}}


def record_of(pin, decision):
    return dict({"schema": fs.DECISION_SCHEMA, "protocol_sha256": pin}, **decision)


def test_projection_trigger_boundary(tmp_path):
    proto, pin = frozen()
    block = proto["fewshot"]
    fb = block["rules"]["compute_fallback"]
    assert fs.FALLBACK_HOURS * 3600 == 36000
    at = compute.fewshot_decision({"total_s": 36000.0}, block)
    assert at["decision"]["mode"] == "primary" and at["trigger"]["projection_fired"] is False
    assert at["decision"]["disclosure"] is None and at["decision"]["base_seeds"] == [0, 1, 2]
    over = compute.fewshot_decision({"total_s": 36000.000001}, block)
    assert over["decision"]["mode"] == "fallback" and over["trigger"]["projection_fired"] is True
    assert over["trigger"]["ladder_failure"] is False and over["trigger"]["ladder_failure_text"] is None
    assert over["decision"]["disclosure"] == fb["disclosure"] == "the results state that base-model variance is not " \
        "captured"
    assert over["decision"]["rule"] == fb["rule"] and over["decision"]["base_seeds"] == [0]
    assert over["trigger"]["rule"] == fb["trigger"] and over["trigger"]["threshold_hours"] == 10
    for name, d in (("at", at), ("over", over)):
        path = write_decision(str(tmp_path / name / "projection.json"), record_of(pin, d))
        assert fs.read_mode(pin, path) == d["decision"]["mode"]


def test_projection_ladder_failure_gives_the_disclosed_fallback(tmp_path):
    proto, pin = frozen()
    block = proto["fewshot"]
    _, rec = committed()
    d = compute.fewshot_decision(rec["projection"], block, "laya base seed 1: MPS out of memory on encoder_frozen")
    assert d["trigger"]["projection_fired"] is False and d["trigger"]["ladder_failure"] is True
    assert d["trigger"]["ladder_failure_text"] == "laya base seed 1: MPS out of memory on encoder_frozen"
    assert d["decision"]["mode"] == "fallback"
    assert d["decision"]["disclosure"] == block["rules"]["compute_fallback"]["disclosure"]
    path = write_decision(str(tmp_path / "projection.json"), record_of(pin, d))
    assert fs.read_mode(pin, path) == "fallback"
    assert fs.allowed_base_seeds(block, "fallback") == [0] == d["decision"]["base_seeds"]
    for bad in ("", "   ", 3):
        with pytest.raises(SystemExit, match="non-empty text"):
            compute.fewshot_decision(rec["projection"], block, bad)


def test_projection_fallback_total_is_one_base_per_model():
    _, rec = committed()
    p = rec["projection"]
    assert p["base_runs_per_model"] == 3 and p["fallback_base_runs_per_model"] == 1
    assert p["fallback_total_s"] < p["total_s"] and p["fallback_total_h"] == pytest.approx(p["fallback_total_s"] / 3600)


def synthetic(tmp_path, synthetic_raw, monkeypatch):
    return synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, splits=SPLITS)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def test_projection_cli_writes_the_record_once(tmp_path, synthetic_raw, monkeypatch, capsys):
    ns = synthetic(tmp_path, synthetic_raw, monkeypatch)
    out = str(tmp_path / "rec" / "projection.json")
    capsys.readouterr()
    rec = runner.main(["--projection", "--out", out])
    assert json.loads(capsys.readouterr().out) == rec
    assert read_bytes(out) == runner.record_bytes(rec)
    assert rec["protocol_sha256"] == ns.pin and rec["decision"]["mode"] == "primary"
    assert rec["projection"]["n_scored"] == {"val": 5, "test": 7}
    assert fs.read_mode(ns.pin, out) == "primary"
    before = read_bytes(out)
    with pytest.raises(SystemExit, match="recorded once") as e:
        runner.main(["--projection", "--out", out])
    assert "\n" not in str(e.value) and read_bytes(out) == before


def test_projection_cli_ladder_failure_rewrites_a_primary_record(tmp_path, synthetic_raw, monkeypatch, capsys):
    ns = synthetic(tmp_path, synthetic_raw, monkeypatch)
    out = str(tmp_path / "rec" / "projection.json")
    rec = runner.main(["--projection", "--out", out])
    text = "berturk base seed 2: MPS out of memory after one rerun"
    new = runner.main(["--projection", "--ladder-failure", text, "--out", out])
    assert read_bytes(out) == runner.record_bytes(new)
    assert new["decision"]["mode"] == "fallback" and new["trigger"]["ladder_failure_text"] == text
    assert new["trigger"]["ladder_failure"] is True and new["trigger"]["projection_fired"] is False
    assert new["decision"]["disclosure"] == ns.block["rules"]["compute_fallback"]["disclosure"]
    assert {k: v for k, v in new.items() if k not in ("trigger", "decision")} == \
        {k: v for k, v in rec.items() if k not in ("trigger", "decision")}
    assert fs.read_mode(ns.pin, out) == "fallback"
    before = read_bytes(out)
    with pytest.raises(SystemExit, match="already records the fallback"):
        runner.main(["--projection", "--ladder-failure", text, "--out", out])
    assert read_bytes(out) == before


@pytest.mark.parametrize("case,match", [
    ("missing", "not found"),
    ("fallback", "already records the fallback"),
    ("other_pin", "protocol sha256"),
    ("no_projection", "no projection"),
    ("empty_text", "non-empty text"),
])
def test_projection_cli_ladder_failure_refuses(tmp_path, synthetic_raw, monkeypatch, case, match):
    ns = synthetic(tmp_path, synthetic_raw, monkeypatch)
    out = str(tmp_path / "rec" / "projection.json")
    text = "laya base seed 0: failed after encoder_frozen"
    if case == "fallback":
        write_decision(out, decision_record(ns.pin, "fallback"))
    elif case == "other_pin":
        write_decision(out, decision_record("f" * 64, "primary"))
    elif case == "no_projection":
        write_decision(out, decision_record(ns.pin, "primary"))
    elif case == "empty_text":
        runner.main(["--projection", "--out", out])
        text = "  "
    before = read_bytes(out) if os.path.exists(out) else None
    with pytest.raises(SystemExit, match=match) as e:
        runner.main(["--projection", "--ladder-failure", text, "--out", out])
    assert "\n" not in str(e.value)
    assert (read_bytes(out) if os.path.exists(out) else None) == before


@pytest.mark.parametrize("argv,match", [
    (["--projection"], "needs --out"),
    (["--counts", "--out", "x.json"], "belong to --projection"),
    (["--support-sets", "--ladder-failure", "x"], "belong to --projection"),
])
def test_projection_cli_refuses_misplaced_flags(argv, match, capsys):
    with pytest.raises(SystemExit):
        runner.main(argv)
    assert match in capsys.readouterr().err


# ---- 09-11: the runs manifest and the once-only scoring target (D-04, D-05, D-09, D-11) ----

RUNS = os.path.join(REPO_ROOT, "results", "fewshot", "runs.json")


def git(*args):
    """stdout of a git command in the repo, captured before any parsing."""
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    return out.stdout


def committed_runs():
    """(runs manifest, its sha256, its bytes) read through pilot.fewshot.read_runs under the frozen pin."""
    proto, pin = frozen()
    runs, sha = fs.read_runs(proto, pin, RUNS)
    assert runs is not None, "results/fewshot/runs.json does not exist"
    return runs, sha, read_bytes(RUNS)


def test_runs_manifest_is_committed_and_valid():
    out = git("ls-files", "--", "results/fewshot/runs.json")
    assert out.strip() == "results/fewshot/runs.json", "results/fewshot/runs.json is not tracked"
    proto, pin = frozen()
    block = proto["fewshot"]
    mode = fs.read_mode(pin)
    runs, sha, raw = committed_runs()
    assert sha == sha256_bytes(raw)
    assert runs["schema"] == fs.RUNS_SCHEMA == "fewshot-runs/1"
    assert runs["protocol_sha256"] == pin == pr.PROTOCOL_SHA256 and runs["mode"] == mode
    entries = runs["entries"]
    have = [(e["kind"], e["stage"], e["seed"], e["draw"], e["k"]) for e in entries]
    assert have == fs.slots(block, mode), have
    n_bases = len(fs.allowed_base_seeds(block, mode))
    n_cont = len(block["draws"]) * len(block["k_values"])
    for kind in fs.KINDS:
        assert sum(e["kind"] == kind and e["stage"] == "base" for e in entries) == n_bases, kind
        assert sum(e["kind"] == kind and e["stage"] == "continued" for e in entries) == n_cont, kind
    if mode == "primary":
        assert (n_bases, n_cont, len(entries)) == (3, 6, 18)
    tags = sorted(block["tags"])
    assert runs["tags"] == tags and len(tags) == 6
    for e in entries:
        proof, name = e["proof"], e["checkpoint_name"]
        assert name == ("ckpt-fewshot-%s-base-s%d" % (e["kind"], e["seed"]) if e["stage"] == "base" else
                        "ckpt-fewshot-%s-d%d-k%d" % (e["kind"], e["draw"], e["k"])), name
        if e["stage"] == "base":
            key = "held_out_items" if e["kind"] == "laya" else "unmasked_held_out_labels"
            assert sorted(proof) == [key], (name, sorted(proof))
            assert proof[key] == {t: 0 for t in tags}, (name, proof[key])
            continue
        sel = fs.selection(block, e["draw"], e["k"])
        n = len(sel["support"])
        assert proof["support_rows"] == n and proof["replay_rows"] == len(sel["replay"]), name
        key = "new_tag_items" if e["kind"] == "laya" else "new_tag_labels"
        assert proof[key] == {t: n for t in tags}, (name, proof[key])
        if e["kind"] == "berturk":
            assert proof["replay_unmasked_held_out_labels"] == 0, name
        assert e["base_seed"] == fs.base_seed_of(block, e["draw"], mode), name
    for kind in fs.KINDS:                           # both models train on the same rows per (draw, k)
        rows = {(e["draw"], e["k"]): (e["proof"]["support_rows"], e["proof"]["replay_rows"])
                for e in entries if e["kind"] == kind and e["stage"] == "continued"}
        assert rows == {(e["draw"], e["k"]): (e["proof"]["support_rows"], e["proof"]["replay_rows"])
                        for e in entries if e["kind"] == fs.KINDS[0] and e["stage"] == "continued"}
    text = raw.decode("utf-8")
    assert "bank_" not in text and "hotel_" not in text
    assert os.path.expanduser("~") not in text and REPO_ROOT not in text
    assert raw.endswith(b"\n") and raw == (json.dumps(runs, ensure_ascii=False, indent=2, sort_keys=True)
                                          + "\n").encode("utf-8")


def makefile():
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        return f.read()


def fewshot_scoring_recipe(text):
    """The recipe lines (tab stripped) of the fewshot-scoring target."""
    m = re.search(r"^fewshot-scoring:\n((?:\t.*\n)+)", text, re.M)
    assert m, "Makefile has no fewshot-scoring target"
    return [line[1:] for line in m.group(1).splitlines()]


def expected_fewshot_lines(runs):
    """Per entry in runs.json order: a base's val line then its test line, a continued run's test line."""
    out = []
    for e in runs["entries"]:
        for split in (("val", "test") if e["stage"] == "base" else ("test",)):
            out.append("$(PYTHON) score.py data/splits/%s.jsonl --model %s --checkpoint $(FEWSHOT_ROOT)/%s "
                       "--device mps --workers 0" % (split, e["kind"], e["checkpoint_name"]))
    return out


def test_makefile_fewshot_scoring_matches_runs():
    proto, _ = frozen()
    assert proto["scoring"]["device"] == "mps" and proto["scoring"]["workers"] == 0
    runs, _, _ = committed_runs()
    text = makefile()
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    assert phony and "fewshot-scoring" in phony.group(1).split(), "fewshot-scoring is not in .PHONY"
    assert re.search(r"^FEWSHOT_ROOT \?= \.$", text, re.M), "Makefile has no `FEWSHOT_ROOT ?= .` line"
    assert re.search(r"^#   make fewshot-scoring  score every few-shot checkpoint of results/fewshot/runs.json once ",
                     text, re.M), "Makefile header has no fewshot-scoring line"
    lines = fewshot_scoring_recipe(text)
    want = expected_fewshot_lines(runs)
    assert lines == want, lines
    n_bases = sum(e["stage"] == "base" for e in runs["entries"])
    assert len(lines) == len(runs["entries"]) + n_bases
    if runs["mode"] == "primary":
        assert len(lines) == 24
    for line in lines:
        words = line.split()
        assert "--force" not in words and "--n" not in words and "--out-root" not in words, line
        assert "hotels_probe" not in line, line
