"""The committed few-shot records: bindings, byte rebuilds and the rules they apply.

The compute projection and decision record:
- results/fewshot/projection.json (fewshot-projection/1) is tracked, bound to the frozen protocol's sha256 and to the
  sha256 of results/compute/compute_checkpoint.json, mode primary with projection_fired false and no ladder failure,
  a total between 8.0 and 8.6 h whose lines sum to it, and pilot.fewshot.read_mode reads primary from it; it holds no
  uid string and no local path.
- the projection and the decision recompute from the committed compute rates and the record's own counts
  (pilot.compute.fewshot_projection, fewshot_decision), offline; the Laya lines use calib_s None (training no longer
  calibrates) and the notes say so.
- the frozen trigger at its boundary: a projection of exactly 36,000 s (10 h) does not fire it ("exceeds 10 h"), one
  microsecond more does, and the fallback carries the frozen disclosure; a recorded ladder failure gives the fallback
  too (base seed 0 only); read_mode accepts every such record.
- `fewshot.py --projection` (synthetic protocol, temp paths): a fresh run refuses an existing --out without writing;
  --ladder-failure TEXT refuses a missing --out, a fallback record, a record of another protocol, a record without a
  projection and an empty text, and rewrites a primary record of this protocol into the fallback record naming the
  failure, everything else unchanged.
- data-marked: `fewshot.py --projection` rebuilds the record byte for byte from the committed data.

The runs manifest and the once-only scoring:
- results/fewshot/runs.json (fewshot-runs/1) is tracked and pilot.fewshot.read_runs accepts it under the frozen pin and
  the recorded compute mode; per kind it lists one masked base per allowed base seed and one continued run per frozen
  (draw, k) (pilot.fewshot.slots), every base proof reads 0 for each held-out tag (Laya held_out_items, BERTurk
  unmasked_held_out_labels), every continued proof counts exactly its frozen support rows per tag and the frozen
  replay size (BERTurk replay_unmasked_held_out_labels 0); its bytes hold no bank_ or hotel_ uid and no local path.
- `make fewshot-scoring` is exactly the command list derived from the committed runs.json: in entry order, per base
  its val line (ungated) then its test line (gated), per continued run its test line, each `score.py ... --model
  <kind> --checkpoint $(FEWSHOT_ROOT)/<checkpoint_name> --device mps --workers 0`; no line holds --force, --n or
  --out-root; the header line, `FEWSHOT_ROOT ?= .` and the .PHONY entry exist.
- every runs.json entry was scored exactly once: a base's dump dir tracks exactly val and test npz + json, a continued
  run's exactly test npz + json (no HOTELS dump: fewshot.rules.test_admission); every dump verifies
  (pilot.dumps.load_dump), is full with the protocol split's message count, was scored on mps, fp32, autocast off,
  in-process, from the entry's checkpoint (base name), with the entry's seed and training run id; every test sidecar
  carries protocol_sha256 == PROTOCOL_SHA256 and fewshot_runs_sha256 == the sha256 of the committed runs.json, a
  base's val sidecar neither (its ungated θ source).
- git order: runs.json has exactly one commit M, M descends from the protocol.json commit, and every tracked few-shot
  test dump file was added in a commit C != M with M an ancestor of C, and each masked base's val dump file exactly
  once (skipped in a shallow clone).

The few-shot results record and the README section:
- results/fewshot/fewshot.json (fewshot-results/1) is tracked and bound: protocol_sha256 the pin, the runs manifest
  and the compute decision record by path and sha256, its mode the decision record's (the disclosure present iff
  fallback), every runs.json entry's dumps by the dump_sha256 of their sidecars; its primary block is at criteria
  (d)'s fewshot k, each per-draw delta is laya - berturk, mean_delta == float(np.mean(draws)) in float64; by_k holds
  k = 0 and every frozen k, the bootstrap block the protocol's B, seed and level, forgetting = old-tag F1 minus the
  base's; its bytes hold no bank_ or hotel_ uid and no local path, written as pilot.fewshot_report.record_bytes.
- README.md's few-shot section (FEWSHOT_HEADING) follows the frozen-protocol section and holds exactly one block
  between <!-- fewshot:start --> and <!-- fewshot:end -->, equal to `python fewshot.py --render
  results/fewshot/fewshot.json`; its prose (heading to the first marker) states what was masked and how it was proven,
  the identical support sets, epoch parity, the once-only scoring behind the allow-list, how to read the primary
  metric and forgetting, the record's compute mode and how the GO block uses the record, and holds no metric value
  (test_final_results.PROSE_NUMBER_RE).
- data-marked: pilot.fewshot_report.build_record rebuilds the committed record byte for byte from the dumps.

The README docs of the few-shot experiment:
- README.md's Files table has a row for every few-shot file (pilot/fewshot.py with its row builders, binding, runs
  manifest and admission, pilot/fewshot_report.py, fewshot.py with every mode, the few-shot flags of ray_train.py and
  berturk_train.py, score.py's few-shot allow-list, report.py --verdict --fewshot, results/fewshot/ and the few-shot
  dumps), and its Run block every few-shot command (each fewshot.py mode, both trainers' base and continued commands,
  make fewshot-scoring marked as run once, make fewshot-reports, the --render and --verdict --fewshot lines).
- the few-shot prose states the caveats a reader needs: BERTurk's new-tag micro-F1 is zero on every continued run
  because its new-tag scores stay below its base's val θ (checked against the committed record), AP is the primary
  metric because it needs no θ, and step parity is deferred.
- the prose names the cause of that zero (the masked BERTurk base's untrained held-out rows near one half on val push
  its 48-column θ up; Laya's θ is the same under both rules) and the record's post hoc variant (θ on each base's
  trained columns only, labelled post hoc, not frozen, descriptive and not a decision input), each claim checked
  against the committed record and dumps; the frozen reading's numbers stay the record's.
Offline except the `data` tests.
"""
import json
import os
import re
import subprocess
import types
from pathlib import Path
from typing import Any

import pytest

import fewshot as runner
from conftest import SyntheticRaw
from fewshot_synthetic import decision_record, synthetic_protocol, write_decision
from pilot import compute
from pilot import dumps
from pilot import fewshot as fs
from pilot import protocol as pr
from pilot.data import REPO_ROOT, sha256_bytes

pytestmark = pytest.mark.unit

PROJECTION = os.path.join(REPO_ROOT, "results", "fewshot", "projection.json")
COMPUTE = os.path.join(REPO_ROOT, "results", "compute", "compute_checkpoint.json")
LINE_IDS = ["fewshot_laya_bases", "fewshot_berturk_bases", "fewshot_laya_continued", "fewshot_berturk_continued",
            "fewshot_scoring"]


def committed() -> tuple[bytes, dict[str, Any]]:
    with open(PROJECTION, "rb") as f:
        raw = f.read()
    return raw, json.loads(raw.decode("utf-8"))


def frozen() -> tuple[dict[str, Any], str]:
    return pr.require_frozen()


def measured() -> dict[str, Any]:
    with open(COMPUTE, encoding="utf-8") as f:
        return json.load(f)["measured"]


def n_scored(proto: dict[str, Any]) -> dict[str, int]:
    return {s: proto["splits"][s]["n_messages"] for s in ("val", "test")}


def test_projection_record_is_committed_and_bound() -> None:
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


def test_projection_recomputes_from_the_committed_rates_and_counts() -> None:
    proto, _ = frozen()
    _, rec = committed()
    block = proto["fewshot"]
    p = compute.fewshot_projection(measured(), rec["counts"], block, n_scored(proto))
    assert json.loads(json.dumps(p)) == rec["projection"]
    d = compute.fewshot_decision(p, block)
    assert {"trigger": rec["trigger"], "decision": rec["decision"]} == json.loads(json.dumps(d))


def test_projection_lines_use_the_measured_terms_without_calibration() -> None:
    proto, _ = frozen()
    _, rec = committed()
    m = measured()
    assert m["laya"]["calib_s"] is not None                   # the compute probe calibrated; few-shot runs do not
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
    assert any("removed training-time calibration" in n and "calib_s None" in n for n in rec["notes"])
    # the fallback: one base per model, the continued runs and their scoring unchanged
    one = 2 * compute.laya_run_s(14580, 3, laya_terms) + 2 * 29 * 46 * m["berturk"]["s_per_step"] + \
        2 * (208 + 413) * (m["scoring"]["s_per_msg"] + m["berturk"]["s_per_msg_infer"])
    assert p["fallback_total_s"] == pytest.approx(p["total_s"] - one)


@pytest.mark.data
def test_projection_rebuilds(tmp_path: Path) -> None:
    raw, _ = committed()
    out = str(tmp_path / "projection.json")
    runner.main(["--projection", "--out", out])
    with open(out, "rb") as f:
        assert f.read() == raw


# ---- the trigger at its boundary, the fallback disclosure and the ladder-failure branch ----

SPLITS = {"val": {"n_messages": 5}, "test": {"n_messages": 7}}


def record_of(pin: str, decision: dict[str, Any]) -> dict[str, Any]:
    return dict({"schema": fs.DECISION_SCHEMA, "protocol_sha256": pin}, **decision)


def test_projection_trigger_boundary(tmp_path: Path) -> None:
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


def test_projection_ladder_failure_gives_the_disclosed_fallback(tmp_path: Path) -> None:
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


def test_projection_fallback_total_is_one_base_per_model() -> None:
    _, rec = committed()
    p = rec["projection"]
    assert p["base_runs_per_model"] == 3 and p["fallback_base_runs_per_model"] == 1
    assert p["fallback_total_s"] < p["total_s"] and p["fallback_total_h"] == pytest.approx(p["fallback_total_s"] / 3600)


def synthetic(tmp_path: Path, synthetic_raw: SyntheticRaw, monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    return synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, splits=SPLITS)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def test_projection_cli_writes_the_record_once(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                               monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
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


def test_projection_cli_ladder_failure_rewrites_a_primary_record(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                                 monkeypatch: pytest.MonkeyPatch,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
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
def test_projection_cli_ladder_failure_refuses(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                               monkeypatch: pytest.MonkeyPatch, case: str, match: str) -> None:
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
def test_projection_cli_refuses_misplaced_flags(argv: list[str], match: str,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        runner.main(argv)
    assert match in capsys.readouterr().err


# ---- the runs manifest and the once-only scoring target ----

RUNS = os.path.join(REPO_ROOT, "results", "fewshot", "runs.json")


def git(*args: str) -> str:
    """stdout of a git command in the repo, captured before any parsing."""
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    return out.stdout


def committed_runs() -> tuple[dict[str, Any], str, bytes]:
    """(runs manifest, its sha256, its bytes) read through pilot.fewshot.read_runs under the frozen pin."""
    proto, pin = frozen()
    runs, sha = fs.read_runs(proto, pin, RUNS)
    assert runs is not None, "results/fewshot/runs.json does not exist"
    return runs, sha, read_bytes(RUNS)


def test_runs_manifest_is_committed_and_valid() -> None:
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


def makefile() -> str:
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        return f.read()


def fewshot_scoring_recipe(text: str) -> list[str]:
    """The recipe lines (tab stripped) of the fewshot-scoring target."""
    m = re.search(r"^fewshot-scoring:\n((?:\t.*\n)+)", text, re.M)
    assert m, "Makefile has no fewshot-scoring target"
    return [line[1:] for line in m.group(1).splitlines()]


def expected_fewshot_lines(runs: dict[str, Any]) -> list[str]:
    """Per entry in runs.json order: a base's val line then its test line, a continued run's test line."""
    out = []
    for e in runs["entries"]:
        for split in (("val", "test") if e["stage"] == "base" else ("test",)):
            out.append("$(PYTHON) score.py data/splits/%s.jsonl --model %s --checkpoint $(FEWSHOT_ROOT)/%s "
                       "--device mps --workers 0" % (split, e["kind"], e["checkpoint_name"]))
    return out


def test_makefile_fewshot_scoring_matches_runs() -> None:
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


FEWSHOT_FILES = {"base": ("test.json", "test.npz", "val.json", "val.npz"), "continued": ("test.json", "test.npz")}
GATE_KEYS = ("protocol_sha256", "fewshot_runs_sha256")


def tracked(*pathspec: str) -> list[str]:
    return [f for f in git("ls-files", "-z", "--", *pathspec).split("\0") if f]


def test_fewshot_scored_exactly_once() -> None:
    proto, pin = frozen()
    runs, runs_sha, _ = committed_runs()
    assert runs_sha == sha256_bytes(read_bytes(RUNS))
    dump_root = os.path.join(REPO_ROOT, "results", "dumps")
    for e in runs["entries"]:
        rid, stage = e["dump_run_id"], e["stage"]
        names = sorted(f.split("/")[3] for f in tracked("results/dumps/%s/" % rid))
        assert names == sorted(FEWSHOT_FILES[stage]), (rid, names)
        for split in (("val", "test") if stage == "base" else ("test",)):
            logits, uids, side, _ = dumps.load_dump(os.path.join(dump_root, rid), split)
            n = proto["splits"][split]["n_messages"]
            assert logits.shape == (n, 48) and len(uids) == n, (rid, split, logits.shape)
            assert side["run_id"] == rid and side["n_subset"] is None, (rid, split)
            assert side["split"]["name"] == split and side["split"]["n_messages"] == n, (rid, split)
            sc = side["scorer"]
            assert (sc["device"], sc["dtype"], sc["amp_enabled"], sc["workers"]) == \
                ("mps", "torch.float32", False, 0), (rid, split, sc)
            assert side["system"]["kind"] == e["kind"], (rid, split)
            assert side["system"]["checkpoint"].rsplit("/", 1)[-1] == e["checkpoint_name"], (rid, split)
            assert side["seed"] == e["seed"], (rid, split, side["seed"])
            assert side["training"]["run_id"] == e["train_run_id"], (rid, split)
            if split == "test":
                assert side.get("protocol_sha256") == pin == pr.PROTOCOL_SHA256, (rid, split)
                assert side.get("fewshot_runs_sha256") == runs_sha, (rid, split)
            else:
                assert not any(key in side for key in GATE_KEYS), (rid, split, "a base's val dump is ungated")
    listed = {e["dump_run_id"] for e in runs["entries"]}
    assert not any("hotels" in f for rid in listed for f in tracked("results/dumps/%s/" % rid))


def require_full_history() -> None:
    """Skip in a shallow clone: its grafted HEAD commit reports every file as added there, and merge-base has no
    history to walk, so the git-order checks need the full history."""
    if git("rev-parse", "--is-shallow-repository").strip() == "true":
        pytest.skip("needs the full git history (git fetch --unshallow)")


def test_fewshot_dumps_descend_from_the_runs_manifest() -> None:
    require_full_history()
    out = git("log", "--format=%H", "--", "results/fewshot/runs.json")
    manifest = out.split()
    assert len(manifest) == 1, manifest
    out = git("log", "--format=%H", "--", "protocol.json")
    freeze = out.split()
    assert len(freeze) == 1, freeze
    r = subprocess.run(["git", "merge-base", "--is-ancestor", freeze[0], manifest[0]], cwd=REPO_ROOT,
                       capture_output=True)
    assert r.returncode == 0, "runs.json (%s) does not descend from the protocol freeze %s" % (manifest[0], freeze[0])
    runs, _, _ = committed_runs()
    paths = ["results/dumps/%s/test.%s" % (e["dump_run_id"], ext) for e in runs["entries"] for ext in ("npz", "json")]
    have = set(tracked("results/dumps/"))
    assert set(paths) <= have, sorted(set(paths) - have)
    for path in paths:
        out = git("log", "--diff-filter=A", "--format=%H", "--", path)
        added = out.split()
        assert len(added) == 1, (path, added)
        assert added[0] != manifest[0], "%s was added in the runs-manifest commit" % path
        r = subprocess.run(["git", "merge-base", "--is-ancestor", manifest[0], added[0]], cwd=REPO_ROOT,
                           capture_output=True)
        assert r.returncode == 0, "%s was added in %s, which does not descend from the runs manifest %s" % (
            path, added[0], manifest[0])
    # each masked base's ungated val dump (its θ source) was added exactly once, never re-added
    vals = ["results/dumps/%s/val.%s" % (e["dump_run_id"], ext) for e in runs["entries"] if e["stage"] == "base"
            for ext in ("npz", "json")]
    assert set(vals) <= have, sorted(set(vals) - have)
    for path in vals:
        added = git("log", "--diff-filter=A", "--format=%H", "--", path).split()
        assert len(added) == 1, (path, added)


# ---- the few-shot results record and the README few-shot section ----

RECORD = os.path.join(REPO_ROOT, "results", "fewshot", "fewshot.json")
DECISION = os.path.join(REPO_ROOT, "results", "fewshot", "projection.json")
README = os.path.join(REPO_ROOT, "README.md")
FEWSHOT_HEADING = "## Few-shot new-intent experiment"
FEWSHOT_START, FEWSHOT_END = "<!-- fewshot:start -->", "<!-- fewshot:end -->"
FINAL_CONF_END = "<!-- final-confidence:end -->"
RENDER_LINE = "Regenerate with `python fewshot.py --render results/fewshot/fewshot.json`."
FEWSHOT_PROSE = ("protocol.json", "unchanged", "no number of its own", "masked", "deleted", "before negative sampling",
                "masked label", "Every message was kept", "runs manifest", "identical frozen support sets", "replay",
                "k = 5 and k = 10", "epoch parity, not step parity", "allow-list", "scored once", "protocol gate",
                "refuses a rerun", "make fewshot-reports", "average precision at k = 10", "mean over the draws",
                "Forgetting", "old-tag micro-F1", "val θ", "never on test", "descriptive", "Compute fallback",
                "Criterion (d)'s few-shot path in the GO block reads this record")
FALLBACK_PROSE = {"primary": "The compute fallback was not used: the decision record reads mode primary",
                  "fallback": "The compute fallback was used: the decision record reads mode fallback"}


def committed_record() -> tuple[dict[str, Any], str, bytes]:
    """(record, its sha256, its bytes) of the tracked results/fewshot/fewshot.json."""
    assert git("ls-files", "--", "results/fewshot/fewshot.json").strip() == "results/fewshot/fewshot.json", \
        "results/fewshot/fewshot.json is not tracked"
    raw = read_bytes(RECORD)
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw), raw


def test_fewshot_record_binds_the_protocol_and_runs() -> None:
    import numpy as np
    from pilot import fewshot_report as fr
    proto, pin = frozen()
    rec, _, raw = committed_record()
    assert rec["schema"] == fr.RESULTS_SCHEMA == "fewshot-results/1"
    assert rec["protocol_sha256"] == pin == pr.PROTOCOL_SHA256
    runs, runs_sha, _ = committed_runs()
    assert rec["runs"] == {"path": "results/fewshot/runs.json", "sha256": runs_sha}
    assert rec["decision"]["path"] == "results/fewshot/projection.json"
    assert rec["decision"]["sha256"] == sha256_bytes(read_bytes(DECISION)) == runs["decision"]["sha256"]
    mode = fs.read_mode(pin)
    assert rec["decision"]["mode"] == mode == runs["mode"]
    disclosure = proto["fewshot"]["rules"]["compute_fallback"]["disclosure"]
    assert rec["decision"]["disclosure"] == (disclosure if mode == "fallback" else None)
    dump_root = os.path.join(REPO_ROOT, "results", "dumps")
    want = {}
    for e in runs["entries"]:
        for split in (("val", "test") if e["stage"] == "base" else ("test",)):
            side = load_json(os.path.join(dump_root, e["dump_run_id"], "%s.json" % split))
            want.setdefault(e["dump_run_id"], {})[split] = side["dump_sha256"]
    assert rec["dumps"] == want
    p = rec["primary"]
    k = [q for q in proto["criteria"]["d"]["paths"] if q["path"] == "fewshot"][0]["k"]
    assert p["k"] == k == proto["fewshot"]["rules"]["primary_metric"]["k"] and p["metric"] == "new_tag_macro_ap"
    assert [d["draw"] for d in p["per_draw"]] == list(range(len(proto["fewshot"]["draws"])))
    for d in p["per_draw"]:
        assert d["delta"] == d["laya"] - d["berturk"], d
    assert p["draws"] == [d["delta"] for d in p["per_draw"]]
    assert p["mean_delta"] == float(np.mean(np.asarray(p["draws"], dtype=np.float64)))
    assert {key: rec["by_k"][str(k)][key] for key in ("per_draw", "draws", "mean_delta")} == \
        {key: p[key] for key in ("per_draw", "draws", "mean_delta")}
    assert sorted(rec["by_k"], key=int) == ["0"] + [str(v) for v in sorted(proto["fewshot"]["k_values"])]
    assert sorted(rec["bootstrap"], key=int) == sorted(rec["by_k"], key=int)
    for key, b in rec["bootstrap"].items():
        assert b["delta"] == rec["by_k"][key]["mean_delta"], key
        assert (b["B"], b["seed"], b["level"]) == (proto["bootstrap"]["B"], proto["bootstrap"]["seed"],
                                                   proto["bootstrap"]["level"]), key
    for kind in fs.KINDS:
        assert set(rec["f1"][kind]["continued"]) == set(rec["wall_clock_s"][kind]), kind
        for key, row in rec["f1"][kind]["continued"].items():
            base = rec["f1"][kind]["base"]["s%d" % row["base_seed"]]
            assert row["forgetting"] == row["old"] - base["old"], (kind, key)
    assert rec["tags"] == list(proto["fewshot"]["tags"]) and rec["k_values"] == list(proto["fewshot"]["k_values"])
    assert [d["digest"] for d in rec["draws"]] == [d["digest"] for d in proto["fewshot"]["draws"]]
    text = raw.decode("utf-8")
    assert "bank_" not in text and "hotel_" not in text
    assert os.path.expanduser("~") not in text and REPO_ROOT not in text
    assert raw == fr.record_bytes(rec)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def readme_text() -> str:
    with open(README, encoding="utf-8") as f:
        return f.read()


def fewshot_section(text: str) -> tuple[int, int]:
    assert text.count(FEWSHOT_HEADING) == 1, "README.md has no single few-shot section"
    i = text.index(FEWSHOT_HEADING)
    j = text.find("\n## ", i + len(FEWSHOT_HEADING))
    return i, (len(text) if j < 0 else j)


def test_readme_fewshot_block_matches_render(capsys: pytest.CaptureFixture[str]) -> None:
    text = readme_text()
    assert text.count(FEWSHOT_START) == 1 and text.count(FEWSHOT_END) == 1, "README.md must hold one fewshot block"
    i, j = text.index(FEWSHOT_START), text.index(FEWSHOT_END)
    lo, hi = fewshot_section(text)
    assert text.index(FINAL_CONF_END) < lo < i < j < hi, "the fewshot block must sit in the few-shot section, after " \
        "the frozen-protocol section"
    assert text.index(RENDER_LINE) < i and text.index(RENDER_LINE) > lo
    capsys.readouterr()
    rendered = runner.main(["--render", RECORD])
    assert capsys.readouterr().out.strip() == rendered.strip()
    assert text[i + len(FEWSHOT_START):j].strip() == rendered.strip(), (
        "the README fewshot block differs from `python fewshot.py --render results/fewshot/fewshot.json`; "
        "regenerate it")


def fewshot_prose(text: str | None = None) -> str:
    """The README's few-shot prose: from the section heading to the first generated block's marker after it."""
    text = readme_text() if text is None else text
    lo, hi = fewshot_section(text)
    j = text.find("<!--", lo)
    assert lo < j < hi, "the few-shot section holds no generated block"
    return " ".join(text[lo:j].split())


def test_readme_fewshot_prose() -> None:
    """The prose states what was masked and how it was proven, the identical support sets and replay, the epoch
    parity, the once-only scoring behind the committed allow-list, how to read the primary metric and forgetting, the
    record's compute mode and how the GO block uses the record, and holds no metric value of its own."""
    from test_final_results import PROSE_NUMBER_RE
    prose = fewshot_prose()
    for want in FEWSHOT_PROSE:
        assert want in prose, "the README few-shot prose must state %r before its generated block" % want
    rec, _, _ = committed_record()
    mode = rec["decision"]["mode"]
    assert FALLBACK_PROSE[mode] in prose, "the few-shot prose must say %r (the record's mode)" % FALLBACK_PROSE[mode]
    other = [m for m in FALLBACK_PROSE if m != mode]
    assert all(FALLBACK_PROSE[m] not in prose for m in other)
    if mode == "fallback":
        assert rec["decision"]["disclosure"] in prose
    hits = [m.group(0) for m in PROSE_NUMBER_RE.finditer(prose)]
    assert hits == [], "the README few-shot prose holds metric values %r; numbers belong in a generated block" % hits


@pytest.mark.data
def test_fewshot_record_rebuilds() -> None:
    from pilot import fewshot_report as fr
    proto, pin = frozen()
    _, _, raw = committed_record()
    assert fr.record_bytes(fr.build_record(proto, pin)) == raw, "results/fewshot/fewshot.json does not rebuild"


# ---- the README Files table and Run block name every few-shot file and command ----

README_FILES = {
    "pilot/fewshot.py": ("masked_rows", "laya_rows", "berturk_rows", "binding", "check_base", "check_continued",
                         "read_mode", "check_runs", "read_runs", "admitted"),
    "pilot/fewshot_report.py": ("fewshot-results/1", "new-tag macro-AP", "forgetting", "bootstrap"),
    "fewshot.py": ("--support-sets", "--counts", "--projection", "--inspect", "--runs-manifest", "--prune",
                   "--drop-storage", "--report", "--render"),
    "ray_train.py": ("--fewshot-base", "--mask-tags", "--fewshot-draw", "--fewshot-k"),
    "berturk_train.py": ("--fewshot-base", "--fewshot-draw", "--fewshot-k"),
    "score.py": ("results/fewshot/runs.json", "test only", "fewshot_runs_sha256"),
    "report.py": ("--verdict --fewshot",),
    "pilot/compute.py": ("fewshot_projection", "fewshot_decision"),
    "results/": ("fewshot/", "projection.json", "runs.json", "fewshot.json", "few-shot dumps"),
}
README_RUN = (
    "python fewshot.py --support-sets",
    "python fewshot.py --counts",
    "python fewshot.py --projection --out results/fewshot/projection.json",
    "--fewshot-base --no-export-epochs --out ckpt-fewshot-laya-base-s0",
    "--fewshot-draw 0 --fewshot-k 10 --no-export-epochs --out ckpt-fewshot-laya-d0-k10",
    "--fewshot-base --no-export-epochs \\\n    --out ckpt-fewshot-berturk-base-s0",
    "--fewshot-draw 0 --fewshot-k 10 --no-export-epochs \\\n    --out ckpt-fewshot-berturk-d0-k10",
    "python fewshot.py --drop-storage",
    "python fewshot.py --inspect",
    "python fewshot.py --runs-manifest CKPT... --out results/fewshot/runs.json",
    "make fewshot-scoring        # ran once; never rerun: the gate refuses an existing dump",
    "python fewshot.py --prune",
    "make fewshot-reports",
    "python fewshot.py --render results/fewshot/fewshot.json",
    "python report.py --verdict --fewshot results/fewshot/fewshot.json --out results/final/verdict.json",
)


def readme_part(text: str, heading: str, end: str) -> str:
    """The README text from `heading` to the next `end` heading after it (exactly one of each)."""
    assert text.count(heading) == 1, "README.md must hold one %r" % heading
    i = text.index(heading)
    j = text.index(end, i + len(heading))
    return text[i:j]


def files_rows(text: str) -> dict[str, str]:
    """{file: description} of the README Files table (a row is | `file` | description |)."""
    rows = {}
    for line in readme_part(text, "\n## Files\n", "\n## Run\n").splitlines():
        m = re.match(r"^\| `([^`]+)` \| (.*) \|$", line)
        if m:
            assert m.group(1) not in rows, "the Files table lists %s twice" % m.group(1)
            rows[m.group(1)] = m.group(2)
    return rows


def run_block(text: str) -> str:
    """The README Run block's fenced code."""
    part = readme_part(text, "\n## Run\n", "\n## Scoring path\n")
    i = part.index("```bash\n")
    return part[i:part.index("\n```", i + 8)]


def test_readme_names_the_fewshot_files_and_commands() -> None:
    text = readme_text()
    rows = files_rows(text)
    for path, wants in README_FILES.items():
        assert path in rows, "the README Files table has no row for %s" % path
        missing = [w for w in wants if w not in rows[path]]
        assert not missing, "the README Files row of %s does not name %s" % (path, ", ".join(missing))
    run = run_block(text)
    missing = [c for c in README_RUN if c not in run]
    assert not missing, "the README Run block does not hold %r" % missing
    i, j = run.index("python ray_train.py --data data/splits/selection_train.jsonl --base multilingual --device mps"), \
        run.index("python berturk_train.py --data data/splits/selection_train.jsonl --base berturk --device mps")
    assert run.index("python fewshot.py --projection") < i < j < run.index("make fewshot-scoring") \
        < run.index("make fewshot-reports"), "the Run block's few-shot commands must be in the order they ran"
    for target in ("fewshot-scoring:", "fewshot-reports:"):
        assert re.search(r"^%s$" % re.escape(target), makefile(), re.M), "the Makefile has no %s target" % target


FEWSHOT_CAVEATS = ("new-tag micro-F1", "zero on every continued run", "below its masked base's val θ",
                  "Average precision needs no θ", "primary metric", "Step parity", "deferred", "fp16-rounded",
                  "The cause is that θ", "held-out classifier rows were never trained", "score near one half on val",
                  "choosing the global θ over all 48 columns pushes it up",
                  "Laya's θ is the same whether its held-out columns are counted or not",
                  "keeps the frozen reading's numbers", "post hoc, not frozen, descriptive and not a decision input",
                  "trained columns only")


def test_readme_fewshot_prose_states_the_caveats() -> None:
    """The caveats the recommendation carries are in the few-shot prose, and the BERTurk new-tag F1 statement is the
    record's."""
    prose = fewshot_prose()
    for want in FEWSHOT_CAVEATS:
        assert want in prose, "the README few-shot prose must state %r" % want
    rec, _, _ = committed_record()
    cont = {k: v for k, v in rec["f1"]["berturk"]["continued"].items()}
    assert cont and all(v["new"] == 0.0 for v in cont.values()), \
        "the prose says BERTurk's new-tag micro-F1 is zero on every continued run; the record disagrees"
    import numpy as np
    from intents import INTENTS
    tags = list(INTENTS)
    cols = [tags.index(t) for t in rec["tags"]]
    by_key = {"d%d-k%d" % (e["draw"], e["k"]): e for e in load_json(RUNS)["entries"]
              if e["kind"] == "berturk" and e["stage"] == "continued"}
    assert set(by_key) == set(cont)
    for key, e in by_key.items():
        theta = rec["theta"]["berturk"]["s%d" % e["base_seed"]]
        assert not set(rec["tags"]) & set(theta["per_tag"]), "a held-out tag has its own θ; the prose says the base θ"
        with np.load(os.path.join(REPO_ROOT, "results", "dumps", e["dump_run_id"], "test.npz")) as z:
            p = 1.0 / (1.0 + np.exp(-z["logits"][:, cols].astype(np.float64)))
        assert p.max() < theta["global"], "BERTurk %s scores a held-out tag at or above its base's val θ" % key


def test_committed_post_hoc_theta_variant_matches_the_prose() -> None:
    """The committed record keeps the frozen reading (theta, f1) and adds theta_trained beside it,
    labelled post hoc, not frozen, descriptive and not a decision input. The prose's claims are the record's: Laya's θ
    (global and per tag) is the same under both rules on every base, so its post hoc F1 rows are its frozen ones;
    every BERTurk base's θ on its trained columns is below its 48-column θ; and the masked BERTurk bases' held-out
    columns score near one half on val (mean P within 0.4 and 0.6, the untrained rows). Criterion (d) reads the
    primary block only (report.fewshot_input), which the variant does not touch (test_fewshot_report)."""
    import numpy as np
    from intents import INTENTS
    from pilot import fewshot_report as fr
    rec, _, _ = committed_record()
    t = rec["theta_trained"]
    assert t["label"] == fr.POST_HOC_LABEL == "post hoc, not frozen, descriptive; not a decision input"
    assert (t["rule"], t["cause"]) == (fr.THETA_TRAINED_RULE, fr.THETA_CAUSE)
    for kind in fs.KINDS:
        assert sorted(t["theta"][kind]) == sorted(rec["theta"][kind])
        assert {s: sorted(v) for s, v in t["f1"][kind].items()} == {s: sorted(v) for s, v in rec["f1"][kind].items()}
    assert t["theta"]["laya"] == rec["theta"]["laya"], "Laya's θ differs between the rules; the prose says it does not"
    assert t["f1"]["laya"] == rec["f1"]["laya"]
    for key, th in rec["theta"]["berturk"].items():
        assert t["theta"]["berturk"][key]["global"] < th["global"], key
    assert any(v["new"] > 0.0 for v in t["f1"]["berturk"]["continued"].values())
    cols = [list(INTENTS).index(x) for x in rec["tags"]]
    for e in load_json(RUNS)["entries"]:
        if (e["kind"], e["stage"]) == ("berturk", "base"):
            with np.load(os.path.join(REPO_ROOT, "results", "dumps", e["dump_run_id"], "val.npz")) as z:
                p = 1.0 / (1.0 + np.exp(-z["logits"][:, cols].astype(np.float64)))
            assert 0.4 < p.mean() < 0.6, (e["dump_run_id"], p.mean())

