"""The once-only scoring run of the six final seeds (plan 08-11, REP-01, ROADMAP Phase 8 criterion 4).

- `make final-scoring` is the exact command list: per protocol final seed (pilot.protocol.require_frozen(), Laya then
  BERTurk, seeds in protocol order) one score.py line on the committed test split and one on hotels_probe, each
  naming the seed's checkpoint_name under $(FINAL_ROOT) with --device mps --workers 0 (the protocol's scoring block),
  then the ungated zero-shot Laya HOTELS reference (08-RESEARCH A15) on its val dump's scorer; no line holds --force,
  --n or --out-root (the gate refuses them anyway)
- each final seed was scored exactly once: its dump dir tracks exactly test and hotels_probe npz + json, every dump
  verifies (pilot.dumps.load_dump, offline), is full (n_subset None, the protocol's message count), carries
  protocol_sha256 == pilot.protocol.PROTOCOL_SHA256, was scored on mps, fp32, autocast off, in-process, with the
  seed, the postfit record sha256 and the chosen T (1e-12) of the protocol, from the seed's checkpoint (base name);
  the zero-shot HOTELS reference is tracked, ungated (no protocol_sha256), on its val dump's scorer (cpu, 2 workers)
- git order (REP-01): protocol.json has exactly one commit, and it is a strict ancestor of the commit that added each
  gated dump file

The published verdict (plan 08-12, REP-02, REP-03, ROADMAP Phase 8 criteria 4 and 5):
- `make final-reports` (no model) is the exact command list: LAYA_FINAL and BERTURK_FINAL are the protocol's final
  dump run ids in protocol order; per final seed one `report.py RUN_DIR --split test --postfit RECORD` line into
  results/final/reports/<id>-test.json, then `report.py --verdict --out results/final/verdict.json`
- results/final/verdict.json is verdict/1 of the frozen protocol (protocol.sha256 == PROTOCOL_SHA256); every criterion
  status and the overall verdict recompute with pilot.verdict's functions from the record's own numbers and the
  frozen criteria block; the six per-seed reports sit at their postfit record's θ (theta_source bound by sha256, the
  shipped val_selected θ == the record's theta.shipped, in_sample false) and their metrics are the verdict's
  per-seed rows
- README.md's Phase 8 section holds exactly one GO block, equal to pilot.verdict.render_go of the committed verdict,
  one final-results block before it, equal to pilot.verdict.render_final_table (every fine-tuned row shows its three
  per-seed values in seed order: no best seed), and one final-confidence block after it, equal to
  `report.py --confidence-table results/final/confidence` with its diagram links under results/final/confidence/
- results/final/confidence/ holds exactly 15 confidence/1 records and 8 reliability SVGs: every final seed on test
  (with its SVG == report.render_svg) and hotels_probe, bound final_seed to its system's postfit record by sha256,
  zero-shot Laya on test and hotels_probe and the label prior on test, bound val_selected; never in-sample, a HOTELS
  record holds ood_fp on the 22 banking-only columns and no calibration block; `make final-reports` lists them
- data marker: verdict.json is report.verdict_report() serialised as report.py writes it, byte for byte, every
  per-seed report is report.run_report(..., postfit=RECORD) and every confidence record (and SVG) is
  report.confidence_report of its dump and binding, byte for byte (needs `make data`)
"""
import glob
import json
import os
import re
import subprocess

import pytest

import pilot.verdict
import report
from pilot import dumps, pins, protocol
from pilot.data import DEFAULT_MANIFEST

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATED_SPLITS = ("test", "hotels_probe")
GATED_FILES = ("test.npz", "test.json", "hotels_probe.npz", "hotels_probe.json")
ZERO_SHOT = "laya-zeroshot-7b928d82"
DUMPS = os.path.join(REPO_ROOT, "results", "dumps")
ZERO_SHOT_LINE = ("$(PYTHON) score.py data/splits/hotels_probe.jsonl --model laya --workers $(SCORE_WORKERS) "
                  "--cpus-per-worker $(SCORE_CPUS)")


def makefile():
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        return f.read()


def final_scoring_recipe(text):
    """The recipe lines (tab stripped) of the final-scoring target."""
    m = re.search(r"^final-scoring:\n((?:\t.*\n)+)", text, re.M)
    assert m, "Makefile has no final-scoring target"
    return [line[1:] for line in m.group(1).splitlines()]


def expected_gated_lines(proto):
    out = []
    for kind in protocol.KINDS:
        for s in protocol.final_seeds(proto, kind):
            for split in GATED_SPLITS:
                out.append("$(PYTHON) score.py data/splits/%s.jsonl --model %s --checkpoint $(FINAL_ROOT)/%s "
                           "--device mps --workers 0" % (split, kind, s["checkpoint_name"]))
    return out


def test_makefile_final_scoring_matches_the_protocol():
    proto, _ = protocol.require_frozen()
    assert proto["scoring"]["device"] == "mps" and proto["scoring"]["workers"] == 0
    assert tuple(proto["scoring"]["splits"]) == GATED_SPLITS
    text = makefile()
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    assert phony and "final-scoring" in phony.group(1).split(), "final-scoring is not in .PHONY"
    assert re.search(r"^FINAL_ROOT \?= \.$", text, re.M), "Makefile has no `FINAL_ROOT ?= .` line"
    lines = final_scoring_recipe(text)
    assert len(lines) == 13, lines
    gated = expected_gated_lines(proto)
    assert len(gated) == 12 and lines[:12] == gated, lines[:12]
    names = {s["checkpoint_name"] for kind in protocol.KINDS for s in protocol.final_seeds(proto, kind)}
    assert len(names) == 6 and all(sum(n == l.split("$(FINAL_ROOT)/")[1].split()[0] for l in lines[:12]) == 2
                                   for n in names)
    assert lines[12] == ZERO_SHOT_LINE and "--checkpoint" not in lines[12]
    for line in lines:
        words = line.split()
        assert "--force" not in words and "--n" not in words and "--out-root" not in words, line


def git(*args):
    """stdout of a git command in the repo, captured before any parsing."""
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    return out.stdout


def tracked(*pathspec):
    return [f for f in git("ls-files", "-z", "--", *pathspec).split("\0") if f]


def gated_paths(proto):
    return ["results/dumps/%s/%s" % (s["dump_run_id"], name) for kind in protocol.KINDS
            for s in protocol.final_seeds(proto, kind) for name in GATED_FILES]


def test_final_seeds_scored_exactly_once():
    proto, pin = protocol.require_frozen()
    assert pin == protocol.PROTOCOL_SHA256
    seeds = [(kind, s) for kind in protocol.KINDS for s in protocol.final_seeds(proto, kind)]
    assert len(seeds) == 6
    for kind, s in seeds:
        rid = s["dump_run_id"]
        names = sorted(f.split("/")[3] for f in tracked("results/dumps/%s/" % rid))
        assert names == sorted(GATED_FILES), (rid, names)
        system = proto["systems"][kind]
        for split in GATED_SPLITS:
            logits, uids, side, _ = dumps.load_dump(os.path.join(DUMPS, rid), split)
            n = proto["splits"][split]["n_messages"]
            assert logits.shape == (n, 48) and len(uids) == n, (rid, split, logits.shape)
            assert side["run_id"] == rid and side["n_subset"] is None, (rid, split)
            assert side["split"]["name"] == split and side["split"]["n_messages"] == n, (rid, split)
            assert side.get("protocol_sha256") == pin, (rid, split, side.get("protocol_sha256"))
            sc = side["scorer"]
            assert (sc["device"], sc["dtype"], sc["amp_enabled"], sc["workers"]) == \
                ("mps", "torch.float32", False, 0), (rid, split, sc)
            assert side["system"]["kind"] == kind, (rid, split)
            assert side["seed"] == s["seed"], (rid, split, side["seed"])
            assert side["training"]["postfit_sha256"] == system["postfit"]["sha256"], (rid, split)
            assert side["training"]["split"] == "final_train", (rid, split)
            assert abs(side["temperature"]["shipped"] - system["chosen"]["temperature"]) <= 1e-12, (rid, split)
            assert side["system"]["checkpoint"].rsplit("/", 1)[-1] == s["checkpoint_name"], (rid, split)
    zs = sorted(f.split("/")[3] for f in tracked("results/dumps/%s/" % ZERO_SHOT))
    assert {"hotels_probe.npz", "hotels_probe.json"} <= set(zs), zs
    logits, uids, side, _ = dumps.load_dump(os.path.join(DUMPS, ZERO_SHOT), "hotels_probe")
    n = proto["splits"]["hotels_probe"]["n_messages"]
    assert logits.shape == (n, 48) and side["n_subset"] is None and side["run_id"] == ZERO_SHOT
    assert "protocol_sha256" not in side, "the zero-shot HOTELS reference is ungated (08-RESEARCH A15)"
    with open(os.path.join(DUMPS, ZERO_SHOT, "val.json"), encoding="utf-8") as f:
        val = json.load(f)
    assert side["scorer"]["device"] == "cpu" and side["scorer"]["workers"] == 2
    for key in ("device", "dtype", "amp_enabled", "workers", "torch_threads"):
        assert side["scorer"][key] == val["scorer"][key], key
    assert side["system"]["checkpoint_identity"] == val["system"]["checkpoint_identity"]


def test_protocol_json_has_one_commit():
    out = git("log", "--format=%H", "--", "protocol.json")
    commits = out.split()
    assert len(commits) == 1, commits


def test_gated_dumps_descend_from_the_freeze():
    proto, _ = protocol.require_frozen()
    out = git("log", "--format=%H", "--", "protocol.json")
    freeze = out.split()
    assert len(freeze) == 1, freeze
    paths = gated_paths(proto) + ["results/dumps/%s/hotels_probe.%s" % (ZERO_SHOT, ext) for ext in ("npz", "json")]
    assert set(paths) <= set(tracked("results/dumps/")), sorted(set(paths) - set(tracked("results/dumps/")))
    for path in paths:
        out = git("log", "--diff-filter=A", "--format=%H", "--", path)
        added = out.split()
        assert len(added) == 1, (path, added)
        assert added[0] != freeze[0], "%s was added in the freeze commit" % path
        r = subprocess.run(["git", "merge-base", "--is-ancestor", freeze[0], added[0]], cwd=REPO_ROOT,
                           capture_output=True)
        assert r.returncode == 0, "%s was added in %s, which does not descend from the freeze %s" % (
            path, added[0], freeze[0])


# ---- 08-12: the published verdict, per-seed reports and README blocks ----

FINAL = os.path.join(REPO_ROOT, "results", "final")
VERDICT = os.path.join(FINAL, "verdict.json")
README = os.path.join(REPO_ROOT, "README.md")
PHASE8_HEADING = "## Frozen protocol and head-to-head verdict (Phase 8)"
GO_START, GO_END = "<!-- go-block:start -->", "<!-- go-block:end -->"
PROTOCOL_END = "<!-- protocol:end -->"
FINAL_VAR = {"laya": "LAYA_FINAL", "berturk": "BERTURK_FINAL"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def committed_verdict():
    assert "results/final/verdict.json" in tracked("results/final/"), "results/final/verdict.json is not tracked"
    return load(VERDICT)


def makefile_list(text, name):
    m = re.search(r"^%s := (.+)$" % name, text, re.M)
    assert m, "Makefile has no %s := line" % name
    return m.group(1).split()


def recipe(text, target):
    m = re.search(r"^%s:\n((?:\t.*\n)+)" % re.escape(target), text, re.M)
    assert m, "Makefile has no %s target" % target
    return [line[1:] for line in m.group(1).splitlines()]


def final_seeds():
    proto, _ = protocol.require_frozen()
    return [(kind, s) for kind in protocol.KINDS for s in protocol.final_seeds(proto, kind)]


def report_lines():
    proto, _ = protocol.require_frozen()
    return ["$(PYTHON) report.py results/dumps/%s --split test --postfit %s --out $(FINAL)/reports/%s-test.json "
            "> /dev/null" % (s["dump_run_id"], proto["systems"][kind]["postfit"]["path"], s["dump_run_id"])
            for kind, s in final_seeds()]


VERDICT_LINE = "$(PYTHON) report.py --verdict --out $(FINAL)/verdict.json > /dev/null"
CONF_SPLITS = ("test", "hotels_probe")
REFERENCES = ((ZERO_SHOT, ("test", "hotels_probe")), ("prior", ("test",)))   # bound val_selected (the val rule)


def confidence_lines():
    proto, _ = protocol.require_frozen()
    out = ["$(PYTHON) report.py --confidence results/dumps/%s --split %s --postfit %s --out "
           "$(FINAL)/confidence/%s-%s.json > /dev/null" % (s["dump_run_id"], split,
                                                          proto["systems"][kind]["postfit"]["path"], s["dump_run_id"],
                                                          split)
           for kind, s in final_seeds() for split in CONF_SPLITS]
    for split in ("test", "hotels_probe"):
        out.append("$(PYTHON) report.py --confidence results/dumps/$(LAYA_ZS) --split %s --out "
                   "$(FINAL)/confidence/$(LAYA_ZS)-%s.json > /dev/null" % (split, split))
    out.append("$(PYTHON) report.py --confidence results/dumps/prior --split test --out "
               "$(FINAL)/confidence/prior-test.json > /dev/null")
    return out


def expected_confidence():
    """{run_id-split: (system kind or None for a reference, split)} of the 15 committed confidence records."""
    out = {"%s-%s" % (s["dump_run_id"], split): (kind, split) for kind, s in final_seeds() for split in CONF_SPLITS}
    for run_id, splits in REFERENCES:
        for split in splits:
            out["%s-%s" % (run_id, split)] = (None, split)
    return out


def phase8_section(text):
    assert text.count(PHASE8_HEADING) == 1, "README.md has no single Phase 8 section"
    i = text.index(PHASE8_HEADING)
    j = text.find("\n## ", i + len(PHASE8_HEADING))
    return i, (len(text) if j < 0 else j)


def readme_block(start, end):
    """(block text between the markers, its start offset) after checking one marker pair inside the Phase 8 section."""
    with open(README, encoding="utf-8") as f:
        text = f.read()
    assert text.count(start) == 1 and text.count(end) == 1, "README.md must hold exactly one %s pair" % start
    i, j = text.index(start), text.index(end)
    assert i < j, "%s comes after its end marker" % start
    lo, hi = phase8_section(text)
    assert lo < i and j < hi, "%s is not inside the Phase 8 section" % start
    return text[i + len(start):j], i, text


def test_makefile_final_reports_matches_the_protocol():
    text = makefile()
    for kind, name in FINAL_VAR.items():
        want = [s["dump_run_id"] for k, s in final_seeds() if k == kind]
        assert makefile_list(text, name) == want, "%s is not the protocol's %s final dump run ids" % (name, kind)
    assert re.search(r"^FINAL := results/final$", text, re.M), "Makefile has no `FINAL := results/final` line"
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    assert phony and "final-reports" in phony.group(1).split(), "final-reports is not in .PHONY"
    lines = recipe(text, "final-reports")
    want = report_lines()
    assert lines[:2] == ["mkdir -p $(FINAL)/reports", "mkdir -p $(FINAL)/confidence"], lines[:2]
    assert lines[2:2 + len(want)] == want, lines[2:2 + len(want)]
    assert lines[2 + len(want)] == VERDICT_LINE, lines[2 + len(want)]
    assert lines[3 + len(want):] == confidence_lines(), lines[3 + len(want):]
    for line in lines:
        assert "score.py" not in line and "--force" not in line.split(), "final-reports runs no model: %s" % line


def test_verdict_binds_the_frozen_protocol():
    proto, pin = protocol.require_frozen()
    v = committed_verdict()
    assert v["schema"] == pilot.verdict.VERDICT_SCHEMA == "verdict/1"
    assert v["protocol"] == {"path": "protocol.json", "sha256": pin} and pin == protocol.PROTOCOL_SHA256
    crit = proto["criteria"]
    h = v["seed_averaged"]["headline"]["metrics"]
    assert v["seed_averaged"]["headline"]["a"] == "laya" and v["seed_averaged"]["headline"]["b"] == "berturk"
    a = pilot.verdict.criterion_a(h["micro_f1"], h["macro_f1"], crit["a"])
    alt = pilot.verdict.alternative_go(h["micro_f1"], crit["alternative_go"])
    cal = [p for p in crit["d"]["paths"] if p["path"] == "calibration"][0]["metric"]
    abst = [p for p in crit["d"]["paths"] if p["path"] == "abstention"][0]["metric"]
    d = pilot.verdict.criterion_d(h[cal], h[abst], None, crit["d"])
    rows = v["collapse"]["rows"]
    for r in rows:                                          # every collapse component recomputes from its own numbers
        for c in r["components"]:
            if c["component"] == "val_floor":
                got = pilot.verdict.collapse_val(c["value"], c["floor"])
            elif c["component"] == "test_floor":
                got = pilot.verdict.collapse_test_floor(c["value"], c["floor"])
            else:
                continue
            assert got["collapsed"] == c["collapsed"], (r["run"], r["kind"], r["seed"], c)
        assert r["collapsed"] == any(c["collapsed"] for c in r["components"]), r
    for r in rows:
        floor = proto["collapse"]["prior_val_fraction" if r["run"] == "selection" else "prior_test_fraction"]
        assert all(c["floor"] == floor for c in r["components"] if c["component"] != "loss"), r
    for kind, s in final_seeds():                           # the test floor is the seed's own counts
        row = [r for r in rows if r["run"] == "final" and r["kind"] == kind and r["seed"] == s["seed"]]
        per = [p for p in v["systems"][kind]["per_seed"] if p["seed"] == s["seed"]]
        assert len(row) == 1 and len(per) == 1, (kind, s["seed"])
        n = per[0]["test_counts"]
        tf = [c for c in row[0]["components"] if c["component"] == "test_floor"][0]
        assert pilot.verdict.parse_fraction(tf["value"]) == pilot.verdict.micro_fraction(n["tp"], n["fp"], n["fn"])
    seeds = {k: [p["seed"] for p in v["systems"][k]["per_seed"]] for k in protocol.KINDS}
    assert seeds == {k: list(proto["systems"][k]["seeds"]) for k in protocol.KINDS}, seeds
    b = pilot.verdict.criterion_b(rows, seeds, crit["b"])
    for key, got in (("a", a), ("b", b), ("d", d), ("alternative_go", alt)):
        assert v["criteria"][key]["status"] == got["status"], (key, v["criteria"][key]["status"], got["status"])
    assert v["criteria"]["c"]["status"] == pilot.verdict.REPORTED
    want = pilot.verdict.decide(a["status"], b["status"], d["status"], alt["status"])
    assert v["verdict"] == want, (v["verdict"], want)
    assert v["fewshot"]["status"] == pilot.verdict.PENDING


def test_per_seed_reports_match_the_verdict():
    proto, _ = protocol.require_frozen()
    v = committed_verdict()
    names = sorted(f for f in tracked("results/final/reports/"))
    want = sorted("results/final/reports/%s-test.json" % s["dump_run_id"] for _, s in final_seeds())
    assert names == want, names
    for kind, s in final_seeds():
        r = load(os.path.join(REPO_ROOT, "results/final/reports/%s-test.json" % s["dump_run_id"]))
        system = proto["systems"][kind]
        record = load(os.path.join(REPO_ROOT, system["postfit"]["path"]))
        assert r["run_id"] == s["dump_run_id"] and r["split"]["name"] == "test" and r["subset"] is False
        assert r["split"]["sha256"] == proto["splits"]["test"]["sha256"]
        src = r["theta_source"]
        assert src["postfit"]["sha256"] == system["postfit"]["sha256"] and src["postfit"]["system_kind"] == kind
        assert src["split"] == "val" and src["chosen"] == record["chosen"]["dump_run_id"], src
        for variant, key in (("shipped", "shipped"), ("raw", "raw")):
            row = r["variants"][variant]["rows"]["val_selected"]
            assert row["in_sample"] is False, (s["dump_run_id"], variant)
            assert row["theta"] == record["chosen"]["theta"][key], (s["dump_run_id"], variant)
        assert abs(r["variants"]["shipped"]["temperature"] - record["chosen"]["temperature"]) <= 1e-12
        per = [p for p in v["systems"][kind]["per_seed"] if p["seed"] == s["seed"]]
        assert len(per) == 1 and per[0]["dump_run_id"] == s["dump_run_id"], (kind, s["seed"])
        row = r["variants"]["shipped"]["rows"]["val_selected"]
        for m in pilot.verdict.TABLE_METRICS + ("n_top1",):
            assert per[0][m] == row[m], (s["dump_run_id"], m, per[0][m], row[m])
        assert per[0]["dump_sha256"] == r["dump_sha256"], s["dump_run_id"]


def test_readme_go_block_matches_render():
    block, i, text = readme_block(GO_START, GO_END)
    v = committed_verdict()
    assert block.strip() == pilot.verdict.render_go(v).strip(), (
        "the README GO block differs from pilot.verdict.render_go of results/final/verdict.json; regenerate it with "
        "`python report.py --go-block results/final/verdict.json`")
    assert text.index(PROTOCOL_END) < i, "the GO block must follow the frozen protocol block"
    assert "**Verdict: %s**" % v["verdict"]["verdict"] in block


def _need_data():
    if not glob.glob(os.path.join(REPO_ROOT, "data", "splits", "*.jsonl")):
        pytest.fail("data/splits/ is empty; run `make data` first")


def as_written(obj):
    """The bytes report.py writes for a JSON report with --out."""
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


@pytest.mark.data
def test_final_results_rebuild():
    _need_data()
    proto, _ = protocol.require_frozen()
    with open(VERDICT, "rb") as f:
        assert as_written(report.verdict_report()) == f.read(), "results/final/verdict.json does not rebuild"
    for kind, s in final_seeds():
        rel = "results/final/reports/%s-test.json" % s["dump_run_id"]
        got = report.run_report(os.path.join(DUMPS, s["dump_run_id"]), "test", DEFAULT_MANIFEST,
                                postfit=os.path.join(REPO_ROOT, proto["systems"][kind]["postfit"]["path"]))
        with open(os.path.join(REPO_ROOT, rel), "rb") as f:
            assert as_written(got) == f.read(), "%s does not rebuild" % rel


FINAL_RESULTS_START, FINAL_RESULTS_END = "<!-- final-results:start -->", "<!-- final-results:end -->"
FINAL_CONF_START, FINAL_CONF_END = "<!-- final-confidence:start -->", "<!-- final-confidence:end -->"
FINAL_CONF_DIR = "results/final/confidence"


def final_confidence():
    """[(repo-relative path, parsed JSON)] of the tracked results/final/confidence/*.json, in sorted path order."""
    out = []
    for rel in sorted(f for f in tracked(FINAL_CONF_DIR + "/") if f.endswith(".json")):
        out.append((rel, load(os.path.join(REPO_ROOT, rel))))
    return out


def test_readme_final_results_block_matches_render():
    block, i, text = readme_block(FINAL_RESULTS_START, FINAL_RESULTS_END)
    assert block.strip() == pilot.verdict.render_final_table(committed_verdict()).strip(), (
        "the README final-results block differs from pilot.verdict.render_final_table of results/final/verdict.json; "
        "regenerate it with `python report.py --final-table results/final/verdict.json`")
    assert text.index(PROTOCOL_END) < i < text.index(GO_START), "the results table must sit between protocol and GO"


def test_results_table_shows_every_seed():
    """REP-02, T-08-38: each fine-tuned row holds its three per-seed values in seed order (never a best seed)."""
    v = committed_verdict()
    table = pilot.verdict.render_final_table(v)
    for kind in protocol.KINDS:
        per = v["systems"][kind]["per_seed"]
        assert [p["seed"] for p in per] == [0, 1, 2], kind
        lines = [l for l in table.splitlines() if l.startswith("| %s fine-tuned (final seeds 0, 1, 2) |" % kind)]
        assert len(lines) == 1, (kind, lines)
        cells = [c.strip() for c in lines[0].strip("|").split("|")]
        for m, cell in zip(pilot.verdict.TABLE_METRICS, cells[2:]):
            mean, std = pilot.verdict.mean_std([p[m] for p in per])
            want = "%.3f ± %.3f (%s)" % (mean, std, " / ".join("%.3f" % p[m] for p in per))
            assert cell == want, (kind, m, cell, want)


def test_final_confidence_records():
    proto, _ = protocol.require_frozen()
    records = final_confidence()
    want = expected_confidence()
    assert sorted(rel for rel, _ in records) == sorted("%s/%s.json" % (FINAL_CONF_DIR, k) for k in want), \
        [rel for rel, _ in records]
    assert len(records) == 15
    svgs = sorted(f for f in tracked(FINAL_CONF_DIR + "/") if f.endswith(".svg"))
    want_svgs = sorted("%s/%s.svg" % (FINAL_CONF_DIR, k) for k, (_, split) in want.items() if split == "test")
    assert svgs == want_svgs and len(svgs) == 8, svgs
    for rel, r in records:
        key = os.path.basename(rel)[:-len(".json")]
        kind, split = want[key]
        assert r["schema"] == report.CONFIDENCE_SCHEMA and r["split"]["name"] == split, rel
        assert r["in_sample"] is False and set(r["variants"]) == {"headline", "raw"}, rel
        assert r["variants"]["raw"]["temperature"] == 1.0, rel
        b = r["binding"]
        assert b["theta_source"]["split"] == "val", (rel, b)
        if kind is not None:                                # a final seed: its system's postfit record, by sha256
            system = proto["systems"][kind]
            record = load(os.path.join(REPO_ROOT, system["postfit"]["path"]))
            assert b["kind"] == "final_seed" and r["system"]["kind"] == kind, rel
            assert b["postfit"] == {"path": system["postfit"]["path"], "sha256": system["postfit"]["sha256"],
                                    "system_kind": kind}, (rel, b)
            assert b["theta_source"]["postfit_sha256"] == system["postfit"]["sha256"], rel
            assert b["theta_source"]["chosen"] == record["chosen"]["dump_run_id"], rel
            assert r["training"]["split"] == "final_train", rel
            assert abs(r["variants"]["headline"]["temperature"] - record["chosen"]["temperature"]) <= 1e-12, rel
        else:                                               # a reference: run_report's val rule on its own val dump
            assert b["kind"] == "val_selected" and b["postfit"] is None and r["training"] is None, (rel, b)
        for name, var in r["variants"].items():
            if split == "hotels_probe":
                assert "calibration" not in var and "selective" not in var and "no_intent_fp" not in var, (rel, name)
                for row in var["ood_fp"].values():
                    assert len(row["cols"]) == 22 and set(row["cols"]) == set(pins.OOD_SCORING_TAGS), (rel, name)
                    assert row["n"] == proto["splits"][split]["n_messages"], (rel, name)
            else:
                assert {"calibration", "selective", "no_intent_fp"} <= set(var) and "ood_fp" not in var, (rel, name)
        if split == "test":
            with open(os.path.join(REPO_ROOT, rel[:-len(".json")] + ".svg"), encoding="utf-8") as f:
                assert f.read() == report.render_svg(r), "%s.svg differs from report.render_svg" % rel


def test_readme_final_confidence_block_matches_render():
    block, i, text = readme_block(FINAL_CONF_START, FINAL_CONF_END)
    paths = [os.path.join(REPO_ROOT, rel) for rel, _ in final_confidence()]
    assert report.confidence_svg_dir(paths) == FINAL_CONF_DIR
    rendered = report.render_confidence([r for _, r in final_confidence()], svg_dir=FINAL_CONF_DIR)
    assert block.strip() == rendered.strip(), (
        "the README final-confidence block differs from the render of the committed records; regenerate it with "
        "`python report.py --confidence-table results/final/confidence`")
    assert text.index(GO_END) < i, "the final-confidence block must follow the GO block"
    links = re.findall(r"\]\(([^)]+\.svg)\)", block)
    assert len(links) == 8 and all(l.startswith(FINAL_CONF_DIR + "/") for l in links), links
    for link in links:
        assert link in tracked(link), link


@pytest.mark.data
def test_final_confidence_records_rebuild():
    _need_data()
    for rel, r in final_confidence():
        pf = r["binding"]["postfit"]
        got = report.confidence_report(os.path.join(DUMPS, r["run_id"]), r["split"]["name"], DEFAULT_MANIFEST,
                                       None if pf is None else os.path.join(REPO_ROOT, pf["path"]))
        with open(os.path.join(REPO_ROOT, rel), "rb") as f:
            assert as_written(got) == f.read(), "%s does not rebuild" % rel
        if r["split"]["name"] == "test":
            with open(os.path.join(REPO_ROOT, rel[:-len(".json")] + ".svg"), encoding="utf-8") as f:
                assert report.render_svg(json.loads(json.dumps(got, ensure_ascii=False))) == f.read(), rel
