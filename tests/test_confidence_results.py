"""The committed Phase 7 confidence records (EVAL-03) stay consistent with the code, the Makefile and the postfit records.

- every tracked results/confidence/*.json is a val record (schema confidence/1, split val) that carries in_sample
  true and report.IN_SAMPLE_CAVEAT, both temperature variants with a calibration block, and, for a dump of a trained
  model that is not a final seed, the postfit_candidate binding (its T and θ come from the postfit record, never from
  the sidecar's shipped T = 1)
- the Makefile's LAYA_CHOSEN and BERTURK_CHOSEN are the chosen dumps of results/postfit/laya.json and berturk.json,
  and the tracked records are exactly those two and zero-shot Laya (LAYA_ZS, a reference row bound by report.py's
  val rule, val_selected): the two chosen ones bound to their own system kind's postfit record (07-RESEARCH Open
  Question 1, A10)
- every tracked record has its reliability diagram results/confidence/<run_id>-val.svg, equal byte for byte to
  report.render_svg of the committed JSON (and every tracked SVG has its JSON); the SVG is plain (only g, rect, line,
  circle and text elements; no script, metadata, href or timestamp) and renders identically twice (A8, T-07-11, T-07-13)
- report.render_confidence of the committed records names every run id with both variants (headline and raw: the
  favourable variant is never shown alone, T-07-12), links each record's SVG, and carries the in-sample caveat and
  every calibration definition text; it refuses anything that is not a confidence/1 record
- EVAL-04: every record and variant carries a selective block at θ 0.5 and at its θ (coverages 50/70/80/90 with the
  integer cut, AURC) and the confidence, coverage and AURC rule texts of pilot.metrics; render_confidence shows the
  abstention table (one row per record, variant and θ), the three rule texts and report.CONFIDENCE_DISCLOSURE (the
  val AURC order of the two fine-tuned models depends on the confidence rule, T-07-14), and refuses records whose
  abstention definitions differ or are missing
- EVAL-05: every record and variant carries a no-intent false-positive block at θ 0.5 and at its θ (the manifest's 7 val
  no-intent messages, rate k / n unrounded, the Wilson interval of those counts) and the false-positive and Wilson rule
  texts; render_confidence shows the no-intent table with k/n, rate and Wilson 95% interval on every row (T-07-17),
  the rule texts and the sample-size line, and refuses records whose false-positive definitions or no-intent counts
  disagree
- data marker: every committed record equals report.confidence_report recomputed from the committed dump, the
  committed manifest and the record it is bound to (needs `make data`, because the gold labels come from the verified
  splits), and a record bound to a postfit candidate reproduces that candidate's nll_fitted (headline) and nll_raw
  (raw) within 1e-8
"""
import glob
import json
import os
import re
import subprocess

import pytest

import report
from pilot.data import DEFAULT_MANIFEST, REPO_ROOT

pytestmark = pytest.mark.unit

DUMPS = os.path.join(REPO_ROOT, "results", "dumps")
LAYA_RECORD = os.path.join(REPO_ROOT, "results", "postfit", "laya.json")
BERTURK_RECORD = os.path.join(REPO_ROOT, "results", "postfit", "berturk.json")
POSTFIT_BY_KIND = {"laya": "results/postfit/laya.json", "berturk": "results/postfit/berturk.json"}
NLL_TOL = 1e-8


def tracked(*pathspec):
    """Tracked paths via git ls-files (never a directory walk, which would see untracked scratch records)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", *pathspec], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True, encoding="utf-8")
    return [f for f in out.stdout.split("\0") if f]


def committed():
    """[(repo-relative path, parsed JSON)] of the tracked confidence records, in sorted path order."""
    out = []
    for rel in sorted(f for f in tracked("results/confidence/") if f.endswith(".json")):
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
            out.append((rel, json.load(f)))
    return out


def _postfit_path(rec):
    pf = rec["binding"]["postfit"]
    return None if pf is None else os.path.join(REPO_ROOT, pf["path"])


def test_committed_records_are_val_and_in_sample():
    records = committed()
    assert records, "no tracked confidence records under results/confidence/"
    for rel, r in records:
        assert r["schema"] == report.CONFIDENCE_SCHEMA, rel
        assert r["split"]["name"] == "val" and r["in_sample"] is True, rel
        assert r["caveat"] == report.IN_SAMPLE_CAVEAT, rel
        assert rel == "results/confidence/%s-val.json" % r["run_id"], rel
        assert set(r["variants"]) == {"headline", "raw"}, rel
        for name, v in r["variants"].items():
            cal = v["calibration"]
            assert {"brier", "nll", "classwise_ece", "per_tag", "reliability"} <= set(cal), (rel, name)
            assert sum(b["count"] for b in cal["reliability"]["pooled"]) == r["n_messages"] * 48, (rel, name)
        assert r["variants"]["raw"]["temperature"] == 1.0, rel
        tr = r["training"]
        if tr is not None and r["binding"]["kind"] != "final_seed":
            assert r["binding"]["kind"] == "postfit_candidate", rel
            assert r["binding"]["postfit"] is not None, rel


def makefile():
    with open(os.path.join(REPO_ROOT, "Makefile"), encoding="utf-8") as f:
        return f.read()


def makefile_var(text, name):
    m = re.search(r"^%s := (\S+)$" % name, text, re.M)
    assert m, "Makefile has no %s := line" % name
    return m.group(1)


def test_makefile_confidence_ids_match_postfit_records():
    text = makefile()
    for name, path in (("LAYA_CHOSEN", LAYA_RECORD), ("BERTURK_CHOSEN", BERTURK_RECORD)):
        run_id = makefile_var(text, name)
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        assert run_id == record["chosen"]["dump_run_id"], "%s is not %s's chosen dump" % (name, path)
        assert "results/confidence/%s-val.json" % run_id in [rel for rel, _ in committed()], name
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    assert phony and "confidence-reports" in phony.group(1).split(), "confidence-reports is not in .PHONY"
    recipe = re.search(r"^confidence-reports:\n((?:\t.*\n)+)", text, re.M)
    assert recipe, "Makefile has no confidence-reports target"
    for line in ("$(PYTHON) report.py --confidence results/dumps/$(LAYA_CHOSEN) --split val --postfit "
                 "results/postfit/laya.json --out $(CONFIDENCE)/$(LAYA_CHOSEN)-val.json",
                 "$(PYTHON) report.py --confidence results/dumps/$(BERTURK_CHOSEN) --split val --postfit "
                 "results/postfit/berturk.json --out $(CONFIDENCE)/$(BERTURK_CHOSEN)-val.json",
                 "$(PYTHON) report.py --confidence results/dumps/$(LAYA_ZS) --split val "
                 "--out $(CONFIDENCE)/$(LAYA_ZS)-val.json"):
        assert line in recipe.group(1), line


def test_committed_records_are_the_three_systems():
    text = makefile()
    chosen = {makefile_var(text, "LAYA_CHOSEN"): "laya", makefile_var(text, "BERTURK_CHOSEN"): "berturk"}
    zs = makefile_var(text, "LAYA_ZS")
    records = {r["run_id"]: r for _, r in committed()}
    assert set(records) == set(chosen) | {zs}, "tracked confidence records: %s" % sorted(records)
    for run_id, kind in chosen.items():
        r = records[run_id]
        assert r["system"]["kind"] == kind, run_id
        assert r["binding"]["kind"] == "postfit_candidate", run_id
        assert r["binding"]["postfit"]["path"] == POSTFIT_BY_KIND[kind], run_id
        assert r["binding"]["postfit"]["system_kind"] == kind, run_id
        assert r["training"] is not None and r["training"]["split"] == "selection_train", run_id
    r = records[zs]
    assert r["system"]["kind"] == "laya" and r["training"] is None, zs
    assert r["binding"]["kind"] == "val_selected" and r["binding"]["postfit"] is None, zs


def _need_data():
    if not glob.glob(os.path.join(REPO_ROOT, "data", "splits", "*.jsonl")):
        pytest.fail("data/splits/ is empty; run `make data` first")


@pytest.mark.data
def test_confidence_records_reproduce_from_dumps():
    _need_data()
    records = committed()
    assert records
    for rel, r in records:
        got = report.confidence_report(os.path.join(DUMPS, r["run_id"]), "val", DEFAULT_MANIFEST, _postfit_path(r))
        assert json.loads(json.dumps(got, ensure_ascii=False)) == r, "%s does not reproduce" % rel


@pytest.mark.data
def test_nll_matches_postfit_record():
    _need_data()
    bound = [(rel, r) for rel, r in committed() if r["binding"]["kind"] == "postfit_candidate"]
    kinds = sorted(r["system"]["kind"] for _, r in bound)
    assert kinds == ["berturk", "laya"], "postfit-bound records: %s" % [r["run_id"] for _, r in bound]
    for rel, r in bound:
        with open(_postfit_path(r), encoding="utf-8") as f:
            record = json.load(f)
        cands = [c for c in record["candidates"] if c["dump_sha256"] == r["dump_sha256"]]
        assert len(cands) == 1, rel
        c = cands[0]
        assert abs(r["variants"]["headline"]["calibration"]["nll"] - c["nll_fitted"]) <= NLL_TOL, r["run_id"]
        assert abs(r["variants"]["raw"]["calibration"]["nll"] - c["nll_raw"]) <= NLL_TOL, r["run_id"]
        assert r["variants"]["headline"]["temperature"] == c["temperature"], r["run_id"]


def committed_svgs():
    return sorted(f for f in tracked("results/confidence/") if f.endswith(".svg"))


def test_svg_matches_render():
    records = committed()
    svgs = committed_svgs()
    assert svgs == sorted(rel[:-len(".json")] + ".svg" for rel, _ in records), "every record needs exactly its SVG"
    for rel, r in records:
        with open(os.path.join(REPO_ROOT, rel[:-len(".json")] + ".svg"), "rb") as f:
            assert f.read() == report.render_svg(r).encode("utf-8"), "%s.svg is not render_svg of its JSON" % rel


ALLOWED_SVG_ELEMENTS = {"svg", "g", "rect", "line", "circle", "text"}


def test_svg_is_plain_and_reproducible():
    for rel, r in committed():
        text = report.render_svg(r)
        assert text == report.render_svg(json.loads(json.dumps(r))), rel
        low = text.lower()
        for banned in ("<script", "<metadata", "href", "<style", "<image", "<foreignobject", "@import", "url("):
            assert banned not in low, (rel, banned)
        assert not re.search(r"\d:\d", text), "%s: a timestamp-like digit:digit pattern in the SVG" % rel
        assert set(re.findall(r"<([A-Za-z][A-Za-z0-9]*)", text)) <= ALLOWED_SVG_ELEMENTS, rel
        assert text.startswith('<svg xmlns="http://www.w3.org/2000/svg" ') and text.endswith("</svg>\n"), rel
        for name in ("headline", "raw"):
            v = r["variants"][name]
            title = "%s %s, T = %.4f, classwise ECE = %.4f" % (r["run_id"], name, v["temperature"],
                                                               v["calibration"]["classwise_ece"])
            assert ">%s</text>" % title in text, (rel, name)
            for b in v["calibration"]["reliability"]["pooled"]:
                if b["count"]:
                    assert ">%d</text>" % b["count"] in text, (rel, name, b)


def test_render_has_tables_links_and_caveat():
    records = [r for _, r in committed()]
    text = report.render_confidence(records)
    assert text == report.render_confidence(json.loads(json.dumps(records)))
    assert text.startswith("### Calibration on val (θ-free)\n")
    for line in text.split("\n"):
        assert not line.startswith(("# ", "## ")), "only level-3 headings: %r" % line
    for r in records:
        for name in ("headline", "raw"):
            cal = r["variants"][name]["calibration"]
            row = "| %s | %s | %.4f | %.4f | %.4f | %.4f |" % (r["run_id"], name, r["variants"][name]["temperature"],
                                                              cal["brier"], cal["nll"], cal["classwise_ece"])
            assert row in text, (r["run_id"], name)
        image = "![reliability diagram %s (val)](results/confidence/%s-val.svg)" % (r["run_id"], r["run_id"])
        assert text.count(image) == 1, r["run_id"]
    assert text.count("![reliability diagram ") == len(records)
    assert text.count(report.IN_SAMPLE_CAVEAT) == 1
    for rule in records[0]["definitions"]["calibration"].values():
        assert rule in text, rule
    for t1 in [r["run_id"] for r in records if r["variants"]["headline"]["temperature"] == 1.0]:
        assert "%s: headline temperature is 1.0" % t1 in text



def test_committed_records_carry_selective_blocks():
    """EVAL-04: every record and variant has a selective block at θ 0.5 and at its θ, under the rule texts in code."""
    from pilot import metrics
    for rel, r in committed():
        assert r["definitions"] == {"calibration": metrics.CALIBRATION_RULES, "confidence": metrics.CONFIDENCE_RULE,
                                    "coverage": metrics.COVERAGE_RULE, "aurc": metrics.AURC_RULE,
                                    "false_positive": metrics.FP_RULE, "wilson": report.WILSON_TEXT}, rel
        n = r["n_messages"]
        for name in ("headline", "raw"):
            sel = r["variants"][name]["selective"]
            assert sorted(sel) == ["0.5", "theta"], (rel, name)
            for key, s in sel.items():
                assert s["n"] == n and list(s["coverages"]) == [str(c) for c in metrics.COVERAGES], (rel, name, key)
                assert [s["coverages"][str(c)]["n_kept"] for c in metrics.COVERAGES] == [
                    (c * n + 99) // 100 for c in metrics.COVERAGES], (rel, name, key)
                assert 0.0 <= s["aurc"] <= 1.0
                for row in s["coverages"].values():
                    assert 1 <= row["n_tied_at_cut"] <= n and 0.0 <= row["accuracy"] <= 1.0


def test_render_has_abstention_table_and_disclosure():
    from pilot import metrics
    records = [r for _, r in committed()]
    text = report.render_confidence(records)
    head = "### Abstention on val (message confidence)"
    assert text.count(head) == 1
    start = text.index(head)
    section = text[start:text.index("\n### ", start)]         # up to the next section
    covs = [str(c) for c in metrics.COVERAGES]
    header = ("| system | run_id | variant | θ | " + " | ".join("precision@%s%%" % c for c in covs) + " | "
              + " | ".join("accuracy@%s%%" % c for c in covs) + " | AURC |")
    assert header in section
    for r in records:
        for name in ("headline", "raw"):
            v = r["variants"][name]
            for key, label in (("0.5", "0.5"), ("theta", "val-selected, global %.2f" % v["theta"]["global"])):
                s = v["selective"][key]
                row = ("| %s | %s | %s | " % (r["run_id"], name, label)
                       + " | ".join("%.3f" % s["coverages"][c]["precision"] for c in covs) + " | "
                       + " | ".join("%.3f" % s["coverages"][c]["accuracy"] for c in covs)
                       + " | %.3f |" % s["aurc"])
                assert section.count(row) == 1, (r["run_id"], name, key)
    table = [line for line in section.split("\n") if line.startswith("|")]
    assert len(table) == 2 + 4 * len(records)               # header, separator, 2 variants x 2 θ rows each
    for rule in (metrics.CONFIDENCE_RULE, metrics.COVERAGE_RULE, metrics.AURC_RULE, report.CONFIDENCE_DISCLOSURE):
        assert text.count(rule) == 1, rule
    assert "weakest-tag" in report.CONFIDENCE_DISCLOSURE and "Phase 8" in report.CONFIDENCE_DISCLOSURE
    other = json.loads(json.dumps(records))
    other[0]["definitions"]["aurc"] = "another risk"
    with pytest.raises(SystemExit, match="abstention definitions"):
        report.render_confidence(other)
    del other[0]["definitions"]["aurc"]
    with pytest.raises(SystemExit, match="abstention definitions"):
        report.render_confidence(other[:1])

def test_committed_records_carry_no_intent_blocks():
    """EVAL-05: every record and variant has a no-intent false-positive block at θ 0.5 and at its θ over all 48 tags:
    the manifest's no-intent count, unrounded rate = k / n and the Wilson interval of those counts."""
    from pilot import metrics
    with open(DEFAULT_MANIFEST, encoding="utf-8") as f:
        splits = json.load(f)["splits"]
    assert report.TEST_NO_INTENT == splits["test"]["n_no_intent"] == 15
    for rel, r in committed():
        n_no_intent = splits[r["split"]["name"]]["n_no_intent"]
        assert n_no_intent == 7, rel
        for name in ("headline", "raw"):
            fp = r["variants"][name]["no_intent_fp"]
            assert sorted(fp) == ["0.5", "theta"], (rel, name)
            for key, f in fp.items():
                assert f["n"] == n_no_intent and f["cols"] == "all" and 0 <= f["k"] <= f["n"], (rel, name, key)
                assert f["rate"] == f["k"] / f["n"] and f["wilson95"] == metrics.wilson(f["k"], f["n"])
                assert sum(f["per_tag_fp"].values()) >= f["k"] and all(v > 0 for v in f["per_tag_fp"].values())


def test_render_has_no_intent_table_with_counts():
    """T-07-17: every no-intent row shows k/n, the rate and the Wilson 95% interval; never a rate alone."""
    from pilot import metrics
    records = [r for _, r in committed()]
    text = report.render_confidence(records)
    head = "### No-intent false positives on val"
    assert text.count(head) == 1
    start = text.index(head)
    section = text[start:text.index("\n### ", start)]
    assert text.index("### Abstention") < start < text.index("### Caveat and definitions")
    assert "| system | run_id | variant | θ | k/n | rate | Wilson 95% |" in section
    for r in records:
        for name in ("headline", "raw"):
            v = r["variants"][name]
            for key, label in (("0.5", "0.5"), ("theta", "val-selected, global %.2f" % v["theta"]["global"])):
                f = v["no_intent_fp"][key]
                row = ("| %s | %s | %s | %d/%d | %.3f | [%.3f, %.3f] |"
                       % (r["run_id"], name, label, f["k"], f["n"], f["rate"], f["wilson95"][0], f["wilson95"][1]))
                assert section.count(row) == 1, (r["run_id"], name, key)
    table = [line for line in section.split("\n") if line.startswith("|")]
    assert len(table) == 2 + 4 * len(records)
    assert re.search(r"\| 2/7 \| 0\.286 \| \[0\.082, 0\.641\] \|", section)   # the research probe's 2/7
    assert section.count("The val no-intent set holds 7 messages (15 on test), so every interval is wide.") == 1
    assert section.count("- false_positive: %s" % metrics.FP_RULE) == 1
    assert section.count("- wilson: %s" % report.WILSON_TEXT) == 1 and repr(metrics.WILSON_Z) in report.WILSON_TEXT
    other = json.loads(json.dumps(records))
    other[0]["definitions"]["wilson"] = "a Wald interval"
    with pytest.raises(SystemExit, match="false-positive definitions"):
        report.render_confidence(other)
    other = json.loads(json.dumps(records))
    other[0]["variants"]["raw"]["no_intent_fp"]["0.5"]["n"] = 6
    with pytest.raises(SystemExit, match="no-intent message count"):
        report.render_confidence(other)


def test_confidence_table_cli_prints_render(capsys):
    out = report.main(["--confidence-table", os.path.join(REPO_ROOT, "results", "confidence")])
    assert out == report.render_confidence([r for _, r in committed()])
    assert capsys.readouterr().out == out


@pytest.mark.parametrize("bad", [{"schema": "postfit/1"}, {"variants": {}}, [], "confidence/1"])
def test_render_refuses_a_non_confidence_record(bad):
    with pytest.raises(SystemExit, match="not a confidence/1 record"):
        report.render_confidence([bad])
    with pytest.raises(SystemExit, match="not a confidence/1 record"):
        report.render_svg(bad)
    with pytest.raises(SystemExit, match="no confidence records"):
        report.render_confidence([])
