"""The recommendation of REP-04 (Phase 10, plan 10-01): the frozen verdict rendered mechanically, never a judgement.

- pilot.memo.recommendation maps the verdict/1 record's verdict field (GO -> GO, NO-GO -> NO-GO, INCONCLUSIVE -> NO-GO;
  collect bank data, D-07) and refuses another schema, a provisional verdict, a verdict whose few-shot path of (d) is
  still PENDING and an unknown verdict word, each in one SystemExit line naming the fix
- pilot.memo.facts reads the block's numbers from the records: test n and no-intent n from the verdict's split block,
  the final_train folds from pilot.data.SPLITS, machine and memory from results/spike/mps_spike.json meta.platform, the
  scorer device from protocol.json scoring
- `report.py --recommendation VERDICT_JSON` prints pilot.memo.render_recommendation only after
  pilot.protocol.require_frozen and pilot.verdict.check_record pass, so a missing, edited or unfrozen verdict is
  refused before anything is printed; render_recommendation refuses a verdict block that disagrees with its own
  render_go text
"""
import copy
import hashlib
import json
import os
import re

import pytest

import pilot.data
import pilot.memo as memo
import pilot.protocol
import pilot.verdict
import report

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERDICT = os.path.join(REPO_ROOT, "results", "final", "verdict.json")
VERDICT_REL = "results/final/verdict.json"
README = os.path.join(REPO_ROOT, "README.md")
SPIKE = os.path.join(REPO_ROOT, "results", "spike", "mps_spike.json")
V = pilot.verdict


def committed():
    with open(VERDICT, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def planted(a, b, d, alt):
    """A deep copy of the committed verdict with only its verdict block replaced by pilot.verdict.decide's output."""
    v = copy.deepcopy(committed()[0])
    v["verdict"] = json.loads(json.dumps(V.decide(a, b, d, alt)))
    return v


def rendered():
    v, sha = committed()
    proto, _ = pilot.protocol.require_frozen()
    return memo.render_recommendation(v, VERDICT_REL, sha, memo.facts(v, proto)), v, sha, proto


def test_recommendation_maps_the_verdict():
    v, _ = committed()
    assert v["verdict"]["verdict"] == V.INCONCLUSIVE and v["verdict"]["rule"] == 4
    assert memo.recommendation(v) == "NO-GO; collect bank data"
    assert memo.RECOMMENDATION == {V.GO: "GO", V.NO_GO: "NO-GO", V.INCONCLUSIVE: "NO-GO; collect bank data"}
    cases = {(V.INCONCLUSIVE, V.PASS, V.PASS, V.MET): "GO",                      # rule 2: alternative GO MET
             (V.INCONCLUSIVE, V.FAIL, V.PASS, V.NOT_MET): "NO-GO",               # rule 1: (b) FAIL
             (V.INCONCLUSIVE, V.PASS, V.PASS, V.NOT_MET): "NO-GO; collect bank data",   # rule 4
             (V.PASS, V.PASS, V.PASS, V.NOT_MET): "GO",                          # rule 5
             (V.PASS, V.PASS, V.FAIL, V.NOT_MET): "NO-GO"}                       # rule 6
    for statuses, want in cases.items():
        p = planted(*statuses)
        assert memo.recommendation(p) == want, (statuses, p["verdict"])
        assert memo.recommendation(p) == memo.RECOMMENDATION[p["verdict"]["verdict"]]


def test_recommendation_quotes_the_frozen_meaning():
    assert "counts as " + memo.RECOMMENDATION[V.INCONCLUSIVE] == V.INCONCLUSIVE_MEANS
    v, _ = committed()
    assert v["verdict"]["means"] == V.INCONCLUSIVE_MEANS
    assert memo.PENDING_FEWSHOT == {"status": V.PENDING, "phase": 9}
    with open(memo.__file__, encoding="utf-8") as f:
        src = f.read()
    import ast
    floats = [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert floats == [], "pilot/memo.py holds float literals %r; it may hold no threshold" % floats


def test_recommendation_refuses_a_provisional_or_pending_verdict():
    p = planted(V.PASS, V.PASS, V.PENDING, V.NOT_MET)
    assert p["verdict"]["provisional"] is True
    with pytest.raises(SystemExit, match="provisional"):
        memo.recommendation(p)
    v = copy.deepcopy(committed()[0])
    v["fewshot"] = {"status": "PENDING", "phase": 9}
    with pytest.raises(SystemExit, match="PENDING"):
        memo.recommendation(v)
    v = copy.deepcopy(committed()[0])
    v["verdict"]["verdict"] = "MAYBE"
    with pytest.raises(SystemExit, match="MAYBE"):
        memo.recommendation(v)
    v = copy.deepcopy(committed()[0])
    v["schema"] = "verdict/2"
    with pytest.raises(SystemExit, match="verdict/1"):
        memo.recommendation(v)
    for bad in (None, [], "INCONCLUSIVE"):
        with pytest.raises(SystemExit):
            memo.recommendation(bad)
    with pytest.raises(SystemExit) as info:
        memo.recommendation(p)
    assert "\n" not in str(info.value) and V.FEWSHOT_FIX in str(info.value)


def test_cli_recommendation_matches_render(capsys, tmp_path):
    want, v, sha, proto = rendered()
    capsys.readouterr()
    got = report.main(["--recommendation", VERDICT])
    assert got == want
    assert capsys.readouterr().out == want
    lines = want.split("\n")
    assert lines[0] == memo.RECOMMENDATION_HEADING and want.endswith("\n") and not want.endswith("\n\n")
    assert "**Recommendation: NO-GO; collect bank data**" in lines
    assert "- Verdict: INCONCLUSIVE, by rule 4: %s." % V.VERDICT_RULES[3] in lines
    assert "- Verdict: INCONCLUSIVE, by rule 4: (a) INCONCLUSIVE -> INCONCLUSIVE (counts as NO-GO; collect bank data)." \
        in lines
    statuses = [l for l in V.render_go(v).split("\n") if l.startswith("Statuses: ")]
    assert len(statuses) == 1 and "- " + statuses[0] in lines
    with open(os.path.join(REPO_ROOT, v["fewshot"]["record"]["path"]), "rb") as f:
        few_sha = hashlib.sha256(f.read()).hexdigest()
    assert few_sha == v["fewshot"]["record"]["sha256"]
    for prefix in (sha[:12], pilot.protocol.PROTOCOL_SHA256[:12], few_sha[:12]):
        assert "`%s`" % prefix in want
    assert VERDICT_REL in want and "protocol.json" in want and v["fewshot"]["record"]["path"] in want
    assert "audit and override" in want
    assert os.path.abspath(REPO_ROOT) not in want                   # repo-relative paths only
    out = tmp_path / "rec.md"
    assert report.main(["--recommendation", VERDICT, "--out", str(out)]) == want
    assert out.read_bytes() == want.encode("utf-8")
    assert memo.render_recommendation(v, VERDICT_REL, sha, memo.facts(v, proto)) == want      # identical input


def test_recommendation_facts_are_the_records(monkeypatch, tmp_path):
    want, v, sha, proto = rendered()
    f = memo.facts(v, proto)
    final = next(s for s in pilot.data.SPLITS if s["name"] == "final_train")["folds"]
    with open(SPIKE, encoding="utf-8") as fh:
        plat = json.load(fh)["meta"]["platform"]
    assert f == {"n": v["split"]["n"], "n_no_intent": v["split"]["n_no_intent"],
                 "folds": "%d-%d" % (min(final), max(final)), "machine": plat["machine"],
                 "memory_gib": plat["total_memory_bytes"] // 2 ** 30, "device": proto["scoring"]["device"]}
    assert "n = %d (%d no-intent)" % (v["split"]["n"], v["split"]["n_no_intent"]) in want
    assert "banking folds %d-%d" % (min(final), max(final)) in want
    assert "%s, %d GiB unified memory (results/spike/mps_spike.json)" % (
        plat["machine"], plat["total_memory_bytes"] // 2 ** 30) in want
    assert "test and HOTELS scored on %s (protocol.json scoring)" % proto["scoring"]["device"] in want
    # a missing spike record or key, a memory that is not a whole number of GiB, non-contiguous folds: refused
    with pytest.raises(SystemExit, match="git checkout -- results/spike/mps_spike.json"):
        memo.facts(v, proto, root=str(tmp_path))
    spike_dir = tmp_path / "results" / "spike"
    spike_dir.mkdir(parents=True)
    (spike_dir / "mps_spike.json").write_text(json.dumps({"meta": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="git checkout -- results/spike/mps_spike.json"):
        memo.facts(v, proto, root=str(tmp_path))
    (spike_dir / "mps_spike.json").write_text(json.dumps(
        {"meta": {"platform": {"machine": "arm64", "total_memory_bytes": 2 ** 30 + 1}}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="GiB"):
        memo.facts(v, proto, root=str(tmp_path))
    (spike_dir / "mps_spike.json").write_text(json.dumps(
        {"meta": {"platform": {"machine": "arm64", "total_memory_bytes": 8 * 2 ** 30}}}), encoding="utf-8")
    assert memo.facts(v, proto, root=str(tmp_path))["memory_gib"] == 8
    splits = [dict(s, folds=[0, 1, 3]) if s["name"] == "final_train" else s for s in pilot.data.SPLITS]
    monkeypatch.setattr(pilot.data, "SPLITS", splits)
    with pytest.raises(SystemExit, match="contiguous"):
        memo.facts(v, proto)


def test_cli_recommendation_refuses_an_edited_or_missing_verdict(capsys, monkeypatch, tmp_path):
    capsys.readouterr()
    with pytest.raises(SystemExit, match="not found; write the verdict with report.py --verdict --fewshot"):
        report.main(["--recommendation", str(tmp_path / "missing.json")])
    assert capsys.readouterr().out == ""
    v, _ = committed()
    edited = copy.deepcopy(v)
    edited["criteria"]["a"]["status"] = V.PASS
    p = tmp_path / "verdict.json"
    p.write_text(json.dumps(edited, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="do not recompute"):
        report.main(["--recommendation", str(p)])
    assert capsys.readouterr().out == ""
    p.write_bytes(b"{not json")
    with pytest.raises(SystemExit, match="not valid JSON") as info:
        report.main(["--recommendation", str(p)])
    assert "\n" not in str(info.value)
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(pilot.protocol, "PROTOCOL_SHA256", None)
    with pytest.raises(SystemExit, match="protocol is not frozen"):
        report.main(["--recommendation", VERDICT])
    assert capsys.readouterr().out == ""
    monkeypatch.undo()
    with pytest.raises(SystemExit) as info:
        report.main(["--recommendation", VERDICT, "--go-block", VERDICT])
    assert info.value.code == 2
    assert "give exactly one of" in capsys.readouterr().err
    assert not (tmp_path / "rec.md").exists()


def test_recommendation_refuses_a_go_render_that_disagrees(monkeypatch):
    want, v, sha, proto = rendered()
    f = memo.facts(v, proto)
    real = V.render_go
    monkeypatch.setattr(pilot.verdict, "render_go", lambda r: real(r).replace("**Verdict: INCONCLUSIVE**",
                                                                               "**Verdict: GO**"))
    with pytest.raises(SystemExit, match="Verdict"):
        memo.render_recommendation(v, VERDICT_REL, sha, f)
    monkeypatch.setattr(pilot.verdict, "render_go", lambda r: real(r).replace("Statuses: ", "Status: "))
    with pytest.raises(SystemExit, match="Statuses"):
        memo.render_recommendation(v, VERDICT_REL, sha, f)
    monkeypatch.undo()
    edited = copy.deepcopy(v)
    edited["verdict"]["statuses"]["a"] = V.PASS                 # the verdict block disagrees with its criteria
    with pytest.raises(SystemExit, match="statuses"):
        memo.render_recommendation(edited, VERDICT_REL, sha, f)
    assert memo.render_recommendation(v, VERDICT_REL, sha, f) == want


# ---- Task 3: the README "## Recommendation (Phase 10)" section (number-free prose, every D-08 caveat, the block) ----

PHASE10_HEADING = "## Recommendation (Phase 10)"
PHASE10_PROSE = ("mechanical", "frozen", "INCONCLUSIVE", "NO-GO", "collect bank data", "audit", "override",
                 "results/final/memo.md", "make memo")
REC_START, REC_END = "<!-- recommendation:start -->", "<!-- recommendation:end -->"
FEWSHOT_END = "<!-- fewshot:end -->"
PHASE3_HEADING = "## Results (Phase 3: baselines and zero-shot, full test split)"
REGEN_LINE = ("Regenerate with `python report.py --recommendation results/final/verdict.json` (a test fails if it "
              "differs).")
POINTER = "(#recommendation-phase-10)"


def readme_text():
    with open(README, encoding="utf-8") as f:
        return f.read()


def phase10_section(text):
    """(start, end) offsets of the README's Recommendation section: its heading to the next "\\n## "."""
    assert text.count("\n" + PHASE10_HEADING + "\n") == 1, "README.md has no single %r section" % PHASE10_HEADING
    i = text.index("\n" + PHASE10_HEADING + "\n") + 1
    j = text.find("\n## ", i + len(PHASE10_HEADING))
    return i, (len(text) if j < 0 else j)


def test_caveats_cover_every_key_without_numbers():
    assert len(memo.CAVEATS) == 8 and len(set(memo.CAVEATS)) == 8
    for key in memo.CAVEAT_KEYS:
        assert any(key in c for c in memo.CAVEATS), "no pilot.memo.CAVEATS bullet states %r" % key
        assert not memo.NUMBER_RE.search(key), key
    for c in memo.CAVEATS:
        hits = [m.group(0) for m in memo.NUMBER_RE.finditer(c)]
        assert hits == [], "the caveat %r holds numbers %r; numbers come from the records" % (c, hits)
    for planted in ("k = 10", "+65.87", "DEC-01", "n = 413", "0.707", "-1.72 points"):
        assert memo.NUMBER_RE.search(planted), planted
    for clean in ("fp16", "neg8", "v2", "fp32", "Multi3NLU++", "Apple M4", "micro-F1", "θ", "rule_4"):
        assert not memo.NUMBER_RE.search(clean), clean


def test_readme_recommendation_block_matches_render(capsys):
    text = readme_text()
    lo, hi = phase10_section(text)
    assert text.index(FEWSHOT_END) < lo, "the Recommendation section must follow the Phase 9 section"
    assert text.count(PHASE3_HEADING) == 1 and text[hi:].startswith("\n" + PHASE3_HEADING), (
        "the Recommendation section must sit right before %r" % PHASE3_HEADING)
    assert text.count(REC_START) == 1 and text.count(REC_END) == 1, "README.md must hold exactly one %s pair" % REC_START
    i, j = text.index(REC_START), text.index(REC_END)
    assert lo < i < j < hi, "the recommendation block must sit inside the Recommendation section"
    assert text.count(REGEN_LINE) == 1 and lo < text.index(REGEN_LINE) < i
    capsys.readouterr()
    got = report.main(["--recommendation", VERDICT])
    assert capsys.readouterr().out == got
    assert got == rendered()[0]
    assert text[i + len(REC_START):j] == "\n" + got, (
        "the README recommendation block differs from `python report.py --recommendation results/final/verdict.json`; "
        "regenerate it")


def test_readme_phase10_prose():
    """The prose (heading to the first generated block) states the recommendation is the frozen protocol's mechanical
    verdict, INCONCLUSIVE counts as NO-GO and means collect bank data, humans can audit and override it, and every
    D-08 caveat keyword; it holds no metric value of its own (PROSE_NUMBER_RE)."""
    from test_final_results import PROSE_NUMBER_RE
    text = readme_text()
    lo, hi = phase10_section(text)
    k = text.find("<!--", lo)
    assert lo < k < hi, "the Recommendation section holds no generated block"
    prose = " ".join(text[lo:k].split())
    for want in PHASE10_PROSE + memo.CAVEAT_KEYS:
        assert want in prose, "the README Recommendation prose must state %r before its generated block" % want
    for want in ("protocol.json", "results/final/verdict.json", "results/fewshot/fewshot.json", "results/dumps/"):
        assert want in prose, "the README Recommendation prose must name the record %r" % want
    hits = [m.group(0) for m in PROSE_NUMBER_RE.finditer(prose)]
    assert hits == [], "the README Recommendation prose holds metric values %r; numbers belong in the block" % hits


def test_readme_points_to_the_recommendation():
    from test_final_results import PROSE_NUMBER_RE
    text = readme_text()
    intro = text[:text.index("\n## ")]
    lines = [l for l in intro.split("\n") if POINTER in l]
    assert len(lines) == 1, "the README intro must hold exactly one line linking to %s" % POINTER
    first_paragraph_end = intro.index("\n\n", intro.index("\n\n") + 2)
    assert intro.index(lines[0]) > first_paragraph_end, "the pointer line must follow the intro's first paragraph"
    assert not PROSE_NUMBER_RE.search(lines[0]) and "Recommendation" in lines[0]
    assert text.count(POINTER) == 1


# ---- 10-02 Task 1: the one-page decision memo results/final/memo.md (REP-05, D-09, D-10) ----

MEMO = os.path.join(REPO_ROOT, "results", "final", "memo.md")
MAKEFILE = os.path.join(REPO_ROOT, "Makefile")
MEMO_RECIPE = ["$(PYTHON) report.py --memo $(FINAL)/verdict.json --out $(FINAL)/memo.md > /dev/null"]
MEMO_HEADER = ("#   make memo  write results/final/memo.md, the one-page decision memo, from the committed records "
               "(no model)")


def memo_text():
    v, _ = committed()
    return memo.render_memo(v, *report.memo_inputs(v))


def test_memo_matches_render():
    want = memo_text()
    assert os.path.isfile(MEMO), "results/final/memo.md is missing; write it with `make memo`"
    with open(MEMO, "rb") as f:
        got = f.read()
    assert got == want.encode("utf-8"), "results/final/memo.md differs from its render; regenerate it with `make memo`"
    lines = want.split("\n")
    assert lines[0] == memo.MEMO_TITLE and lines[1] == memo.MEMO_GENERATED
    assert want.endswith("\n") and not want.endswith("\n\n")
    assert "**Recommendation: NO-GO; collect bank data**" in lines


def test_cli_memo_prints_the_memo(capsys, tmp_path):
    want = memo_text()
    capsys.readouterr()
    assert report.main(["--memo", VERDICT]) == want
    assert capsys.readouterr().out == want
    out = tmp_path / "memo.md"
    assert report.main(["--memo", VERDICT, "--out", str(out)]) == want
    assert out.read_bytes() == want.encode("utf-8")
    with pytest.raises(SystemExit, match="not found; write the verdict with report.py --verdict --fewshot"):
        report.main(["--memo", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit) as info:
        report.main(["--memo", VERDICT, "--recommendation", VERDICT])
    assert info.value.code == 2
    assert "--memo VERDICT_JSON" in capsys.readouterr().err


def test_makefile_memo_target():
    from test_final_results import recipe
    with open(MAKEFILE, encoding="utf-8") as f:
        text = f.read()
    assert recipe(text, "memo") == MEMO_RECIPE
    phony = re.search(r"^\.PHONY: (.+)$", text, re.M)
    assert phony and "memo" in phony.group(1).split(), "memo is not in .PHONY"
    assert MEMO_HEADER in text.split("\n"), "the Makefile header has no `make memo` line"
    assert "memo.md" not in "\n".join(recipe(text, "final-reports")), "final-reports writes no memo (10-RESEARCH)"


# ---- 10-02 Task 2: the memo writes no number of its own, fits one page, is bound to the verdict's records ----

DECIMAL_RE = re.compile(r"[-+]?\d+\.\d+")
POINTS_RE = re.compile(r"[-+]?\d+(?:\.\d+)? points")


def renders():
    """{source: text} of the five renders the memo may copy lines from, built here exactly as report.memo_inputs
    builds them (the few-shot record the verdict binds, results/final/confidence, results/latency)."""
    import pilot.fewshot_report
    import pilot.latency
    v, _ = committed()
    rec, sha = pilot.fewshot_report.read_record(os.path.join(REPO_ROOT, v["fewshot"]["record"]["path"]))
    assert sha == v["fewshot"]["record"]["sha256"]
    paths = report.table_inputs([os.path.join(REPO_ROOT, "results", "final", "confidence")])
    reports = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            reports.append(json.load(f))
    assert report.confidence_svg_dir(paths) == "results/final/confidence"
    return {"go": V.render_go(v), "table": V.render_final_table(v), "fewshot": pilot.fewshot_report.render_markdown(rec),
            "confidence": report.render_confidence(reports, svg_dir=report.confidence_svg_dir(paths)),
            "latency": pilot.latency.render_markdown(pilot.latency.load_reports(
                os.path.join(REPO_ROOT, "results", "latency")))}


def test_memo_numbers_are_render_lines():
    src = renders()
    assert tuple(src) == memo.MEMO_SOURCES
    v, _ = committed()
    text = memo.render_memo(v, src["fewshot"], src["confidence"], src["latency"])
    with open(MEMO, encoding="utf-8") as f:
        assert f.read() == text
    render_lines = {l for t in src.values() for l in t.split("\n")}
    numbered = 0
    for line in text.split("\n"):
        if memo.NUMBER_RE.search(line):
            numbered += 1
            assert line in render_lines, "the memo line %r holds a number and is not a line of a generated render" % line
        else:
            assert not memo.NUMBER_RE.search(line)
        if line not in render_lines:
            hits = [m.group(0) for m in memo.NUMBER_RE.finditer(line)]
            assert hits == [], "the memo's own line %r holds numbers %r" % (line, hits)
    assert numbered >= 12, "the memo should carry the GO rows and the results rows, not %d numbered lines" % numbered
    gated = src["go"] + src["table"]
    decimals, points = set(DECIMAL_RE.findall(gated)), set(POINTS_RE.findall(gated))
    got_dec, got_pts = DECIMAL_RE.findall(text), POINTS_RE.findall(text)
    assert got_dec and got_pts, "the memo carries no decimal or points value"
    assert "+65.87 points" in got_pts and "0.826" in got_dec          # the few-shot (d) row, a seed-averaged row
    assert [t for t in got_dec if t not in decimals] == [], "memo decimals outside the GO block and results table"
    assert [t for t in got_pts if t not in points] == [], "memo points values outside the GO block and results table"


def test_memo_fits_one_page(monkeypatch):
    text = memo_text()
    assert memo.MEMO_MAX_LINES == 60
    n = len(text.splitlines())
    assert n <= memo.MEMO_MAX_LINES, "the memo is %d lines, more than one page" % n
    v, _ = committed()
    inputs = report.memo_inputs(v)
    monkeypatch.setattr(memo, "CAVEATS", memo.CAVEATS + ("an extra caveat",) * (memo.MEMO_MAX_LINES - n))
    assert len(memo.render_memo(v, *inputs).splitlines()) == memo.MEMO_MAX_LINES      # the cap itself is one page
    monkeypatch.setattr(memo, "CAVEATS", memo.CAVEATS + ("one caveat too many",))
    with pytest.raises(SystemExit, match="one page: %d lines, at most %d" % (memo.MEMO_MAX_LINES + 1,
                                                                          memo.MEMO_MAX_LINES)):
        memo.render_memo(v, *inputs)


def test_memo_module_holds_no_number_literal():
    import ast
    with open(memo.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    floats = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert floats == [], "pilot/memo.py holds float literals %r; every memo number comes from a render" % floats
    own = (memo.MEMO_TITLE, memo.MEMO_GENERATED, memo.MEMO_OVERRIDE, memo.MEMO_AUDIT, memo.MEMO_REBUILD)
    for value in own + memo.MEMO_HEADINGS:
        assert not memo.NUMBER_RE.search(value), "the memo text %r holds a number" % value


def test_memo_states_the_recommendation_and_caveats():
    v, _ = committed()
    text = memo_text()
    lines = text.split("\n")
    assert "**Recommendation: %s**" % memo.recommendation(v) in lines
    rule = "- Rule %d: %s" % (v["verdict"]["rule"], V.VERDICT_RULES[v["verdict"]["rule"] - 1])
    assert rule in lines and rule in V.render_go(v).split("\n")
    assert lines.index("**Recommendation: %s**" % memo.recommendation(v)) < lines.index(rule)
    for c in memo.CAVEATS:
        assert "- " + c in lines, "the memo drops the caveat %r" % c
    for want in (memo.MEMO_OVERRIDE, memo.MEMO_AUDIT, memo.MEMO_REBUILD):
        assert want in lines
    at = [lines.index(h) for h in memo.MEMO_HEADINGS]
    assert at == sorted(at) and all(lines[i - 1] == "" for i in at), "the memo headings are out of order"
    assert [l for l in lines if l.startswith("## ")] == list(memo.MEMO_HEADINGS)
    for want in ("- Protocol: ", "- Split test: ", "| criterion | status | numbers |", "| alternative GO",
                 "| system | run | micro-F1 | macro-F1 | exact match | top-1 |", "- Few-shot path (Phase ",
                 "Criterion (d)'s few-shot path reads", "The test no-intent set holds",
                 "Latency is reported, not gated:"):
        assert sum(l.startswith(want) for l in lines) == 1, "the memo must hold one line starting %r" % want


def test_memo_inputs_refuse_an_unbound_fewshot_record():
    v, _ = committed()
    inputs = report.memo_inputs(v)
    bad = copy.deepcopy(v)
    bad["fewshot"]["record"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit) as info:
        report.memo_inputs(bad)
    msg = str(info.value)
    assert v["fewshot"]["record"]["sha256"] in msg and "0" * 64 in msg and "make fewshot-reports" in msg
    for path in ("../results/fewshot/fewshot.json", os.path.join(REPO_ROOT, "results", "fewshot", "fewshot.json")):
        moved = copy.deepcopy(v)
        moved["fewshot"]["record"]["path"] = path
        with pytest.raises(SystemExit, match="repo-relative"):
            report.memo_inputs(moved)
    pending = copy.deepcopy(v)
    pending["fewshot"] = {"status": "PENDING", "phase": 9}
    with pytest.raises(SystemExit):
        report.memo_inputs(pending)
    with pytest.raises(SystemExit, match="PENDING"):
        memo.render_memo(pending, *inputs)
    p = planted(V.PASS, V.PASS, V.PENDING, V.NOT_MET)
    with pytest.raises(SystemExit, match="provisional"):
        memo.render_memo(p, *inputs)


def test_memo_holds_no_text_or_paths():
    text = memo_text()
    assert "bank_" not in text and "hotel_" not in text, "the memo holds a message uid"
    assert os.path.expanduser("~") not in text, "the memo holds a local home path"
    assert REPO_ROOT not in text and os.path.realpath(REPO_ROOT) not in text, "the memo holds an absolute repo path"
