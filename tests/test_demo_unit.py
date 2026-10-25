"""The demo's record views, offline and model-free: the verdict view against report.py's --go-block, --final-table and
--recommendation renders of the committed results/final/verdict.json, byte for byte, with every verdict string a whole
line (or a table row) of a frozen render; the val selection view against postfit.py --render and --render-berturk, with
the candidates, the reference check and the superseded BERTurk runs from the committed records; served by
demo.create_app's GET /api/<view> without any model loaded.

Per-view refusal: a tampered verdict copy (check_record) or a changed postfit record (protocol.json's input sha256)
breaks only its own view (503 {error, source}), every other view keeps answering; no view holds a local path; the
status tones come from pilot.verdict's constants. Runs in `make test`.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import demo
import pilot.data
import pilot.memo
from pilot import demo as pd

pytestmark = pytest.mark.unit

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600


def run_stdout(*argv: str) -> str:
    """The stdout of `sys.executable argv...` run in the repo root; the command must exit 0."""
    p = subprocess.run([sys.executable, *argv], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                       timeout=TIMEOUT)
    assert p.returncode == 0, "%s exited %d: %s" % (" ".join(argv), p.returncode, p.stderr)
    return p.stdout


def get(view: str, views: dict[str, dict[str, Any]] | None = None) -> Any:
    """GET /api/<view> of an app over `views` (default pd.build_views(REPO)), no model loaded: the response."""
    from starlette.testclient import TestClient
    with ThreadPoolExecutor(max_workers=1) as ex:
        app = demo.create_app(demo.new_state("cpu"), ex, pd.build_views(REPO) if views is None else views)
        with TestClient(app, base_url="http://127.0.0.1") as c:
            return c.get("/api/%s" % view)


def md_row(row: list[str]) -> str:
    return "| " + " | ".join(row) + " |"


def test_verdict_view_matches_report_renders() -> None:
    view = pd.verdict_view()
    r = view["renders"]
    for key, flag in (("go", "--go-block"), ("final_table", "--final-table"), ("recommendation", "--recommendation")):
        assert r[key] == run_stdout("report.py", flag, "results/final/verdict.json"), "renders[%s] != report.py %s" \
            % (key, flag)
    go_lines, rec_lines = r["go"].split("\n"), r["recommendation"].split("\n")
    lines = view["lines"]
    assert "**" + lines["verdict"] + "**" in go_lines
    assert "- " + lines["rule"] in go_lines
    assert "- " + lines["means"] in go_lines
    assert go_lines.index("- " + lines["means"]) == go_lines.index("- " + lines["rule"]) + 1
    assert "**" + lines["recommendation"] + "**" in rec_lines
    assert len(view["criteria"]) == 5
    for c in view["criteria"]:
        assert md_row([c["criterion"], c["status"], c["numbers"]]) in go_lines, c["criterion"]
        assert c["tone"] == pd.tone(c["status"])
    table_lines = r["final_table"].split("\n")
    for block in ("results", "settings"):
        assert md_row(view[block]["header"]) in table_lines
        assert view[block]["rows"], block
        for row in view[block]["rows"]:
            assert md_row(row) in table_lines, (block, row)
    with open(os.path.join(REPO, "results/final/verdict.json"), encoding="utf-8") as f:
        record = json.load(f)
    assert view["verdict"] == record["verdict"]["verdict"]
    assert view["means"] == record["verdict"]["means"]
    assert view["tone"] == pd.tone(record["verdict"]["verdict"])
    assert [(d["criterion"], d["metric"]) for d in view["deltas"]] == [(a, b) for a, b, *_ in pd.DELTA_ROWS]
    for d in view["deltas"]:
        assert d["delta_text"] in r["go"], d["delta_text"]
        assert d["ci_text"] in r["go"], d["ci_text"]
    assert view["caveats"] == list(pilot.memo.CAVEATS)
    assert view["sources"] == ["results/final/verdict.json", "protocol.json"]
    resp = get("verdict")
    assert resp.status_code == 200
    assert resp.json() == view


LAYA_RENDER = ("postfit.py", "--render", "results/postfit/laya.json", "results/postfit/laya-final.json")
BERTURK_RENDER = ("postfit.py", "--render-berturk", "results/postfit/berturk.json",
                  "results/postfit/berturk-reference.json", "results/postfit/berturk-final.json")


def read_json(rel: str) -> Any:
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return json.load(f)


def test_selection_view_matches_postfit_renders() -> None:
    s = pd.selection_view()
    laya, bert = s["systems"]["laya"], s["systems"]["berturk"]
    assert laya["render"] == run_stdout(*LAYA_RENDER)
    assert bert["render"] == run_stdout(*BERTURK_RENDER)
    laya_lines = laya["render"].split("\n")
    assert len(laya["edge_lines"]) == 4
    at = []
    for prefix, line in zip(pd.EDGE_PREFIXES, laya["edge_lines"]):
        assert ("- " + line).startswith(prefix), (prefix, line)
        assert "- " + line in laya_lines, line
        at.append(laya_lines.index("- " + line))
    assert at == sorted(at)
    assert bert["edge_lines"] == []
    assert [k for k in ("laya", "berturk") if s["systems"][k]["kind"] == k] == ["laya", "berturk"]
    for kind in ("laya", "berturk"):
        assert s["systems"][kind]["record"] == {"path": "results/postfit/%s.json" % kind,
                                                "sha256": pilot.data.sha256_file(
                                                    os.path.join(REPO, "results/postfit/%s.json" % kind))}
    resp = get("selection")
    assert resp.status_code == 200
    assert resp.json() == s


def test_selection_candidates_and_reference() -> None:
    import pilot.berturk
    import pilot.postfit
    s = pd.selection_view()
    for kind, n in (("laya", 4), ("berturk", 30)):
        view = s["systems"][kind]
        record = read_json("results/postfit/%s.json" % kind)
        assert len(view["candidates"]) == n == len(record["candidates"])
        for c, rc in zip(view["candidates"], record["candidates"]):
            o = rc["objective"]
            assert c["epoch"] == rc["epoch"]
            assert c["fraction"] == [2 * o["tp"], 2 * o["tp"] + o["fp"] + o["fn"]]
            assert c["score"] == float(pilot.postfit.score(rc))
            assert "%.4f (%d/%d)" % (c["val_micro_f1_at_global_theta"], *c["fraction"]) in view["render"]
            assert c["temperature"] == rc["temperature"]
            assert c["theta_global"] == rc["theta"]["shipped"]["global"]
        ch, rch = view["chosen"], record["chosen"]
        assert ch["epoch"] == rch["epoch"]
        assert ch["epochs_schedule"] == rch["epochs_schedule"]
        assert ch["temperature"] == rch["temperature"]
        assert ch["theta_shipped_global"] == rch["theta"]["shipped"]["global"]
        assert ch["theta_raw_global"] == rch["theta"]["raw"]["global"]
        assert ch["n_own_theta"] == len(rch["theta"]["shipped"]["per_tag"])
        assert ch["n_own_theta_raw"] == len(rch["theta"]["raw"]["per_tag"])
        assert ch["encoder_lr"] == rch["encoder_lr"]
        assert ch["head_lr"] == rch["canonical"]["head_lr"]
    ref, rref = s["reference"], read_json("results/postfit/berturk-reference.json")
    assert ref["fraction"] == rref["value"]["fraction"] == [389, 478]
    assert ref["value"] == rref["value"]["val_micro_f1_at_global_theta"]
    assert ref["threshold"] == rref["threshold"]
    assert ref["status"] == rref["status"]
    assert ref["tone"] == pd.tone(rref["status"])
    assert ref["reference"] == rref["reference"]
    assert ref["threshold_rule"] == rref["threshold_rule"]
    assert ref["record"]["path"] == "results/postfit/berturk-reference.json"
    assert [x["label"] for x in s["superseded"]] == ["superseded: berturk-tied-head-lr",
                                                     "superseded: berturk-10-epoch-grid"]
    for x, entry in zip(s["superseded"], pilot.berturk.SUPERSEDED):
        assert x["name"] == entry["name"]
        assert x["change"] == entry["change"]
        assert x["user_choice"] == entry["user_choice"]
        rec, refx = read_json(entry["record"]), read_json(entry["reference"])
        assert x["fraction"] == refx["value"]["fraction"]
        assert x["value"] == refx["value"]["val_micro_f1_at_global_theta"]
        assert x["candidates"] == [{"epoch": c["epoch"], "score": float(pilot.postfit.score(c))}
                                   for c in rec["candidates"]]
    assert s["sources"][:5] == ["results/postfit/laya.json", "results/postfit/laya-final.json",
                                "results/postfit/berturk.json", "results/postfit/berturk-final.json",
                                "results/postfit/berturk-reference.json"]


def tampered_verdict(tmp_path: Any) -> str:
    """A copy of the committed verdict whose criterion (a) status is edited to PASS (check_record recomputes it)."""
    import pilot.verdict
    v = read_json("results/final/verdict.json")
    assert v["criteria"]["a"]["status"] != pilot.verdict.PASS
    v["criteria"]["a"]["status"] = pilot.verdict.PASS
    path = os.path.join(str(tmp_path), "verdict.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(v, f, ensure_ascii=False, indent=2)
    return path


def test_tampered_verdict_refused_in_its_view_only(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import functools
    copy = tampered_verdict(tmp_path)
    with pytest.raises(SystemExit) as e:
        pd.verdict_view(path=copy)
    want = {"error": str(e.value.code).replace("\n", " "), "source": "results/final/verdict.json"}
    monkeypatch.setitem(pd.VIEWS, "verdict", functools.partial(pd.verdict_view, path=copy))
    views = pd.build_views(REPO)
    assert views["verdict"] == want
    for name in ("selection", "intents"):
        assert not pd.view_refused(views[name]), name
    assert views["selection"] == pd.selection_view()
    resp = get("verdict", views)
    assert resp.status_code == 503
    assert resp.json() == want
    for name in ("selection", "intents"):
        resp = get(name, views)
        assert resp.status_code == 200, name
        assert resp.json() == views[name]


def test_invalid_verdict_json_refused(tmp_path: Any) -> None:
    path = os.path.join(str(tmp_path), "verdict.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"schema": "verdict/1",')
    with pytest.raises(SystemExit) as e:
        pd.verdict_view(path=path)
    msg = str(e.value.code)
    assert "is not valid JSON" in msg
    assert "restore it with `git checkout -- results/final/verdict.json`" in msg


def test_changed_postfit_refused_in_selection_only(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil
    root = str(tmp_path / "repo")
    for rel in ("results/postfit", "results/superseded"):
        shutil.copytree(os.path.join(REPO, rel), os.path.join(root, rel))
    assert pd.selection_view(root) == pd.selection_view()     # the untouched copy is the committed view
    with open(os.path.join(root, "results/postfit/laya.json"), "ab") as f:
        f.write(b" ")
    with pytest.raises(SystemExit) as e:
        pd.selection_view(root)
    assert "protocol input results/postfit/laya.json changed" in str(e.value.code)
    assert "git checkout -- results/postfit/laya.json" in str(e.value.code)
    monkeypatch.setitem(pd.VIEWS, "selection", lambda _root: pd.selection_view(root))
    views = pd.build_views(REPO)
    assert views["selection"] == {"error": str(e.value.code).replace("\n", " "), "source": "results/postfit/laya.json"}
    assert views["verdict"] == pd.verdict_view()
    assert not pd.view_refused(views["intents"])
    assert get("selection", views).status_code == 503
    assert get("verdict", views).status_code == 200


def paths_of(x: Any) -> list[Any]:
    """Every value under a "path" key, anywhere in a JSON-like value."""
    if isinstance(x, dict):
        return [v for k, v in x.items() if k == "path"] + [p for v in x.values() for p in paths_of(v)]
    if isinstance(x, list):
        return [p for v in x for p in paths_of(v)]
    return []


def test_views_hold_no_local_paths() -> None:
    views = pd.build_views(REPO)
    assert not any(pd.view_refused(v) for v in views.values()), views
    text = json.dumps(views, ensure_ascii=False)
    assert os.path.realpath(REPO) not in text and REPO not in text
    assert os.path.expanduser("~") not in text
    paths = paths_of(views)
    assert paths
    for p in paths:
        assert isinstance(p, str) and p and not os.path.isabs(p), p
        assert not p.startswith("..") and not p.startswith("<outside-repo>") and "\\" not in p, p
        assert os.path.exists(os.path.join(REPO, p)), p
    for v in views.values():
        for p in v.get("sources", []):
            assert not os.path.isabs(p) and os.path.exists(os.path.join(REPO, p)), p


def test_status_tone() -> None:
    import pilot.verdict as pv
    assert pd.tone(pv.PASS) == "positive"
    assert pd.tone(pv.FAIL) == "negative"
    assert pd.tone(pv.INCONCLUSIVE) == "caution"
    for s in (pv.MET, pv.NOT_MET, pv.REPORTED, pv.PENDING, "ok", "investigate", None):
        assert pd.tone(s) == "neutral", s
