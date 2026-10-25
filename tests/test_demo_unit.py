"""The demo's record views, offline and model-free: the verdict view against report.py's --go-block, --final-table and
--recommendation renders of the committed results/final/verdict.json, byte for byte, with every verdict string a whole
line (or a table row) of a frozen render; served by demo.create_app's GET /api/<view> without any model loaded.
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
