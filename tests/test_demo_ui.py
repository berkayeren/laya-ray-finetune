"""The demo UI (ui/, Vite + React + TypeScript): the built bundle served by demo.create_app on 127.0.0.1, and the npm
outputs gitignored.

test_built_bundle_is_served needs `npm --prefix ui ci && npm --prefix ui run build` first (it skips, naming that
command, while ui/dist/index.html is absent); every other test reads only tracked files. Runs in `make test`.
"""
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

import demo
import pilot.data

pytestmark = pytest.mark.unit

REPO = pilot.data.REPO_ROOT
UI = os.path.join(REPO, "ui")
BUILD = "npm --prefix ui ci && npm --prefix ui run build"


def ignored(rel: str) -> bool:
    """True when `git check-ignore --no-index` ignores the repo-relative path rel."""
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", "--", rel], cwd=REPO).returncode == 0


def test_built_bundle_is_served() -> None:
    if not os.path.isfile(os.path.join(demo.UI_DIST, "index.html")):
        pytest.skip("ui/dist/index.html is absent; build the UI first: %s" % BUILD)
    from starlette.testclient import TestClient
    with ThreadPoolExecutor(max_workers=1) as ex:
        app = demo.create_app(demo.new_state("cpu"), ex, {}, demo.UI_DIST)
        with TestClient(app, base_url="http://127.0.0.1") as c:
            r = c.get("/")
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/html")
            assert 'id="root"' in r.text
            assets = sorted(set(re.findall(r'(?:src|href)="(/assets/[^"]+)"', r.text)))
            assert assets, "the served index.html references no /assets/ file"
            for path in assets:
                a = c.get(path)
                assert a.status_code == 200, "%s answered %d" % (path, a.status_code)
            s = c.get("/api/status")
            assert s.status_code == 200
            assert s.json()["chat"] == "loading"


def test_ui_outputs_are_gitignored() -> None:
    for rel in ("ui/node_modules/x/index.js", "ui/dist/index.html"):
        assert ignored(rel), "%s must be gitignored" % rel
    for rel in ("ui/src/App.tsx", "ui/package-lock.json"):
        assert not ignored(rel), "%s must not be gitignored" % rel
