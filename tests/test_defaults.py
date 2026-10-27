"""The tuned defaults of new training runs (pilot/defaults.py) and what they must never move.

- pilot.defaults.LAYA and BERTURK are exactly the `best.config` of the committed search records results/tune/laya.json
  and results/tune/berturk.json, read through pilot.tune.read_record at their pinned sha256s and compared by JSON text,
  so a rounded value or an int written for a float (0 for 0.0) fails; SOURCES names each record, its sha256, the best
  trial and its epoch
- pilot.defaults imports nothing outside the stdlib, no other pilot module, and reads no file

Every test is unit-marked and offline.
"""
import ast
import json
import os
import subprocess
import sys

import pytest

from pilot import defaults
from pilot import tune as pt
from pilot.data import REPO_ROOT

pytestmark = pytest.mark.unit

SYSTEMS = ("laya", "berturk")
CONSTANTS = {"laya": defaults.LAYA, "berturk": defaults.BERTURK}
# the committed records' bytes, pinned (the same sha256s tests/test_tune_results.py pins)
RECORD_SHA256 = {"laya": "b111b6e21db971fa765650c46a9534d530317fce620cea7f9e9f0c61f26feef5",
                 "berturk": "acb143c13bc3b64dfbab5c819a435790e64671c39422654cc2342f2f10d3e70d"}
INT_KEYS = ("neg", "epochs", "batch")
DEFAULTS_PY = os.path.join(REPO_ROOT, "pilot", "defaults.py")


def test_defaults_are_the_best_trials_of_the_committed_search() -> None:
    for s in SYSTEMS:
        src = defaults.SOURCES[s]
        assert src["record"] == "results/tune/%s.json" % s
        rec, sha = pt.read_record(os.path.join(REPO_ROOT, src["record"]))
        assert sha == RECORD_SHA256[s] == src["sha256"]
        best, constant = rec["best"], CONSTANTS[s]
        assert json.dumps(constant, sort_keys=True) == json.dumps(best["config"], sort_keys=True)
        assert list(constant) == list(best["config"])
        assert src["trial"] == best["index"]
        assert src["epoch"] == best["epoch"] == constant["epochs"]
        for key, v in constant.items():
            if key in INT_KEYS:
                assert type(v) is int, (s, key, v)
            else:
                assert type(v) is float, (s, key, v)
    assert repr(defaults.BERTURK["weight_decay"]) == "0.0"
    assert set(defaults.SOURCES) == set(SYSTEMS)


def test_defaults_module_imports_the_stdlib_only() -> None:
    code = ("import sys; before = set(sys.modules); import pilot.defaults; new = set(sys.modules) - before; "
            "bad = sorted(m for m in new if m.split('.')[0] not in sys.stdlib_module_names and m.split('.')[0] != "
            "'pilot') + sorted(m for m in new if m.startswith('pilot.') and m != 'pilot.defaults'); print(bad)")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"
    with open(DEFAULTS_PY, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            assert name != "open", "pilot/defaults.py reads a file at line %d" % node.lineno
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names, alias.name
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and (node.module or "").split(".")[0] in sys.stdlib_module_names, node.module
