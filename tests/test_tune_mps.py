"""tune.py on the Apple GPU with the tiny models: a short search of each system on --device mps, one trial at a
time, every trial training and scoring tuning-dev on mps in fp32.

The searches reuse the CPU smoke's enlarged synthetic snapshot and tiny bases (tests/test_tune_smoke.py). BERTurk
uses the smoke's three pinned points (strong, weak, strong), Laya four points ordered for MPS (LAYA_POINTS: strong,
strong, weak, strong). ASHA must stop at least one trial, and every trial started after the first stopped one must
still train and report (the stopped trial released the GPU). The BERTurk search runs the HF Trainer inside a Tune
trial on mps. No weights are kept and nothing under results/, data/ or protocol.json changes. The quiet-machine
preflight runs with --min-available-gib 0, recorded as an override.

Needs Apple MPS, so it is excluded from `make test`: run `pytest -m mps tests/test_tune_mps.py` on the M4, with no
other MPS job running.
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import pilot.data
from pilot import tune as pt
from test_tune_smoke import BERTURK_POINTS, _guarded_status, snapshot  # noqa: F401 (snapshot: fixture)

pytestmark = pytest.mark.mps

REPO = pilot.data.REPO_ROOT
TIMEOUT = 1800
# MPS is not bit-reproducible: with the CPU smoke's Laya order the strong 4-epoch trial 0 can score 0.0 at the epoch-3
# rung, so the weak trial is never below the cutoff. Two strong trials go first (a rung-3 cutoff above the weak
# trial's predict-everything score), then the weak one at the space's lowest LRs, then a strong one that must run
# after the stop.
LAYA_POINTS = [{"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 6},
               {"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 8},
               {"encoder_lr": 5e-06, "head_lr": 3e-05, "neg": 4, "epochs": 4},
               {"encoder_lr": 0.0001, "head_lr": 0.001, "neg": 8, "epochs": 4}]


def _require_mps() -> None:
    import torch
    if not torch.backends.mps.is_available():
        pytest.fail("the mps marker needs Apple MPS; run it on the M4, or deselect it with -m 'not mps'")


def _run(*args: Any) -> subprocess.CompletedProcess:
    """Run a repo script with the test interpreter (no PYTHONPATH); assert it succeeds."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable] + [str(x) for x in args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TIMEOUT)
    assert r.returncode == 0, "%s failed (exit %d):\n%s" % (args[0], r.returncode, r.stderr[-4000:])
    return r


def _search(system: str, points: list[dict[str, Any]], base: str, data: str, manifest: str, storage: Path,
            out: Path) -> dict[str, Any]:
    """Run the MPS search of `system` over `points` and return its checked record, after the assertions both systems
    share."""
    before = _guarded_status()
    _run("tune.py", "--system", system, "--device", "mps", "--cpus", "4", "--min-available-gib", "0",
         "--trials", len(points), "--seed", "0", "--base", base, "--data", data, "--manifest", manifest,
         "--storage", storage, "--out", out, "--points", json.dumps(points))
    record, _sha = pt.read_record(str(out))
    assert record["system"] == system and record["schema"] == pt.SCHEMA
    assert record["device"] == "mps" and record["search"]["points_to_evaluate"] == points
    assert [t["config"] for t in record["trials"]] == points
    assert len(record["segments"]) == 1
    pf = record["segments"][0]["preflight"]
    assert pf is not None and pf["override"] is True and pf["min_available_gib"] == 0, pf
    stopped = [t["index"] for t in record["trials"] if t["asha_stopped"]]
    assert stopped and record["n_asha_stopped"] == len(stopped), record["trials"]
    # every trial after the first stopped one trained and reported: the stopped trial freed the GPU
    after = [t for t in record["trials"] if t["index"] > stopped[0]]
    assert after and all(t["curve"] for t in after), [(t["index"], len(t["curve"])) for t in record["trials"]]
    assert all(t["curve"] for t in record["trials"])
    assert all(t["attempts"] == 1 for t in record["trials"])
    for root, dirs, files in os.walk(str(storage)):
        assert not [n for n in files if n.endswith(".safetensors")], root
        assert "exports" not in dirs and not [n for n in dirs if n.startswith("epoch-")], root
    assert _guarded_status() == before                      # nothing under results/, data/ or protocol.json changed
    return record


@pytest.mark.timeout(TIMEOUT)
def test_laya_search_on_mps(snapshot: tuple[Path, str, str, str, str]) -> None:  # noqa: F811
    """The Laya search on mps: every trial's one-worker TorchTrainer trains on the Apple GPU (the worker refuses a
    parameter off mps) and scores tuning-dev in fp32; an ASHA stop is followed by a trial that trains."""
    _require_mps()
    d, data, manifest, base, _ = snapshot
    _search("laya", LAYA_POINTS, base, data, manifest, d / "mps-laya", d / "mps-laya.json")


@pytest.mark.timeout(TIMEOUT)
def test_berturk_search_on_mps(snapshot: tuple[Path, str, str, str, str]) -> None:  # noqa: F811
    """The BERTurk search on mps: berturk_train.fit's HF Trainer runs inside a Tune trial with every parameter on mps
    in float32, autocast off; an ASHA stop is followed by a trial that trains."""
    _require_mps()
    d, data, manifest, _, bert = snapshot
    storage = d / "mps-berturk"
    _search("berturk", BERTURK_POINTS, bert, data, manifest, storage, d / "mps-berturk.json")
    done = glob.glob(os.path.join(str(storage), "berturk-train", "trial_*-a1", "*", "done.json"))
    assert done                                             # at least one trial ran to its last epoch
    for path in done:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)["record"]
        assert (rec["device"], rec["param_device"], rec["dtype"], rec["amp"]) == ("mps", "mps", "torch.float32",
                                                                                  False), path
