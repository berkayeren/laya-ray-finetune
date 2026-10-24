"""latency.py on the Apple GPU: a tiny Laya final seed and a tiny BERTurk export are timed per message on real MPS
through score.py's load path, every parameter on mps in float32 with autocast off, and the device is synchronised
exactly as pilot.latency.time_messages promises: once after the warm-up pass over every message (the
full_pass_per_seed policy, recorded as warmup n), then right before t0 and right after the host copy of every timed
message (1 + 2 x n calls of torch.mps.synchronize per seed).

The tiny checkpoints, the synthetic postfit-final/1 manifests and the recorded preflight come from
tests/test_latency_cpu.py (the BERTurk export is trained on CPU there; it is timed on MPS here). Needs Apple MPS, so
it is excluded from `make test`: run `pytest -m mps tests/test_latency_mps.py` on the M4, with no other MPS job running.
"""
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

import latency
from pilot import latency as plat
# berturk_built is the CPU test's module-scoped fixture (a tiny BERTurk export), imported so pytest uses it here
from test_latency_cpu import PREFLIGHT, berturk_built, berturk_setup, laya_setup, run_args, val_cases

pytestmark = pytest.mark.mps

TIMEOUT = 600
MPS_STATE = {"device": "mps", "dtype": "torch.float32", "amp_enabled": False}


@pytest.fixture
def counted_sync(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """torch.mps.synchronize wrapped by a counter that still calls the original; the preflight is a recorded dict and
    torch's thread count is restored afterwards."""
    import torch
    if not torch.backends.mps.is_available():
        pytest.fail("the mps marker needs Apple MPS; run it on the M4, or deselect it with -m 'not mps'")
    threads = torch.get_num_threads()
    monkeypatch.setattr(latency, "preflight", lambda min_gib: dict(PREFLIGHT, min_available_gib=min_gib))
    original, calls = torch.mps.synchronize, []

    def counting() -> Any:
        calls.append(1)
        return original()
    monkeypatch.setattr(torch.mps, "synchronize", counting)
    yield calls
    torch.set_num_threads(threads)


def mps_args(system: str, splits: str, manifest: str, final: str, root: str, out: Path) -> list[str]:
    """run_args of the CPU test, moved to --device mps."""
    argv = run_args(system, splits, manifest, final, root, out)
    argv[argv.index("--device") + 1] = "mps"
    return argv


def check_mps_report(report: dict[str, Any], splits: str, system: str, calls: list[int]) -> None:
    """Assert an mps latency/1 report's shape and the sync count: one after the warm-up pass, two per timed message."""
    n = len(val_cases(splits))
    assert report["system"] == system and report["device"] == "mps"
    assert report["warmup"] == n and report["warmup_policy"] == plat.WARMUP_POLICY
    assert report["dtype"] == "torch.float32" and report["amp_enabled"] is False
    assert report["split"]["n_messages"] == n > 0
    assert [s["state"] for s in report["seeds"]] == [MPS_STATE] * len(report["seeds"])
    for s in report["seeds"]:
        assert len(s["times_s"]) == s["summary"]["n"] == n and s["summary"]["min"] > 0
    assert len(calls) == len(report["seeds"]) * (1 + 2 * n), (len(calls), n)   # the pass's sync, two per message


@pytest.mark.timeout(TIMEOUT)
def test_tiny_models_latency_on_mps(tmp_path: Path, synthetic_raw: Callable[..., tuple[str, dict[str, str]]],
                                    berturk_built: tuple[str, str, str], counted_sync: list[int]) -> None:
    splits, manifest, m, root = laya_setup(tmp_path, synthetic_raw)
    report = latency.main(mps_args("laya", splits, manifest, m, root, tmp_path / "latency" / "laya-mps.json"))
    check_mps_report(report, splits, "laya", counted_sync)

    del counted_sync[:]
    b_splits, b_manifest, _ = berturk_built
    bm, b_root = berturk_setup(berturk_built, tmp_path)
    report = latency.main(mps_args("berturk", b_splits, b_manifest, bm, b_root,
                                   tmp_path / "latency" / "berturk-mps.json"))
    check_mps_report(report, b_splits, "berturk", counted_sync)
