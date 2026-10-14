"""score.py on a Laya checkpoint, offline: the tiny checkpoint (tests/tiny_laya.py) on the synthetic snapshot.

The dump must hold exactly the adapter's raw noul margins (la.noul_logits) in split-file order and the temperature
laya.load serves (la.noul_temperature), and report.py must read it like any baseline dump.
"""
import json
import os

import numpy as np
import pytest

import pilot.data
import report
import score
from conftest import prepared
from intents import QUESTIONS
from pilot import laya_adapter as la
from pilot.weights import identity
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.unit

TINY_T = 1.3


@pytest.fixture
def tiny(tmp_path, synthetic_raw):
    """(out_dir, manifest_path, checkpoint dir) of the synthetic snapshot and a tiny checkpoint with T = 1.3."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    return out_dir, manifest, tiny_checkpoint(tmp_path / "ckpt", temperature=TINY_T)


def _split_cases(path):
    with open(path, "rb") as f:
        return pilot.data.parse_jsonl(f.read())


def test_tiny_checkpoint_round_trips_through_score_and_report(tmp_path, tiny):
    import laya
    out_dir, manifest, ckpt = tiny
    test_jsonl = os.path.join(out_dir, "splits", "test.jsonl")
    run_dir, split = score.main([test_jsonl, "--model", "laya", "--checkpoint", ckpt, "--workers", "0",
                                 "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    assert os.path.basename(run_dir) == "laya-" + identity(ckpt)["model_sha256"][:12] and split == "test"
    cases = _split_cases(test_jsonl)
    z = np.load(os.path.join(run_dir, "test.npz"), allow_pickle=False)
    assert list(z["uids"]) == [c["uid"] for c in cases]
    want = la.noul_logits(laya.load(ckpt, device="cpu"), [c["state"] for c in cases], QUESTIONS)
    assert z["logits"].dtype == np.float32 and z["logits"].shape == (len(cases), 48)
    np.testing.assert_allclose(z["logits"], want, rtol=0, atol=1e-6)
    with open(os.path.join(run_dir, "test.json"), encoding="utf-8") as f:
        side_text = f.read()
    assert str(tmp_path) not in side_text and os.path.expanduser("~") not in side_text
    side = json.loads(side_text)
    assert side["temperature"]["shipped"] == TINY_T
    sc = side["scorer"]
    assert sc["device"] == "cpu" and sc["amp_enabled"] is False and sc["dtype"] == "torch.float32"
    assert sc["workers"] == 0 and sc["laya_version"] == la.LAYA_VERSION
    assert side["system"]["kind"] == "laya" and side["system"]["checkpoint"] == "<outside-repo>/ckpt"
    assert side["system"]["checkpoint_identity"] == identity(ckpt)
    shipped = report.run_report(run_dir, "test", manifest)["variants"]["shipped"]
    assert shipped["temperature"] == TINY_T and shipped["rows"]["0.5"]["n"] == 7
