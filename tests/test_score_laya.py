"""score.py on a Laya checkpoint, offline: the tiny checkpoint (tests/tiny_laya.py) on the synthetic snapshot.

The dump must hold exactly the adapter's raw noul margins (la.noul_logits) in split-file order and the temperature
laya.load serves (la.noul_temperature), and report.py must read it like any baseline dump.
"""
import json
import os
import sys
import types

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
    """(out_dir, manifest_path, checkpoint dir) of the synthetic snapshot and a tiny checkpoint with T = 1.3.

    Scoring sets torch's process-wide thread count; it is restored so later tests keep the session's setting.
    """
    import torch
    threads = torch.get_num_threads()
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    yield out_dir, manifest, tiny_checkpoint(tmp_path / "ckpt", temperature=TINY_T)
    torch.set_num_threads(threads)


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


# --- Ray reassembly and the Predictor actor (one code path for in-process and Ray scoring) -------------------------

def _rows(*uids):
    return [{"uid": u, "logits": np.full(48, float(ord(u[-1])), dtype=np.float32)} for u in uids]


def test_assemble_reorders_by_uid():
    got = score.assemble(_rows("c", "a", "b"), ["a", "b", "c"])
    assert got.dtype == np.float32 and got.shape == (3, 48)
    assert list(got[:, 0]) == [float(ord("a")), float(ord("b")), float(ord("c"))]


@pytest.mark.parametrize("rows,uids,match", [
    (_rows("a", "a", "b"), ["a", "b"], "duplicate uid a"),
    (_rows("a"), ["a", "b"], "uid b was not scored"),
    (_rows("a", "b", "x"), ["a", "b"], "unexpected uid x"),
], ids=["dup", "missing", "extra"])
def test_assemble_refuses_duplicate_missing_and_extra(rows, uids, match):
    with pytest.raises(SystemExit, match=match):
        score.assemble(rows, uids)


def test_predictor_returns_logits_and_state(tiny):
    out_dir, _, ckpt = tiny
    model_dir, expected = score.resolve_checkpoint(ckpt)
    p = score.Predictor(model_dir, "cpu", expected, 1)
    cases = _split_cases(os.path.join(out_dir, "splits", "test.jsonl"))[:3]
    out = p({"uid": np.array([c["uid"] for c in cases]), "text": np.array([c["state"] for c in cases])})
    assert list(out["uid"]) == [c["uid"] for c in cases]
    assert out["logits"].dtype == np.float32 and out["logits"].shape == (3, 48)
    assert list(out["temperature"]) == [TINY_T] * 3
    states = {s for s in out["agent_state"]}
    assert len(states) == 1
    state = json.loads(states.pop())
    assert state["device"] == "cpu" and state["dtype"] == "torch.float32" and state["amp_enabled"] is False
    assert state["torch_threads"] == 1
    secs = list(out["infer_secs"])
    assert len(secs) == 3 and len(set(secs)) == 1 and secs[0] >= 0


# --- refusals: agent state, arguments, checkpoint -------------------------------------------------------------------

@pytest.mark.parametrize("agent,device,match", [
    (types.SimpleNamespace(device="cpu", dtype="torch.bfloat16", amp_enabled=False), "cpu", "scoring is fp32"),
    (types.SimpleNamespace(device="cpu", dtype="torch.float32", amp_enabled=True), "cpu", "LAYA_CPU_AMP"),
    (types.SimpleNamespace(device="cpu", dtype="torch.float32", amp_enabled=False), "cuda", "requested cuda"),
], ids=["bf16_dtype", "amp_on", "device_mismatch"])
def test_check_agent_refuses(agent, device, match):
    with pytest.raises(SystemExit, match=match):
        score.check_agent(agent, device)


def test_check_agent_passes_fp32_cpu_and_returns_its_state():
    agent = types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"), dtype="torch.float32", amp_enabled=False)
    assert score.check_agent(agent, "cpu") == {"device": "cpu", "dtype": "torch.float32", "amp_enabled": False}


def test_baseline_with_checkpoint_is_an_argument_error(tmp_path, synthetic_raw, capsys):
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    with pytest.raises(SystemExit) as e:
        score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "prior", "--checkpoint", "x",
                    "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    assert e.value.code == 2 and "--checkpoint applies only to --model laya" in capsys.readouterr().err
    assert not (tmp_path / "dumps").exists()


def test_score_refuses_missing_checkpoint_before_ray(tmp_path, synthetic_raw, monkeypatch):
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    monkeypatch.chdir(tmp_path)                             # a stray ray_results/ would land in the cwd
    with pytest.raises(SystemExit) as e:
        score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "laya", "--checkpoint", "no_such_ckpt",
                    "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    assert "--checkpoint must be" in str(e.value.code)
    assert not (tmp_path / "dumps").exists() and not (tmp_path / "ray_results").exists()
    assert "ray" not in sys.modules or not sys.modules["ray"].is_initialized()
