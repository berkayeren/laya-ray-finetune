"""score.py on a Laya checkpoint, offline: the tiny checkpoint (tests/tiny_laya.py) on the synthetic snapshot.

The dump must hold exactly the adapter's raw noul margins (la.noul_logits) in split-file order and the temperature
laya.load serves (la.noul_temperature), and report.py must read it like any baseline dump.
"""
import json
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest

import pilot.data
import report
import score
from conftest import SyntheticRaw, prepared
from intents import QUESTIONS
from pilot import dumps
from pilot import laya_adapter as la
from pilot.weights import identity
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.unit

TINY_T = 1.3


@pytest.fixture
def tiny(tmp_path: Path, synthetic_raw: SyntheticRaw) -> Iterator[tuple[str, str, str]]:
    """(out_dir, manifest_path, checkpoint dir) of the synthetic snapshot and a tiny checkpoint with T = 1.3.

    Scoring sets torch's process-wide thread count; it is restored so later tests keep the session's setting.
    """
    import torch
    threads = torch.get_num_threads()
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    yield out_dir, manifest, tiny_checkpoint(tmp_path / "ckpt", temperature=TINY_T)
    torch.set_num_threads(threads)


def _split_cases(path: str) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        return pilot.data.parse_jsonl(f.read())


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a dir the user cannot write")
def test_a_tokenizer_config_laya_could_not_rewrite_is_refused(tiny: tuple[str, str, str]) -> None:
    """On a read-only tokenizer/ laya.load only warns and loads the un-normalized config, which the
    identity (bound by the normalized view) does not tell apart from the rewritten one; score.load_agent refuses."""
    import warnings
    _, _, ckpt = tiny
    model_dir, expected = score.resolve_checkpoint(ckpt)
    before = identity(ckpt)
    tok_dir = os.path.join(ckpt, "tokenizer")
    os.chmod(tok_dir, 0o555)
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            with pytest.raises(SystemExit, match="checkpoint: laya could not rewrite tokenizer/tokenizer_config.json"):
                score.load_agent(model_dir, "cpu", expected)
        assert any("could not patch" in str(w.message) for w in seen)
    finally:
        os.chmod(tok_dir, 0o755)
    score.load_agent(model_dir, "cpu", expected)                     # writable: laya rewrites it and the load passes
    assert identity(ckpt) == before                                  # the rewrite keeps the identity


@pytest.mark.timeout(60)
@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a dir the user cannot write")
def test_score_with_ray_actors_refuses_a_read_only_tokenizer_dir_before_ray(tmp_path: Path, tiny: tuple[str, str, str],
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """With --workers 1 (the default) a refusal in the Ray Data actor's constructor would make Ray restart the actor
    forever, so the driver refuses a tokenizer-config rewrite laya cannot make before ray.init."""
    import ray

    def no_ray(*a: Any, **kw: Any) -> NoReturn:
        raise AssertionError("ray.init reached; the read-only tokenizer/ must be refused first")
    out_dir, manifest, ckpt = tiny
    test_jsonl = os.path.join(out_dir, "splits", "test.jsonl")
    tok_dir = os.path.join(ckpt, "tokenizer")
    monkeypatch.setattr(ray, "init", no_ray)
    os.chmod(tok_dir, 0o555)
    try:
        with pytest.raises(SystemExit, match="checkpoint: tokenizer/tokenizer_config.json needs laya's load-time "
                                             "rewrite, but tokenizer/ is not writable"):
            score.main([test_jsonl, "--model", "laya", "--checkpoint", ckpt, "--workers", "1",
                        "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    finally:
        os.chmod(tok_dir, 0o755)
    assert not (tmp_path / "dumps").exists()


def test_tiny_checkpoint_round_trips_through_score_and_report(tmp_path: Path, tiny: tuple[str, str, str]) -> None:
    import laya
    out_dir, manifest, ckpt = tiny
    test_jsonl = os.path.join(out_dir, "splits", "test.jsonl")
    before = identity(ckpt)
    run_dir, split = score.main([test_jsonl, "--model", "laya", "--checkpoint", ckpt, "--workers", "0",
                                 "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    ident = identity(ckpt)
    assert ident == before                  # laya's tokenizer_config.json rewrite on load keeps it
    assert sorted(ident["files_sha256"]) == ["encoder/config.json", "tokenizer/tokenizer.json",
                                             "tokenizer/tokenizer_config.json#normalized"]
    assert os.path.basename(run_dir) == "laya-%s-%s-%s" % (ident["model_sha256"][:12], ident["config_sha256"][:8],
                                                           dumps.files_digest(ident["files_sha256"])[:8])
    assert split == "test"
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
    assert sc["laya_default_amp"] is False and sc["laya_default_dtype"] == "torch.float32"   # laya's cpu default
    assert sc["workers"] == 0 and sc["laya_version"] == la.LAYA_VERSION
    assert side["system"]["kind"] == "laya" and side["system"]["checkpoint"] == "<outside-repo>/ckpt"
    assert side["system"]["checkpoint_identity"] == identity(ckpt)
    shipped = report.run_report(run_dir, "test", manifest)["variants"]["shipped"]
    assert shipped["temperature"] == TINY_T and shipped["rows"]["0.5"]["n"] == 7


# --- seed and training provenance in the sidecar -------------------------------------------------------------------

TRAIN_CANON = {"split": "selection_train", "seed": 3, "neg_seed": 3, "epochs": 4, "encoder_lr": 3e-05, "head_lr": 0.0001}


def _as_trained(src: str, dst: Path, epoch: int = 2, **canon: Any) -> str:
    """Copy the tiny checkpoint to dst and give it a ray_train.py training block (canonical TRAIN_CANON + canon)."""
    import shutil
    shutil.copytree(src, dst)
    path = os.path.join(str(dst), "rl_agent_config.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["fine_tuned"] = True
    cfg["training"] = {"ray_train": {"run_id": "laya-selection_train-neg8-s3-0123abcd",
                                     "canonical": dict(TRAIN_CANON, **canon),
                                     "record": {"rung": "none", "fallback": False, "weaker": False,
                                                "epochs": [{"loss": 0.5}, {"loss": 0.4}]}},
                       "epoch": epoch, "calibrated": False}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return str(dst)


def _score_val(tiny: tuple[str, str, str], ckpt: str, root: Path) -> tuple[str, dict[str, Any]]:
    out_dir, manifest, _ = tiny
    run_dir, _ = score.main([os.path.join(out_dir, "splits", "val.jsonl"), "--checkpoint", ckpt, "--workers", "0",
                             "--manifest", manifest, "--out-root", str(root)])
    with open(os.path.join(run_dir, "val.json"), encoding="utf-8") as f:
        return run_dir, json.load(f)


def test_sidecar_records_the_training_seed_and_config(tmp_path: Path, tiny: tuple[str, str, str]) -> None:
    ckpt = _as_trained(tiny[2], tmp_path / "s3")
    run_dir, side = _score_val(tiny, ckpt, tmp_path / "dumps")
    assert side["seed"] == 3
    t = side["training"]
    assert t["run_id"] == "laya-selection_train-neg8-s3-0123abcd" and t["split"] == "selection_train"
    assert t["epoch"] == 2 and t["epochs_schedule"] == 4 and t["stop_after_epoch"] is None
    assert t["encoder_lr"] == 3e-05 and t["head_lr"] == 0.0001 and t["neg_seed"] == 3
    assert (t["rung"], t["fallback"], t["weaker"]) == ("none", False, False)
    assert t["loss_by_epoch"] == [0.5, 0.4] and t["calibrated"] is False and t["postfit_sha256"] is None
    assert t["canonical"] == TRAIN_CANON
    assert t["config_sha256_sans_seed"] == dumps.training_digest(TRAIN_CANON, 2)
    assert not {"state", "text"} & set(t)
    # another seed of the same recipe: another dump run dir, the same seed-free digest
    other_dir, other = _score_val(tiny, _as_trained(tiny[2], tmp_path / "s4", seed=4, neg_seed=4), tmp_path / "dumps")
    assert other_dir != run_dir and other["seed"] == 4
    assert other["training"]["config_sha256_sans_seed"] == t["config_sha256_sans_seed"]
    # another learning rate or epoch is another recipe
    assert dumps.training_digest(dict(TRAIN_CANON, encoder_lr=2e-05), 2) != t["config_sha256_sans_seed"]
    assert dumps.training_digest(TRAIN_CANON, 3) != t["config_sha256_sans_seed"]
    assert dumps.training_digest(dict(TRAIN_CANON, split="final_train"), 2) != t["config_sha256_sans_seed"]
    assert dumps.training_digest(dict(TRAIN_CANON, seed=9, neg_seed=7), 2) == t["config_sha256_sans_seed"]


def test_training_block_is_none_without_a_ray_train_config(tmp_path: Path, tiny: tuple[str, str, str]) -> None:
    _, _, ckpt = tiny
    assert score.training_block(ckpt, identity(ckpt)) == (None, None)
    assert score.training_block("multilingual", None) == (None, None)       # no fetch, no identity needed
    _, side = _score_val(tiny, ckpt, tmp_path / "dumps")
    assert side["seed"] is None and side["training"] is None


def test_training_block_refuses_a_config_changed_after_hashing(tmp_path: Path, tiny: tuple[str, str, str]) -> None:
    ckpt = _as_trained(tiny[2], tmp_path / "s3")
    ident = identity(ckpt)
    assert score.training_block(ckpt, ident)[0] == 3
    path = os.path.join(ckpt, "rl_agent_config.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["training"]["ray_train"]["canonical"]["seed"] = 5
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    with pytest.raises(SystemExit, match="changed after it was hashed; rescore"):
        score.training_block(ckpt, ident)


def test_check_checkpoint_arg_refuses_an_invalid_config(tmp_path: Path) -> None:
    """A corrupt rl_agent_config.json is a refusal naming the file, not a traceback (checked with the --checkpoint
    argument: the protocol gate keys on the identity, never on the config's contents)."""
    (tmp_path / "rl_agent_config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="rl_agent_config.json is not valid JSON"):
        score.check_checkpoint_arg(str(tmp_path))


# --- Ray reassembly and the Predictor actor (one code path for in-process and Ray scoring) -------------------------

def _rows(*uids: str) -> list[dict[str, Any]]:
    return [{"uid": u, "logits": np.full(48, float(ord(u[-1])), dtype=np.float32)} for u in uids]


def test_assemble_reorders_by_uid() -> None:
    got = score.assemble(_rows("c", "a", "b"), ["a", "b", "c"])
    assert got.dtype == np.float32 and got.shape == (3, 48)
    assert list(got[:, 0]) == [float(ord("a")), float(ord("b")), float(ord("c"))]


@pytest.mark.parametrize("rows,uids,match", [
    (_rows("a", "a", "b"), ["a", "b"], "duplicate uid a"),
    (_rows("a"), ["a", "b"], "uid b was not scored"),
    (_rows("a", "b", "x"), ["a", "b"], "unexpected uid x"),
], ids=["dup", "missing", "extra"])
def test_assemble_refuses_duplicate_missing_and_extra(rows: list[dict[str, Any]], uids: list[str], match: str) -> None:
    with pytest.raises(SystemExit, match=match):
        score.assemble(rows, uids)


def test_predictor_returns_logits_and_state(tiny: tuple[str, str, str]) -> None:
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
    assert state["laya_default_amp"] is False and state["laya_default_dtype"] == "torch.float32"
    assert state["torch_threads"] == 1
    secs = list(out["infer_secs"])
    assert len(secs) == 3 and len(set(secs)) == 1 and secs[0] >= 0


# --- refusals: agent state, arguments, checkpoint -------------------------------------------------------------------

def _agent(device: str, dtype: str = "torch.float32", amp: bool = False, **kw: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(device=types.SimpleNamespace(type=device), dtype=dtype, amp_enabled=amp, **kw)


@pytest.mark.parametrize("agent,device,match", [
    (_agent("cpu", "torch.bfloat16"), "cpu", r"torch.bfloat16 with autocast False on cpu; scoring is fp32$"),
    (_agent("cpu", amp=True), "cpu", r"laya runs torch.float32 with autocast True on cpu; scoring is fp32$"),
    (_agent("mps", "torch.float16", True), "mps", r"runs torch.float16 with autocast True on mps; scoring is fp32"),
    (_agent("mps", amp=True), "mps", r"autocast True on mps; scoring is fp32$"),
    (_agent("cuda", "torch.float16", True), "cuda", r"runs torch.float16 with autocast True on cuda; scoring is fp32"),
    (_agent("cuda", "torch.bfloat16", False), "cuda", r"torch.bfloat16 with autocast False on cuda; scoring is fp32"),
    (_agent("cpu"), "cuda", "laya loaded on cpu, requested cuda"),
    (_agent("cpu"), "mps", "laya loaded on cpu, requested mps"),
    (_agent("mps", cpu_fallback_count=1, last_fallback_reason="MPS backend out of memory"), "mps",
     r"fell back from mps to CPU 1 time\(s\) mid-run \(MPS backend out of memory\)"),
], ids=["cpu_bf16", "cpu_amp_on", "mps_fp16_amp", "mps_amp_on", "cuda_fp16_amp", "cuda_bf16", "cpu_for_cuda",
        "cpu_for_mps", "oom_fallback"])
def test_check_agent_refuses(agent: Any, device: str, match: str) -> None:
    with pytest.raises(SystemExit, match=match):
        score.check_agent(agent, device)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_check_agent_passes_fp32_and_returns_its_state(device: str) -> None:
    assert score.check_agent(_agent(device, cpu_fallback_count=0), device) == {
        "device": device, "dtype": "torch.float32", "amp_enabled": False}


def test_predictor_rechecks_the_agent_after_every_batch(tiny: tuple[str, str, str],
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """laya's _infer can demote the agent mid-run (GPU OOM -> CPU); the batch that saw it is refused, not dumped."""
    out_dir, _, ckpt = tiny
    model_dir, expected = score.resolve_checkpoint(ckpt)
    p = score.Predictor(model_dir, "cpu", expected, 1)
    case = _split_cases(os.path.join(out_dir, "splits", "test.jsonl"))[0]
    batch = {"uid": np.array([case["uid"]]), "text": np.array([case["state"]])}
    p(batch)
    real = la.noul_logits

    def demoting(agent: Any, texts: list[str], questions: dict[str, Any]) -> np.ndarray:
        out = real(agent, texts, questions)
        setattr(agent, "cpu_fallback_count", 1)             # what laya's OOM retry leaves behind
        return out
    monkeypatch.setattr(la, "noul_logits", demoting)
    with pytest.raises(SystemExit, match="fell back from cpu to CPU 1 time"):
        p(batch)


@pytest.mark.parametrize("args,match", [
    (["--model", "prior", "--device", "cpu"], "--device applies only to --model laya"),
    (["--model", "majority", "--device", "mps"], "--device applies only to --model laya"),
    (["--device", "cpu", "--gpus-per-worker", "1"], "it conflicts with --device cpu"),
    (["--device", "mps", "--gpus-per-worker", "1", "--workers", "0"], "it conflicts with --device mps"),
    (["--device", "cuda", "--workers", "2"], "--device cuda with Ray actors needs --gpus-per-worker > 0"),
    (["--device", "mps", "--workers", "1"], "--device mps scores in-process: use --workers 0"),
    (["--device", "mps"], "--device mps scores in-process: use --workers 0"),          # --workers defaults to 1
    (["--device", "tpu"], "invalid choice"),
], ids=["prior", "majority", "cpu_gpu", "mps_gpu", "cuda_no_gpu", "mps_ray", "mps_default_workers", "unknown"])
def test_device_conflicts_are_argument_errors(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                              capsys: pytest.CaptureFixture[str], args: list[str], match: str) -> None:
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    with pytest.raises(SystemExit) as e:
        score.main([os.path.join(out_dir, "splits", "test.jsonl"), *args, "--manifest", manifest,
                    "--out-root", str(tmp_path / "dumps")])
    assert e.value.code == 2 and match in capsys.readouterr().err
    assert not (tmp_path / "dumps").exists()


@pytest.mark.parametrize("args,want", [
    ([], "cpu"), (["--gpus-per-worker", "1"], "cuda"), (["--device", "cuda", "--workers", "0"], "cuda"),
    (["--device", "cuda", "--gpus-per-worker", "1"], "cuda"), (["--device", "mps", "--workers", "0"], "mps"),
    (["--device", "cpu"], "cpu"),
], ids=["default_cpu", "gpu_implies_cuda", "cuda_inprocess", "cuda_ray", "mps_inprocess", "explicit_cpu"])
def test_device_resolution(tmp_path: Path, synthetic_raw: SyntheticRaw, monkeypatch: pytest.MonkeyPatch,
                           args: list[str], want: str) -> None:
    """Omitting --device keeps today's behaviour (cuda iff --gpus-per-worker > 0); the resolved device reaches
    score_laya, which is stubbed so no model loads."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    seen = []

    def stop(a: Any, checkpoint: str, cases: list[dict[str, Any]], device: str, ident: Any) -> NoReturn:
        seen.append(device)
        raise SystemExit("stub")
    monkeypatch.setattr(score, "score_laya", stop)
    with pytest.raises(SystemExit, match="stub"):
        score.main([os.path.join(out_dir, "splits", "test.jsonl"), *args, "--manifest", manifest,
                    "--out-root", str(tmp_path / "dumps")])
    assert seen == [want]


def test_mps_unavailable_is_refused_before_any_load(tmp_path: Path, tiny: tuple[str, str, str],
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    out_dir, manifest, ckpt = tiny
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(score, "load_agent", lambda *x: pytest.fail("loaded a model"))
    with pytest.raises(SystemExit) as e:
        score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--checkpoint", ckpt, "--device", "mps",
                    "--workers", "0", "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    assert "--device mps, but MPS is not available here; use --device cpu" in str(e.value.code)
    assert not (tmp_path / "dumps").exists()


def test_baseline_with_checkpoint_is_an_argument_error(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    with pytest.raises(SystemExit) as e:
        score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "prior", "--checkpoint", "x",
                    "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    assert e.value.code == 2 and "--checkpoint applies only to --model laya" in capsys.readouterr().err
    assert not (tmp_path / "dumps").exists()


def test_score_refuses_missing_checkpoint_before_ray(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    monkeypatch.chdir(tmp_path)                             # a stray ray_results/ would land in the cwd
    with pytest.raises(SystemExit) as e:
        score.main([os.path.join(out_dir, "splits", "test.jsonl"), "--model", "laya", "--checkpoint", "no_such_ckpt",
                    "--manifest", manifest, "--out-root", str(tmp_path / "dumps")])
    assert "--checkpoint must be" in str(e.value.code)
    assert not (tmp_path / "dumps").exists() and not (tmp_path / "ray_results").exists()
    assert "ray" not in sys.modules or not sys.modules["ray"].is_initialized()
