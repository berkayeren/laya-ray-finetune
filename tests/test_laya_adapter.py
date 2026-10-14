"""The Laya adapter contract (D-15): a laya version or private-signature change fails here, not silently in training.

(a) laya.__version__ must be exactly pilot.laya_adapter.LAYA_VERSION; (b) every wrapped laya symbol keeps the
inspect.signature string in SIGNATURES; (c) an offline tiny-model round trip (items -> encode -> collate -> forward ->
losses -> calibration records -> temperature fit -> save -> public laya.load -> noul logits) still runs, and the
adapter's logit route reproduces the public predict_batch probabilities.

SIGNATURES is the laya 0.4.6 snapshot. Pins move only with `make lock`; re-snapshot only after reading laya's changelog.
"""
import ast
import inspect
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from intents import QUESTIONS
from pilot import laya_adapter as la
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.unit

SIGNATURES = {
    "laya.train.load_checkpoint": "(model_dir: str)",
    "laya.train.items_from_rows": "(tok, rows: Iterable[Dict[str, Any]], max_len: int, head_max_len: int, "
                                  "label_smoothing: float = 0.0) -> Tuple[List[Dict[str, Any]], Dict[str, int]]",
    "laya.train.split_calibration": "(items: Sequence[Any], calib_max: int, calib_frac: float, seed: int) "
                                    "-> Tuple[List[Any], List[Any]]",
    "laya.train.encode_item": "(tok, item: Dict[str, Any], max_len: int, head_max_len: int, "
                              "option_order: Optional[List[int]] = None, parallel: bool = False) -> Dict[str, Any]",
    "laya.train._forward": "(model, batch, device, amp: bool, detach_encoder: bool)",
    "laya.train.rlcd_loss": "(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, qtype: torch.Tensor, "
                            "sigma: float, samples: int = 4, w_sph: float = 0.75, w_rps: float = 1.0) -> torch.Tensor",
    "laya.train.soft_ce_loss": "(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor",
    "laya.train.calibration_records": "(model, tok, items: Sequence[Dict[str, Any]], device: torch.device, "
                                      "max_len: int, head_max_len: int, batch_size: int = 16, parallel: bool = False)",
    "laya.train.calibration_report": "(records: Sequence[Tuple[int, Any, Any, int]], temperature: Sequence[float]) "
                                     "-> Dict[str, Dict[str, Any]]",
    "laya.train.save_checkpoint": "(model, tok, cfg: Dict[str, Any], path: str) -> None",
    "laya.train.sigma_at": "(epoch: int, epochs: int, start: float, end: float) -> float",
    "laya.train.draw_option_order": "(item: Dict[str, Any], rng: random.Random, shuffle_types: Sequence[str]) "
                                    "-> Optional[List[int]]",
    "laya.train.optimizer_updates": "(n_items: int, micro_batch: int, grad_accum: int, epochs: int) -> int",
    "laya.train.read_jsonl": "(path: str) -> List[Dict[str, Any]]",
    "laya.common.collate_items": "(batch, pad_id: int)",
    "laya.common.uses_parallel_layout": "(cfg: Dict) -> bool",
    "laya.common.DecisionModel": "(encoder: torch.nn.modules.module.Module, head_layers: int = 2, n_act: int = 2, "
                                 "dropout: float = 0.1, no_init: bool = False)",
    "laya.common.temp_bucket": "(qtype: int, k: int) -> str",
    "laya.calibrate.fit_temperature_map": "(records: Iterable, compute_ece: bool = False, seed: int = 0) "
                                          "-> Dict[str, Any]",
    "laya.calibrate.records_from_labeled": "(agent, pairs: Sequence) -> List[Tuple[int, Any, Any, int]]",
    "laya.load": "(model_id_or_path: str = 'convaiinnovations/laya', device: Optional[str] = None, "
                 "token: Optional[str] = None, subfolder: Optional[str] = None, fast: bool = False, "
                 "compile: bool = False, revision: Optional[str] = None, "
                 "expected_sha256: Optional[Dict[str, str]] = None, "
                 "lang_temperatures: Optional[Dict[str, Dict[str, Any]]] = None, hooks=None, on_predict_start=None, "
                 "on_predict_end=None, hooks_raise: bool = True, hooks_concurrent: bool = True, "
                 "hooks_timeout: Optional[float] = None, calibration: Optional[str] = None, "
                 "backend: Optional[str] = None, onnx_path: Optional[str] = None, compile_warmup: bool = True, "
                 "compile_cache: bool = False, compile_mode: str = 'default') "
                 "-> Union[laya.agent.Agent, ForwardRef('ONNXAgent')]",
}
TAGS = list(QUESTIONS)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER = "pilot/laya_adapter.py"
PRIVATE = ("laya.train", "laya.common", "laya.calibrate", "laya.agent")
INTERNALS = ("head_checkpointing", "gradient_checkpointing_enable", "tok_embeddings", "records_from_labeled")
# an agent's autocast switch and dtype (laya's autocast reads both): writable only through la.force_fp32, as an
# attribute store or a setattr with the constant name; reads stay allowed. AUTOCAST_WRITERS may write them: this file
# sets laya's non-fp32 load-time defaults to test force_fp32 (an explicit entry, not a bypass)
AUTOCAST = ("amp_enabled", "dtype")
AUTOCAST_WRITERS = ("tests/test_laya_adapter.py",)


@pytest.fixture(scope="module")
def tiny_agent(tmp_path_factory):
    import laya
    path = tiny_checkpoint(tmp_path_factory.mktemp("tiny") / "c", temperature=1.3)
    return laya.load(path, device="cpu")


def test_laya_version_is_pinned(monkeypatch):
    import laya
    la.check_version()
    for bad in ("0.4.60", "0.4.6.post1", "0.4.5"):
        monkeypatch.setattr(laya, "__version__", bad)
        with pytest.raises(SystemExit, match="pinned to 0.4.6"):
            la.check_version()


def test_wrapped_signatures_match_snapshot():
    have = la.wrapped()
    assert set(have) == set(SIGNATURES), sorted(set(have) ^ set(SIGNATURES))
    diff = ["%s\n  have %s\n  want %s" % (n, inspect.signature(have[n]), SIGNATURES[n])
            for n in sorted(SIGNATURES) if str(inspect.signature(have[n])) != SIGNATURES[n]]
    assert not diff, ("laya changed: read laya's changelog before re-snapshotting (pins move only with `make lock`)\n"
                      + "\n".join(diff))


def _rows():
    tags = TAGS[:3]
    return [{"state": "Kartım kayboldu", "uid": "u0", "questions": {t: QUESTIONS[t] for t in tags},
             "expected": {tags[0]: True, tags[1]: False, tags[2]: False}},
            {"state": "Evet.", "uid": "u1", "questions": {t: QUESTIONS[t] for t in tags},
             "expected": {tags[0]: False, tags[1]: False, tags[2]: True}}]


def test_training_round_trip_on_tiny_model(tmp_path):
    import torch
    model, tok, cfg = la.load_checkpoint(tiny_checkpoint(tmp_path / "base"))
    rows = _rows()
    items, skipped = la.items_from_rows(tok, rows, 64, 32)
    assert len(items) == 6, (len(items), skipped)
    batch = la.collate_items([[la.encode_item(tok, it, 64, 32, None, la.uses_parallel_layout(cfg)) for it in items]],
                             tok.pad_token_id)
    model.train()
    logits = la.forward(model, batch, torch.device("cpu"), False, False)
    rlcd = la.rlcd_loss(logits, batch["target"], batch["marker_mask"], batch["qtype"], 0.4)
    soft = la.soft_ce_loss(logits, batch["target"], batch["marker_mask"])
    for loss in (rlcd, soft):
        assert loss.dim() == 0 and bool(torch.isfinite(loss)), loss
    rlcd.backward()
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in head)

    records = la.calibration_records(model.eval(), tok, items, torch.device("cpu"), 64, 32)
    assert len(records) == len(items)
    fitted = la.fit_temperature_map(records)
    assert "temperature" in fitted and "temperature_by_options" in fitted, sorted(fitted)
    assert isinstance(la.calibration_report(records, fitted["temperature"]), dict)

    assert [len(x) for x in la.split_calibration(list(range(100)), 40, 0.1, 0)] == [90, 10]
    assert la.optimizer_updates(10, 4, 2, 1) == 2
    assert la.sigma_at(0, 2, 0.4, 0.1) == 0.4
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert la.read_jsonl(str(path)) == rows


def test_noul_logits_match_predict_batch(tiny_agent):
    texts = ["Kartım kayboldu 0", "Evet.", "x"]
    m = la.noul_logits(tiny_agent, texts, QUESTIONS)
    assert m.dtype == np.float32 and m.shape == (3, 48)
    t = la.noul_temperature(tiny_agent)
    assert t == pytest.approx(1.3)
    p = 1.0 / (1.0 + np.exp(-m.astype(np.float64) / t))
    out = tiny_agent.predict_batch(texts, QUESTIONS, batch_size=8, sort_by_length=True)
    want = np.array([[out[i]["answers"][tag]["noul"] for tag in TAGS] for i in range(len(texts))])
    assert np.allclose(p, want, atol=1e-4, rtol=0), float(np.abs(p - want).max())


def test_noul_logits_empty_single_and_order(tiny_agent):
    empty = la.noul_logits(tiny_agent, [], QUESTIONS)
    assert empty.shape == (0, 48) and empty.dtype == np.float32
    assert la.noul_logits(tiny_agent, ["Evet."], QUESTIONS).shape == (1, 48)
    a, b = "Kartım kayboldu", "Evet."
    ab, ba = la.noul_logits(tiny_agent, [a, b], QUESTIONS), la.noul_logits(tiny_agent, [b, a], QUESTIONS)
    assert np.allclose(ab, ba[::-1], atol=1e-6, rtol=0)


def test_force_fp32_turns_autocast_off(tmp_path, tiny_agent):
    """force_fp32 returns the load-time state and leaves an fp32 agent with autocast off whose margins equal those
    of an agent that was never switched (tiny_agent, cpu fp32 since load)."""
    import laya
    import torch
    agent = laya.load(tiny_checkpoint(tmp_path / "c", temperature=1.3), device="cpu")
    agent.amp_enabled = True                                # allowed here only (AUTOCAST_WRITERS)
    agent.dtype = torch.bfloat16                            # laya's cuda default on sm_80+; LAYA_CPU_AMP's on cpu
    before = la.force_fp32(agent)
    assert before == {"amp_enabled": True, "dtype": "torch.bfloat16"}
    assert agent.amp_enabled is False and agent.dtype == torch.float32
    texts = ["Kartım kayboldu 0", "Evet."]
    got, want = la.noul_logits(agent, texts, QUESTIONS), la.noul_logits(tiny_agent, texts, QUESTIONS)
    assert np.allclose(got, want, atol=1e-6, rtol=0), float(np.abs(got - want).max())
    assert la.force_fp32(agent) == {"amp_enabled": False, "dtype": "torch.float32"}


def test_noul_logits_refuses_bad_records(tiny_agent, monkeypatch):
    def fake(recs):
        return types.SimpleNamespace(records_from_labeled=lambda agent, pairs: recs)
    good = (2, np.array([0.0, 1.0], dtype=np.float32), [0.0, 1.0], 2)
    monkeypatch.setattr(la, "_lcal", lambda: fake([good] * 47))
    with pytest.raises(SystemExit, match="47 records for 1 messages x 48 questions"):
        la.noul_logits(tiny_agent, ["Evet."], QUESTIONS)
    for bad in ((0, good[1], good[2], 2), (2, good[1], good[2], 3), (2, np.zeros(3, np.float32), good[2], 2)):
        monkeypatch.setattr(la, "_lcal", lambda b=bad: fake([good] * 47 + [b]))
        with pytest.raises(SystemExit, match="record 47 is not a 2-option noul"):
            la.noul_logits(tiny_agent, ["Evet."], QUESTIONS)


# ---- D-13/D-14: no private laya use outside the adapter (tests included, no exceptions) ----

def _private(name):
    return any(name == m or name.startswith(m + ".") for m in PRIVATE)


def _private_laya_uses(path, label=None, autocast_ok=False):
    """"path:line what" for every private laya import (any form), laya.<private> attribute chain, dynamic import of a
    private module, DecisionModel internal / logit-route attribute, or write to an agent's autocast switch or dtype
    (`x.amp_enabled = ...`, `setattr(x, "dtype", ...)`; reads are fine; autocast_ok skips only these writes). Strings
    are not code: only the first argument of importlib.import_module / __import__ and the second of setattr are
    read."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    bad = []
    for node in ast.walk(tree):
        what = None
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names if _private(a.name)]
            what = "import %s" % ", ".join(names) if names else None
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if _private(node.module):
                what = "from %s import ..." % node.module
            elif node.module == "laya":
                names = [a.name for a in node.names if _private("laya." + a.name)]
                what = "from laya import %s" % ", ".join(names) if names else None
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "laya" and _private("laya." + node.attr):
                what = "laya.%s" % node.attr
            elif node.attr in INTERNALS:
                what = ".%s" % node.attr
            elif node.attr in AUTOCAST and isinstance(node.ctx, ast.Store) and not autocast_ok:
                what = ".%s = (write)" % node.attr
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
            arg = node.args[0] if node.args else None
            if name in ("import_module", "__import__") and isinstance(arg, ast.Constant) \
                    and isinstance(arg.value, str) and _private(arg.value):
                what = "%s(%r)" % (name, arg.value)
            elif name == "setattr" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) \
                    and node.args[1].value in AUTOCAST and not autocast_ok:
                what = "setattr(..., %r) (write)" % node.args[1].value
        if what:
            bad.append("%s:%d %s" % (label or path, node.lineno, what))
    return bad


def test_guard_no_private_laya_use_outside_adapter():
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    files = [f for f in out.split() if f != ADAPTER]
    assert ADAPTER in out.split() and {"ray_train.py", "mps_spike.py", "tests/test_spike_unit.py"} <= set(files)
    bad = [b for f in files for b in _private_laya_uses(os.path.join(REPO, f), f, f in AUTOCAST_WRITERS)]
    assert not bad, "private laya use outside %s (route it through the adapter):\n%s" % (ADAPTER, "\n".join(bad))


def test_guard_catches_every_form(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("import laya.train\n"
                 "import laya.common as c\n"
                 "from laya.calibrate import fit_temperature_map\n"
                 "from laya import train as lt\n"
                 "from laya import agent\n"
                 "x = laya.common.collate_items\n"
                 "importlib.import_module(\"laya.train\")\n"
                 "__import__(\"laya.calibrate\")\n"
                 "m.head_checkpointing = True\n"
                 "m.encoder.gradient_checkpointing_enable()\n"
                 "m.encoder.embeddings.tok_embeddings\n"
                 "c.records_from_labeled(a, p)\n"
                 "agent.amp_enabled = False\n"
                 "agent.dtype = torch.bfloat16\n"
                 "setattr(agent, \"amp_enabled\", True)\n"
                 "setattr(agent, \"dtype\", torch.float16)\n", encoding="utf-8")
    found = _private_laya_uses(str(p))
    assert len(found) == 16 and sorted(int(b.split(":")[-1].split()[0]) for b in found) == list(range(1, 17)), found
    assert len(_private_laya_uses(str(p), autocast_ok=True)) == 12                # the allow-list skips only writes
    q = tmp_path / "clean.py"
    q.write_text("import laya\n"
                 "laya.load(path)\n"
                 "from pilot import laya_adapter as la\n"
                 "la.forward(m, b, d, False, False)\n"
                 "s = \"laya.train\"\n"
                 "ok = agent.amp_enabled is False\n"
                 "d = agent.dtype\n"
                 "setattr(agent, \"temperature\", 1.0)\n"
                 "la.force_fp32(agent)\n", encoding="utf-8")
    assert _private_laya_uses(str(q)) == []


def test_laya_internals_covered():
    import mps_spike
    missing = [n for n in mps_spike.LAYA_INTERNALS
               if n not in la.COVERS or not callable(getattr(la, la.COVERS[n], None))]
    assert not missing, missing


def test_adapter_import_is_torch_free():
    code = ("import sys, pilot.laya_adapter, mps_spike\n"
            "print(sorted(m for m in ('torch', 'ray', 'laya.train', 'laya.common', 'laya.calibrate', 'laya.agent') "
            "if m in sys.modules))")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", r.stdout
