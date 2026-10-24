"""The Laya adapter contract: a laya version or private-signature change fails here, not silently in training.

(a) laya.__version__ must be exactly pilot.laya_adapter.LAYA_VERSION; (b) every wrapped laya symbol keeps the
inspect.signature string in SIGNATURES; (c) an offline tiny-model round trip (items -> encode -> collate -> forward ->
losses -> save -> public laya.load -> noul logits) still runs, and the adapter's logit route reproduces the public
predict_batch probabilities; (d) no tracked .py file names laya's item-level calibration (the temperature is fitted
on val dumps by pilot.postfit; la.fit_temperature_map is its test oracle in tests/test_postfit.py); (e) no tracked .py
file but the adapter uses laya's private API.

SIGNATURES is the laya 0.4.6 snapshot. Pins move only with `make lock`; re-snapshot only after reading laya's changelog.
"""
import ast
import inspect
import json
import os
import subprocess
import sys
import types
import warnings
from pathlib import Path
from typing import Any

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
    "laya.train.encode_item": "(tok, item: Dict[str, Any], max_len: int, head_max_len: int, "
                              "option_order: Optional[List[int]] = None, parallel: bool = False) -> Dict[str, Any]",
    "laya.train._forward": "(model, batch, device, amp: bool, detach_encoder: bool)",
    "laya.train.rlcd_loss": "(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, qtype: torch.Tensor, "
                            "sigma: float, samples: int = 4, w_sph: float = 0.75, w_rps: float = 1.0) -> torch.Tensor",
    "laya.train.soft_ce_loss": "(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor",
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
    "laya.agent._fix_tokenizer_config": "(path: str)",
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
def tiny_agent(tmp_path_factory: pytest.TempPathFactory) -> Any:
    import laya
    path = tiny_checkpoint(tmp_path_factory.mktemp("tiny") / "c", temperature=1.3)
    return laya.load(path, device="cpu")


def test_laya_version_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    import laya
    la.check_version()
    for bad in ("0.4.60", "0.4.6.post1", "0.4.5"):
        monkeypatch.setattr(laya, "__version__", bad)
        with pytest.raises(SystemExit, match="pinned to 0.4.6"):
            la.check_version()


def test_wrapped_signatures_match_snapshot() -> None:
    have = la.wrapped()
    assert set(have) == set(SIGNATURES), sorted(set(have) ^ set(SIGNATURES))
    diff = ["%s\n  have %s\n  want %s" % (n, inspect.signature(have[n]), SIGNATURES[n])
            for n in sorted(SIGNATURES) if str(inspect.signature(have[n])) != SIGNATURES[n]]
    assert not diff, ("laya changed: read laya's changelog before re-snapshotting (pins move only with `make lock`)\n"
                      + "\n".join(diff))


def _rows() -> list[dict[str, Any]]:
    tags = TAGS[:3]
    return [{"state": "Kartım kayboldu", "uid": "u0", "questions": {t: QUESTIONS[t] for t in tags},
             "expected": {tags[0]: True, tags[1]: False, tags[2]: False}},
            {"state": "Evet.", "uid": "u1", "questions": {t: QUESTIONS[t] for t in tags},
             "expected": {tags[0]: False, tags[1]: False, tags[2]: True}}]


def test_training_round_trip_on_tiny_model(tmp_path: Path) -> None:
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

    assert la.optimizer_updates(10, 4, 2, 1) == 2
    assert la.sigma_at(0, 2, 0.4, 0.1) == 0.4
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert la.read_jsonl(str(path)) == rows


def test_items_per_question_counts_items_by_their_intents_instruction(tmp_path: Path) -> None:
    """The few-shot masking proof's counter: items_per_question maps each laya item to its tag by the exact INTENTS
    instruction of its question, over all 48 tags: unmasked rows give one item per labelled question, a deleted
    (masked) key gives none, and an item whose instruction is not one of the 48 is refused."""
    from intents import INTENTS
    _model, tok, _cfg = la.load_checkpoint(tiny_checkpoint(tmp_path / "base"))
    rows = _rows()
    tags = TAGS[:3]
    per_q = la.items_per_question(la.items_from_rows(tok, rows, 64, 32)[0])
    assert list(per_q) == list(INTENTS) and per_q[tags[0]] == per_q[tags[1]] == per_q[tags[2]] == 2
    assert sum(per_q.values()) == 6
    masked = [dict(r, expected={t: v for t, v in r["expected"].items() if t != tags[1]}) for r in rows]
    items = la.items_from_rows(tok, masked, 64, 32)[0]
    per_q = la.items_per_question(items)
    assert per_q[tags[1]] == 0 and per_q[tags[0]] == per_q[tags[2]] == 2 and sum(per_q.values()) == len(items) == 4
    bad = dict(items[0], q=dict(items[0]["q"], ins=items[0]["q"]["ins"] + " "))
    with pytest.raises(SystemExit, match="training item 1 asks .* not one of the 48 intents.py instructions"):
        la.items_per_question([items[0], bad])
    with pytest.raises(SystemExit, match="training item 0 asks None"):
        la.items_per_question([{"k": 2}])


def test_noul_logits_match_predict_batch(tiny_agent: Any) -> None:
    texts = ["Kartım kayboldu 0", "Evet.", "x"]
    m = la.noul_logits(tiny_agent, texts, QUESTIONS)
    assert m.dtype == np.float32 and m.shape == (3, 48)
    t = la.noul_temperature(tiny_agent)
    assert t == pytest.approx(1.3)
    p = 1.0 / (1.0 + np.exp(-m.astype(np.float64) / t))
    out = tiny_agent.predict_batch(texts, QUESTIONS, batch_size=8, sort_by_length=True)
    want = np.array([[out[i]["answers"][tag]["noul"] for tag in TAGS] for i in range(len(texts))])
    assert np.allclose(p, want, atol=1e-4, rtol=0), float(np.abs(p - want).max())


def test_noul_logits_empty_single_and_order(tiny_agent: Any) -> None:
    empty = la.noul_logits(tiny_agent, [], QUESTIONS)
    assert empty.shape == (0, 48) and empty.dtype == np.float32
    assert la.noul_logits(tiny_agent, ["Evet."], QUESTIONS).shape == (1, 48)
    a, b = "Kartım kayboldu", "Evet."
    ab, ba = la.noul_logits(tiny_agent, [a, b], QUESTIONS), la.noul_logits(tiny_agent, [b, a], QUESTIONS)
    assert np.allclose(ab, ba[::-1], atol=1e-6, rtol=0)


def test_force_fp32_turns_autocast_off(tmp_path: Path, tiny_agent: Any) -> None:
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


def test_noul_logits_refuses_bad_records(tiny_agent: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(recs: list[Any]) -> types.SimpleNamespace:
        return types.SimpleNamespace(records_from_labeled=lambda agent, pairs: recs)
    good = (2, np.array([0.0, 1.0], dtype=np.float32), [0.0, 1.0], 2)
    monkeypatch.setattr(la, "_lcal", lambda: fake([good] * 47))
    with pytest.raises(SystemExit, match="47 records for 1 messages x 48 questions"):
        la.noul_logits(tiny_agent, ["Evet."], QUESTIONS)
    for bad in ((0, good[1], good[2], 2), (2, good[1], good[2], 3), (2, np.zeros(3, np.float32), good[2], 2)):
        monkeypatch.setattr(la, "_lcal", lambda b=bad: fake([good] * 47 + [b]))
        with pytest.raises(SystemExit, match="record 47 is not a 2-option noul"):
            la.noul_logits(tiny_agent, ["Evet."], QUESTIONS)


# ---- no private laya use outside the adapter (tests included, no exceptions) ----

def _private(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in PRIVATE)


def _private_laya_uses(path: str, label: str | None = None, autocast_ok: bool = False) -> list[str]:
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


def test_guard_no_private_laya_use_outside_adapter() -> None:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    files = [f for f in out.split() if f != ADAPTER]
    assert ADAPTER in out.split() and {"ray_train.py", "mps_spike.py", "tests/test_spike_unit.py"} <= set(files)
    bad = [b for f in files for b in _private_laya_uses(os.path.join(REPO, f), f, f in AUTOCAST_WRITERS)]
    assert not bad, "private laya use outside %s (route it through the adapter):\n%s" % (ADAPTER, "\n".join(bad))


# laya's item-level calibration split, records and report, and the train-message holdout that fed them (all removed:
# the temperature is fitted on message-level val dumps by pilot.postfit)
ITEM_CALIBRATION = ("split_calibration", "calibration_records", "calibration_report", "holdout_messages")


def _item_calibration_uses(path: str, label: str | None = None) -> list[str]:
    """"path:line name" for every Name, Attribute, function definition or import alias named in ITEM_CALIBRATION;
    string constants are not code, so a test may still name them."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    bad = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [node.name]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n for a in node.names for n in (a.name.split(".")[-1], a.asname) if n]
        bad += ["%s:%d %s" % (label or path, node.lineno, n) for n in names if n in ITEM_CALIBRATION]
    return bad


def test_no_item_level_calibration_anywhere(tmp_path: Path) -> None:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    files = out.split()
    assert ADAPTER in files and "ray_train.py" in files
    bad = [b for f in files for b in _item_calibration_uses(os.path.join(REPO, f), f)]
    assert not bad, "item-level calibration is gone (T comes from pilot.postfit):\n%s" % "\n".join(bad)
    assert not set(ITEM_CALIBRATION) & set(dir(la))
    p = tmp_path / "bad.py"                                  # the guard sees every form
    p.write_text("from laya.train import split_calibration\n"
                 "import x.calibration_report as calibration_records\n"
                 "la.calibration_records(1)\n"
                 "def holdout_messages():\n"
                 "    pass\n"
                 "ok = 'split_calibration'\n", encoding="utf-8")
    found = {b.split()[-1] for b in _item_calibration_uses(str(p))}
    assert found == set(ITEM_CALIBRATION), found


def test_guard_catches_every_form(tmp_path: Path) -> None:
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


def test_laya_internals_covered() -> None:
    import mps_spike
    missing = [n for n in mps_spike.LAYA_INTERNALS
               if n not in la.COVERS or not callable(getattr(la, la.COVERS[n], None))]
    assert not missing, missing


def test_adapter_import_is_torch_free() -> None:
    code = ("import sys, pilot.laya_adapter, mps_spike\n"
            "print(sorted(m for m in ('torch', 'ray', 'laya.train', 'laya.common', 'laya.calibrate', 'laya.agent') "
            "if m in sys.modules))")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", r.stdout


@pytest.mark.parametrize("tcfg", [
    {"backend": "tokenizers", "is_local": True, "tokenizer_class": "TokenizersBackend", "pad_token": "[PAD]"},
    {"pad_token": "[PAD]", "model_max_length": 128},                          # no tokenizer_class
    {"tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": ["<a>", "<b>"]},
    {"tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {"extra_0": "<a>"}, "backend": "x"},
    {"tokenizer_class": "BertTokenizer", "do_lower_case": False},
])
def test_the_tokenizer_config_mirror_matches_laya(tmp_path: Path, tcfg: dict[str, Any]) -> None:
    """A local identity binds tokenizer_config.json by pilot.weights.normalized_tokenizer_config, a
    mirror of the rewrite laya.load makes; it must equal what laya writes, and laya's output must be a fixed point."""
    from pilot import weights
    path = tmp_path / "tokenizer" / "tokenizer_config.json"
    path.parent.mkdir()
    path.write_text(json.dumps(tcfg), encoding="utf-8")
    before = weights.tokenizer_config_sha256(str(path))
    la.fix_tokenizer_config(str(tmp_path))
    written = json.loads(path.read_text(encoding="utf-8"))
    assert weights.normalized_tokenizer_config(tcfg) == written
    assert weights.normalized_tokenizer_config(written) == written
    assert weights.tokenizer_config_sha256(str(path)) == before


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a dir the user cannot write")
def test_a_tokenizer_config_rewrite_laya_could_not_make_is_refused(tmp_path: Path) -> None:
    """laya only warns when it cannot rewrite tokenizer_config.json (here a read-only tokenizer/) and
    loads the raw file. pilot.weights.check_tokenizer_config_rewritten, which ray_train's worker runs after
    la.load_checkpoint and score.load_agent after laya.load, refuses that file, and passes once the rewrite is made."""
    from pilot import weights
    ckpt = tiny_checkpoint(tmp_path / "base")                 # transformers saves tokenizer_class TokenizersBackend
    tok_dir = os.path.join(ckpt, "tokenizer")
    os.chmod(tok_dir, 0o555)
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            la.load_checkpoint(ckpt)
        assert any("could not patch" in str(w.message) for w in seen), [str(w.message) for w in seen]
        with pytest.raises(SystemExit, match="base checkpoint: laya could not rewrite tokenizer/tokenizer_config.json"):
            weights.check_tokenizer_config_rewritten(ckpt, "base checkpoint")
    finally:
        os.chmod(tok_dir, 0o755)
    la.load_checkpoint(ckpt)
    weights.check_tokenizer_config_rewritten(ckpt, "base checkpoint")
