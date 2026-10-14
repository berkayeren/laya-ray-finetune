"""The Laya adapter contract (D-15): a laya version or private-signature change fails here, not silently in training.

(a) laya.__version__ must be exactly pilot.laya_adapter.LAYA_VERSION; (b) every wrapped laya symbol keeps the
inspect.signature string in SIGNATURES; (c) an offline tiny-model round trip (items -> encode -> collate -> forward ->
losses -> calibration records -> temperature fit -> save -> public laya.load -> noul logits) still runs, and the
adapter's logit route reproduces the public predict_batch probabilities.

SIGNATURES is the laya 0.4.6 snapshot. Pins move only with `make lock`; re-snapshot only after reading laya's changelog.
"""
import inspect
import json
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
