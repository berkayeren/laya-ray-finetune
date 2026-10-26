"""ray_train.tune_score, the per-epoch tuning-dev score a tune.py search trial runs on rank 0, between optimizer steps
on the live model: it must leave the training trajectory untouched (every parameter and the RNG state byte-identical
to an unscored run), restore the training modes, refuse non-fp32 scoring, and return exactly pilot.tune's objective.

CPU torch on the offline tiny checkpoint. The training loop here is a test fixture mirroring the worker's optimizer
step (the encoder and head AdamW groups, the cosine schedule, the RLCD loss at laya's sigma anneal, the gradient clip
at ray_train.CLIP_GRAD_NORM); it never ships.
"""
import random
from typing import Any

import numpy as np
import pytest

ray_train = pytest.importorskip("ray_train")                # imports ray, torch and laya (the locked env has them)

import torch                                                # noqa: E402

import pilot.tune                                           # noqa: E402
from intents import INTENTS, QUESTIONS                      # noqa: E402
from pilot import laya_adapter as la                        # noqa: E402
from tiny_laya import tiny_checkpoint                       # noqa: E402

pytestmark = pytest.mark.unit

TAGS = list(INTENTS)
CPU = torch.device("cpu")
MB, ACCUM, EPOCHS = 4, 2, 2
OBJECTIVE_KEYS = ["dev_micro_f1", "dev_theta", "dev_macro_f1", "dev_tp", "dev_fp", "dev_fn"]


def _row(uid: str, text: str, positives: tuple[str, ...]) -> dict[str, Any]:
    """A row in the split schema: all 48 questions, all 48 tags labelled."""
    return {"state": text, "uid": uid, "questions": QUESTIONS, "expected": {t: t in positives for t in TAGS}}


TRAIN_ROWS = [_row("bank_0_0", "Kartım kayboldu", (TAGS[0],)), _row("bank_0_1", "Evet .", (TAGS[1], TAGS[2]))]
DEV_ROWS = [_row("bank_12_0", "Kartım kayboldu .", (TAGS[0],)), _row("bank_13_0", "Evet", ()),
            _row("bank_13_1", "kayboldu Kartım", (TAGS[3],))]


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory: pytest.TempPathFactory) -> str:
    return tiny_checkpoint(tmp_path_factory.mktemp("tiny") / "c")


def _load(path: str, freeze_encoder: bool = False) -> tuple[Any, Any, dict[str, Any]]:
    """The worker's model set-up: load, freeze or checkpoint, then train mode (the frozen encoder in eval)."""
    model, tok, cfg = la.load_checkpoint(path)
    if freeze_encoder:
        la.freeze_encoder(model)
    else:
        assert la.enable_gradient_checkpointing(model)
    model.train()
    if freeze_encoder:
        model.encoder.eval()
    return model, tok, cfg


def _score(model: Any, tok: Any, cfg: dict[str, Any], freeze_encoder: bool = False) -> dict[str, Any]:
    return ray_train.tune_score(model, tok, DEV_ROWS, cfg["max_len"], cfg["head_max_len"],
                                la.uses_parallel_layout(cfg), CPU, freeze_encoder)


def _train(path: str, score_after: int | None) -> tuple[Any, Any, dict[str, Any] | None]:
    """Four optimizer steps of the worker's recipe from a fixed torch seed; tune_score after optimizer step
    `score_after` (None: never). Returns (model, torch RNG state at the end, the score or None)."""
    torch.manual_seed(0)
    model, tok, cfg = _load(path)
    ml, hml, parallel = cfg["max_len"], cfg["head_max_len"], la.uses_parallel_layout(cfg)
    items, _skipped = la.items_from_rows(tok, TRAIN_ROWS, ml, hml)
    groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("encoder.")], "lr": 2e-05},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": 1e-04}]
    opt = torch.optim.AdamW(groups, weight_decay=ray_train.WEIGHT_DECAY)
    per_epoch = -(-(-(-len(items) // MB)) // ACCUM)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * per_epoch, eta_min=1e-6)
    params = [p for g in groups for p in g["params"]]
    rng = random.Random(0)
    steps, score = 0, None
    for epoch in range(EPOCHS):
        order = list(items)
        random.Random(1000 * epoch).shuffle(order)
        sigma = la.sigma_at(epoch, EPOCHS, 0.4, 0.1)
        opt.zero_grad(set_to_none=True)
        n = 0
        for s in range(0, len(order), MB):
            batch = la.collate_items([[la.encode_item(tok, it, ml, hml, la.draw_option_order(it, rng, ("choice",)),
                                                      parallel) for it in order[s:s + MB]]], tok.pad_token_id)
            logits = la.forward(model, batch, CPU, False, False)
            loss = la.rlcd_loss(logits, batch["target"], batch["marker_mask"], batch["qtype"], sigma)
            (loss / ACCUM).backward()
            n += 1
            if n % ACCUM == 0 or s + MB >= len(order):
                torch.nn.utils.clip_grad_norm_(params, ray_train.CLIP_GRAD_NORM)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
                if steps == score_after:
                    score = _score(model, tok, cfg)
                if steps == 4:
                    return model, torch.get_rng_state(), score
    raise AssertionError("fewer than 4 optimizer steps (%d)" % steps)


def test_tune_score_leaves_training_unchanged(ckpt: str) -> None:
    """Scoring tuning-dev after optimizer step 2 of 4 leaves every parameter and the torch RNG state byte-identical
    to the unscored run: dropout and the checkpointed recompute draw from the same RNG stream, so a draw or a weight
    change in tune_score shows in the parameters after steps 3 and 4."""
    a, rng_a, score = _train(ckpt, 2)
    b, rng_b, none = _train(ckpt, None)
    assert none is None and score is not None and score["dev_tp"] + score["dev_fn"] == 2
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert list(pa) == list(pb)
    differ = [n for n in pa if not torch.equal(pa[n], pb[n])]
    assert not differ, differ
    assert torch.equal(rng_a, rng_b)
    assert a.training and all(m.training for m in a.modules())
    fresh = dict(la.load_checkpoint(ckpt)[0].named_parameters())
    assert any(not torch.equal(pa[n], fresh[n]) for n in pa)                # the fixture really trained


def test_tune_score_restores_the_modes(ckpt: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Train mode after scoring (eval while scoring); with --freeze-encoder the encoder stays in eval mode; also when
    the scoring route raises."""
    seen = []
    real = la.noul_margins

    def spy(model: Any, *a: Any) -> Any:
        seen.append(model.training or any(m.training for m in model.modules()))
        return real(model, *a)
    monkeypatch.setattr(la, "noul_margins", spy)
    model, tok, cfg = _load(ckpt)
    _score(model, tok, cfg)
    assert seen == [False] and all(m.training for m in model.modules())

    model, tok, cfg = _load(ckpt, freeze_encoder=True)
    _score(model, tok, cfg, freeze_encoder=True)
    assert seen == [False, False] and model.training and not model.encoder.training
    assert not any(m.training for m in model.encoder.modules())
    assert all(m.training for n, m in model.named_modules() if n and not n.startswith("encoder"))

    def boom(*a: Any) -> Any:
        raise RuntimeError("scoring failed")
    monkeypatch.setattr(la, "noul_margins", boom)
    for freeze in (False, True):
        model, tok, cfg = _load(ckpt, freeze_encoder=freeze)
        with pytest.raises(RuntimeError, match="scoring failed"):
            _score(model, tok, cfg, freeze_encoder=freeze)
        assert model.training and model.encoder.training is not freeze


def test_tune_score_refuses_non_fp32(ckpt: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """float16 parameters, and autocast on for the scoring device, are refused before any forward and before the
    mode switch."""
    calls = []
    monkeypatch.setattr(la, "noul_margins", lambda *a: calls.append(a))
    model, tok, cfg = _load(ckpt)
    model.half()
    with pytest.raises(SystemExit, match=r"tuning-dev scoring is fp32 on cpu, but parameters are cpu torch\.float16"):
        _score(model, tok, cfg)
    assert model.training
    model, tok, cfg = _load(ckpt)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(SystemExit, match="autocast is on for cpu while scoring tuning-dev"):
            _score(model, tok, cfg)
    assert calls == [] and model.training


def test_tune_score_matches_pilot_tune(ckpt: str) -> None:
    """The keys are pilot.tune.dev_objective's, then dev_s; the values are dev_objective of la.noul_margins on the
    same model and rows."""
    model, tok, cfg = _load(ckpt)
    out = _score(model, tok, cfg)
    assert list(out) == OBJECTIVE_KEYS + ["dev_s"]
    assert isinstance(out["dev_s"], float) and out["dev_s"] >= 0
    model.eval()
    margins = la.noul_margins(model, tok, [r["state"] for r in DEV_ROWS], cfg["max_len"], cfg["head_max_len"],
                              la.uses_parallel_layout(cfg), CPU)
    want = pilot.tune.dev_objective(pilot.tune.labels(DEV_ROWS), margins)
    assert list(want) == OBJECTIVE_KEYS
    assert {k: out[k] for k in OBJECTIVE_KEYS} == want
    assert want["dev_tp"] + want["dev_fn"] == 2 and np.isfinite(want["dev_micro_f1"])

