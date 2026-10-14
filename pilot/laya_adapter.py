"""The one module that uses Laya's private API (LAYA-01, D-13..D-16).

laya.train, laya.common, laya.calibrate (and laya.agent) are not public: `laya-train` and `laya.load` are. This module
is the only tracked .py file that imports them or touches DecisionModel internals (head_checkpointing, the encoder's
gradient_checkpointing_enable, tok_embeddings) or the records_from_labeled logit route; an AST guard in
tests/test_laya_adapter.py fails on any other file, tests included.

Every laya import lives inside a function, so `import pilot.laya_adapter` stays torch-free (the spike orchestrator
and report.py depend on that; `import laya` alone does not import torch, laya.train/common/calibrate do). Callers use
`from pilot import laya_adapter as la` and call `la.forward(...)` through the module attribute, and every wrapper looks
its laya function up at call time, so a test can monkeypatch either side.

Pinned to laya 0.4.6 (LAYA_VERSION). tests/test_laya_adapter.py snapshots the inspect.signature of every wrapped
symbol and runs an offline tiny-model round trip; when the lock moves laya (`make lock`), read laya's changelog before
re-snapshotting.

usage: from pilot import laya_adapter as la; la.check_version(); model, tok, cfg = la.load_checkpoint(base_dir)
"""

LAYA_VERSION = "0.4.6"

# mps_spike.LAYA_INTERNALS (the names the committed spike report records) -> the adapter function covering each (D-14)
COVERS = {"lt.load_checkpoint": "load_checkpoint", "lt.items_from_rows": "items_from_rows",
          "lt.encode_item": "encode_item", "lt._forward": "forward", "lt.rlcd_loss": "rlcd_loss",
          "laya.common.collate_items": "collate_items", "laya.common.uses_parallel_layout": "uses_parallel_layout",
          "laya.common.DecisionModel": "decision_model",
          "DecisionModel.head_checkpointing": "enable_gradient_checkpointing",
          "encoder.gradient_checkpointing_enable": "enable_gradient_checkpointing",
          "encoder.embeddings.tok_embeddings": "freeze_embeddings"}

_TRAIN = ("load_checkpoint", "items_from_rows", "split_calibration", "encode_item", "_forward", "rlcd_loss",
          "soft_ce_loss", "calibration_records", "calibration_report", "save_checkpoint", "sigma_at",
          "draw_option_order", "optimizer_updates", "read_jsonl")
_COMMON = ("collate_items", "uses_parallel_layout", "DecisionModel", "temp_bucket")
_CALIBRATE = ("fit_temperature_map", "records_from_labeled")


def _lt():
    from laya import train
    return train


def _lc():
    from laya import common
    return common


def _lcal():
    from laya import calibrate
    return calibrate


def check_version():
    """Refuse unless the installed laya is exactly LAYA_VERSION (0.4.60, 0.4.6.post1 and 0.4.5 are all refused)."""
    import laya
    if laya.__version__ != LAYA_VERSION:
        raise SystemExit("laya %s installed, the adapter is pinned to %s; run `make env`"
                         % (laya.__version__, LAYA_VERSION))


def wrapped():
    """{snapshot name: laya object} for every laya symbol this module wraps (the contract test's signature set)."""
    import laya
    lt, lc, lcal = _lt(), _lc(), _lcal()
    out = {"laya.train." + n: getattr(lt, n) for n in _TRAIN}
    out.update({"laya.common." + n: getattr(lc, n) for n in _COMMON})
    out.update({"laya.calibrate." + n: getattr(lcal, n) for n in _CALIBRATE})
    out["laya.load"] = laya.load
    return out


# ---- laya.train ----

def load_checkpoint(model_dir):
    return _lt().load_checkpoint(model_dir)


def items_from_rows(tok, rows, max_len, head_max_len):
    return _lt().items_from_rows(tok, rows, max_len, head_max_len)


def split_calibration(items, calib_max, calib_frac, seed):
    return _lt().split_calibration(items, calib_max, calib_frac, seed)


def encode_item(tok, item, max_len, head_max_len, option_order=None, parallel=False):
    return _lt().encode_item(tok, item, max_len, head_max_len, option_order, parallel)


def forward(model, batch, device, amp, detach_encoder):
    return _lt()._forward(model, batch, device, amp, detach_encoder)


def rlcd_loss(logits, target, mask, qtype, sigma):
    """laya's RLCD loss with laya's own defaults named (samples 4, w_sph 0.75, w_rps 1.0), the values ray_train.py
    and mps_spike.py passed positionally before the adapter."""
    return _lt().rlcd_loss(logits, target, mask, qtype, sigma, samples=4, w_sph=0.75, w_rps=1.0)


def soft_ce_loss(logits, target, mask):
    return _lt().soft_ce_loss(logits, target, mask)


def calibration_records(model, tok, items, device, max_len, head_max_len, batch_size=16, parallel=False):
    return _lt().calibration_records(model, tok, items, device, max_len, head_max_len, batch_size=batch_size,
                                     parallel=parallel)


def calibration_report(records, temperature):
    return _lt().calibration_report(records, temperature)


def save_checkpoint(model, tok, cfg, path):
    return _lt().save_checkpoint(model, tok, cfg, path)


def sigma_at(epoch, epochs, start, end):
    return _lt().sigma_at(epoch, epochs, start, end)


def draw_option_order(item, rng, shuffle_types):
    return _lt().draw_option_order(item, rng, shuffle_types)


def optimizer_updates(n_items, micro_batch, grad_accum, epochs):
    return _lt().optimizer_updates(n_items, micro_batch, grad_accum, epochs)


def read_jsonl(path):
    return _lt().read_jsonl(path)


# ---- laya.common ----

def collate_items(batch, pad_id):
    return _lc().collate_items(batch, pad_id)


def uses_parallel_layout(cfg):
    return _lc().uses_parallel_layout(cfg)


def decision_model(encoder, head_layers=2, n_act=2):
    return _lc().DecisionModel(encoder, head_layers, n_act)


# ---- laya.calibrate ----

def fit_temperature_map(records):
    return _lcal().fit_temperature_map(records)


# ---- DecisionModel internals ----

def enable_gradient_checkpointing(model):
    """Gradient checkpointing on the encoder (use_reentrant False) and the Laya head; False (nothing changed) when the
    encoder has no gradient_checkpointing_enable."""
    if not hasattr(model.encoder, "gradient_checkpointing_enable"):
        return False
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    return True


def freeze_embeddings(model):
    """Freeze only the encoder's token embeddings (the spike's embeddings_frozen rung)."""
    model.encoder.embeddings.tok_embeddings.weight.requires_grad_(False)


def freeze_encoder(model):
    """Freeze every encoder parameter (--freeze-encoder: head-only training)."""
    for p in model.encoder.parameters():
        p.requires_grad_(False)


def head_checkpointing_on(model):
    return bool(getattr(model, "head_checkpointing", False))


# ---- scoring: raw noul logits and the temperature laya applies ----

def noul_logits(agent, texts, questions):
    """float32 [len(texts), len(questions)] margins z_true - z_false for 2-option noul questions.

    Goes through laya.calibrate.records_from_labeled, which runs one message per forward, so a row never depends on
    batching. Row i is texts[i], column j is list(questions)[j]. A margin m gives P = sigmoid(m / T), which is laya's
    softmax(z / T)[1] with T = noul_temperature(agent). The targets passed to laya are placeholders (unused here).
    """
    import numpy as np
    ids = list(questions)
    nq = len(ids)
    if not texts:
        return np.zeros((0, nq), dtype=np.float32)
    noul = _lc().QTYPES["noul"]
    recs = _lcal().records_from_labeled(agent, [(t, questions, {q: [0.0, 1.0] for q in ids}) for t in texts])
    if len(recs) != len(texts) * nq:
        raise SystemExit("laya returned %d records for %d messages x %d questions" % (len(recs), len(texts), nq))
    out = np.empty((len(texts), nq), dtype=np.float32)
    for i, (qtype, z, _target, k) in enumerate(recs):
        if qtype != noul or k != 2 or len(z) != 2:
            raise SystemExit("record %d is not a 2-option noul (qtype %s, k %s)" % (i, qtype, k))
        out[i // nq, i % nq] = float(z[1]) - float(z[0])
    return out


def noul_temperature(agent):
    """The temperature laya.load applies to a 2-option noul (already clamped by laya); one scalar for all 48 tags."""
    lc = _lc()
    noul = lc.QTYPES["noul"]
    return float(agent.temperature_by_options.get(lc.temp_bucket(noul, 2), agent.temperature[noul]))
