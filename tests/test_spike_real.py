"""MPS spike numbers on the real selection_train split and the pinned Laya tokenizer (data and weights markers).

The verdict projects an epoch over the neg8 item count and reads its truncation clause from token-length stats; these
tests prove both on the real data: 14,854 neg8 items (3,254 positives + 8 x 1,450), no message over 128 or 256
tokens, and a padded batch that is exactly (micro_batch, max_len) with a full attention mask.
"""
import os
from typing import Any

import pytest

import mps_spike
from pilot import laya_adapter as la
from pilot import pins, spike
from pilot.data import parse_jsonl, verify_split_file

SPLIT = mps_spike.SPLIT


def _rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(manifest entry, rows) of the verified selection_train split."""
    with open(SPLIT, "rb") as f:
        raw = f.read()
    entry = verify_split_file(SPLIT, roles=("train",), data=raw)
    return entry, parse_jsonl(raw)


@pytest.fixture(scope="module")
def laya_dir() -> str:
    from pilot.weights import fetch
    return fetch(pins.LAYA)


@pytest.mark.data
def test_neg8_count_on_real_split() -> None:
    entry, rows = _rows()
    assert len(rows) == entry["n_messages"] == 1450
    assert sum(1 for r in rows for v in r["expected"].values() if v) == 3254
    assert sum(len(r["expected"]) for r in spike.neg8_rows(rows, 0)) == 14854


@pytest.mark.data
@pytest.mark.weights
def test_stats_child_on_real_split(laya_dir: str) -> None:
    entry, _ = _rows()
    payload = mps_spike.launch(mps_spike.stats_cfg(laya_dir, entry["sha256"], 0, 1), "stats")
    assert payload["status"] == "ok", payload
    ts = payload["token_stats"]
    assert ts["n_messages"] == payload["n_messages"] == 1450
    assert ts["pct_messages_over_128"] == 0.0 and ts["pct_messages_over_256"] == 0.0
    assert ts["seq_len"]["max"] < 128
    assert payload["n_neg8_items"] == 14854 and payload["n_positives"] == 3254


@pytest.mark.data
@pytest.mark.weights
def test_padded_batch_is_exactly_mb_by_max_len(laya_dir: str) -> None:
    _, rows = _rows()
    model, tok, cfg = la.load_checkpoint(laya_dir)
    del model
    head_max_len, parallel = cfg.get("head_max_len", 192), la.uses_parallel_layout(cfg)
    for max_len in (128, 256):
        items = la.items_from_rows(tok, spike.neg8_rows(rows[:2], 0), max_len, head_max_len)[0][:4]
        assert len(items) == 4
        batch = la.collate_items([[la.encode_item(tok, spike.pad_item_to(it, max_len), max_len, head_max_len, None,
                                                  parallel) for it in items]], tok.pad_token_id)
        assert tuple(batch["input_ids"].shape) == (4, max_len), (max_len, tuple(batch["input_ids"].shape))
        assert bool(batch["attention_mask"].all())
        assert os.path.isdir(laya_dir)
