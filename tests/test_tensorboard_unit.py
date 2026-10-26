"""pilot/tensorboard.py: the scalar builders (tags and units), the tag-checked write and the lazy writer, offline.

The writer round trip goes through tensorboardX and is read back with tests/tb_events.py (crc-checked, no tensorboard
package); the builders and write are checked on a fake writer that records its add_scalar calls.
"""
import os
from pathlib import Path
from typing import Any

import pytest

import tb_events
from pilot import tensorboard as ptb

pytestmark = pytest.mark.unit


class FakeWriter:
    """Records every add_scalar(tag, value, step) call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.calls.append((tag, value, step))


def test_memory_scalars_convert_bytes_and_keep_none() -> None:
    m = ptb.memory_scalars(3 * ptb.GIB, None, ptb.GIB, 0)
    assert m == {"memory/device_current_gib": 3.0, "memory/mps_driver_gib": None, "memory/rss_gib": 1.0,
                 "memory/swap_used_gib": 0.0}


def test_step_scalars_rate_and_tags() -> None:
    mem = ptb.memory_scalars(None, None, ptb.GIB, 0)
    s = ptb.step_scalars(0.5, None, 1e-4, 2.0, None, "train/items_per_s", 12, 2.0, mem)
    assert s["train/items_per_s"] == 6.0 and "train/messages_per_s" not in s
    assert s["train/loss"] == 0.5 and s["train/lr_head"] == 1e-4 and s["train/grad_norm"] == 2.0
    assert s["train/lr_encoder"] is None and s["train/sigma"] is None
    assert s["memory/rss_gib"] == 1.0 and s["memory/device_current_gib"] is None
    assert set(s) <= set(ptb.TAGS)
    z = ptb.step_scalars(0.5, 2e-5, 1e-4, 2.0, 0.3, "train/messages_per_s", 12, 0.0, mem)
    assert z["train/messages_per_s"] is None and "train/items_per_s" not in z
    assert z["train/lr_encoder"] == 2e-5 and z["train/sigma"] == 0.3


def test_step_scalars_refuse_an_unknown_rate_tag() -> None:
    with pytest.raises(ValueError, match="train/x"):
        ptb.step_scalars(0.5, None, 1e-4, 2.0, None, "train/x", 12, 2.0, {})


def test_epoch_scalars_convert_units() -> None:
    e = ptb.epoch_scalars(0.25, 10.0, 12.5, None, 2 * ptb.GIB, -3 * ptb.MIB)
    assert e == {"epoch/loss": 0.25, "epoch/train_s": 10.0, "epoch/wall_s": 12.5, "epoch/save_s": None,
                 "epoch/peak_total_gib": 2.0, "epoch/swap_delta_mib": -3.0}
    assert set(e) == set(ptb.EPOCH_TAGS)


def test_write_refuses_an_unknown_tag() -> None:
    w = FakeWriter()
    with pytest.raises(ValueError, match="train/text"):
        ptb.write(w, 3, {"train/loss": 1.0, "train/text": 2.0})
    assert w.calls == []                                    # nothing written before the refusal


def test_write_skips_none_in_sorted_order() -> None:
    w = FakeWriter()
    ptb.write(w, 7, {"train/loss": 1, "memory/rss_gib": 0.5, "train/sigma": None, "epoch/loss": 2.0})
    assert w.calls == [("epoch/loss", 2.0, 7), ("memory/rss_gib", 0.5, 7), ("train/loss", 1.0, 7)]
    assert all(type(v) is float for _, v, _ in w.calls)


def test_open_writer_round_trip(tmp_path: Path) -> None:
    logdir = str(tmp_path / "tb" / "rid")
    w = ptb.open_writer(logdir, 3, ptb.segment_suffix(0))
    ptb.write(w, 3, {"train/loss": 0.5})
    ptb.write(w, 4, {"train/loss": 0.25, "epoch/loss": 0.375})
    w.close()
    files = tb_events.event_files(logdir)
    assert len(files) == 1 and files[0].endswith(".segment-000"), files
    events = tb_events.read_events(logdir)
    assert {kind for _, _, kind, _ in events} == {"file_version", "session_log", "train/loss", "epoch/loss"}
    assert [(step, value) for _, step, kind, value in events if kind == "session_log"] == [(3, 1)]
    assert tb_events.scalars(events, "train/loss") == {3: 0.5, 4: 0.25}
    assert tb_events.scalars(events, "epoch/loss") == {4: 0.375}


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a dir the user cannot write")
def test_open_writer_refuses_an_unwritable_dir(tmp_path: Path) -> None:
    parent = tmp_path / "ro"
    parent.mkdir()
    os.chmod(parent, 0o555)
    try:
        with pytest.raises(SystemExit) as e:
            ptb.open_writer(str(parent / "tb"), 1, ptb.segment_suffix(0))
    finally:
        os.chmod(parent, 0o755)
    msg: Any = e.value.code
    assert str(msg).startswith("--tensorboard: cannot write"), msg
